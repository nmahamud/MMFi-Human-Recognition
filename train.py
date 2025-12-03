"""
Action-only training for MMFi mmWave segments.

Model: PointNet (per-frame) -> Temporal ConvNet (dilated) -> Masked attention pooling -> Action head
Data:  Uses CSV-defined segments only; subject-wise session split; per-environment ZMUV (train-only stats)
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from action_dataloader import (
    build_segment_index_from_csv,
    build_action_map,
    ActionSegmentDataset,
    _list_frames,
    _read_frame_bin,
)

from model import ActionNet_PointNet_TCN


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    root: Path
    csv_path: Path
    outdir: Path = Path("./outputs_action")
    seed: int = 42

    # dataset
    min_len: int = 11
    verify_files: bool = True
    accept_mmwave_subdir: bool = True
    max_frames: int = 31       # center-padded to this T
    max_points: int = 150
    normalize: str = "per_env_zmuv"

    # split
    train_ratio: float = 0.8
    balance_by_action: bool = True

    # training
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 3e-4
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    num_workers: int = 0  # Colab/Windows friendly

    # model
    input_features: int = 4
    frame_dim: int = 256
    tcn_layers: int = 3
    tcn_dropout: float = 0.2
    kernel: int = 3

    # augs (train dataset)
    yaw_deg: float = 8.0
    point_jitter: float = 0.02
    drop_point: float = 0.05
    drop_frame: float = 0.10

    # early stopping
    patience: int = 8


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def group_sessions(index: List[Dict]) -> Dict[Tuple[str, str, str, str], Dict]:
    groups: Dict[Tuple[str,str,str,str], Dict] = {}
    for i, row in enumerate(index):
        key = (row["env"], row["subject"], row["action"], row["session_dir"])
        g = groups.setdefault(key, {"rows": [], "n_segments": 0})
        g["rows"].append(i)
        g["n_segments"] += 1
    return groups


def per_subject_sessions(index: List[Dict]) -> Dict[str, List[Tuple[Tuple[str,str,str,str], Dict]]]:
    sessions = group_sessions(index)
    by_subj: Dict[str, List[Tuple[Tuple[str,str,str,str], Dict]]] = {}
    for key, info in sessions.items():
        _, subj, _, _ = key
        by_subj.setdefault(subj, []).append((key, info))
    return by_subj


def stratified_subject_split(
    index: List[Dict],
    train_ratio: float = 0.8,
    balance_by_action: bool = True,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Session-level split per subject to avoid leakage,
    optionally preserving per-subject action proportions.
    """
    rng = random.Random(seed)
    by_subj = per_subject_sessions(index)

    train_rows: List[int] = []
    val_rows: List[int] = []

    for subj, sess_list in by_subj.items():
        action_to_sessions: Dict[str, List[Tuple[Tuple[str,str,str,str], Dict]]] = {}
        total_segments = 0
        for (key, info) in sess_list:
            act = key[2]
            action_to_sessions.setdefault(act, []).append((key, info))
            total_segments += info["n_segments"]

        for act in action_to_sessions:
            rng.shuffle(action_to_sessions[act])

        target_train_segments = math.ceil(train_ratio * total_segments)

        session_to_rows: Dict[Tuple[str,str,str,str], List[int]] = {}
        for (key, info) in sess_list:
            session_to_rows[key] = info["rows"]

        per_action_total = {act: sum(info["n_segments"] for _, info in action_to_sessions[act])
                            for act in action_to_sessions}
        per_action_target_train = {
            act: round(train_ratio * per_action_total[act]) for act in per_action_total
        }

        action_queues = {act: list(action_to_sessions[act]) for act in action_to_sessions}

        subj_train_rows: List[int] = []
        subj_val_rows: List[int] = []

        while len(subj_train_rows) < target_train_segments and any(action_queues.values()):
            if balance_by_action:
                current_train_counts: Dict[str,int] = {}
                for ridx in subj_train_rows:
                    a = index[ridx]["action"]
                    current_train_counts[a] = current_train_counts.get(a, 0) + 1

                deficits = []
                for act in action_queues:
                    if not action_queues[act]:
                        continue
                    deficit = per_action_target_train[act] - current_train_counts.get(act, 0)
                    deficits.append((deficit, act))

                if not deficits:
                    break
                deficits.sort(reverse=True, key=lambda x: x[0])
                if deficits[0][0] > 0:
                    chosen_act = deficits[0][1]
                else:
                    chosen_act = next((a for a in action_queues if action_queues[a]), None)
                    if chosen_act is None:
                        break
            else:
                non_empty = [a for a in action_queues if action_queues[a]]
                if not non_empty:
                    break
                chosen_act = rng.choice(non_empty)

            sess_key, sess_info = action_queues[chosen_act].pop(0)
            rows = session_to_rows[sess_key]
            subj_train_rows.extend(rows)
            if len(subj_train_rows) >= target_train_segments:
                break

        chosen_train_session_keys = set()
        for (key, info) in sess_list:
            if any(r in subj_train_rows for r in info["rows"]):
                chosen_train_session_keys.add(key)

        for (key, info) in sess_list:
            if key not in chosen_train_session_keys:
                subj_val_rows.extend(info["rows"])

        train_rows.extend(subj_train_rows)
        val_rows.extend(subj_val_rows)

    return train_rows, val_rows


def compute_env_stats_from_index(
    index: List[Dict],
    sample_stride_frames: int = 3,
    max_points_per_frame_for_stats: int = 512,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute per-environment mean/std over xyz from TRAIN ONLY.
    Returns: { 'E01': {'mean':[mx,my,mz], 'std':[sx,sy,sz]}, ... }
    """
    env_sums   = {}
    env_sumsq  = {}
    env_counts = {}

    for row in index:
        env   = row["env"]
        sdir  = Path(row["session_dir"])
        start = int(row["start"])
        end   = int(row["end"])

        frame_map = _list_frames(sdir)
        if not frame_map:
            continue

        fids = [f for f in range(start, end + 1) if f in frame_map]
        fids = fids[::sample_stride_frames] if sample_stride_frames > 1 else fids
        if not fids:
            continue

        for fid in fids:
            arr = _read_frame_bin(frame_map[fid], target_feats=4)
            if arr.size == 0:
                continue
            xyz = arr[:, :3]
            if xyz.shape[0] > max_points_per_frame_for_stats:
                idx = np.random.choice(xyz.shape[0], max_points_per_frame_for_stats, replace=False)
                xyz = xyz[idx]

            if env not in env_sums:
                env_sums[env]   = np.zeros(3, dtype=np.float64)
                env_sumsq[env]  = np.zeros(3, dtype=np.float64)
                env_counts[env] = 0

            env_sums[env]   += xyz.sum(axis=0, dtype=np.float64)
            env_sumsq[env]  += (xyz * xyz).sum(axis=0, dtype=np.float64)
            env_counts[env] += xyz.shape[0]

    env_stats: Dict[str, Dict[str, List[float]]] = {}
    for env in env_counts:
        n   = max(int(env_counts[env]), 1)
        sumv   = env_sums[env]
        sumsq  = env_sumsq[env]
        mean   = sumv / n
        var    = np.maximum(sumsq / n - mean * mean, 0.0)
        std    = np.sqrt(var + 1e-8)
        env_stats[env] = {
            "mean": mean.astype(np.float32).tolist(),
            "std":  std.astype(np.float32).tolist(),
        }
    return env_stats


def evaluate(model, loader, device, label_smoothing: float = 0.0):
    model.eval()
    total = 0
    correct = 0
    ce = nn.CrossEntropyLoss(reduction="sum", label_smoothing=label_smoothing)
    loss_total = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["sequence"].to(device)      # (B,T,P,F)
            y = batch["action_id"].to(device)
            logits = model(x)
            loss = ce(logits, y)
            loss_total += loss.item()
            total += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    return (
        loss_total / max(total, 1),
        100.0 * correct / max(total, 1),
    )


def plot_history(hist, outpath: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(hist["train_loss"], label="Train")
    axes[0].plot(hist["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(True); axes[0].legend()

    axes[1].plot(hist["train_acc"], label="Train")
    axes[1].plot(hist["val_acc"], label="Val")
    axes[1].set_title("Action Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].grid(True); axes[1].legend()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"[Saved] {outpath}")


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = Config(
        root=Path("/content/mmfi_mmwave_data"),                        # <<< EDIT
        csv_path=Path("/content/cleaned_compacted_mmwave_action_segment_manual.csv")  # <<< EDIT
    )
    set_seed(cfg.seed)
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    # 1) Build index from CSV
    index = build_segment_index_from_csv(
        root=cfg.root,
        csv_path=cfg.csv_path,
        min_len=cfg.min_len,
        verify_files=cfg.verify_files,
        accept_mmwave_subdir=cfg.accept_mmwave_subdir,
    )
    if not index:
        raise RuntimeError("No samples found. Check paths/CSV and min_len.")

    # 2) Action encoder
    act2id = build_action_map(index)
    np.save(cfg.outdir / "action_encoder_classes.npy", np.array(list(act2id.keys())))
    num_actions = len(act2id)
    print(f"Actions: {num_actions} | Total samples: {len(index)}")

    # 3) Subject-wise split (session-granular)
    train_rows, val_rows = stratified_subject_split(
        index, train_ratio=cfg.train_ratio, balance_by_action=cfg.balance_by_action, seed=cfg.seed
    )

    # 4) Per-environment stats from TRAIN only
    train_index = [index[i] for i in train_rows]
    env_stats = compute_env_stats_from_index(train_index)
    with open(cfg.outdir / "env_stats.json", "w") as f:
        json.dump(env_stats, f, indent=2)
    print("Per-environment stats:", env_stats)

    # 5) Datasets & loaders
    train_ds = ActionSegmentDataset(
        index=train_index,
        action_to_id=act2id,
        max_frames=cfg.max_frames, max_points=cfg.max_points,
        normalize=cfg.normalize, env_stats=env_stats,
        train=True,
        aug_cfg={"yaw_deg":cfg.yaw_deg, "point_jitter":cfg.point_jitter,
                 "drop_point":cfg.drop_point, "drop_frame":cfg.drop_frame},
    )
    val_ds = ActionSegmentDataset(
        index=[index[i] for i in val_rows],
        action_to_id=act2id,
        max_frames=cfg.max_frames, max_points=cfg.max_points,
        normalize=cfg.normalize, env_stats=env_stats,
        train=False
    )

    # sanity check: fixed T
    _check = train_ds[0]["sequence"].shape
    assert _check[0] == cfg.max_frames, f"Expected T={cfg.max_frames}, got {_check[0]}."

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

    # 6) Model
    model = ActionNet_PointNet_TCN(
        num_actions=num_actions,
        input_features=cfg.input_features,
        frame_dim=cfg.frame_dim,
        tcn_layers=cfg.tcn_layers,
        tcn_dropout=cfg.tcn_dropout,
        kernel=cfg.kernel,
    ).to(cfg.device)

    # 7) Optimizer, scheduler, loss
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    best_epoch = -1

    # 8) Train loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs}", leave=False)
        for batch in pbar:
            x = batch["sequence"].to(cfg.device)  # (B,T,P,F)
            y = batch["action_id"].to(cfg.device)

            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            opt.step()

            total += x.size(0)
            loss_sum += loss.item()
            correct += (logits.argmax(1) == y).sum().item()

            pbar.set_postfix(loss=loss.item(), acc=100.0*correct/max(total,1))

        train_loss = loss_sum / max(total, 1)
        train_acc  = 100.0 * correct / max(total, 1)

        # Val
        val_loss, val_acc = evaluate(model, val_loader, cfg.device, label_smoothing=cfg.label_smoothing)

        # Step LR
        sched.step()

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"Train: loss {train_loss:.4f}, A {train_acc:.2f}% | "
              f"Val: loss {val_loss:.4f}, A {val_acc:.2f}%")

        # Checkpoint & early stopping
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            (cfg.outdir / "models").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg.outdir / "models/best_model_action.pth")
            print("[Saved] best_model_action.pth")

        if epoch - best_epoch >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {cfg.patience} epochs).")
            break

    # 9) Final artifacts
    torch.save(model.state_dict(), cfg.outdir / "models/final_model_action.pth")
    plot_history(history, cfg.outdir / "training_history_action.png")

    with open(cfg.outdir / "model_config_action.json", "w") as f:
        json.dump({
            "num_actions": num_actions,
            "input_features": cfg.input_features,
            "frame_dim": cfg.frame_dim,
            "tcn_layers": cfg.tcn_layers,
            "tcn_dropout": cfg.tcn_dropout,
            "kernel": cfg.kernel,
            "max_frames": cfg.max_frames,
            "max_points": cfg.max_points,
            "normalize": cfg.normalize,
            "label_smoothing": cfg.label_smoothing,
            "weight_decay": cfg.weight_decay,
            "lr": cfg.lr,
        }, f, indent=2)

    print("\nDone!")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Outputs in: {cfg.outdir.resolve()}")


if __name__ == "__main__":
    main()
