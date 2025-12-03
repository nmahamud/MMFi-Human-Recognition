# action_dataloader.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------
# Helpers
# ---------------------------

FRAME_RE = re.compile(r"^frame(\d+)\.bin$", re.IGNORECASE)

def _parse_segment(seg: str) -> Optional[Tuple[int, int]]:
    """Parse 'start-end' into inclusive integer bounds. Return None if invalid."""
    if not isinstance(seg, str) or "-" not in seg:
        return None
    try:
        a, b = seg.split("-", 1)
        start, end = int(a), int(b)
        if start <= 0 or end <= 0 or end < start:
            return None
        return start, end
    except Exception:
        return None

def _list_frames(session_dir: Path) -> Dict[int, Path]:
    """Map frame index -> file path for all frame*.bin under session_dir."""
    mapping = {}
    for p in session_dir.glob("frame*.bin"):
        m = FRAME_RE.match(p.name)
        if not m:
            continue
        mapping[int(m.group(1))] = p
    return dict(sorted(mapping.items()))

def _read_frame_bin(path: Path, target_feats: int = 4) -> np.ndarray:
    """
    Read one frame*.bin as float32 → (N, target_feats).
    Tries feature counts (4,5,6,3); truncates/pads to target_feats; cleans NaN/Inf and clips extremes.
    """
    raw = np.fromfile(path, dtype=np.float32)
    for feats in (4, 5, 6, 3):
        if raw.size % feats == 0 and raw.size > 0:
            arr = raw.reshape(-1, feats).astype(np.float32, copy=False)
            break
    else:
        feats = 4
        n = raw.size // feats
        arr = raw[: n * feats].reshape(-1, feats).astype(np.float32, copy=False)

    if arr.shape[1] > target_feats:
        arr = arr[:, :target_feats]
    elif arr.shape[1] < target_feats:
        pad = np.zeros((arr.shape[0], target_feats - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)

    arr[~np.isfinite(arr)] = 0.0
    np.clip(arr, -1e4, 1e4, out=arr)
    return arr


# ---------------------------
# 1) INDEX BUILDER
# ---------------------------

def build_segment_index_from_csv(
    root: Path,
    csv_path: Path,
    min_len: int = 5,
    verify_files: bool = True,
    accept_mmwave_subdir: bool = True,
) -> List[Dict]:
    """
    Build a list of segment samples from the cleaned CSV.

    Each sample:
      {
        'env': 'E02', 'subject': 'S14', 'action': 'A22',
        'session_dir': '/.../E02/S14/A22' (or .../A22/mmwave),
        'start': 1, 'end': 18
      }

    - Only keeps segments with length >= min_len
    - Strict: no fallback to full session; verifies requested frames exist (if verify_files=True)
    - Handles both:
        a) First-cell CSV rows like "E02,S14,A22,1-18,19-40,..."
        b) Proper columns: col0=env, col1=subj, col2=act, col3+=segments
    """
    df = pd.read_csv(csv_path, header=None)
    samples: List[Dict] = []

    for _, row in df.iterrows():
        first = row.iloc[0]
        if isinstance(first, str) and first.startswith("E") and "," in first:
            parts = [p.strip() for p in first.split(",") if p.strip()]
            if len(parts) < 3:
                continue
            env, subj, act = parts[0], parts[1], parts[2]
            seg_cols = list(parts[3:]) + [str(x).strip() for x in row.iloc[1:].dropna().tolist()]
        else:
            env, subj, act = row.iloc[:3].tolist()
            if not (isinstance(env, str) and isinstance(subj, str) and isinstance(act, str)):
                continue
            seg_cols = [str(x).strip() for x in row.iloc[3:].dropna().tolist()]

        session_dir = root / env / subj / act
        if accept_mmwave_subdir:
            mmwave_dir = session_dir / "mmwave"
            if mmwave_dir.exists():
                session_dir = mmwave_dir

        for seg in seg_cols:
            bounds = _parse_segment(seg)
            if not bounds:
                continue
            start, end = bounds
            if end - start + 1 < min_len:
                continue

            if verify_files:
                frames = _list_frames(session_dir)
                if not frames:
                    continue
                needed = list(range(start, end + 1))
                if not all(f in frames for f in needed):
                    continue

            samples.append({
                "env": env, "subject": subj, "action": act,
                "session_dir": str(session_dir),
                "start": int(start), "end": int(end),
            })
    return samples


# ---------------------------
# Label maps (action-only primary)
# ---------------------------

def build_action_map(index: List[Dict]) -> Dict[str, int]:
    # Sort Axx numerically if possible
    def _anum(s: str) -> int:
        import re
        m = re.findall(r"\d+", s)
        return int(m[0]) if m else 10**9
    actions = sorted({row["action"] for row in index}, key=_anum)
    return {a: i for i, a in enumerate(actions)}


# ---------------------------
# 2) DATASET (ACTION-ONLY)
# ---------------------------

class ActionSegmentDataset(Dataset):
    """
    Action-only dataset.
    Loads frames for [start, end], applies env normalization (optional),
    pads/truncates points per frame, and **center-pads time** to max_frames.

    Returns:
      {
        'sequence':  FloatTensor (T, P, F)  -- T == max_frames (after center pad/crop)
        'action_id': int
        'action':    str
        'meta':      dict(env, subject, action, start, end, session_dir)
        # optionally 'subject' if include_subject=True (for analysis only)
      }
    """

    def __init__(
        self,
        index: List[Dict],
        action_to_id: Dict[str, int],
        max_frames: Optional[int] = 30,
        max_points: Optional[int] = 150,
        crop_mode: str = "center",             # used when len>max_frames
        normalize: Optional[str] = "per_env_zmuv",  # None | 'per_frame_mean' | 'per_sequence_mean' | 'per_env_zmuv'
        target_feats: int = 4,
        env_stats: Optional[Dict[str, Dict[str, List[float]]]] = None,
        include_subject: bool = False,         # set True if you still want subject string in output
        train: bool = False,                   # enables augmentations if True
        aug_cfg: Optional[Dict] = None,        # {'yaw_deg':8, 'point_jitter':0.02, 'drop_point':0.05, 'drop_frame':0.1}
    ):
        self.index = index
        self.action_to_id = action_to_id
        self.max_frames = max_frames
        self.max_points = max_points
        self.crop_mode = crop_mode
        self.normalize = normalize
        self.target_feats = target_feats
        self.env_stats = env_stats or {}
        self.include_subject = include_subject
        self.train = train
        # default augment config
        self.aug = {
            "yaw_deg": 0.0,
            "point_jitter": 0.0,
            "drop_point": 0.0,
            "drop_frame": 0.0,
        }
        if aug_cfg:
            self.aug.update(aug_cfg)

    def __len__(self):
        return len(self.index)

    # ---------- Normalization helpers ----------
    def _normalize_points(self, arr: np.ndarray) -> np.ndarray:
        if self.normalize is None or arr.size == 0:
            return arr
        if self.normalize == "per_frame_mean":
            arr[:, :3] -= arr[:, :3].mean(axis=0, keepdims=True)
        return arr

    def _apply_env_norm(self, env: str, arr: np.ndarray) -> np.ndarray:
        if self.normalize != "per_env_zmuv" or arr.size == 0:
            return arr
        stats = self.env_stats.get(env)
        if not stats:
            return arr
        mu = np.asarray(stats.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(1, 3)
        std = np.asarray(stats.get("std",  [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 3)
        eps = 1e-6
        arr[:, :3] = (arr[:, :3] - mu) / (std + eps)
        return arr

    # ---------- Augmentations (train only, light & physics-plausible) ----------
    def _yaw_rotate(self, arr: np.ndarray, deg: float) -> np.ndarray:
        if arr.size == 0 or deg == 0.0:
            return arr
        th = np.deg2rad(deg)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        arr[:, :3] = arr[:, :3] @ R.T
        return arr

    def _jitter(self, arr: np.ndarray, sigma: float) -> np.ndarray:
        if arr.size == 0 or sigma <= 0.0:
            return arr
        noise = np.random.normal(0.0, sigma, size=arr[:, :3].shape).astype(np.float32)
        arr[:, :3] += noise
        return arr

    def _drop_points(self, arr: np.ndarray, p: float) -> np.ndarray:
        if arr.size == 0 or p <= 0.0:
            return arr
        keep = np.random.rand(arr.shape[0]) > p
        if keep.any():
            return arr[keep]
        return arr  # avoid empty

    # ---------- Core ----------
    def __getitem__(self, i: int):
        meta = self.index[i]
        env, subj, act = meta["env"], meta["subject"], meta["action"]
        session_dir = Path(meta["session_dir"])
        start, end = int(meta["start"]), int(meta["end"])

        frame_map = _list_frames(session_dir)
        needed = [f for f in range(start, end + 1) if f in frame_map]
        if not needed:
            raise RuntimeError(f"No frames found for {session_dir} [{start}-{end}]")

        # crop if too long (we'll center-pad later if too short)
        if self.max_frames is not None and len(needed) > self.max_frames:
            T = self.max_frames
            if self.crop_mode == "center":
                mid = len(needed) // 2
                half = T // 2
                s = max(0, mid - half)
                needed = needed[s:s+T]
            elif self.crop_mode == "tail":
                needed = needed[-T:]
            else:
                needed = needed[:T]

        frames_np: List[np.ndarray] = []
        for fid in needed:
            arr = _read_frame_bin(frame_map[fid], target_feats=self.target_feats)

            # env normalization (train+val) computed from train stats outside
            if self.normalize == "per_env_zmuv":
                arr = self._apply_env_norm(env, arr)
            elif self.normalize == "per_frame_mean":
                arr = self._normalize_points(arr)
            # per-sequence mean handled after we load all frames

            # train-time augs
            if self.train:
                if self.aug["yaw_deg"] > 0:
                    # sample uniform in [-yaw, +yaw]
                    deg = (np.random.rand() * 2 - 1.0) * self.aug["yaw_deg"]
                    arr = self._yaw_rotate(arr, deg)
                if self.aug["point_jitter"] > 0:
                    arr = self._jitter(arr, self.aug["point_jitter"])
                if self.aug["drop_point"] > 0:
                    arr = self._drop_points(arr, self.aug["drop_point"])

            frames_np.append(arr)

        # per-sequence mean centering (if selected)
        if self.normalize == "per_sequence_mean":
            xyz_list = [a[:, :3] for a in frames_np if a.size > 0 and a.shape[0] > 0]
            mu = np.mean(np.concatenate(xyz_list, axis=0), axis=0, keepdims=True) if xyz_list else np.zeros((1,3), np.float32)
            for k in range(len(frames_np)):
                if frames_np[k].size > 0:
                    frames_np[k][:, :3] -= mu

        # pad/truncate points per frame
        if self.max_points is not None:
            P = self.max_points
            padded = []
            for arr in frames_np:
                if arr.shape[0] >= P:
                    padded.append(arr[:P])
                else:
                    pad = np.zeros((P - arr.shape[0], arr.shape[1]), dtype=np.float32)
                    padded.append(np.vstack([arr, pad]))
            frames_np = padded

        # stack to (T_cur, P, F)
        seq = np.stack(frames_np, axis=0).astype(np.float32) if frames_np else \
              np.zeros((0, self.max_points or 0, self.target_feats), dtype=np.float32)

        # random frame drop (train only), then center pad to max_frames
        if self.train and self.aug["drop_frame"] > 0 and seq.shape[0] > 0:
            keep_mask = (np.random.rand(seq.shape[0]) > self.aug["drop_frame"])
            if keep_mask.any():
                seq = seq[keep_mask]

        if self.max_frames is not None:
            T_cur, P, F = seq.shape
            T_tar = self.max_frames
            if T_cur < T_tar:
                pad_total = T_tar - T_cur
                pad_left  = pad_total // 2
                pad_right = pad_total - pad_left
                left = np.zeros((pad_left,  P, F), dtype=np.float32)
                right = np.zeros((pad_right, P, F), dtype=np.float32)
                seq = np.concatenate([left, seq, right], axis=0)
            elif T_cur > T_tar:
                # final safety crop to center
                mid = T_cur // 2
                half = T_tar // 2
                s = max(0, mid - half)
                seq = seq[s:s+T_tar]

        sample = {
            "sequence": torch.from_numpy(seq),                  # (T, P, F) with T==max_frames
            "action_id": int(self.action_to_id[act]),
            "action": act,
            "meta": {
                "env": env, "subject": subj, "action": act,
                "start": start, "end": end, "session_dir": str(session_dir)
            }
        }
        if self.include_subject:
            sample["subject"] = subj  # optional, not used for training
        return sample
