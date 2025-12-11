#!/usr/bin/env python3
"""
Test script for a two-stage mmWave -> Skeleton -> Action pipeline.

- Stage 1 (PointNet2Regressor): Evaluates MPJPE on predicted skeletons vs. ground-truth.
- Stage 2 (STGCNActionClassifier): Evaluates classification accuracy (optionally using
  Stage 1 predictions or raw ground-truth skeletons).

This script mirrors the shapes/logic used in the provided dataset and training code.
"""

import os
import argparse
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# --------------------------
# Try both dataset module names (your train script imports `dataset`, but the file provided is `dataset_ver4_val.py`)
# --------------------------
from dataset import MMFiDataset


from model_stage1 import PointNet2Regressor, MPJPELoss
from model_stage2 import STGCNActionClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smooth_predictions(skeletons, window_size=5):
    """Same anti-jitter smoothing as training."""
    B, T, V, C = skeletons.shape
    data = skeletons.view(B, T, -1).permute(0, 2, 1)  # [B, 51, T]
    pad = window_size // 2
    smoothed = torch.nn.functional.avg_pool1d(
        data, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False
    )
    smoothed = smoothed[:, :, :T]
    return smoothed.permute(0, 2, 1).view(B, T, V, C)


@torch.no_grad()
def evaluate_stage1(regressor, data_loader):
    """Compute MPJPE for the Stage-1 regressor over the entire loader."""
    regressor.eval()

    mpjpe_fn = MPJPELoss()
    total_loss = 0.0
    total_frames = 0

    # Optional: keep per-action stats
    per_action_sum = defaultdict(float)
    per_action_count = defaultdict(int)

    for mmwave_seq, gt_skel_seq, labels in data_loader:
        # mmwave_seq: [B, T, N, C=5]
        B, T, N, C = mmwave_seq.shape

        inputs = mmwave_seq.view(-1, N, C).to(DEVICE)          # [B*T, N, 5]
        targets = gt_skel_seq.view(-1, 17, 3).to(DEVICE)       # [B*T, 17, 3]

        preds = regressor(inputs)                               # [B*T, 17, 3]
        loss = mpjpe_fn(preds, targets)

        # Accumulate by frames (B*T frames)
        total_loss += loss.item() * (B * T)
        total_frames += (B * T)

        # Per-action logging: one label per sequence -> count sequence once using mean over its frames
        # Compute per-sequence MPJPE
        seq_preds = preds.view(B, T, 17, 3)
        seq_targets = targets.view(B, T, 17, 3)
        seq_loss = torch.norm(seq_preds - seq_targets, dim=-1).mean(dim=(1, 2))  # [B]
        for i in range(B):
            action_id = labels[i].item()
            per_action_sum[action_id] += seq_loss[i].item()
            per_action_count[action_id] += 1

    mean_mpjpe = total_loss / max(total_frames, 1)

    # Build per-action MPJPE (if any)
    per_action = {}
    for k, s in per_action_sum.items():
        c = max(per_action_count[k], 1)
        per_action[k] = s / c

    return mean_mpjpe, per_action


@torch.no_grad()
def evaluate_stage2(classifier, data_loader, regressor=None, use_ground_truth=False, smooth=True):
    """Compute classification accuracy for Stage-2.

    If use_ground_truth=False, we first generate skeletons with Stage-1 regressor.
    """
    classifier.eval()
    if regressor is not None:
        regressor.eval()

    correct = 0
    total = 0

    # Optional: per-action accuracy
    per_action_correct = defaultdict(int)
    per_action_total = defaultdict(int)

    for mmwave_seq, gt_skel_seq, labels in data_loader:
        B, T, N, C = mmwave_seq.shape

        if use_ground_truth:
            skeletons = gt_skel_seq.to(DEVICE)
        else:
            if regressor is None:
                raise ValueError("regressor must be provided when use_ground_truth=False")
            mmwave_input = mmwave_seq.to(DEVICE)               # [B, T, N, 5]
            flat_input = mmwave_input.view(-1, N, C)           # [B*T, N, 5]
            flat_skel = regressor(flat_input)                  # [B*T, 17, 3]
            skeletons = flat_skel.view(B, T, 17, 3)            # [B, T, 17, 3]

        if smooth:
            skeletons = smooth_predictions(skeletons, window_size=5)

        # Center skeletons at joint-0 (consistent with training)

        center_joint = skeletons[:, :, 0:1, :]
        skeletons_rel = skeletons - center_joint               # [B, T, 17, 3]

        outputs = classifier(skeletons_rel)                    # [B, num_classes]
        preds = outputs.argmax(dim=1)
        labels = labels.to(DEVICE)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i in range(labels.size(0)):
            action_id = labels[i].item()
            per_action_total[action_id] += 1
            if preds[i].item() == action_id:
                per_action_correct[action_id] += 1

    acc = 100.0 * correct / max(total, 1)

    # Per-action accuracies
    per_action_acc = {}
    for k, c in per_action_total.items():
        per_action_acc[k] = 100.0 * per_action_correct[k] / max(c, 1)

    return acc, per_action_acc


def make_loader(data_root, csv_file, batch_size=64, num_workers=4, use_sampler=False):
    # Build dataset
    dataset = MMFiDataset(
        data_root=data_root,
        csv_file=csv_file,
        dataframe=None,
        points_per_frame=512
    )

    # Optional weighted sampler (typically for training; off by default for eval)
    if use_sampler:
        weights = dataset.get_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader, dataset


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Evaluation Script")
    parser.add_argument("--mode", type=str, default="both", choices=["stage1", "stage2", "both"],
                        help="What to evaluate.")
    parser.add_argument("--data_root", type=str, default="./MMFi")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="CSV with segments (same format used in training).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stage1_ckpt", type=str, default="stage1_best.pth")
    parser.add_argument("--stage2_ckpt", type=str, default="stage2_best.pth")
    parser.add_argument("--use_gt_for_stage2", action="store_true",
                        help="Evaluate Stage-2 on ground-truth skeletons instead of Stage-1 predictions.")
    parser.add_argument("--no_smooth", action="store_true",
                        help="Disable skeleton smoothing before classification.")

    args = parser.parse_args()

    # Data
    loader, dataset = make_loader(
        data_root=args.data_root,
        csv_file=args.csv_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_sampler=False
    )

    # Label map (action string -> idx) lives inside dataset; we invert for readability if needed
    action_to_idx = dataset.action_to_idx
    idx_to_action = {v: k for k, v in action_to_idx.items()}

    # -------------------
    # Stage 1
    # -------------------
    if args.mode in ("stage1", "both"):
        if not os.path.exists(args.stage1_ckpt):
            raise FileNotFoundError(f"Missing Stage 1 checkpoint: {args.stage1_ckpt}")
        regressor = PointNet2Regressor(num_joints=17).to(DEVICE)
        regressor.load_state_dict(torch.load(args.stage1_ckpt, map_location=DEVICE))
        mean_mpjpe, per_action = evaluate_stage1(regressor, loader)

        print("\n[Stage 1] MPJPE (mean over all frames): {:.4f}".format(mean_mpjpe))
        if len(per_action) > 0:
            print("[Stage 1] Per-Action MPJPE:")
            for k in sorted(per_action.keys()):
                print(f"  - {idx_to_action.get(k, str(k))}: {per_action[k]:.4f}")

    # -------------------
    # Stage 2
    # -------------------
    if args.mode in ("stage2", "both"):
        if not os.path.exists(args.stage2_ckpt):
            raise FileNotFoundError(f"Missing Stage 2 checkpoint: {args.stage2_ckpt}")
        classifier = STGCNActionClassifier(num_classes=len(action_to_idx)).to(DEVICE)
        classifier.load_state_dict(torch.load(args.stage2_ckpt, map_location=DEVICE))

        regressor = None
        if not args.use_gt_for_stage2:
            if not os.path.exists(args.stage1_ckpt):
                raise FileNotFoundError(
                    "Stage-2 evaluation on predictions requires Stage-1 checkpoint. "
                    f"Not found: {args.stage1_ckpt}"
                )
            regressor = PointNet2Regressor(num_joints=17).to(DEVICE)
            regressor.load_state_dict(torch.load(args.stage1_ckpt, map_location=DEVICE))

        acc, per_action_acc = evaluate_stage2(
            classifier,
            loader,
            regressor=regressor,
            use_ground_truth=args.use_gt_for_stage2,
            smooth=(not args.no_smooth)
        )
        print("\n[Stage 2] Overall Accuracy: {:.2f}%".format(acc))
        if len(per_action_acc) > 0:
            print("[Stage 2] Per-Action Accuracy:")
            for k in sorted(per_action_acc.keys()):
                print(f"  - {idx_to_action.get(k, str(k))}: {per_action_acc[k]:.2f}%")


if __name__ == "__main__":
    main()
