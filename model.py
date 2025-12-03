"""
Action-only model for MMFi mmWave segments.
Architecture: PointNet (per frame) -> Temporal ConvNet (dilated 1D convs)
-> Masked attention pooling -> Action classifier.

Input:  x  (B, T, P, F)  with center-padded zeros in time dimension
Output: action_logits (B, num_actions)

Notes:
- Ignores padded frames by using a learned temporal attention with masking.
- Fast, stable, and well-suited for short sequences like T≈25–35.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------ Frame encoder (PointNet-style) ------------
class PointNetFrameEncoder(nn.Module):
    def __init__(self, in_feats: int = 4, out_dim: int = 256, proj_dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_feats, 64, 1),  nn.BatchNorm1d(64),  nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),       nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, out_dim, 1),  nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x_btp: torch.Tensor) -> torch.Tensor:
        # x_btp: (B*T, P, F)
        x = x_btp.transpose(1, 2)          # (B*T, F, P)
        feat = self.mlp(x)                 # (B*T, D, P)
        global_feat = feat.max(dim=2).values   # (B*T, D)
        return self.proj(global_feat)          # (B*T, D)


# ------------ Temporal Conv Block (Res + Dilation) ------------
class ResTCNBlock(nn.Module):
    def __init__(self, dim: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):  # x: (B, D, T)
        y = self.net(x)
        return self.act(x + y)


# ------------ Masked temporal attention pooling ------------
class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, seq, mask):
        """
        seq:  (B, T, D)
        mask: (B, T)  True for real frames, False for padded
        """
        B, T, D = seq.shape
        logits = self.score(seq).squeeze(-1)  # (B, T)

        # very negative for padded positions so they get ~0 weight
        neg_inf = torch.finfo(seq.dtype).min / 2
        logits = logits.masked_fill(~mask, neg_inf)

        attn = torch.softmax(logits, dim=1)   # (B, T)
        attn = attn.unsqueeze(-1)             # (B, T, 1)
        pooled = (seq * attn).sum(dim=1)      # (B, D)
        return pooled, attn.squeeze(-1)


# ------------ Full action model ------------
class ActionNet_PointNet_TCN(nn.Module):
    def __init__(
        self,
        num_actions: int,
        input_features: int = 4,
        frame_dim: int = 256,
        tcn_layers: int = 3,
        tcn_dropout: float = 0.2,
        kernel: int = 3,
    ):
        super().__init__()
        self.frame_enc = PointNetFrameEncoder(input_features, frame_dim, proj_dropout=0.1)

        # stack of dilated ResTCN blocks: dilation = 1,2,4,...
        blocks = []
        for i in range(tcn_layers):
            blocks.append(ResTCNBlock(frame_dim, kernel=kernel, dilation=2**i, dropout=tcn_dropout))
        self.tcn = nn.Sequential(*blocks)

        self.pool = MaskedAttentionPool(frame_dim)

        self.cls = nn.Sequential(
            nn.Linear(frame_dim, frame_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(frame_dim, num_actions),
        )

    @torch.no_grad()
    def _frame_mask(self, x):
        # x: (B, T, P, F); a frame is padded if all P,F are zero
        return (x.abs().sum(dim=(2, 3)) > 0)  # (B, T) True = real

    def forward(self, x):
        """
        x: (B, T, P, F)
        returns: action_logits (B, num_actions)
        """
        B, T, P, F = x.shape
        mask = self._frame_mask(x)                    # (B,T)

        x_btp = x.reshape(B * T, P, F)
        frame_emb = self.frame_enc(x_btp).view(B, T, -1)  # (B,T,D)

        # TCN expects (B, D, T)
        z = frame_emb.transpose(1, 2)                 # (B,D,T)
        z = self.tcn(z)                                # (B,D,T)
        z = z.transpose(1, 2)                          # (B,T,D)

        pooled, _ = self.pool(z, mask)                 # (B,D)
        logits = self.cls(pooled)                      # (B,num_actions)
        return logits
