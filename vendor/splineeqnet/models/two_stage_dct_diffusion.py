#!/usr/bin/env python3
"""
Two-stage DCT coarse + conditional diffusion refiner (TIME DOMAIN diffusion) with v-pred.

Drop-in replacement for `two_stage_dct_diffusion.py`, keeping the same public API:

- TwoStageDCTDiffusionConfig
- TwoStageDCTDiffusionForecaster with:
    - .coarse(history) -> (B, T_out, N, 3)
    - .diffusion_loss(history, future_gt) -> (loss, coarse_future)
    - .predict(history, deterministic=True, seed=0) -> refined (B, T_out, N, 3)
    - .forward(history): returns coarse in train mode, refined in eval mode

Stage A (coarse): predicts low-frequency DCT content and reconstructs coarse future in time domain.
Stage B (refiner): conditional diffusion over time-domain residual r = y_gt - y_coarse.
Diffusion is trained and sampled directly in residual space:
    x0 = r

Denoiser predicts v (v-pred):
    x_t = sqrt(ab) * x0 + sqrt(1-ab) * eps
    v   = sqrt(ab) * eps - sqrt(1-ab) * x0
This avoids the unstable division by sqrt(ab) used by epsilon-pred when computing x0 at high noise.

This file assumes your project already provides:
- models.simlpe_dct.SiMLPeConfig, SiMLPeBackbone, _build_dct_matrix
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from models.simlpe_dct import SiMLPeConfig, SiMLPeBackbone, _build_dct_matrix


# ----------------------------- Small utilities --------------------------------

def _assert_shape(t: torch.Tensor, shape: Tuple[Optional[int], ...], name: str) -> None:
    if t.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dims, got {t.ndim} with shape {tuple(t.shape)}")
    for i, s in enumerate(shape):
        if s is not None and int(t.shape[i]) != int(s):
            raise ValueError(f"{name} dim {i} expected {s}, got {t.shape[i]}")

def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    if timesteps.ndim != 1:
        timesteps = timesteps.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.size(0), 1, device=emb.device, dtype=emb.dtype)], dim=1)
    return emb

def _make_ddim_timesteps(T: int, steps: int, device: torch.device) -> torch.Tensor:
    """
    Make a strictly decreasing integer timestep schedule with no duplicates.
    Ensures inclusion of T-1 and 0.
    """
    if steps <= 1:
        return torch.tensor([T - 1], device=device, dtype=torch.long)
    base = torch.linspace(0, T - 1, steps, device=device)
    idx = torch.round(base).long()
    idx = torch.cat([idx, torch.tensor([0, T - 1], device=device, dtype=torch.long)], dim=0)
    idx = torch.unique(idx, sorted=True)
    idx = torch.flip(idx, dims=[0])  # descending
    return idx


# ----------------------------- Configs ----------------------------------------

@dataclass
class TwoStageDCTDiffusionConfig:
    input_length: int
    pred_length: int
    num_nodes: int

    # Coarse backbone (siMLPe-like)
    hidden_dim: int
    num_layers: int
    simlpe_use_norm: bool = True
    simlpe_spatial_fc_only: bool = False
    simlpe_mix_spatial_temporal: bool = False
    simlpe_norm_axis: str = "spatial"
    simlpe_add_last_offset: bool = True

    # DCT split (still used by coarse)
    k_low: int = 16

    # Diffusion
    diffusion_steps: int = 100
    ddim_steps: int = 50
    beta_schedule: str = "cosine"  # "cosine" or "linear"

    # Denoiser architecture
    denoiser_dim: int = 256
    denoiser_depth: int = 6
    denoiser_heads: int = 8
    dropout: float = 0.0

    # Training
    freeze_coarse: bool = True
    stopgrad_coarse_condition: bool = True
    cond_use_history: bool = True
    cond_use_coarse: bool = True
    allow_no_conditioning: bool = False
    coarse_target_lowpass_only: bool = False
    diffusion_loss_type: str = "mahalanobis_mse"

    # Sampling stabilizers
    x0_clip: float = 0  # clip x0_hat in normalized space (0 disables)

    # Wrist-anchored covariance knobs
    mobility_palm_var: float = 0.15
    mobility_depth1_var: float = 0.35
    mobility_depth2_var: float = 0.70
    mobility_depth3plus_var: float = 1.00
    graph_edge_strength: float = 0.6
    graph_two_hop_strength: float = 0.2
    covariance_jitter: float = 1e-4

    @property
    def total_length(self) -> int:
        return int(self.input_length + self.pred_length)

    @property
    def feature_dim(self) -> int:
        return int(self.num_nodes * 3)

    @property
    def k_high(self) -> int:
        return int(max(0, self.total_length - self.k_low))


# ----------------------------- Stage A ----------------------------------------

class CoarseDCTForecaster(nn.Module):
    """Predict low-frequency DCT coefficients and reconstruct a coarse future."""

    def __init__(self, cfg: TwoStageDCTDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        T = cfg.total_length
        if not (1 <= cfg.k_low <= T):
            raise ValueError(f"k_low must be in [1, {T}], got {cfg.k_low}")

        dct = _build_dct_matrix(T)
        idct = dct.t()
        self.register_buffer("_dct_low", dct[: cfg.k_low, :], persistent=False)
        self.register_buffer("_idct_low", idct[:, : cfg.k_low], persistent=False)

        sim_cfg = SiMLPeConfig(
            input_length=cfg.input_length,
            pred_length=cfg.pred_length,
            num_nodes=cfg.num_nodes,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            use_norm=cfg.simlpe_use_norm,
            use_spatial_fc_only=cfg.simlpe_spatial_fc_only,
            mix_spatial_temporal=cfg.simlpe_mix_spatial_temporal,
            norm_axis=cfg.simlpe_norm_axis,
            add_last_offset=cfg.simlpe_add_last_offset,
        )
        self.backbone = SiMLPeBackbone(sim_cfg)

    def _pad_total(self, history: torch.Tensor) -> torch.Tensor:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        B = history.size(0)
        flat = history.reshape(B, self.cfg.input_length, -1)
        last = flat[:, -1:, :].expand(-1, self.cfg.pred_length, -1)
        padded = torch.cat([flat, last], dim=1)
        return padded

    def predict_low_coeffs(self, history: torch.Tensor) -> torch.Tensor:
        padded = self._pad_total(history)
        dct_low = self._dct_low.unsqueeze(0).to(device=padded.device, dtype=padded.dtype)
        low_coeffs = torch.matmul(dct_low, padded)

        B, _, F = low_coeffs.shape
        coeffs_full = torch.zeros(B, self.cfg.total_length, F, device=padded.device, dtype=padded.dtype)
        coeffs_full[:, : self.cfg.k_low, :] = low_coeffs

        pred_full = self.backbone(coeffs_full)
        pred_low = pred_full[:, : self.cfg.k_low, :]
        return pred_low

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        pred_low = self.predict_low_coeffs(history)
        idct_low = self._idct_low.unsqueeze(0).to(device=pred_low.device, dtype=pred_low.dtype)
        seq_low = torch.matmul(idct_low, pred_low)

        future = seq_low[:, -self.cfg.pred_length :, :].reshape(
            history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3
        )
        if self.cfg.simlpe_add_last_offset:
            future = future + history[:, -1:, :, :]
        return future

    def lowpass_future_target(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        _assert_shape(future, (history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3), "future")
        full = torch.cat([history, future], dim=1).reshape(history.size(0), self.cfg.total_length, -1)
        dct_low = self._dct_low.unsqueeze(0).to(device=full.device, dtype=full.dtype)
        idct_low = self._idct_low.unsqueeze(0).to(device=full.device, dtype=full.dtype)
        seq_low = torch.matmul(idct_low, torch.matmul(dct_low, full))
        future_low = seq_low[:, -self.cfg.pred_length :, :].reshape(
            history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3
        )
        if self.cfg.simlpe_add_last_offset:
            future_low = future_low + history[:, -1:, :, :]
        return future_low


# ----------------------------- Diffusion scheduler ----------------------------

class DiffusionSchedule:
    def __init__(self, steps: int, schedule: str = "cosine", device: Optional[torch.device] = None) -> None:
        if steps <= 1:
            raise ValueError("diffusion steps must be > 1")
        self.steps = int(steps)
        self.schedule = str(schedule)
        self.device = device

        betas = self._make_betas(self.steps, self.schedule)
        self.betas = betas.to(device=device)
        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.alpha_bars = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @staticmethod
    def _make_betas(T: int, schedule: str) -> torch.Tensor:
        if schedule == "linear":
            beta_start = 1e-4
            beta_end = 2e-2
            return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        if schedule == "cosine":
            s = 0.008
            steps = T
            x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 1e-8, 0.999).float()
            return betas
        raise ValueError(f"Unknown beta schedule: {schedule}")


# ----------------------------- Denoiser (factorized ST with joint tokens) -----

class _FiLM(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, dim)
        self.to_shift = nn.Linear(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shape = [cond.size(0)] + [1] * (x.ndim - 2) + [cond.size(-1)]
        scale = self.to_scale(cond).view(*shape)
        shift = self.to_shift(cond).view(*shape)
        return x * (1.0 + scale) + shift


class _MLP(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SpatialSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        y = self.norm1(x).reshape(B * T, N, D)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y.reshape(B, T, N, D)
        x = x + self.mlp(self.norm2(x))
        return x


class _TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        y = self.norm1(x).permute(0, 2, 1, 3).reshape(B * N, T, D)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = y.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class _TemporalCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, T_out, N, D = x.shape
        _, T_mem, N_mem, D_mem = memory.shape
        if N_mem != N or D_mem != D:
            raise ValueError(
                f"memory shape mismatch: expected joint dim {N} and feature dim {D}, "
                f"got {(B, T_mem, N_mem, D_mem)}"
            )
        if T_mem == 0:
            x = x + self.mlp(self.norm2(x))
            return x
        q = self.norm_q(x).permute(0, 2, 1, 3).reshape(B * N, T_out, D)
        m = self.norm_m(memory).permute(0, 2, 1, 3).reshape(B * N, T_mem, D)
        y, _ = self.attn(q, m, m, need_weights=False)
        y = y.reshape(B, N, T_out, D).permute(0, 2, 1, 3)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class _FactorizedSTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.film_spatial = _FiLM(dim, dim)
        self.spatial = _SpatialSelfAttentionBlock(dim, n_heads, dropout)

        self.film_temporal = _FiLM(dim, dim)
        self.temporal = _TemporalSelfAttentionBlock(dim, n_heads, dropout)

        self.film_cross = _FiLM(dim, dim)
        self.cross = _TemporalCrossAttentionBlock(dim, n_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = self.film_spatial(x, t_emb)
        x = self.spatial(x)
        x = self.film_temporal(x, t_emb)
        x = self.temporal(x)
        x = self.film_cross(x, t_emb)
        x = self.cross(x, memory)
        return x


class MRTimeTransformerDenoiser(nn.Module):
    """Factorized spatio-temporal denoiser with explicit joint tokens."""

    def __init__(
        self,
        *,
        in_feat: int,
        t_in: int,
        t_out: int,
        d_model: int,
        depth: int,
        n_heads: int,
        dropout: float,
        cond_use_history: bool = True,
        cond_use_coarse: bool = True,
        allow_no_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.in_feat = int(in_feat)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.d_model = int(d_model)
        self.cond_use_history = bool(cond_use_history)
        self.cond_use_coarse = bool(cond_use_coarse)
        self.allow_no_conditioning = bool(allow_no_conditioning)

        if self.in_feat % 3 != 0:
            raise ValueError(f"in_feat must be divisible by 3, got {self.in_feat}")
        self.num_nodes = self.in_feat // 3

        self.proj_q = nn.Linear(3, d_model)
        self.proj_hist = nn.Linear(3, d_model)
        self.proj_coarse = nn.Linear(3, d_model)
        self.proj_mamp = nn.LazyLinear(d_model)

        self.pos_q_time = nn.Parameter(torch.zeros(1, t_out, 1, d_model))
        self.pos_hist_time = nn.Parameter(torch.zeros(1, t_in, 1, d_model))
        self.pos_coarse_time = nn.Parameter(torch.zeros(1, t_out, 1, d_model))
        self.pos_joint = nn.Parameter(torch.zeros(1, 1, self.num_nodes, d_model))
        self.pos_mamp = nn.Parameter(torch.zeros(1, 1, d_model))

        self.t_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mamp_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, depth // 2))

        self.blocks = nn.ModuleList([
            _FactorizedSTBlock(d_model, n_heads, dropout) for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 3)

        nn.init.normal_(self.pos_q_time, std=0.02)
        nn.init.normal_(self.pos_hist_time, std=0.02)
        nn.init.normal_(self.pos_coarse_time, std=0.02)
        nn.init.normal_(self.pos_joint, std=0.02)
        nn.init.normal_(self.pos_mamp, std=0.02)

    def _flat_to_joint_tokens(self, x: torch.Tensor, T: int, name: str) -> torch.Tensor:
        B = x.size(0)
        _assert_shape(x, (B, T, self.in_feat), name)
        return x.reshape(B, T, self.num_nodes, 3)

    def _encode_mamp_per_joint(
        self,
        mamp_feat: Optional[torch.Tensor],
        *,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if mamp_feat is None:
            return None
        if int(mamp_feat.shape[0]) != B:
            raise ValueError(f"mamp_feat batch mismatch: expected {B}, got {mamp_feat.shape[0]}")
        if mamp_feat.ndim == 2:
            token = self.proj_mamp(mamp_feat.to(device=device, dtype=dtype)).unsqueeze(1) + self.pos_mamp
            token = token.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
            return token
        if mamp_feat.ndim == 3:
            if int(mamp_feat.shape[1]) == self.num_nodes:
                joint_tok = self.proj_mamp(mamp_feat.to(device=device, dtype=dtype))
                return joint_tok.unsqueeze(1)
            tok = self.proj_mamp(mamp_feat.to(device=device, dtype=dtype))
            tok = tok + self.pos_mamp
            tok = self.mamp_encoder(tok)
            tok = tok.mean(dim=1, keepdim=True)
            tok = tok.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
            return tok
        raise ValueError(
            f"mamp_feat must be (B, D), (B, K, D), or (B, N, D), got {tuple(mamp_feat.shape)}"
        )

    def forward(
        self,
        x_noisy: torch.Tensor,        # (B, T_out, F)
        timesteps: torch.Tensor,      # (B,)
        history: torch.Tensor,        # (B, T_in, F)
        coarse_future: torch.Tensor,  # (B, T_out, F)
        mamp_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x_noisy.size(0)
        _assert_shape(x_noisy, (B, self.t_out, self.in_feat), "x_noisy")
        _assert_shape(history, (B, self.t_in, self.in_feat), "history_flat")
        _assert_shape(coarse_future, (B, self.t_out, self.in_feat), "coarse_future_flat")

        x_noisy_j = self._flat_to_joint_tokens(x_noisy, self.t_out, "x_noisy")
        history_j = self._flat_to_joint_tokens(history, self.t_in, "history_flat")
        coarse_j = self._flat_to_joint_tokens(coarse_future, self.t_out, "coarse_future_flat")

        x = self.proj_q(x_noisy_j) + self.pos_q_time + self.pos_joint

        memory_parts = []
        if self.cond_use_history:
            h_hist = self.proj_hist(history_j) + self.pos_hist_time + self.pos_joint
            memory_parts.append(h_hist)
        if self.cond_use_coarse:
            h_coarse = self.proj_coarse(coarse_j) + self.pos_coarse_time + self.pos_joint
            memory_parts.append(h_coarse)

        if not memory_parts and mamp_feat is None and not self.allow_no_conditioning:
            raise ValueError("No conditioning enabled.")

        if memory_parts:
            memory = torch.cat(memory_parts, dim=1)
        else:
            memory = torch.zeros(B, 0, self.num_nodes, self.d_model, device=x.device, dtype=x.dtype)

        mamp_joint = self._encode_mamp_per_joint(mamp_feat, B=B, device=x.device, dtype=x.dtype)
        if mamp_joint is not None:
            x = x + mamp_joint
            if memory.size(1) > 0:
                memory = memory + mamp_joint.expand(-1, memory.size(1), -1, -1)

        te = _timestep_embedding(timesteps, self.d_model).to(dtype=x.dtype, device=x.device)
        te = self.t_embed(te)

        for block in self.blocks:
            x = block(x, memory, te)

        v_hat = self.out(self.out_norm(x))
        v_hat = v_hat.reshape(B, self.t_out, self.in_feat)
        return v_hat


# ----------------------------- Hand covariance builder ------------------------

class HandKinematicCovariance:
    def __init__(
        self,
        *,
        num_nodes: int,
        wrist_index: int,
        edges: Iterable[Tuple[int, int]],
        palm_var: float,
        depth1_var: float,
        depth2_var: float,
        depth3plus_var: float,
        edge_strength: float,
        two_hop_strength: float,
        jitter: float,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.wrist_index = int(wrist_index)
        self.edges = [(int(i), int(j)) for i, j in edges]
        self.palm_var = float(palm_var)
        self.depth1_var = float(depth1_var)
        self.depth2_var = float(depth2_var)
        self.depth3plus_var = float(depth3plus_var)
        self.edge_strength = float(edge_strength)
        self.two_hop_strength = float(two_hop_strength)
        self.jitter = float(jitter)

        if not (0 <= self.wrist_index < self.num_nodes):
            raise ValueError(f"Invalid wrist_index={self.wrist_index} for num_nodes={self.num_nodes}")

        self.free_indices = [i for i in range(self.num_nodes) if i != self.wrist_index]
        self.orig_to_free = {orig: k for k, orig in enumerate(self.free_indices)}
        self.num_free_nodes = len(self.free_indices)

    def _make_adjacency_full(self) -> List[List[int]]:
        adj = [[] for _ in range(self.num_nodes)]
        for i, j in self.edges:
            if not (0 <= i < self.num_nodes and 0 <= j < self.num_nodes):
                raise ValueError(f"Edge {(i, j)} out of bounds for num_nodes={self.num_nodes}")
            if j not in adj[i]:
                adj[i].append(j)
            if i not in adj[j]:
                adj[j].append(i)
        return adj

    def _dist_from_wrist(self) -> List[int]:
        adj = self._make_adjacency_full()
        dist = [-1] * self.num_nodes
        q: deque[int] = deque([self.wrist_index])
        dist[self.wrist_index] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if any(d < 0 for d in dist):
            raise ValueError("Hand graph is disconnected from the wrist.")
        return dist

    def _node_variances_free(self) -> torch.Tensor:
        dist = self._dist_from_wrist()
        vars_free: List[float] = []
        for orig_idx in self.free_indices:
            d = dist[orig_idx]
            if d <= 1:
                vars_free.append(self.palm_var)
            elif d == 2:
                vars_free.append(self.depth1_var)
            elif d == 3:
                vars_free.append(self.depth2_var)
            else:
                vars_free.append(self.depth3plus_var)
        return torch.tensor(vars_free, dtype=torch.float32)

    def _free_graph_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        Nf = self.num_free_nodes
        A1 = torch.zeros(Nf, Nf, dtype=torch.float32)
        full_adj = self._make_adjacency_full()
        for i_orig in self.free_indices:
            i_free = self.orig_to_free[i_orig]
            for j_orig in full_adj[i_orig]:
                if j_orig == self.wrist_index:
                    continue
                j_free = self.orig_to_free[j_orig]
                A1[i_free, j_free] = 1.0
        A1 = torch.maximum(A1, A1.t())
        A1.fill_diagonal_(0.0)
        A2 = ((A1 @ A1) > 0).float()
        A2 = A2 - A1
        A2.fill_diagonal_(0.0)
        A2 = torch.clamp(A2, min=0.0, max=1.0)
        A2 = torch.maximum(A2, A2.t())
        return A1, A2

    def build_feature_covariance(self) -> torch.Tensor:
        Nf = self.num_free_nodes
        if Nf <= 0:
            raise ValueError("No free joints available after removing wrist.")
        node_vars = self._node_variances_free()
        D_half = torch.diag(torch.sqrt(node_vars))
        A1, A2 = self._free_graph_matrices()
        C = torch.eye(Nf, dtype=torch.float32)
        if self.edge_strength != 0.0:
            C = C + self.edge_strength * A1
        if self.two_hop_strength != 0.0:
            C = C + self.two_hop_strength * A2
        C = 0.5 * (C + C.t())
        Sigma_node = D_half @ C @ D_half
        Sigma_node = 0.5 * (Sigma_node + Sigma_node.t())
        eigvals = torch.linalg.eigvalsh(Sigma_node)
        min_eig = float(eigvals.min().item())
        if min_eig <= 0.0:
            Sigma_node = Sigma_node + (abs(min_eig) + self.jitter) * torch.eye(Nf, dtype=torch.float32)
        else:
            Sigma_node = Sigma_node + self.jitter * torch.eye(Nf, dtype=torch.float32)
        tr = float(torch.trace(Sigma_node).item())
        # Sigma_node = Sigma_node * (Nf / max(tr, 1e-12))
        Sigma_feat = torch.kron(Sigma_node, torch.eye(3, dtype=torch.float32))
        Sigma_feat = 0.5 * (Sigma_feat + Sigma_feat.t())
        eigvals_feat = torch.linalg.eigvalsh(Sigma_feat)
        min_eig_feat = float(eigvals_feat.min().item())
        if min_eig_feat <= 0.0:
            Sigma_feat = Sigma_feat + (abs(min_eig_feat) + self.jitter) * torch.eye(3 * Nf, dtype=torch.float32)
        return Sigma_feat


# ----------------------------- Diffusion module -------------------------------

class TimeResidualConditionalDiffusion(nn.Module):
    def __init__(self, cfg: TwoStageDCTDiffusionConfig, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.schedule = DiffusionSchedule(cfg.diffusion_steps, cfg.beta_schedule)

        metadata = dict(metadata or {})
        if "wrist_index" not in metadata:
            raise ValueError("Missing required metadata['wrist_index'].")
        self.wrist_index = int(metadata["wrist_index"])
        edges_raw = metadata.get("edges", ())
        self.edges = [(int(i), int(j)) for i, j in edges_raw] if edges_raw else []

        if not (0 <= self.wrist_index < cfg.num_nodes):
            raise ValueError(f"Invalid wrist_index={self.wrist_index} for num_nodes={cfg.num_nodes}")
        self.free_indices = [i for i in range(cfg.num_nodes) if i != self.wrist_index]
        self.num_free_nodes = len(self.free_indices)
        self.free_feature_dim = 3 * self.num_free_nodes

        if self.edges and self.num_free_nodes > 0:
            cov_builder = HandKinematicCovariance(
                num_nodes=cfg.num_nodes,
                wrist_index=self.wrist_index,
                edges=self.edges,
                palm_var=cfg.mobility_palm_var,
                depth1_var=cfg.mobility_depth1_var,
                depth2_var=cfg.mobility_depth2_var,
                depth3plus_var=cfg.mobility_depth3plus_var,
                edge_strength=cfg.graph_edge_strength,
                two_hop_strength=cfg.graph_two_hop_strength,
                jitter=cfg.covariance_jitter,
            )
            cov = cov_builder.build_feature_covariance()
        else:
            cov = torch.eye(self.free_feature_dim, dtype=torch.float32)

        chol = torch.linalg.cholesky(cov)
        inv_chol = torch.inverse(chol)
        self.register_buffer("cov_base", cov.to(torch.float32), persistent=True)
        self.register_buffer("cov_base_chol", chol.to(torch.float32), persistent=True)
        self.register_buffer("cov_base_inv_chol", inv_chol.to(torch.float32), persistent=True)
        self.denoiser = MRTimeTransformerDenoiser(
            in_feat=self.free_feature_dim,
            t_in=cfg.input_length,
            t_out=cfg.pred_length,
            d_model=cfg.denoiser_dim,
            depth=cfg.denoiser_depth,
            n_heads=cfg.denoiser_heads,
            dropout=cfg.dropout,
            cond_use_history=cfg.cond_use_history,
            cond_use_coarse=cfg.cond_use_coarse,
            allow_no_conditioning=cfg.allow_no_conditioning,
        )

    def _select_free_joints(self, x: torch.Tensor, T: int, name: str) -> torch.Tensor:
        _assert_shape(x, (None, T, self.cfg.num_nodes, 3), name)
        return x[:, :, self.free_indices, :]

    def _flatten_free(self, x: torch.Tensor, T: int, name: str) -> torch.Tensor:
        x_free = self._select_free_joints(x, T, name)
        return x_free.reshape(x_free.size(0), T, -1)

    def _restore_full_from_free(self, x_free_flat: torch.Tensor, T: int, dtype: torch.dtype) -> torch.Tensor:
        B = x_free_flat.size(0)
        _assert_shape(x_free_flat, (B, T, self.free_feature_dim), "x_free_flat")
        x_free = x_free_flat.reshape(B, T, self.num_free_nodes, 3)
        full = torch.zeros(B, T, self.cfg.num_nodes, 3, device=x_free.device, dtype=dtype)
        full[:, :, self.free_indices, :] = x_free
        return full

    def _schedule_tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.schedule.alpha_bars.to(device=device, dtype=dtype),
            self.schedule.sqrt_alpha_bars.to(device=device, dtype=dtype),
            self.schedule.sqrt_one_minus_alpha_bars.to(device=device, dtype=dtype),
        )

    def training_loss(
        self,
        *,
        residual_gt: torch.Tensor,
        history_full: torch.Tensor,
        coarse_future_full: torch.Tensor,
        mamp_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = residual_gt.size(0)
        _assert_shape(residual_gt, (B, self.cfg.pred_length, self.cfg.num_nodes, 3), "residual_gt")
        _assert_shape(history_full, (B, self.cfg.input_length, self.cfg.num_nodes, 3), "history_full")
        _assert_shape(coarse_future_full, (B, self.cfg.pred_length, self.cfg.num_nodes, 3), "coarse_future_full")

        residual_gt_free = self._flatten_free(residual_gt, self.cfg.pred_length, "residual_gt")
        history_free = self._flatten_free(history_full, self.cfg.input_length, "history_full")
        coarse_free = self._flatten_free(coarse_future_full, self.cfg.pred_length, "coarse_future_full")
        x0 = residual_gt_free.float()
        device = residual_gt_free.device
        _, sqrt_alpha_bars, sqrt_one_minus_alpha_bars = self._schedule_tensors(device, x0.dtype)

        t = torch.randint(0, self.cfg.diffusion_steps, (B,), device=device, dtype=torch.long)
        eps = torch.randn_like(x0)
        chol = self.cov_base_chol.to(device=device, dtype=x0.dtype)
        inv_chol = self.cov_base_inv_chol.to(device=device, dtype=x0.dtype)
        eta = torch.matmul(eps, chol.t())
        sqrt_ab = sqrt_alpha_bars[t].view(B, 1, 1)
        sqrt_omb = sqrt_one_minus_alpha_bars[t].view(B, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_omb * eta
        v = sqrt_ab * eta - sqrt_omb * x0

        v_hat = self.denoiser(
            x_t.to(dtype=history_free.dtype),
            t,
            history_free,
            coarse_free,
            mamp_feat=mamp_feat,
        ).float()

        err = v_hat - v
        if self.cfg.diffusion_loss_type == "mahalanobis_mse":
            err_w = torch.matmul(err, inv_chol.t())
            v_loss = torch.mean(err_w ** 2)
        elif self.cfg.diffusion_loss_type == "mse":
            v_loss = torch.mean(err ** 2)
        else:
            raise ValueError(
                f"Unsupported diffusion_loss_type={self.cfg.diffusion_loss_type!r}. "
                "Expected 'mahalanobis_mse' or 'mse'."
            )
        return v_loss

    @torch.no_grad()
    def sample_ddim(
        self,
        *,
        history_full: torch.Tensor,
        coarse_future_full: torch.Tensor,
        mamp_feat: Optional[torch.Tensor] = None,
        seed: int = 0,
        steps: Optional[int] = None,
        eta: float = 0.0,
        return_score: bool = False,
    ) -> torch.Tensor:
        if steps is None:
            steps = self.cfg.ddim_steps
        steps = int(steps)
        if steps <= 0:
            raise ValueError("DDIM steps must be positive")

        B = history_full.size(0)
        device = history_full.device
        _assert_shape(history_full, (B, self.cfg.input_length, self.cfg.num_nodes, 3), "history_full")
        _assert_shape(coarse_future_full, (B, self.cfg.pred_length, self.cfg.num_nodes, 3), "coarse_future_full")

        history_free = self._flatten_free(history_full, self.cfg.input_length, "history_full")
        coarse_free = self._flatten_free(coarse_future_full, self.cfg.pred_length, "coarse_future_full")

        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        x = torch.randn(
            (B, self.cfg.pred_length, self.free_feature_dim),
            device=device, dtype=torch.float32, generator=gen,
        )
        x = torch.matmul(x, self.cov_base_chol.to(device=device, dtype=x.dtype).t())

        T = self.cfg.diffusion_steps
        t_seq = _make_ddim_timesteps(T, steps, device=device)
        alpha_bars, _, _ = self._schedule_tensors(device, x.dtype)
        clip = float(self.cfg.x0_clip)
        score = torch.zeros(B, device=device, dtype=torch.float32) if return_score else None

        for idx in range(t_seq.numel()):
            t_int = int(t_seq[idx].item())
            t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)
            ab_t = alpha_bars[t_int]
            sqrt_ab_t = torch.sqrt(ab_t)
            sqrt_omb_t = torch.sqrt(1.0 - ab_t)

            x_input = x.to(dtype=history_free.dtype)

            v_hat = self.denoiser(
                x_input, t_batch, history_free, coarse_free, mamp_feat=mamp_feat,
            ).float()

            x0 = sqrt_ab_t * x - sqrt_omb_t * v_hat
            if clip > 0.0:
                x0 = torch.clamp(x0, -clip, clip)
            eta_hat = sqrt_omb_t * x + sqrt_ab_t * v_hat
            if score is not None:
                score = score + eta_hat.float().pow(2).sum(dim=(1, 2))

            if idx == t_seq.numel() - 1:
                ab_prev = torch.tensor(1.0, device=device, dtype=x.dtype)
            else:
                t_prev = int(t_seq[idx + 1].item())
                ab_prev = alpha_bars[t_prev]
            sqrt_ab_prev = torch.sqrt(ab_prev)
            sqrt_omb_prev = torch.sqrt(1.0 - ab_prev)

            if eta != 0.0:
                sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
                dir_coeff = torch.sqrt(torch.clamp(sqrt_omb_prev**2 - sigma**2, min=0.0))
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
                noise = torch.matmul(noise, self.cov_base_chol.to(device=device, dtype=x.dtype).t())
                x = sqrt_ab_prev * x0 + dir_coeff * eta_hat + sigma * noise
            else:
                x = sqrt_ab_prev * x0 + sqrt_omb_prev * eta_hat

        out_free = x.to(dtype=history_full.dtype)
        out_full = self._restore_full_from_free(out_free, self.cfg.pred_length, history_full.dtype)
        if score is None:
            return out_full
        return out_full, score


# ----------------------------- Wrapper model ----------------------------------

class TwoStageDCTDiffusionForecaster(nn.Module):
    def __init__(self, cfg: TwoStageDCTDiffusionConfig, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.coarse = CoarseDCTForecaster(cfg)
        self.diffusion = TimeResidualConditionalDiffusion(cfg, metadata=metadata)
        if cfg.freeze_coarse:
            for p in self.coarse.parameters():
                p.requires_grad = False

    def diffusion_loss(
        self,
        history: torch.Tensor,
        future_gt: torch.Tensor,
        mamp_feat: Optional[torch.Tensor] = None,
        coarse_future: Optional[torch.Tensor] = None,
        allow_coarse_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        _assert_shape(future_gt, (None, self.cfg.pred_length, self.cfg.num_nodes, 3), "future_gt")

        if coarse_future is None:
            coarse_future = self.coarse(history)
        else:
            _assert_shape(
                coarse_future,
                (history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3),
                "coarse_future",
            )
        if allow_coarse_grad:
            coarse_cond = coarse_future if not self.cfg.stopgrad_coarse_condition else coarse_future.detach()
            residual_gt = future_gt - coarse_future
        else:
            coarse_cond = coarse_future.detach() if self.cfg.stopgrad_coarse_condition else coarse_future
            residual_gt = future_gt - coarse_future.detach()

        loss = self.diffusion.training_loss(
            residual_gt=residual_gt,
            history_full=history,
            coarse_future_full=coarse_cond,
            mamp_feat=mamp_feat,
        )
        return loss, coarse_future

    @torch.no_grad()
    def predict(
        self,
        history: torch.Tensor,
        mamp_feat: Optional[torch.Tensor] = None,
        coarse_future: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        seed: int = 0,
        return_score: bool = False,
    ) -> torch.Tensor:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        if coarse_future is None:
            coarse_future = self.coarse(history)
        else:
            _assert_shape(
                coarse_future,
                (history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3),
                "coarse_future",
            )

        coarse_cond = coarse_future.detach() if self.cfg.stopgrad_coarse_condition else coarse_future
        residual_pred = self.diffusion.sample_ddim(
            history_full=history,
            coarse_future_full=coarse_cond,
            mamp_feat=mamp_feat,
            seed=seed,
            steps=self.cfg.ddim_steps,
            eta=0.0 if deterministic else 1.0,
            return_score=return_score,
        )

        if return_score:
            residual_pred, score = residual_pred
        future = coarse_future + residual_pred
        if 0 <= int(self.diffusion.wrist_index) < self.cfg.num_nodes:
            future[:, :, int(self.diffusion.wrist_index), :] = 0.0
        if return_score:
            return future, score
        return future

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.coarse(history)
        return self.predict(history, deterministic=True, seed=0)
