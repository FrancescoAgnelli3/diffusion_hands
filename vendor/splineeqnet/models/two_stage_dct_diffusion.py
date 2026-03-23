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
Diffusion is trained and sampled in NORMALIZED residual space using EMA RMS scale s:
    x0 = r / s

Denoiser predicts v (v-pred):
    x_t = sqrt(ab) * x0 + sqrt(1-ab) * eps
    v   = sqrt(ab) * eps - sqrt(1-ab) * x0
This avoids the unstable division by sqrt(ab) used by epsilon-pred when computing x0 at high noise.

This file assumes your project already provides:
- models.simlpe_dct.SiMLPeConfig, SiMLPeBackbone, _build_dct_matrix
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

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
    # Ensure endpoints exist
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
    stopgrad_coarse_condition: bool = True  # stop-grad coarse conditioning into diffusion during training
    cond_use_history: bool = True
    cond_use_coarse: bool = True

    # Residual normalization (EMA RMS)
    residual_ema_decay: float = 0.99
    residual_rms_eps: float = 1e-6

    # Sampling stabilizers
    x0_clip: float = 0  # clip x0_hat in normalized space (0 disables)

    @property
    def total_length(self) -> int:
        return int(self.input_length + self.pred_length)

    @property
    def feature_dim(self) -> int:
        return int(self.num_nodes * 3)

    # Kept for backward-compat; not used by diffusion stage
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

        dct = _build_dct_matrix(T)  # (T, T) orthonormal
        idct = dct.t()              # (T, T)
        self.register_buffer("_dct_low", dct[: cfg.k_low, :], persistent=False)     # (K_low, T)
        self.register_buffer("_idct_low", idct[:, : cfg.k_low], persistent=False)  # (T, K_low)

        sim_cfg = SiMLPeConfig(
            input_length=cfg.input_length,
            pred_length=cfg.pred_length,
            num_nodes=cfg.num_nodes,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            use_norm=cfg.simlpe_use_norm,
            use_spatial_fc_only=cfg.simlpe_spatial_fc_only,
            mix_spatial_temporal=cfg.simlpe_mix_spatial_temporal,
            norm_axis=cfg.simlpe_norm_axis,  # type: ignore[arg-type]
            add_last_offset=cfg.simlpe_add_last_offset,
        )
        self.backbone = SiMLPeBackbone(sim_cfg)

    def _pad_total(self, history: torch.Tensor) -> torch.Tensor:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        B = history.size(0)
        flat = history.reshape(B, self.cfg.input_length, -1)
        last = flat[:, -1:, :].expand(-1, self.cfg.pred_length, -1)
        padded = torch.cat([flat, last], dim=1)  # (B, T_total, F)
        return padded

    def predict_low_coeffs(self, history: torch.Tensor) -> torch.Tensor:
        """Return predicted low-band DCT coefficients for the total sequence. Shape: (B, K_low, F)."""
        padded = self._pad_total(history)  # (B, T, F)
        dct_low = self._dct_low.unsqueeze(0).to(device=padded.device, dtype=padded.dtype)  # (1, K_low, T)
        low_coeffs = torch.matmul(dct_low, padded)  # (B, K_low, F)

        B, _, F = low_coeffs.shape
        coeffs_full = torch.zeros(B, self.cfg.total_length, F, device=padded.device, dtype=padded.dtype)
        coeffs_full[:, : self.cfg.k_low, :] = low_coeffs

        pred_full = self.backbone(coeffs_full)      # (B, T, F) treated as coefficient sequence
        pred_low = pred_full[:, : self.cfg.k_low, :]
        return pred_low

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Return coarse future in time domain. Shape: (B, T_out, N, 3)."""
        pred_low = self.predict_low_coeffs(history)  # (B, K_low, F)
        idct_low = self._idct_low.unsqueeze(0).to(device=pred_low.device, dtype=pred_low.dtype)  # (1, T, K_low)
        seq_low = torch.matmul(idct_low, pred_low)  # (B, T, F)

        future = seq_low[:, -self.cfg.pred_length :, :].reshape(
            history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3
        )
        if self.cfg.simlpe_add_last_offset:
            future = future + history[:, -1:, :, :]
        return future


# ----------------------------- Diffusion scheduler ----------------------------

class DiffusionSchedule:
    def __init__(self, steps: int, schedule: str = "cosine", device: Optional[torch.device] = None) -> None:
        if steps <= 1:
            raise ValueError("diffusion steps must be > 1")
        self.steps = int(steps)
        self.schedule = str(schedule)
        self.device = device

        betas = self._make_betas(self.steps, self.schedule)
        self.betas = betas.to(device=device)  # (T,)
        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)
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
        # x: (..., D), cond: (B, D)
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
    """
    Self-attention across joints within each time step.
    Input/output: (B, T, N, D)
    """
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
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
    """
    Self-attention across time for each joint.
    Input/output: (B, T, N, D)
    """
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
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
    """
    Cross-attention over conditioning time tokens for each joint independently.

    Query:  (B, T_out, N, D)
    Memory: (B, T_mem, N, D)
    Output: (B, T_out, N, D)
    """
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
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

        q = self.norm_q(x).permute(0, 2, 1, 3).reshape(B * N, T_out, D)
        m = self.norm_m(memory).permute(0, 2, 1, 3).reshape(B * N, T_mem, D)
        y, _ = self.attn(q, m, m, need_weights=False)
        y = y.reshape(B, N, T_out, D).permute(0, 2, 1, 3)

        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class _FactorizedSTBlock(nn.Module):
    """
    One factorized spatio-temporal block with timestep FiLM and temporal cross-attention.
    """
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
        x: torch.Tensor,        # (B, T_out, N, D)
        memory: torch.Tensor,   # (B, T_mem, N, D)
        t_emb: torch.Tensor,    # (B, D)
    ) -> torch.Tensor:
        x = self.film_spatial(x, t_emb)
        x = self.spatial(x)

        x = self.film_temporal(x, t_emb)
        x = self.temporal(x)

        x = self.film_cross(x, t_emb)
        x = self.cross(x, memory)

        return x


class MRTimeTransformerDenoiser(nn.Module):
    """
    Factorized spatio-temporal denoiser with explicit joint tokens.

    Signature kept intact.

    Query tokens:
        x_noisy: (B, T_out, F) with F = N * 3
    Conditioning:
        history: (B, T_in, F)
        coarse_future: (B, T_out, F)

    Internally everything is reshaped to joint tokens:
        (B, T, N, 3)

    Output:
        predicted v with same shape as input residual, i.e. (B, T_out, F)
    """

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
    ) -> None:
        super().__init__()
        self.in_feat = int(in_feat)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.d_model = int(d_model)
        self.cond_use_history = bool(cond_use_history)
        self.cond_use_coarse = bool(cond_use_coarse)

        if self.in_feat % 3 != 0:
            raise ValueError(f"in_feat must be divisible by 3, got {self.in_feat}")
        self.num_nodes = self.in_feat // 3

        # Per-joint input projection
        self.proj_q = nn.Linear(3, d_model)
        self.proj_hist = nn.Linear(3, d_model)
        self.proj_coarse = nn.Linear(3, d_model)
        self.proj_mamp = nn.LazyLinear(d_model)

        # Positional embeddings
        self.pos_q_time = nn.Parameter(torch.zeros(1, t_out, 1, d_model))
        self.pos_hist_time = nn.Parameter(torch.zeros(1, t_in, 1, d_model))
        self.pos_coarse_time = nn.Parameter(torch.zeros(1, t_out, 1, d_model))
        self.pos_joint = nn.Parameter(torch.zeros(1, 1, self.num_nodes, d_model))
        self.pos_mamp = nn.Parameter(torch.zeros(1, 1, d_model))

        # Diffusion timestep embedding
        self.t_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Optional conditioning token encoder for mamp_feat only
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

        # Main factorized ST blocks
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
        """
        Returns joint-shaped additive conditioning tensor of shape (B, 1, N, D),
        or None if mamp_feat is None.

        Supported inputs:
        - (B, Dm): global token broadcast to all joints
        - (B, K, Dm): K global tokens encoded then pooled and broadcast
        - (B, N, Dm): per-joint features
        """
        if mamp_feat is None:
            return None

        if int(mamp_feat.shape[0]) != B:
            raise ValueError(f"mamp_feat batch mismatch: expected {B}, got {mamp_feat.shape[0]}")

        if mamp_feat.ndim == 2:
            # (B, Dm) -> (B, 1, D) -> broadcast over joints
            token = self.proj_mamp(mamp_feat.to(device=device, dtype=dtype)).unsqueeze(1) + self.pos_mamp
            token = token.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)  # (B, 1, N, D)
            return token

        if mamp_feat.ndim == 3:
            if int(mamp_feat.shape[1]) == self.num_nodes:
                # (B, N, Dm): per-joint features
                joint_tok = self.proj_mamp(mamp_feat.to(device=device, dtype=dtype))  # (B, N, D)
                joint_tok = joint_tok.unsqueeze(1)  # (B, 1, N, D)
                return joint_tok

            # Otherwise treat as K global tokens, encode, pool, broadcast
            tok = self.proj_mamp(mamp_feat.to(device=device, dtype=dtype))  # (B, K, D)
            tok = tok + self.pos_mamp
            tok = self.mamp_encoder(tok)
            tok = tok.mean(dim=1, keepdim=True)  # (B, 1, D)
            tok = tok.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)  # (B, 1, N, D)
            return tok

        raise ValueError(
            f"mamp_feat must be (B, D), (B, K, D), or (B, N, D), got {tuple(mamp_feat.shape)}"
        )

    def forward(
        self,
        x_noisy: torch.Tensor,       # (B, T_out, F)
        timesteps: torch.Tensor,     # (B,)
        history: torch.Tensor,       # (B, T_in, F)
        coarse_future: torch.Tensor, # (B, T_out, F)
        mamp_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x_noisy.size(0)
        _assert_shape(x_noisy, (B, self.t_out, self.in_feat), "x_noisy")
        _assert_shape(history, (B, self.t_in, self.in_feat), "history_flat")
        _assert_shape(coarse_future, (B, self.t_out, self.in_feat), "coarse_future_flat")

        # Reshape flat features -> joint tokens
        x_noisy_j = self._flat_to_joint_tokens(x_noisy, self.t_out, "x_noisy")
        history_j = self._flat_to_joint_tokens(history, self.t_in, "history_flat")
        coarse_j = self._flat_to_joint_tokens(coarse_future, self.t_out, "coarse_future_flat")

        # Input embeddings
        x = self.proj_q(x_noisy_j) + self.pos_q_time + self.pos_joint  # (B, T_out, N, D)

        memory_parts = []
        if self.cond_use_history:
            h_hist = self.proj_hist(history_j) + self.pos_hist_time + self.pos_joint  # (B, T_in, N, D)
            memory_parts.append(h_hist)
        if self.cond_use_coarse:
            h_coarse = self.proj_coarse(coarse_j) + self.pos_coarse_time + self.pos_joint  # (B, T_out, N, D)
            memory_parts.append(h_coarse)

        if not memory_parts and mamp_feat is None:
            raise ValueError("No conditioning enabled. Set cond_use_history/coarse or provide mamp_feat.")

        if memory_parts:
            memory = torch.cat(memory_parts, dim=1)  # (B, T_mem, N, D)
        else:
            memory = torch.zeros(
                B, 0, self.num_nodes, self.d_model,
                device=x.device, dtype=x.dtype
            )

        mamp_joint = self._encode_mamp_per_joint(
            mamp_feat,
            B=B,
            device=x.device,
            dtype=x.dtype,
        )
        if mamp_joint is not None:
            # Add global/per-joint side information to both query and memory
            x = x + mamp_joint
            if memory.size(1) > 0:
                memory = memory + mamp_joint.expand(-1, memory.size(1), -1, -1)

        # Diffusion timestep embedding
        te = _timestep_embedding(timesteps, self.d_model).to(dtype=x.dtype, device=x.device)
        te = self.t_embed(te)  # (B, D)

        # Factorized spatio-temporal processing
        for block in self.blocks:
            x = block(x, memory, te)

        v_hat = self.out(self.out_norm(x))  # (B, T_out, N, 3)
        v_hat = v_hat.reshape(B, self.t_out, self.in_feat)  # back to (B, T_out, F)
        return v_hat
# ----------------------------- Diffusion module (time-domain residual, v-pred) -

class TimeResidualConditionalDiffusion(nn.Module):
    def __init__(self, cfg: TwoStageDCTDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.schedule = DiffusionSchedule(cfg.diffusion_steps, cfg.beta_schedule)

        # Keep EMA in fp32 for numerical stability
        self.register_buffer("ema_rms", torch.tensor(1.0, dtype=torch.float32))

        self.denoiser = MRTimeTransformerDenoiser(
            in_feat=cfg.feature_dim,
            t_in=cfg.input_length,
            t_out=cfg.pred_length,
            d_model=cfg.denoiser_dim,
            depth=cfg.denoiser_depth,
            n_heads=cfg.denoiser_heads,
            dropout=cfg.dropout,
            cond_use_history=cfg.cond_use_history,
            cond_use_coarse=cfg.cond_use_coarse,
        )

    def _schedule_tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.schedule.alpha_bars.to(device=device, dtype=dtype),
            self.schedule.sqrt_alpha_bars.to(device=device, dtype=dtype),
            self.schedule.sqrt_one_minus_alpha_bars.to(device=device, dtype=dtype),
        )

    def _residual_scale(self, residual_gt: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar s (fp32) on residual_gt.device.
        Updates EMA during training using detached residual RMS.
        """
        eps = float(self.cfg.residual_rms_eps)
        decay = float(self.cfg.residual_ema_decay)

        batch_rms = residual_gt.detach().float().pow(2).mean().sqrt()  # fp32 scalar on device
        if self.training:
            self.ema_rms.mul_(decay).add_(
                batch_rms.to(device=self.ema_rms.device, dtype=self.ema_rms.dtype),
                alpha=1.0 - decay,
            )

        s = torch.clamp(self.ema_rms.to(device=residual_gt.device), min=eps)  # fp32
        return s

    def training_loss(
        self,
        *,
        residual_gt: torch.Tensor,       # (B, T_out, F) in data units
        history_flat: torch.Tensor,      # (B, T_in, F)
        coarse_future_flat: torch.Tensor, # (B, T_out, F)
        mamp_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = residual_gt.size(0)
        _assert_shape(residual_gt, (B, self.cfg.pred_length, self.cfg.feature_dim), "residual_gt")

        # Normalize residual: x0 = r / s  (work in fp32 for stability)
        s = self._residual_scale(residual_gt)  # fp32 scalar
        x0 = residual_gt.float() / s

        device = residual_gt.device
        alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars = self._schedule_tensors(device, x0.dtype)

        t = torch.randint(0, self.cfg.diffusion_steps, (B,), device=device, dtype=torch.long)
        eps = torch.randn_like(x0)

        sqrt_ab = sqrt_alpha_bars[t].view(B, 1, 1)
        sqrt_omb = sqrt_one_minus_alpha_bars[t].view(B, 1, 1)

        x_t = sqrt_ab * x0 + sqrt_omb * eps

        # v target: v = sqrt(ab)*eps - sqrt(1-ab)*x0
        v = sqrt_ab * eps - sqrt_omb * x0

        v_hat = self.denoiser(
            x_t.to(dtype=history_flat.dtype),
            t,
            history_flat,
            coarse_future_flat,
            mamp_feat=mamp_feat,
        ).float()
        return torch.mean((v - v_hat) ** 2)

    @torch.no_grad()
    def sample_ddim(
        self,
        *,
        history_flat: torch.Tensor,       # (B, T_in, F)
        coarse_future_flat: torch.Tensor, # (B, T_out, F)
        mamp_feat: Optional[torch.Tensor] = None,
        seed: int = 0,
        steps: Optional[int] = None,
        eta: float = 0.0,
        return_score: bool = False,
    ) -> torch.Tensor:
        """
        DDIM sampling for residual sequence in data units.

        Sampling is done in normalized space; final output is multiplied by s.
        """
        if steps is None:
            steps = self.cfg.ddim_steps
        steps = int(steps)
        if steps <= 0:
            raise ValueError("DDIM steps must be positive")

        B = history_flat.size(0)
        device = history_flat.device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        # Use EMA scale (fp32) without updating
        s = torch.clamp(self.ema_rms.to(device=device), min=float(self.cfg.residual_rms_eps))  # fp32 scalar

        # Sample in normalized space (fp32 core)
        x = torch.randn(
            (B, self.cfg.pred_length, self.cfg.feature_dim),
            device=device,
            dtype=torch.float32,
            generator=gen,
        )

        T = self.cfg.diffusion_steps
        t_seq = _make_ddim_timesteps(T, steps, device=device)  # descending unique ints

        t_max = int(1 * (T - 1))  # try 0.4–0.8
        t_seq = t_seq[t_seq <= t_max]
        if t_seq.numel() == 0 or int(t_seq[-1]) != 0:
            t_seq = torch.cat([t_seq, torch.tensor([0], device=device, dtype=torch.long)])

        alpha_bars, _, _ = self._schedule_tensors(device, x.dtype)

        clip = float(self.cfg.x0_clip)

        score = torch.zeros(B, device=device, dtype=torch.float32) if return_score else None

        for idx in range(t_seq.numel()):
            t = int(t_seq[idx].item())
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            ab_t = alpha_bars[t]  # scalar
            sqrt_ab_t = torch.sqrt(ab_t)
            sqrt_omb_t = torch.sqrt(1.0 - ab_t)

            # Predict v (network runs in its dtype; cast to fp32 for math)
            v_hat = self.denoiser(
                x.to(dtype=history_flat.dtype),
                t_batch,
                history_flat,
                coarse_future_flat,
                mamp_feat=mamp_feat,
            ).float()

            # Reconstruct x0 from v-pred (stable; no division by sqrt_ab_t)
            x0 = sqrt_ab_t * x - sqrt_omb_t * v_hat

            if clip > 0.0:
                x0 = torch.clamp(x0, -clip, clip)

            # Need eps_hat for DDIM direction
            eps_hat = (x - sqrt_ab_t * x0) / (sqrt_omb_t + 1e-12)
            if score is not None:
                score = score + eps_hat.float().pow(2).sum(dim=(1, 2))

            if idx == t_seq.numel() - 1:
                ab_prev = torch.tensor(1.0, device=device, dtype=x.dtype)
            else:
                t_prev = int(t_seq[idx + 1].item())
                ab_prev = alpha_bars[t_prev]

            sqrt_ab_prev = torch.sqrt(ab_prev)
            sqrt_omb_prev = torch.sqrt(1.0 - ab_prev)

            if eta != 0.0:
                # DDIM with noise
                sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
                noise = torch.randn(
                    x.shape,
                    device=x.device,
                    dtype=x.dtype,
                    generator=gen,
                )
                x = (
                    sqrt_ab_prev * x0
                    + torch.sqrt(torch.clamp(sqrt_omb_prev**2 - sigma**2, min=0.0)) * eps_hat
                    + sigma * noise
                )
            else:
                # Deterministic DDIM
                x = sqrt_ab_prev * x0 + sqrt_omb_prev * eps_hat

        # Denormalize once at end; return in the caller's expected dtype
        out = (x * s).to(dtype=history_flat.dtype)
        if score is None:
            return out
        return out, score

# ----------------------------- Wrapper model ----------------------------------

class TwoStageDCTDiffusionForecaster(nn.Module):
    def __init__(self, cfg: TwoStageDCTDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.coarse = CoarseDCTForecaster(cfg)
        self.diffusion = TimeResidualConditionalDiffusion(cfg)

        if cfg.freeze_coarse:
            for p in self.coarse.parameters():
                p.requires_grad = False

    def _flat_history(self, history: torch.Tensor) -> torch.Tensor:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        return history.reshape(history.size(0), self.cfg.input_length, -1)

    def _flat_future(self, future: torch.Tensor) -> torch.Tensor:
        _assert_shape(future, (None, self.cfg.pred_length, self.cfg.num_nodes, 3), "future")
        return future.reshape(future.size(0), self.cfg.pred_length, -1)

    def diffusion_loss(
        self,
        history: torch.Tensor,
        future_gt: torch.Tensor,
        mamp_feat: Optional[torch.Tensor] = None,
        coarse_future: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute diffusion training loss (v-pred in normalized residual space) and return coarse prediction for logging.
        Signature unchanged for your training code.
        """
        history_flat = self._flat_history(history)

        if coarse_future is None:
            coarse_future = self.coarse(history)  # (B, T_out, N, 3)
        else:
            _assert_shape(
                coarse_future,
                (history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3),
                "coarse_future",
            )
        coarse_future_flat = coarse_future.reshape(history.size(0), self.cfg.pred_length, -1)

        coarse_cond = coarse_future_flat.detach() if self.cfg.stopgrad_coarse_condition else coarse_future_flat

        future_gt_flat = self._flat_future(future_gt)
        residual_gt = future_gt_flat - coarse_future_flat.detach()

        loss = self.diffusion.training_loss(
            residual_gt=residual_gt,
            history_flat=history_flat,
            coarse_future_flat=coarse_cond,
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
        """Full two-stage prediction (single reconstruction)."""
        history_flat = self._flat_history(history)

        if coarse_future is None:
            coarse_future = self.coarse(history)
        else:
            _assert_shape(
                coarse_future,
                (history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3),
                "coarse_future",
            )
        coarse_future_flat = coarse_future.reshape(history.size(0), self.cfg.pred_length, -1)
        coarse_cond = coarse_future_flat.detach() if self.cfg.stopgrad_coarse_condition else coarse_future_flat

        residual_pred = self.diffusion.sample_ddim(
            history_flat=history_flat,
            coarse_future_flat=coarse_cond,
            mamp_feat=mamp_feat,
            seed=seed,
            steps=self.cfg.ddim_steps,
            eta=0.0 if deterministic else 1.0,
            return_score=return_score,
        )

        if return_score:
            residual_pred, score = residual_pred
        future_flat = coarse_future_flat + residual_pred
        future = future_flat.reshape(history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3)
        if return_score:
            return future, score
        return future

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.coarse(history)
        return self.predict(history, deterministic=True, seed=0)
