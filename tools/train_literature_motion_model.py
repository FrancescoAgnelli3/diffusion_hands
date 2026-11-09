#!/usr/bin/env python3
"""Shared-protocol adapters for CacheFlow and STARS.

The upstream projects hard-code Human3.6M/AMASS readers.  This entry point
keeps their modelling mechanisms while deliberately using diffusion_hands'
fixed splits, windows and metric implementation, making comparisons on the
hand datasets meaningful.
"""
from __future__ import annotations

import argparse
import csv
import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vendor.belfusion.run_belfusion import (  # shared window construction/evaluation
    WindowDataset, evaluate_model, load_split_samples, make_loader,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CacheFlowAdapter(nn.Module):
    """Conditional cached normalizing flow over the flattened future motion."""
    def __init__(self, frame_dim: int, future_dim: int, hidden: int, flow_layers: int):
        super().__init__()
        self.future_dim = future_dim
        self.frame_encoder = nn.GRU(frame_dim, hidden, batch_first=True)
        self.base = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, future_dim))
        self.log_scale = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, future_dim))
        self.transforms = nn.ModuleList([
            nn.Sequential(nn.Linear(future_dim + hidden, hidden), nn.SiLU(), nn.Linear(hidden, future_dim * 2))
            for _ in range(flow_layers)
        ])
        mask = (torch.arange(future_dim) % 2).float()
        self.register_buffer("mask", mask)

    def _context(self, obs: torch.Tensor) -> torch.Tensor:
        _b, _t, _d = obs.shape
        _, h = self.frame_encoder(obs)
        return h[-1]

    def _params(self, value: torch.Tensor, context: torch.Tensor, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        masked = value * (self.mask if layer % 2 == 0 else 1.0 - self.mask)
        shift, log_s = self.transforms[layer](torch.cat([masked, context], dim=1)).chunk(2, dim=1)
        log_s = torch.tanh(log_s) * 0.75
        return shift, log_s

    def nll(self, obs: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        context = self._context(obs)
        mean = self.base(context)
        scale = self.log_scale(context).clamp(-4, 3)
        z = (future - mean) * torch.exp(-scale)
        logdet = -scale.sum(dim=1)
        for i in range(len(self.transforms)):
            mask = self.mask if i % 2 == 0 else 1.0 - self.mask
            shift, log_s = self._params(z, context, i)
            z = z * mask + (1.0 - mask) * (z - shift) * torch.exp(-log_s)
            logdet = logdet - ((1.0 - mask) * log_s).sum(dim=1)
        return (0.5 * (z.square() + np.log(2 * np.pi)).sum(dim=1) - logdet).mean()

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, num_samples: int) -> torch.Tensor:
        context = self._context(obs)
        b = obs.shape[0]
        z = torch.randn(num_samples, b, self.future_dim, device=obs.device)
        ctx = context.unsqueeze(0).expand(num_samples, -1, -1).reshape(num_samples * b, -1)
        z = z.reshape(num_samples * b, -1)
        for i in reversed(range(len(self.transforms))):
            mask = self.mask if i % 2 == 0 else 1.0 - self.mask
            shift, log_s = self._params(z, ctx, i)
            z = z * mask + (1.0 - mask) * (z * torch.exp(log_s) + shift)
        mean = self.base(ctx)
        scale = self.log_scale(ctx).clamp(-4, 3)
        return (mean + z * torch.exp(scale)).reshape(num_samples, b, -1)


class STARSAdapter(nn.Module):
    """Spatial-temporal-anchor predictor with learned, diverse future anchors."""
    def __init__(self, frame_dim: int, future_dim: int, hidden: int, anchors: int):
        super().__init__()
        self.encoder = nn.GRU(frame_dim, hidden, batch_first=True)
        self.anchors = nn.Parameter(torch.randn(anchors, hidden) * 0.15)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, future_dim)
        )

    def sample(self, obs: torch.Tensor, num_samples: int) -> torch.Tensor:
        _, h = self.encoder(obs)
        context = h[-1]
        # Repeat/cycle anchors if evaluation uses more candidates than training.
        codes = self.anchors[torch.arange(num_samples, device=obs.device) % len(self.anchors)]
        joint = torch.cat([
            context.unsqueeze(0).expand(num_samples, -1, -1),
            codes.unsqueeze(1).expand(-1, obs.shape[0], -1),
        ], dim=-1)
        return self.decoder(joint.reshape(num_samples * obs.shape[0], -1)).reshape(num_samples, obs.shape[0], -1)

    def loss(
        self,
        obs: torch.Tensor,
        future: torch.Tensor,
        candidates: int,
        diversity_weight: float,
        min_diversity: float,
    ) -> torch.Tensor:
        prediction = self.sample(obs, candidates)
        per_candidate = (prediction - future.unsqueeze(0)).square().mean(dim=2)
        best_of_k = per_candidate.min(dim=0).values.mean()
        # A best-of-K term alone leaves every losing anchor unpenalized.  The
        # former inverse-distance reward then pushed those anchors to infinity,
        # corrupting APD/FID.  Fit *all* candidates and only penalize diversity
        # below a finite target: extra separation receives no reward.
        all_candidates = per_candidate.mean()
        reconstruction = 0.5 * best_of_k + 0.5 * all_candidates
        if candidates < 2 or diversity_weight <= 0 or min_diversity <= 0:
            return reconstruction
        flat = prediction.permute(1, 0, 2)  # [B, K, D]
        distances = torch.cdist(flat, flat, p=2) / float(prediction.shape[-1]) ** 0.5
        pair_index = torch.triu_indices(candidates, candidates, offset=1, device=flat.device)
        pairs = distances[:, pair_index[0], pair_index[1]]
        diversity_shortfall = torch.relu(float(min_diversity) - pairs).mean()
        return reconstruction + diversity_weight * diversity_shortfall


def _batch(batch, device: torch.device):
    obs, future, _norm, _sid = batch
    obs = obs[..., :3] if obs.shape[-1] == 3 else obs[..., 4:7]
    future = future[..., :3] if future.shape[-1] == 3 else future[..., 4:7]
    return obs.to(device).reshape(obs.shape[0], obs.shape[1], -1), future.to(device).reshape(future.shape[0], -1)


def train(cfg: Dict[str, object]) -> Dict[str, float]:
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples, val_samples, test_samples = load_split_samples(cfg)
    if not train_samples or not test_samples:
        raise RuntimeError("The selected split produced no train/test windows.")
    data, model_cfg, train_cfg, eval_cfg, runtime = (cfg["data"], cfg["model"], cfg["train"], cfg["eval"], cfg["runtime"])
    subset_path = str(runtime.get("eval_subset_indices_path", "") or "")
    if subset_path:
        selected = [int(index) for index in json.loads(Path(subset_path).read_text())]
        if not selected or min(selected) < 0 or max(selected) >= len(test_samples):
            raise ValueError("invalid mobile evaluation subset indices")
        test_samples = [test_samples[index] for index in selected]
    input_n, output_n, n_nodes = int(data["input_n"]), int(data["output_n"]), int(data.get("n_nodes", 21))
    frame_dim, future_dim = n_nodes * 3, output_n * n_nodes * 3
    method = str(cfg["method"]).lower()
    if method == "cacheflow":
        model: nn.Module = CacheFlowAdapter(frame_dim, future_dim, int(model_cfg.get("hidden_dim", 256)), int(model_cfg.get("flow_layers", 4)))
    elif method == "stars":
        model = STARSAdapter(frame_dim, future_dim, int(model_cfg.get("hidden_dim", 256)), int(model_cfg.get("anchors", 20)))
    else:
        raise ValueError(f"Unsupported method {method!r}")
    model = model.to(device)
    reuse_checkpoint = str(runtime.get("reuse_checkpoint_path", "") or "")
    if bool(runtime.get("checkpoint_only", False)) and not reuse_checkpoint:
        raise RuntimeError(f"{method} checkpoint-only evaluation is missing a reused checkpoint")
    if reuse_checkpoint:
        checkpoint = torch.load(reuse_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        print(f"[{method}] reused checkpoint: {reuse_checkpoint}")
    else:
        train_loader = make_loader(train_samples, int(train_cfg.get("batch_size", 64)), True, int(train_cfg.get("num_workers", 0)))
        opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-5)))
        best, stale, best_state = float("inf"), 0, None
        epochs, patience = int(train_cfg.get("epochs", 50)), int(train_cfg.get("patience", 10))
        for epoch in range(1, epochs + 1):
            model.train(); total = 0.0; batches = 0
            for batch in train_loader:
                obs, future = _batch(batch, device)
                loss = model.nll(obs, future) if method == "cacheflow" else model.loss(
                    obs, future, int(model_cfg.get("anchors", 20)),
                    float(train_cfg.get("diversity_weight", 0.01)),
                    float(train_cfg.get("min_diversity", 0.05)),
                )
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                total += float(loss.detach()); batches += 1
            current = total / max(1, batches)
            print(f"[{method}][epoch {epoch}/{epochs}] train_loss={current:.6f}")
            if current < best - float(train_cfg.get("min_delta", 1e-4)):
                best, stale, best_state = current, 0, copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if patience and stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
    out_dir = Path(str(runtime["output_dir"])); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "method": method, "config": cfg}, out_dir / "checkpoints" / "final.pt")

    class Evaluator(nn.Module):
        def __init__(self, wrapped: nn.Module): super().__init__(); self.wrapped = wrapped
        def sample(self, obs_flat: torch.Tensor, num_samples: int):
            obs = obs_flat.reshape(obs_flat.shape[0], input_n, frame_dim)
            return self.wrapped.sample(obs, num_samples)
    metrics = evaluate_model(Evaluator(model), make_loader(test_samples, int(eval_cfg.get("batch_size", 128)), False, 0),
        int(eval_cfg["num_candidates"]), float(eval_cfg["multimodal_threshold"]), output_n, n_nodes, device,
        Path(str(runtime["eval_samples_path"])) if runtime.get("eval_samples_path") else None,
        {"model": method, "num_candidates": int(eval_cfg["num_candidates"]), "adapter": "shared_protocol"})
    metrics_path = Path(str(runtime["metrics_csv"])); metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["APD", "ADE", "FDE", "MMADE", "MMFDE", "CMD", "FID"]); writer.writeheader(); writer.writerow(metrics)
    (out_dir / "run_meta.json").write_text(json.dumps({"method": method, "num_train_windows": len(train_samples), "num_val_windows": len(val_samples), "num_test_windows": len(test_samples)}, indent=2) + "\n")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        train(yaml.safe_load(f))

if __name__ == "__main__":
    main()
