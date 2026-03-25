#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.metrics import splineeqnet_diffusion_batch_eval
from common.preprocessing import select_most_active_hand, split_train_val_test


@dataclass
class WindowSample:
    obs: np.ndarray
    fut: np.ndarray
    norm: float


class WindowDataset(Dataset):
    def __init__(self, samples: Sequence[WindowSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        obs = torch.from_numpy(s.obs).float()
        fut = torch.from_numpy(s.fut).float()
        norm = torch.tensor(float(s.norm), dtype=torch.float32)
        return obs, fut, norm


class BelFusionCVAE(nn.Module):
    def __init__(self, obs_dim: int, fut_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.post_enc = nn.Sequential(
            nn.Linear(obs_dim + fut_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, fut_dim),
        )

    def _obs_feat(self, obs_flat: torch.Tensor) -> torch.Tensor:
        return self.obs_enc(obs_flat)

    def forward(self, obs_flat: torch.Tensor, fut_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_obs = self._obs_feat(obs_flat)
        h_post = self.post_enc(torch.cat([obs_flat, fut_flat], dim=1))
        mu = self.mu(h_post)
        logvar = self.logvar(h_post)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        recon = self.dec(torch.cat([h_obs, z], dim=1))
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, obs_flat: torch.Tensor, num_samples: int) -> torch.Tensor:
        h_obs = self._obs_feat(obs_flat)
        bs = obs_flat.shape[0]
        h_obs_rep = h_obs.unsqueeze(0).expand(num_samples, bs, h_obs.shape[-1])
        z = torch.randn(num_samples, bs, self.mu.out_features, device=obs_flat.device, dtype=obs_flat.dtype)
        dec_in = torch.cat([h_obs_rep, z], dim=-1)
        out = self.dec(dec_in.reshape(num_samples * bs, -1)).reshape(num_samples, bs, -1)
        return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize(text: str) -> str:
    return (text or "all").replace("/", "_").replace(" ", "_")


def _build_windows(
    files: Sequence[str],
    input_n: int,
    output_n: int,
    stride: int,
    time_interp: int | None,
    window_norm: int | None,
) -> List[WindowSample]:
    samples: List[WindowSample] = []
    total = input_n + output_n
    for p in files:
        selected = select_most_active_hand(p, time_interp=time_interp, window_norm=window_norm)
        if selected is None:
            continue
        seq, norm = selected
        if seq.shape[0] < total:
            continue
        for st in range(0, seq.shape[0] - total + 1, max(1, int(stride))):
            obs = seq[st : st + input_n]
            fut = seq[st + input_n : st + total]
            samples.append(WindowSample(obs=obs.astype(np.float32), fut=fut.astype(np.float32), norm=float(norm)))
    return samples


def load_split_samples(cfg: Dict[str, object]) -> Tuple[List[WindowSample], List[WindowSample], List[WindowSample]]:
    data_cfg = cfg["data"]
    train_files, val_files, test_files = split_train_val_test(
        str(data_cfg["data_dir"]),
        str(data_cfg.get("action_filter", "")),
        int(cfg["seed"]),
    )
    kwargs = dict(
        input_n=int(data_cfg["input_n"]),
        output_n=int(data_cfg["output_n"]),
        stride=int(data_cfg["stride"]),
        time_interp=data_cfg.get("time_interp", None),
        window_norm=data_cfg.get("window_norm", None),
    )
    train = _build_windows(train_files, **kwargs)
    val = _build_windows(val_files, **kwargs)
    test = _build_windows(test_files, **kwargs)
    return train, val, test


def make_loader(samples: List[WindowSample], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(WindowDataset(samples), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)


def evaluate_model(
    model: BelFusionCVAE,
    test_loader: DataLoader,
    num_samples: int,
    threshold: float,
    output_n: int,
    n_nodes: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    mpjpe_vals: List[torch.Tensor] = []
    mpjpe_norm_vals: List[torch.Tensor] = []
    hm_acc = {"APD": 0.0, "ADE": 0.0, "FDE": 0.0, "MMADE": 0.0, "MMFDE": 0.0, "CMD": 0.0, "FID": 0.0}
    n_batches = 0

    with torch.no_grad():
        for obs, fut, norm in test_loader:
            obs = obs.to(device)
            fut = fut.to(device)
            norm = norm.to(device)
            bs = obs.shape[0]

            obs_flat = obs.reshape(bs, -1)
            sampled = model.sample(obs_flat, num_samples=num_samples)
            pred = sampled.reshape(num_samples, bs, output_n, n_nodes * 3)
            gt = fut.reshape(bs, output_n, n_nodes * 3)
            start_pose = obs[:, -1, :, :]

            res = splineeqnet_diffusion_batch_eval(
                pred_candidates=pred,
                gt_future=gt,
                start_pose=start_pose,
                norm_factor=norm,
                threshold=threshold,
            )
            mpjpe_vals.append(res["per_sample_mpjpe"].detach().cpu())
            mpjpe_norm_vals.append(res["per_sample_mpjpe_norm"].detach().cpu())
            hm = res["humanmac"]
            for k in hm_acc:
                hm_acc[k] += float(hm[k])
            n_batches += 1

    if not mpjpe_vals:
        raise RuntimeError("No valid test windows were produced for BeLFusion evaluation.")

    out = {
        "MPJPE": float(torch.cat(mpjpe_vals).mean().item()),
        "MPJPE_norm": float(torch.cat(mpjpe_norm_vals).mean().item()),
    }
    denom = float(max(1, n_batches))
    out.update({k: v / denom for k, v in hm_acc.items()})
    return out


def train(cfg: Dict[str, object]) -> Dict[str, float]:
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples, val_samples, test_samples = load_split_samples(cfg)
    if not train_samples:
        raise RuntimeError("BeLFusion train split has zero windows after preprocessing.")
    if not test_samples:
        raise RuntimeError("BeLFusion test split has zero windows after preprocessing.")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    model_cfg = cfg["model"]
    runtime_cfg = cfg["runtime"]

    input_n = int(data_cfg["input_n"])
    output_n = int(data_cfg["output_n"])
    n_nodes = int(data_cfg.get("n_nodes", 21))
    obs_dim = input_n * n_nodes * 3
    fut_dim = output_n * n_nodes * 3

    model = BelFusionCVAE(
        obs_dim=obs_dim,
        fut_dim=fut_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
    ).to(device)

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = make_loader(train_samples, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_samples, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_samples else None
    test_loader = make_loader(test_samples, batch_size=int(eval_cfg.get("batch_size", batch_size)), shuffle=False, num_workers=num_workers)

    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    epochs = int(train_cfg["epochs"])
    kl_weight = float(train_cfg.get("kl_weight", 1e-3))
    l1_weight = float(train_cfg.get("l1_weight", 1.0))
    es_enabled = bool(train_cfg.get("early_stopping_enabled", False))
    es_patience = max(1, int(train_cfg.get("early_stopping_patience", 20)))
    es_min_delta = float(train_cfg.get("early_stopping_min_delta", 1e-4))
    es_warmup = max(0, int(train_cfg.get("early_stopping_warmup", 0)))
    es_monitor = str(train_cfg.get("early_stopping_monitor", "auto")).strip().lower()
    es_best = None
    es_bad_epochs = 0
    es_best_state = None

    for _epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        for obs, fut, _norm in train_loader:
            obs = obs.to(device)
            fut = fut.to(device)
            bs = obs.shape[0]
            obs_flat = obs.reshape(bs, -1)
            fut_flat = fut.reshape(bs, -1)

            recon, mu, logvar = model(obs_flat, fut_flat)
            recon_loss = F.mse_loss(recon, fut_flat) + l1_weight * F.l1_loss(recon, fut_flat)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_weight * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss_sum += float(loss.item())
            train_steps += 1

        train_loss_epoch = train_loss_sum / max(1, train_steps)
        val_loss_epoch = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for obs, fut, _norm in val_loader:
                    obs = obs.to(device)
                    fut = fut.to(device)
                    bs = obs.shape[0]
                    obs_flat = obs.reshape(bs, -1)
                    fut_flat = fut.reshape(bs, -1)
                    recon, mu, logvar = model(obs_flat, fut_flat)
                    recon_loss = F.mse_loss(recon, fut_flat) + l1_weight * F.l1_loss(recon, fut_flat)
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    val_loss = recon_loss + kl_weight * kl
                    val_loss_sum += float(val_loss.item())
                    val_steps += 1
            val_loss_epoch = val_loss_sum / max(1, val_steps)

        if es_enabled and _epoch > es_warmup:
            if es_monitor == "val_loss":
                monitored = val_loss_epoch
            elif es_monitor in {"train_loss", "loss"}:
                monitored = train_loss_epoch
            else:
                monitored = val_loss_epoch if val_loss_epoch is not None else train_loss_epoch

            if monitored is not None and np.isfinite(float(monitored)):
                monitored_f = float(monitored)
                improved = es_best is None or monitored_f < (float(es_best) - es_min_delta)
                if improved:
                    es_best = monitored_f
                    es_bad_epochs = 0
                    es_best_state = copy.deepcopy(model.state_dict())
                else:
                    es_bad_epochs += 1
                    if es_bad_epochs >= es_patience:
                        print(
                            f"[BeLFusion][EarlyStop] epoch={_epoch} monitor={es_monitor or 'auto'} "
                            f"best={es_best:.6f} current={monitored_f:.6f}"
                        )
                        break

    if es_best_state is not None:
        model.load_state_dict(es_best_state)

    out_dir = Path(str(runtime_cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        num_samples=int(eval_cfg["num_candidates"]),
        threshold=float(eval_cfg["multimodal_threshold"]),
        output_n=output_n,
        n_nodes=n_nodes,
        device=device,
    )

    out_csv = Path(str(runtime_cfg["metrics_csv"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["MPJPE", "MPJPE_norm", "APD", "ADE", "FDE", "MMADE", "MMFDE", "CMD", "FID"])
        w.writeheader()
        w.writerow(metrics)

    meta = {
        "num_train_windows": len(train_samples),
        "num_val_windows": len(val_samples),
        "num_test_windows": len(test_samples),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return metrics


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BeLFusion (vendored) training + eval for diffusion_hands")
    p.add_argument("--config", required=True, help="Path to resolved belfusion YAML config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--action-filter", default=None)
    p.add_argument("--num-candidates", type=int, default=None)
    p.add_argument("--multimodal-threshold", type=float, default=None)
    return p


def main() -> None:
    args = create_parser().parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.data_dir is not None:
        cfg.setdefault("data", {})["data_dir"] = str(args.data_dir)
    if args.action_filter is not None:
        cfg.setdefault("data", {})["action_filter"] = str(args.action_filter)
    if args.num_candidates is not None:
        cfg.setdefault("eval", {})["num_candidates"] = int(args.num_candidates)
    if args.multimodal_threshold is not None:
        cfg.setdefault("eval", {})["multimodal_threshold"] = float(args.multimodal_threshold)

    train(cfg)


if __name__ == "__main__":
    main()
