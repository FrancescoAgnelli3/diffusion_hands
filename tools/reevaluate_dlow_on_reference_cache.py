#!/usr/bin/env python3
"""Sample a DLow VAE on the exact histories stored in a GRIP reference cache.

This deliberately avoids a second dataset/windowing implementation.  DLow uses
the same root-relative 20-joint representation as training, while targets and
sequence IDs are copied from the reference cache in exactly the same order.
"""
import argparse
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "dlow"))
from models.motion_pred import get_vae_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_candidates < 1 or args.batch_size < 1:
        parser.error("--num-candidates and --batch-size must both be positive")

    cfg = SimpleNamespace(**(yaml.safe_load(args.config.read_text()) or {}))
    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    context = reference["conditioning_context"].float()
    target = reference["gt_future"].float()
    if context.ndim != 4 or context.shape[-2:] != (21, 3):
        raise ValueError("reference cache must contain [N,H,21,3] Assembly histories")
    if target.shape[-2:] != (21, 3):
        raise ValueError("reference cache must contain [N,T,21,3] Assembly targets")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The legacy VAE creates recurrent hidden states with torch.zeros() without
    # passing a dtype; match its original float64 evaluation environment.
    torch.set_default_dtype(torch.float64)
    model = get_vae_model(cfg, traj_dim=60).to(device=device, dtype=torch.float64)
    state = pickle.load(args.checkpoint.open("rb")).get("model_dict")
    model.load_state_dict(state)
    model.eval()
    all_predictions = []
    total = int(context.shape[0])
    batches = (total + args.batch_size - 1) // args.batch_size
    print(
        f"[DLow cache] Sampling {total} windows × {args.num_candidates} candidates "
        f"in {batches} batch(es) on {device}.",
        flush=True,
    )
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, total, args.batch_size), start=1):
            # DatasetAssembly trains DLow in a wrist-relative coordinate
            # system (``seq[:, 1:] -= seq[:, :1]``).  Dropping joint zero by
            # itself leaves the remaining joints in global coordinates, which
            # makes DLow predictions and targets incomparable.  Apply the
            # same per-frame subtraction used by DatasetAssembly.
            history_batch = context[start : start + args.batch_size]
            history = (history_batch[:, :, 1:, :] - history_batch[:, :, :1, :]).to(
                device=device, dtype=torch.float64
            )
            x = history.reshape(history.shape[0], history.shape[1], -1).permute(1, 0, 2)
            x_rep = x.repeat((1, args.num_candidates, 1))
            sampled = model.sample_prior(x_rep)
            # ``repeat`` lays the expanded batch out candidate-major:
            # [all samples for candidate 0, all samples for candidate 1, ...].
            # Preserve that layout while restoring [K, B, T, J, 3].  Reshaping
            # directly to [B, K, ...] silently associates a prediction with a
            # different reference-cache target whenever B > 1.
            pred = sampled.permute(1, 0, 2).reshape(
                args.num_candidates, x.shape[1], int(cfg.t_pred), 20, 3
            ).cpu().float()
            all_predictions.append(pred)
            if batch_index == 1 or batch_index == batches or batch_index % 50 == 0:
                print(f"[DLow cache] {min(start + args.batch_size, total)}/{total} windows", flush=True)
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "pred_candidates": torch.cat(all_predictions, dim=1),
        # Keep every tensor in the same wrist-relative representation as the
        # DLow model.  ``sweep_multimodal_threshold.py`` validates these
        # tensors against the GRIP cache after it canonicalizes GRIP in the
        # identical way.
        "gt_future": (target[:, :, 1:, :] - target[:, :, :1, :]).contiguous(),
        "conditioning_context": (context[:, :, 1:, :] - context[:, :, :1, :]).contiguous(),
        "sequence_ids": reference.get("sequence_ids"),
        "metadata": {"model": "dlow_cvae", "reference_cache": str(args.reference_cache),
                     "num_candidates": args.num_candidates, "root_joint_removed": True},
    }, args.output_cache)
    print(f"[DLow cache] Wrote {args.output_cache}", flush=True)


if __name__ == "__main__":
    main()
