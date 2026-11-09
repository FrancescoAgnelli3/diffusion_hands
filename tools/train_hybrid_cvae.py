#!/usr/bin/env python3
"""Train Hybrid-CVAE with diffusion_hands' shared data protocol.

Only the Hybrid-CVAE network and its loss are loaded from the upstream
checkout.  Dataset loading, file splitting, preprocessing, windows, action
filtering, candidate count, and benchmark metrics are the shared suite's.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.evaluation import save_eval_samples_npz
from common.metrics import humanmac_metrics
from common.dataset_graphs import get_dataset_graph_metadata


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_classes(source_root: Path):
    if not (source_root / "models" / "hybrid_hyperbolic_cvae.py").is_file():
        raise FileNotFoundError(f"Hybrid-CVAE models not found under {source_root}")
    # ``models`` is an upstream package; the shared pipeline has no package of
    # that name, so this does not replace its data/metrics imports.
    sys.path.insert(0, str(source_root))
    from models.hybrid_hyperbolic_cvae import HybridConfig, HybridHyperbolicCVAE
    return HybridConfig, HybridHyperbolicCVAE


def _wrist_first_order(dataset: str) -> torch.Tensor:
    """Return the shared loader's active-hand order with its wrist first."""
    groups = tuple(get_dataset_graph_metadata(dataset)["hand_groups"])
    if not groups:
        raise ValueError(f"No hand graph configured for dataset={dataset!r}")
    group = groups[0]
    nodes = tuple(int(node) for node in group["nodes"])
    wrist = int(group["wrist_index"])
    try:
        wrist_local = nodes.index(wrist)
    except ValueError as exc:
        raise ValueError(f"Configured wrist is absent from {dataset!r} hand nodes") from exc
    return torch.tensor((wrist_local, *[i for i in range(len(nodes)) if i != wrist_local]))


def _coordinates(batch: tuple, device: torch.device, wrist_order: torch.Tensor):
    observed, future, norm, aux = batch
    # Shared SplineEqNet datasets expose [unit velocity direction, speed, xyz].
    # Hybrid-CVAE consumes only wrist-relative Cartesian coordinates and its FK
    # definition requires that wrist at joint 0.  This is a model-input view,
    # not an alternate data preprocessing pipeline.
    order = wrist_order.to(device)
    return (
        observed[..., 4:].to(device).float().index_select(2, order),
        future[..., 4:].to(device).float().index_select(2, order),
        norm,
        aux,
    )


@torch.no_grad()
def _evaluate(model, loader, device: torch.device, candidates: int, threshold: float, wrist_order: torch.Tensor, save_path: Path | None = None):
    model.eval()
    sampled, targets, contexts, sequence_ids = [], [], [], []
    obs_for_cache, pred_for_cache = [], []
    for batch in loader:
        observed, future, _norm, aux = _coordinates(batch, device, wrist_order)
        predictions = model.sample_predict(observed, candidates)  # B,K,T,J,3
        sampled.append(predictions.permute(1, 0, 2, 3, 4).flatten(-2).cpu())
        targets.append(future.flatten(-2).cpu())
        contexts.append(observed[:, -1].flatten(1).cpu())
        if isinstance(aux, dict) and "sequence_id" in aux:
            sequence_ids.append(aux["sequence_id"].cpu())
        obs_for_cache.append(observed.cpu())
        pred_for_cache.append(predictions.cpu())

    pred = torch.cat(sampled, dim=1)
    target = torch.cat(targets, dim=0)
    context = torch.cat(contexts, dim=0)
    ids = torch.cat(sequence_ids, dim=0) if sequence_ids else None
    metrics = humanmac_metrics(
        pred, target, context, threshold=threshold, sequence_ids=ids
    )
    if save_path is not None:
        save_eval_samples_npz(
            save_path,
            obs=torch.cat(obs_for_cache), target=target.reshape(target.shape[0], target.shape[1], -1, 3),
            pred=pred[0].reshape(target.shape[0], target.shape[1], -1, 3),
            pred_all=pred.permute(1, 0, 2, 3).reshape(target.shape[0], candidates, target.shape[1], -1, 3),
            metadata={"model": "hybrid_cvae", "num_candidates": candidates},
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--action-filter", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-n", type=int, required=True)
    parser.add_argument("--output-n", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--time-interp", type=int)
    parser.add_argument("--window-norm", type=int)
    parser.add_argument("--eval-batch-mult", type=int, default=1)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--components", type=int, default=1)
    parser.add_argument("--geometry", choices=("hyperbolic", "euclidean"), default="hyperbolic")
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--hierarchy-weight", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--velocity-weight", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-eval-samples")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if min(args.input_n, args.output_n, args.stride, args.batch_size, args.epochs, args.num_candidates) < 1:
        parser.error("window, training, and candidate sizes must be positive")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)
    device = torch.device(args.device)

    # Import the shared loader before loading the upstream model package.
    spline_root = ROOT / "vendor" / "splineeqnet"
    sys.path.insert(0, str(spline_root))
    from config import DatasetCfg
    from data import build_loaders
    HybridConfig, HybridHyperbolicCVAE = _load_model_classes(Path(args.source_root))
    wrist_order = _wrist_first_order(args.dataset)
    loaders = build_loaders(DatasetCfg(
        dataset=args.dataset, data_dir=args.data_dir, action_filter=args.action_filter,
        input_n=args.input_n, output_n=args.output_n, stride=args.stride,
        time_interp=args.time_interp, window_norm=args.window_norm,
        batch_size=args.batch_size, eval_batch_mult=args.eval_batch_mult,
        seed=args.seed, data_seed=args.data_seed,
    ))
    train_loader, val_loader, test_loader = loaders
    model = HybridHyperbolicCVAE(HybridConfig(
        n_joints=21, tin=args.input_n, tout=args.output_n, hidden=args.hidden,
        latent=args.latent, components=args.components, geometry=args.geometry,
        curvature=args.curvature, hierarchy_weight=args.hierarchy_weight,
        beta=args.beta, velocity_weight=args.velocity_weight,
    )).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = args.beta * min(1.0, epoch / max(1, args.warmup))
        total, count = 0.0, 0
        for batch in train_loader:
            observed, future, _norm, _aux = _coordinates(batch, device, wrist_order)
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_losses(observed, future, beta=beta)
            if not torch.isfinite(losses["total"]):
                raise ValueError("Nonfinite Hybrid-CVAE loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(losses["total"].detach()) * len(observed)
            count += len(observed)
        with torch.random.fork_rng():
            torch.manual_seed(args.seed + 1000)
            validation = _evaluate(model, val_loader, device, args.num_candidates, args.threshold, wrist_order)
        checkpoint = {"model_state_dict": model.state_dict(), "cfg": model.cfg.to_dict(), "epoch": epoch, "args": vars(args)}
        torch.save(checkpoint, out_dir / "ckpt_last.pt")
        if validation["ADE"] < best - 1e-4:
            best, stale = validation["ADE"], 0
            torch.save(checkpoint, out_dir / "ckpt_best.pt")
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": total / max(1, count), "validation": validation})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=True) + "\n")
        if args.patience and stale >= args.patience:
            break

    best_checkpoint = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    metrics = _evaluate(
        model, test_loader, device, args.num_candidates, args.threshold, wrist_order,
        Path(args.save_eval_samples) if args.save_eval_samples else None,
    )
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
