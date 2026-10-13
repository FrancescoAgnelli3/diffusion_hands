from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


TRAVERSAL_LEVELS: Tuple[float, ...] = (-2.0, -0.25, 0.0, 0.25, 2.0)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _representatives_by_action(dataset) -> List[Tuple[int, str, tuple]]:
    id_to_action = list(getattr(dataset, "id_to_action", []))
    representatives: Dict[int, tuple] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        if not isinstance(item, (tuple, list)) or len(item) < 4 or not isinstance(item[3], dict):
            continue
        action_id = item[3].get("action_id")
        if not torch.is_tensor(action_id):
            continue
        action_idx = int(action_id.item())
        representatives.setdefault(action_idx, tuple(item))
        if len(representatives) >= len(id_to_action):
            break
    return [
        (
            action_idx,
            id_to_action[action_idx] if 0 <= action_idx < len(id_to_action) else str(action_idx),
            representatives[action_idx],
        )
        for action_idx in sorted(representatives)
    ]


@torch.no_grad()
def generate_latent_explainability_artifacts(
    model,
    eval_loader,
    *,
    device: torch.device,
    output_dir: str,
    ddim_steps: int,
    seed: int = 0,
    traversal_levels: Sequence[float] = TRAVERSAL_LEVELS,
) -> Dict[str, object]:
    if not output_dir or not bool(getattr(model, "explainable_latent_decomposition", False)):
        return {}
    dataset = getattr(eval_loader, "dataset", None)
    if dataset is None:
        return {}
    representatives = _representatives_by_action(dataset)
    if not representatives:
        return {}

    out_dir = Path(output_dir)
    plot_dir = out_dir / "plots"
    traversal_dir = out_dir / "traversals"
    plot_dir.mkdir(parents=True, exist_ok=True)
    traversal_dir.mkdir(parents=True, exist_ok=True)
    levels = tuple(float(value) for value in traversal_levels)
    if -0.25 not in levels or 0.25 not in levels:
        raise ValueError("Traversal levels must contain -0.25 and 0.25 for central differences.")
    neg_idx, pos_idx = levels.index(-0.25), levels.index(0.25)
    latent_dim = int(model.cfg.latent_dim)

    all_dimension_scores: List[np.ndarray] = []
    action_ids: List[int] = []
    action_names: List[str] = []
    summary_rows: List[Dict[str, object]] = []

    was_training = model.training
    model.eval()
    try:
        for action_idx, action_name, item in representatives:
            inp, target = item[0], item[1]
            history = inp[..., 4:].unsqueeze(0).to(device=device, dtype=torch.float32)
            target_xyz = target[..., 4:].to(dtype=torch.float32)
            action = torch.tensor([action_idx], device=device, dtype=torch.long)
            coarse = model._zero_coarse_future(history) if model.cfg.diffusion_only else model.coarse(history)
            _sample, prior_mu, prior_logvar = model.sample_prior_latent(
                history, action_id=action, coarse_future=coarse, seed=seed + action_idx
            )
            prior_std = torch.exp(0.5 * prior_logvar)

            variants: List[torch.Tensor] = []
            latent_values = torch.empty(latent_dim, len(levels), device=device)
            for dim_idx in range(latent_dim):
                for level_idx, level in enumerate(levels):
                    latent = prior_mu.clone()
                    latent[:, dim_idx] = prior_mu[:, dim_idx] + float(level) * prior_std[:, dim_idx]
                    variants.append(latent)
                    latent_values[dim_idx, level_idx] = latent[0, dim_idx]
            latent_batch = torch.cat(variants, dim=0)
            batch_size = int(latent_batch.size(0))
            history_batch = history.expand(batch_size, -1, -1, -1).contiguous()
            coarse_batch = coarse.expand(batch_size, -1, -1, -1).contiguous()
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) + action_idx * 1009)
            base_noise = torch.randn(
                1,
                int(model.cfg.pred_length),
                int(model.diffusion.free_feature_dim),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            noise_batch = base_noise.expand(batch_size, -1, -1).contiguous()
            predictions = model.predict_with_latent(
                history_batch,
                latent_batch,
                coarse_future=coarse_batch,
                seed=seed + action_idx,
                steps=int(ddim_steps),
                initial_noise=noise_batch,
            )
            predictions = predictions.reshape(
                latent_dim,
                len(levels),
                int(model.cfg.pred_length),
                int(model.cfg.num_nodes),
                3,
            )
            sensitivity = (predictions[:, pos_idx] - predictions[:, neg_idx]) / 0.5

            sensitivity_np = sensitivity.cpu().numpy()
            all_dimension_scores.append(np.linalg.norm(sensitivity_np, axis=-1).mean(axis=(1, 2)))
            action_ids.append(action_idx)
            action_names.append(action_name)
            np.savez_compressed(
                traversal_dir / f"{action_idx:04d}_{_safe_name(action_name)}.npz",
                action_id=np.asarray(action_idx, dtype=np.int64),
                action_name=np.asarray(action_name),
                history=history[0].cpu().numpy(),
                target=target_xyz.cpu().numpy(),
                latent_values=latent_values.cpu().numpy(),
                traversal_levels=np.asarray(levels, dtype=np.float32),
                trajectories=predictions.cpu().numpy(),
                sensitivity=sensitivity_np,
            )
            _plot_action_sensitivity(
                plot_dir / f"{action_idx:04d}_{_safe_name(action_name)}_joint_time.png",
                action_name=action_name,
                sensitivity=sensitivity_np,
                latent_slices=model.latent_slices,
            )

            for dim_idx in range(latent_dim):
                factor = next(
                    name for name, component_slice in model.latent_slices.items()
                    if component_slice.start <= dim_idx < component_slice.stop
                )
                sens = sensitivity[dim_idx]
                extreme_delta = predictions[dim_idx, -1] - predictions[dim_idx, 0]
                velocity_delta = extreme_delta[1:] - extreme_delta[:-1]
                summary_rows.append({
                    "action": action_name,
                    "action_id": action_idx,
                    "factor": factor,
                    "dimension": dim_idx,
                    "sensitivity_l2": float(torch.linalg.norm(sens, dim=-1).mean().item()),
                    "terminal_effect": float(torch.linalg.norm(extreme_delta[-1], dim=-1).mean().item()),
                    "mean_velocity_effect": float(
                        torch.linalg.norm(velocity_delta, dim=-1).mean().item()
                        if velocity_delta.numel() else 0.0
                    ),
                })
    finally:
        if was_training:
            model.train()

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "action_ids": action_ids,
            "action_names": action_names,
            "latent_slices": {
                name: [component.start, component.stop] for name, component in model.latent_slices.items()
            },
            "traversal_levels": levels,
            "ddim_steps": int(ddim_steps),
            "shared_initial_noise": True,
            "sensitivity_delta": 0.25,
            "per_action_directory": "traversals",
        }, handle, indent=2, sort_keys=True)

    summary_path = out_dir / "latent_sensitivity.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    factor_rows: List[Dict[str, object]] = []
    for action_name in action_names:
        for factor in model.latent_slices:
            rows = [r for r in summary_rows if r["action"] == action_name and r["factor"] == factor]
            factor_rows.append({
                "action": action_name,
                "factor": factor,
                "sensitivity_l2": float(np.mean([r["sensitivity_l2"] for r in rows])),
                "terminal_effect": float(np.mean([r["terminal_effect"] for r in rows])),
                "mean_velocity_effect": float(np.mean([r["mean_velocity_effect"] for r in rows])),
            })
    with (out_dir / "factor_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(factor_rows[0]))
        writer.writeheader()
        writer.writerows(factor_rows)

    _plot_artifacts(
        plot_dir,
        action_names=action_names,
        dimension_scores=np.stack(all_dimension_scores),
        latent_slices=model.latent_slices,
    )
    return {
        "actions": len(action_names),
        "dimensions": latent_dim,
        "output_dir": str(out_dir),
    }


def _plot_artifacts(
    plot_dir: Path,
    *,
    action_names: Sequence[str],
    dimension_scores: np.ndarray,
    latent_slices: Dict[str, slice],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim_scores = dimension_scores
    fig, ax = plt.subplots(
        figsize=(max(8, dim_scores.shape[1] * 0.65), min(24, max(5, len(action_names) * 0.22)))
    )
    image = ax.imshow(dim_scores, aspect="auto", cmap="magma")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Action")
    ax.set_xticks(range(dim_scores.shape[1]))
    if len(action_names) <= 60:
        ax.set_yticks(range(len(action_names)), labels=action_names, fontsize=6)
    fig.colorbar(image, ax=ax, label="Mean sensitivity")
    fig.tight_layout()
    fig.savefig(plot_dir / "action_dimension_sensitivity.png", dpi=180)
    plt.close(fig)

    factor_names = list(latent_slices)
    factor_scores = np.stack([
        dim_scores[:, component].mean(axis=1) for component in latent_slices.values()
    ], axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(factor_names, factor_scores.mean(axis=0))
    ax.set_ylabel("Mean sensitivity")
    ax.set_title("Latent factor influence")
    fig.tight_layout()
    fig.savefig(plot_dir / "factor_sensitivity.png", dpi=180)
    plt.close(fig)



def _plot_action_sensitivity(
    path: Path,
    *,
    action_name: str,
    sensitivity: np.ndarray,
    latent_slices: Dict[str, slice],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    magnitude = np.linalg.norm(sensitivity, axis=-1)  # D,T,N
    factor_names = list(latent_slices)
    factor_time_joint = np.stack([
        magnitude[component].mean(axis=0) for component in latent_slices.values()
    ])
    fig, axes = plt.subplots(1, len(factor_names), figsize=(4 * len(factor_names), 3), squeeze=False)
    for factor_idx, factor_name in enumerate(factor_names):
        ax = axes[0, factor_idx]
        image = ax.imshow(factor_time_joint[factor_idx].T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(factor_name)
        ax.set_xlabel("Future frame")
        ax.set_ylabel("Joint")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(action_name)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
