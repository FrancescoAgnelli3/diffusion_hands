from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset_graphs import resolve_local_hand_graph_metadata


CACHE_PATH = ROOT / "out/bayesian_signed_rank/prediction_cache/star_hand_predictions.pt"
OUTPUT_DIR = ROOT / "results/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "belfusion": ROOT
    / "out/diffusion_hands_runs/belfusion/assembly_pick_up_screwd_20260420_103637/eval_samples.npz",
    "gsps": ROOT
    / "out/diffusion_hands_runs/gsps/assembly_pick_up_screwd_20260420_103637/eval_samples.npz",
    "humanmac": ROOT
    / "out/diffusion_hands_runs/humanmac/assembly_pick_up_screwd_20260420_103637/eval_samples.npz",
}

TITLES = {"star_hand": "STAR-Hand", "belfusion": "BeLFusion", "gsps": "GSPS", "humanmac": "HumanMAC"}
LIGHT = {"star_hand": "#D99A22", "belfusion": "#6A9FD0", "gsps": "#C06CA9", "humanmac": "#6FAF63"}
DARK = {"star_hand": "#8A5900", "belfusion": "#245B8A", "gsps": "#7B2C68", "humanmac": "#2F6F28"}
GT_COLOR = "#2CA02C"
PROJECTION = (0, 1)
MOMENTS = (0, 7, 14, 21, 29)


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def digest(array: np.ndarray) -> bytes:
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).digest()


def adapt(array: np.ndarray, model: str) -> np.ndarray:
    """Transform canonical 21-joint arrays to a competitor's saved representation."""
    if model == "belfusion":
        return array
    if model == "gsps":
        # Integrated GSPS drops canonical node 0 and prepends an artificial zero root.
        zeros = np.zeros((*array.shape[:-2], 1, 3), dtype=array.dtype)
        return np.concatenate([zeros, array[..., 1:, :]], axis=-2)
    if model == "humanmac":
        # Integrated HumanMAC stores the canonical hand without node 0.
        return array[..., 1:, :]
    raise ValueError(model)


def graph_for(model: str, canonical_links, canonical_wrist: int):
    if model in {"belfusion", "gsps"}:
        # GSPS does not retain canonical node 0; omit its incident bone.
        links = canonical_links if model == "belfusion" else tuple((a, b) for a, b in canonical_links if a != 0 and b != 0)
        return tuple(links), canonical_wrist
    if model == "humanmac":
        links = tuple((a - 1, b - 1) for a, b in canonical_links if a != 0 and b != 0)
        return links, canonical_wrist - 1
    raise ValueError(model)


def map_saved_examples(saved_obs: np.ndarray, canonical_context: np.ndarray, model: str) -> list[int]:
    adapted_context = adapt(canonical_context, model)
    lookup = {digest(row): index for index, row in enumerate(adapted_context)}
    indices = [lookup.get(digest(row)) for row in saved_obs]
    if any(index is None for index in indices):
        missing = [position for position, index in enumerate(indices) if index is None]
        raise RuntimeError(f"{model}: could not align saved examples at positions {missing}")
    return [int(index) for index in indices]


def oracle_mpjpe(candidates: np.ndarray, target: np.ndarray):
    errors = np.linalg.norm(candidates - target[None], axis=-1).mean(axis=(1, 2))
    best = int(errors.argmin())
    return best, float(errors[best])


def motion_values(target: np.ndarray, wrist: int):
    rooted = target - target[:, wrist : wrist + 1]
    visible = rooted[..., list(PROJECTION)]
    endpoint = float(np.linalg.norm(visible[-1] - visible[0], axis=-1).mean())
    spread = float(np.linalg.norm(visible - visible[0:1], axis=-1).mean())
    return endpoint, spread


def select_two(rows: list[dict]) -> list[dict]:
    """Use the more visibly moving half, then prefer good oracle coverage."""
    endpoints = np.asarray([row["endpoint_motion"] for row in rows])
    spreads = np.asarray([row["temporal_spread"] for row in rows])
    endpoint_rank = np.argsort(np.argsort(endpoints)) / max(1, len(rows) - 1)
    spread_rank = np.argsort(np.argsort(spreads)) / max(1, len(rows) - 1)
    visible_score = np.minimum(endpoint_rank, spread_rank)
    pool_size = max(4, int(math.ceil(len(rows) / 2)))
    pool_indices = np.argsort(-visible_score)[:pool_size]
    pool = [rows[int(index)] for index in pool_indices]
    pool.sort(key=lambda row: (row["best_any_mpjpe"], -row["endpoint_motion"], -row["temporal_spread"]))
    return pool[:2]


def project(points):
    return np.asarray(points)[..., list(PROJECTION)]


def row_limits(poses, wrist: int, padding: float = 0.06):
    points = np.concatenate([project(pose - pose[wrist]).reshape(-1, 2) for pose in poses], axis=0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return lo - padding * span, hi + padding * span


def plot_skeleton(ax, pose, links, wrist, color, lw=1.2, alpha=1.0, zorder=2):
    xy = project(pose - pose[wrist])
    for a, b in links:
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=color, lw=lw, alpha=alpha, zorder=zorder)
    ax.scatter(xy[:, 0], xy[:, 1], s=4.5, color=color, alpha=alpha, zorder=zorder + 1, linewidths=0)


def style_axis(ax, lo, hi):
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure(row: dict, model: str, ordinal: int, links, wrist):
    target = row["target"]
    star = row["star_candidates"]
    competitor = row["competitor_candidates"]
    candidates_by_name = {"star_hand": star, model: competitor}
    best_by_name = {"star_hand": row["star_best_index"], model: row["competitor_best_index"]}

    limits = {"ground_truth": row_limits([target[frame] for frame in MOMENTS], wrist)}
    for name, candidates in candidates_by_name.items():
        limits[name] = row_limits([pose for frame in MOMENTS for pose in candidates[:, frame]], wrist)

    fig, axes = plt.subplots(3, 5, figsize=(15.5, 8.3))
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.04, top=0.83, hspace=0.08, wspace=0.035)
    gt_lo, gt_hi = limits["ground_truth"]
    for column, frame in enumerate(MOMENTS):
        ax = axes[0, column]
        plot_skeleton(ax, target[frame], links, wrist, GT_COLOR, lw=2.15, zorder=4)
        style_axis(ax, gt_lo, gt_hi)
        ax.set_title(f"$t={frame + 1}$", fontsize=22, pad=8)
        if column == 0:
            ax.set_ylabel("Ground truth", fontsize=24, fontweight="semibold", rotation=0, labelpad=82, va="center")

    for model_row, name in enumerate(("star_hand", model), start=1):
        lo, hi = limits[name]
        model_candidates = candidates_by_name[name]
        for column, frame in enumerate(MOMENTS):
            ax = axes[model_row, column]
            for pose in model_candidates[:, frame]:
                plot_skeleton(ax, pose, links, wrist, LIGHT[name], lw=0.75, alpha=0.18, zorder=1)
            best_pose = model_candidates[best_by_name[name], frame]
            plot_skeleton(ax, best_pose, links, wrist, DARK[name], lw=2.15, alpha=0.98, zorder=5)
            style_axis(ax, lo, hi)
            if column == 0:
                ax.set_ylabel(TITLES[name], fontsize=24, fontweight="semibold", rotation=0, labelpad=72, va="center")

    legend = [
        Line2D([0], [0], color=GT_COLOR, lw=2.0, label="Ground truth"),
        Line2D([0], [0], color="#999999", lw=1.0, alpha=0.5, label="Sampled hands (K=10)"),
        Line2D([0], [0], color="#444444", lw=2.1, label="Oracle best-of-10"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3, frameon=False, fontsize=18)
    stem = OUTPUT_DIR / f"additional_competitors_{model}_example_{ordinal:02d}_index_{row['test_index']}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return stem


def main():
    graph = resolve_local_hand_graph_metadata("assembly", (5, 26))
    canonical_wrist = int(graph["wrist_index"])
    canonical_links = tuple(graph["links"])
    cache = load_torch(CACHE_PATH)
    canonical_context = cache["context"].numpy()
    canonical_target = cache["gt_future"].numpy()
    canonical_star = cache["pred_candidates"].numpy()

    report = {
        "selection": "top moving half of each sparse saved set, then lowest oracle MPJPE from STAR-Hand or competitor",
        "models": {},
    }
    for model, source in SOURCES.items():
        saved = np.load(source, allow_pickle=False)
        indices = map_saved_examples(saved["obs"], canonical_context, model)
        links, wrist = graph_for(model, canonical_links, canonical_wrist)
        rows = []
        for saved_position, test_index in enumerate(indices):
            target = saved["target"][saved_position]
            expected_target = adapt(canonical_target[test_index], model)
            if not np.array_equal(target, expected_target):
                raise RuntimeError(f"{model}: target mismatch for aligned test index {test_index}")
            star_candidates = adapt(canonical_star[test_index], model)
            competitor_candidates = saved["pred_all"][saved_position]
            star_best_index, star_error = oracle_mpjpe(star_candidates, target)
            competitor_best_index, competitor_error = oracle_mpjpe(competitor_candidates, target)
            endpoint, spread = motion_values(target, wrist)
            rows.append(
                {
                    "saved_position": saved_position,
                    "test_index": test_index,
                    "target": target,
                    "star_candidates": star_candidates,
                    "competitor_candidates": competitor_candidates,
                    "star_best_index": star_best_index,
                    "competitor_best_index": competitor_best_index,
                    "star_hand_mpjpe": star_error,
                    f"{model}_mpjpe": competitor_error,
                    "best_any_mpjpe": min(star_error, competitor_error),
                    "endpoint_motion": endpoint,
                    "temporal_spread": spread,
                }
            )
        selected = select_two(rows)
        report_rows = []
        for ordinal, row in enumerate(selected, start=1):
            stem = make_figure(row, model, ordinal, links, wrist)
            winner = "star_hand" if row["star_hand_mpjpe"] <= row[f"{model}_mpjpe"] else model
            report_rows.append(
                {
                    "figure": ordinal,
                    "test_index": row["test_index"],
                    "star_hand_mpjpe": row["star_hand_mpjpe"],
                    f"{model}_mpjpe": row[f"{model}_mpjpe"],
                    "winner": winner,
                    "endpoint_motion": row["endpoint_motion"],
                    "temporal_spread": row["temporal_spread"],
                    "png": str(stem.with_suffix(".png")),
                    "pdf": str(stem.with_suffix(".pdf")),
                }
            )
            print(f"Saved {stem.with_suffix('.png')}")
        report["models"][model] = {
            "source": str(source),
            "available_saved_examples": len(rows),
            "selected": report_rows,
        }

    report_path = OUTPUT_DIR / "additional_competitor_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
