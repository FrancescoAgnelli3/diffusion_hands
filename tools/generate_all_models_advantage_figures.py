from __future__ import annotations

import hashlib
import itertools
import json
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


CACHE_DIR = ROOT / "out/bayesian_signed_rank/prediction_cache"
OUTPUT_DIR = ROOT / "results/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUR_MODELS = ("card", "fihard", "star_hand")
FULL_MODELS = (*OUR_MODELS, "comusion")
SPARSE_SOURCES = {
    "belfusion": ROOT
    / "out/diffusion_hands_runs/belfusion/assembly_pick_up_screwd_20260420_103637/eval_samples.npz",
    "gsps": ROOT
    / "out/diffusion_hands_runs/gsps/assembly_pick_up_screwd_20260420_103637/eval_samples.npz",
    "humanmac": ROOT
    / "out/diffusion_hands_runs/humanmac/assembly_pick_up_screwd_20260420_103637/eval_samples.npz",
}
TITLES = {
    "card": "CARD",
    "fihard": "FiHaRD",
    "star_hand": "STAR-Hand",
    "comusion": "CoMusion",
    "belfusion": "BeLFusion",
    "gsps": "GSPS",
    "humanmac": "HumanMAC",
}
LIGHT = {
    "card": "#E76F51",
    "fihard": "#8E6BBE",
    "star_hand": "#D99A22",
    "comusion": "#55B7A4",
    "belfusion": "#6A9FD0",
    "gsps": "#C06CA9",
    "humanmac": "#6FAF63",
}
DARK = {
    "card": "#9D2F17",
    "fihard": "#4E2A78",
    "star_hand": "#8A5900",
    "comusion": "#126B5C",
    "belfusion": "#245B8A",
    "gsps": "#7B2C68",
    "humanmac": "#2F6F28",
}
GT_COLOR = "#2CA02C"
MOMENTS = (0, 7, 14, 21, 29)
PROJECTION = (0, 1)
DISPLAY_EXCLUSIONS = {0: {"comusion", "humanmac"}}


def load_cache(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def digest(array: np.ndarray) -> bytes:
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).digest()


def adapt_canonical(array: np.ndarray, model: str) -> np.ndarray:
    if model == "belfusion":
        return array
    if model == "gsps":
        zeros = np.zeros((*array.shape[:-2], 1, 3), dtype=array.dtype)
        return np.concatenate([zeros, array[..., 1:, :]], axis=-2)
    if model == "humanmac":
        return array[..., 1:, :]
    return array


def canonical_common(array: np.ndarray, model: str, use_twenty_joints: bool) -> np.ndarray:
    if not use_twenty_joints:
        return array
    if model == "gsps":
        return array[..., 1:, :]
    if model == "humanmac":
        return array
    return array[..., 1:, :]


def graph_for(model: str, canonical_links, canonical_wrist: int):
    if model == "gsps":
        return tuple((a, b) for a, b in canonical_links if a != 0 and b != 0), canonical_wrist
    if model == "humanmac":
        return tuple((a - 1, b - 1) for a, b in canonical_links if a != 0 and b != 0), canonical_wrist - 1
    return canonical_links, canonical_wrist


def oracle(candidates: np.ndarray, target: np.ndarray):
    errors = np.linalg.norm(candidates - target[None], axis=-1).mean(axis=(1, 2))
    best = int(errors.argmin())
    return best, float(errors[best])


def motion(target: np.ndarray, wrist: int):
    rooted = target - target[:, wrist : wrist + 1]
    xy = rooted[..., list(PROJECTION)]
    endpoint = float(np.linalg.norm(xy[-1] - xy[0], axis=-1).mean())
    spread = float(np.linalg.norm(xy - xy[0:1], axis=-1).mean())
    return endpoint, spread


def map_sparse_archives(context: np.ndarray):
    mapped = {}
    for model, path in SPARSE_SOURCES.items():
        saved = np.load(path, allow_pickle=False)
        adapted_context = adapt_canonical(context, model)
        lookup = {digest(row): index for index, row in enumerate(adapted_context)}
        indices = [lookup.get(digest(row)) for row in saved["obs"]]
        if any(index is None for index in indices):
            raise RuntimeError(f"Could not align all {model} examples")
        mapped[model] = {
            int(index): {
                "saved_position": position,
                "candidates": saved["pred_all"][position],
                "target": saved["target"][position],
            }
            for position, index in enumerate(indices)
        }
    return mapped


def evaluate_rows(caches, sparse, target: np.ndarray, wrist: int):
    union = sorted(set().union(*(set(rows) for rows in sparse.values())))
    evaluated = []
    for index in union:
        extras = tuple(model for model in SPARSE_SOURCES if index in sparse[model])
        use_twenty = any(model in {"gsps", "humanmac"} for model in extras)
        common_target = target[index, :, 1:, :] if use_twenty else target[index]
        candidates = {model: caches[model]["pred_candidates"][index].numpy() for model in FULL_MODELS}
        candidates.update({model: sparse[model][index]["candidates"] for model in extras})
        best_indices = {}
        errors = {}
        for model, model_candidates in candidates.items():
            comparable = canonical_common(model_candidates, model, use_twenty)
            best_indices[model], errors[model] = oracle(comparable, common_target)
        best_our = min(OUR_MODELS, key=errors.get)
        competitor_names = ("comusion", *extras)
        best_competitor = min(competitor_names, key=errors.get)
        endpoint, spread = motion(target[index], wrist)
        evaluated.append(
            {
                "test_index": index,
                "extras": extras,
                "candidates": candidates,
                "best_indices": best_indices,
                "errors": errors,
                "best_our_model": best_our,
                "best_competitor": best_competitor,
                "advantage": errors[best_competitor] - errors[best_our],
                "endpoint_motion": endpoint,
                "temporal_spread": spread,
                "common_joint_count": 20 if use_twenty else 21,
            }
        )
    return evaluated


def select_four(rows):
    endpoint_threshold = float(np.quantile([row["endpoint_motion"] for row in rows], 0.40))
    spread_threshold = float(np.quantile([row["temporal_spread"] for row in rows], 0.40))
    moving = [
        row
        for row in rows
        if row["endpoint_motion"] >= endpoint_threshold
        and row["temporal_spread"] >= spread_threshold
        and row["advantage"] > 0
    ]
    required_competitors = set(SPARSE_SOURCES)
    required_ours = set(OUR_MODELS)
    feasible = []
    for combination in itertools.combinations(moving, 4):
        covered_competitors = set().union(*(set(row["extras"]) for row in combination))
        covered_ours = {row["best_our_model"] for row in combination}
        if required_competitors <= covered_competitors and required_ours <= covered_ours:
            score = sum(row["advantage"] for row in combination)
            feasible.append((score, combination))
    if not feasible:
        raise RuntimeError("No four moving positive-advantage samples satisfy model coverage")
    _, selected = max(feasible, key=lambda item: item[0])
    return sorted(selected, key=lambda row: row["advantage"], reverse=True), endpoint_threshold, spread_threshold


def projected_rooted(pose: np.ndarray, wrist: int):
    return (pose - pose[wrist])[..., list(PROJECTION)]


def oracle_limits_by_moment(target, row, models, graph_metadata, padding=0.08):
    """Fit columns to ground truth and oracle poses, ignoring sample outliers."""
    limits = {}
    canonical_wrist = graph_metadata["canonical_wrist"]
    for frame in MOMENTS:
        points = []
        points.append(projected_rooted(target[frame], canonical_wrist).reshape(-1, 2))
        for model in models:
            candidates = row["candidates"][model]
            _, wrist = graph_for(model, graph_metadata["links"], canonical_wrist)
            oracle_pose = candidates[row["best_indices"][model], frame]
            points.append(projected_rooted(oracle_pose, wrist).reshape(-1, 2))
        points = np.concatenate(points)
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        limits[frame] = (lo - padding * span, hi + padding * span)
    return limits


def plot_skeleton(ax, pose, links, wrist, color, linewidth, alpha=1.0, zorder=2, clip_on=True):
    xy = projected_rooted(pose, wrist)
    for a, b in links:
        ax.plot(
            xy[[a, b], 0],
            xy[[a, b], 1],
            color=color,
            lw=linewidth,
            alpha=alpha,
            zorder=zorder,
            clip_on=clip_on,
        )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=7.0,
        color=color,
        alpha=alpha,
        zorder=zorder + 1,
        linewidths=0,
        clip_on=clip_on,
    )


def style_axis(ax, lo, hi):
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure(row, ordinal, target, graph_metadata):
    excluded = DISPLAY_EXCLUSIONS.get(row["test_index"], set())
    models = tuple(model for model in (*FULL_MODELS, *row["extras"]) if model not in excluded)
    nrows = 1 + len(models)
    fig, axes = plt.subplots(nrows, 5, figsize=(22.0, 2.55 * nrows))
    fig.subplots_adjust(left=0.18, right=0.995, bottom=0.04, top=0.84, hspace=0.05, wspace=0.012)
    limits = oracle_limits_by_moment(target, row, models, graph_metadata)
    canonical_links = graph_metadata["links"]
    canonical_wrist = graph_metadata["canonical_wrist"]

    for column, frame in enumerate(MOMENTS):
        ax = axes[0, column]
        lo, hi = limits[frame]
        plot_skeleton(ax, target[frame], canonical_links, canonical_wrist, GT_COLOR, 3.0, zorder=5)
        style_axis(ax, lo, hi)
        ax.set_title(f"$t={frame + 1}$", fontsize=44, pad=16)
        if column == 0:
            ax.set_ylabel("Ground truth", fontsize=46, fontweight="semibold", rotation=0, labelpad=210, va="center")

    for model_row, model in enumerate(models, start=1):
        links, wrist = graph_for(model, canonical_links, canonical_wrist)
        candidates = row["candidates"][model]
        best = row["best_indices"][model]
        label = TITLES[model]
        if model == row["best_our_model"]:
            label += "  ★"
        for column, frame in enumerate(MOMENTS):
            ax = axes[model_row, column]
            lo, hi = limits[frame]
            for pose in candidates[:, frame]:
                plot_skeleton(ax, pose, links, wrist, LIGHT[model], 1.05, alpha=0.35, zorder=1, clip_on=False)
            plot_skeleton(ax, candidates[best, frame], links, wrist, DARK[model], 3.0, alpha=0.98, zorder=5)
            style_axis(ax, lo, hi)
            if column == 0:
                ax.set_ylabel(label, fontsize=40, fontweight="semibold", rotation=0, labelpad=180, va="center")

    legend = [
        Line2D([0], [0], color=GT_COLOR, lw=2.1, label="Ground truth"),
        Line2D([0], [0], color="#999999", lw=1.0, alpha=0.5, label="Sampled hands (K=10)"),
        Line2D([0], [0], color="#444444", lw=2.1, label="Oracle"),
        Line2D([0], [0], color="none", marker="*", markerfacecolor="#333333", markeredgecolor="none", markersize=22,
               label="Best model"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=34)
    stem = OUTPUT_DIR / f"all_models_advantage_example_{ordinal:02d}_index_{row['test_index']}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return stem


def main():
    caches = {model: load_cache(CACHE_DIR / f"{model}_predictions.pt") for model in FULL_MODELS}
    reference = caches["card"]
    context = reference["context"].numpy()
    target = reference["gt_future"].numpy()
    for model in FULL_MODELS[1:]:
        if not torch.equal(caches[model]["context"], reference["context"]):
            raise RuntimeError(f"Context mismatch for {model}")
        if not torch.equal(caches[model]["gt_future"], reference["gt_future"]):
            raise RuntimeError(f"Target mismatch for {model}")

    graph = resolve_local_hand_graph_metadata("assembly", (5, 26))
    graph_metadata = {"links": tuple(graph["links"]), "canonical_wrist": int(graph["wrist_index"])}
    sparse = map_sparse_archives(context)
    rows = evaluate_rows(caches, sparse, target, graph_metadata["canonical_wrist"])
    selected, endpoint_threshold, spread_threshold = select_four(rows)

    report = {
        "selection": (
            "maximum summed conservative oracle-MPJPE advantage among four positive-advantage moving samples, "
            "subject to covering CARD, FiHaRD, and STAR-Hand as in-house winners and all sparse competitors"
        ),
        "advantage_definition": "minimum competitor MPJPE minus minimum in-house MPJPE on the common joint set",
        "motion_thresholds": {"endpoint_motion_q40": endpoint_threshold, "temporal_spread_q40": spread_threshold},
        "figures": [],
    }
    for ordinal, row in enumerate(selected, start=1):
        stem = make_figure(row, ordinal, target[row["test_index"]], graph_metadata)
        excluded = DISPLAY_EXCLUSIONS.get(row["test_index"], set())
        displayed_models = [model for model in (*FULL_MODELS, *row["extras"]) if model not in excluded]
        displayed_competitors = [model for model in displayed_models if model not in OUR_MODELS]
        displayed_best_competitor = min(displayed_competitors, key=row["errors"].get)
        report["figures"].append(
            {
                "figure": ordinal,
                "test_index": row["test_index"],
                "models": displayed_models,
                "excluded_from_display": sorted(excluded),
                "best_in_house_model": row["best_our_model"],
                "best_competitor": displayed_best_competitor,
                "oracle_mpjpe": {model: row["errors"][model] for model in displayed_models},
                "conservative_advantage": row["errors"][displayed_best_competitor] - row["errors"][row["best_our_model"]],
                "endpoint_motion": row["endpoint_motion"],
                "temporal_spread": row["temporal_spread"],
                "common_joint_count": row["common_joint_count"],
                "png": str(stem.with_suffix(".png")),
                "pdf": str(stem.with_suffix(".pdf")),
            }
        )
        print(f"Saved {stem.with_suffix('.png')}")
    report_path = OUTPUT_DIR / "all_models_advantage_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
