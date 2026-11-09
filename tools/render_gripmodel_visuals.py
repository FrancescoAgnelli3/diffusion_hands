#!/usr/bin/env python3
"""Render auditable GripModel-versus-competitor visualizations from E1.

The default inputs are the exactly matched Assembly/pick, seed-0 forecasting
suite archives for GripModel and CoMusion.  The examples are deliberately
favourable qualitative cases: among positive-advantage sequences, they
maximize GripModel's best-of-K MPJPE advantage while preferring motion.  The
generated manifest states this rule so the media cannot be mistaken for a
random sample.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common.dataset_graphs import resolve_local_hand_graph_metadata


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    color: str
    source: Path


DEFAULT_METHODS = (
    Method("gripmodel", "GripModel", "#C44900", ROOT / "results/forecasting_suite/e1_pick_last_frame/E1/runs/model_runs/cit_grip_phase_coupled_soft/assembly_pick_20261103_152940_cit_grip_phase_coupled_soft_seed0/eval_samples.npz"),
    Method("comusion", "CoMusion", "#377EB8", ROOT / "results/forecasting_suite/e1_pick_last_frame/E1/runs/model_runs/comusion/assembly_pick_20261103_152940_comusion_seed0/eval_samples.npz"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/gripmodel_visualization")
    parser.add_argument("--plan", type=Path, help="Render the all-action E2 plan written by plan_gripmodel_visual_subset.py.")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--fps", type=int, default=6, help="Playback rate; default is half the original 12 fps.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args()


DISPLAY = {
    "cit_grip_phase_coupled_soft": ("GripModel", "#C44900"),
    "comusion": ("CoMusion", "#377EB8"),
    "fihard": ("FiHaRD", "#6A3D9A"),
    "gsps": ("GSPS", "#4DAF4A"),
    "stars": ("STARS", "#984EA3"),
}

# Keep the qualitative media visually consistent with
# metrics_table_by_dataset_action.ipynb: ground truth is black, every sampled
# trajectory is a faint yellow-to-grey hand, and the oracle trajectory is the
# clear coloured hand for that model.
PLOT_STYLE = {
    "gt_color": "#111111",
    "candidate_color_start": "#d8a600",
    "candidate_color_end": "#5f6670",
    "candidate_alpha_min": 0.10,
    "candidate_alpha_max": 0.28,
    "joint_edge": "#f8f8f8",
    "axis_edge": "0.82",
    "label_color": "0.18",
}
CANDIDATE_CMAP = LinearSegmentedColormap.from_list(
    "candidate_yellow_to_grey",
    [PLOT_STYLE["candidate_color_start"], PLOT_STYLE["candidate_color_end"]],
)


def load(method: Method) -> dict[str, np.ndarray]:
    if not method.source.is_file():
        raise FileNotFoundError(f"Missing {method.label} archive: {method.source}")
    with np.load(method.source, allow_pickle=False) as archive:
        required = {"obs", "target", "pred_all"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{method.source} lacks {sorted(missing)}")
        return {key: archive[key].copy() for key in required}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_errors(pred_all: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each candidate's MPJPE and its oracle index, per sequence."""
    errors = np.linalg.norm(pred_all - target[:, None], axis=-1).mean(axis=(2, 3))
    best = errors.argmin(axis=1)
    return errors[np.arange(len(best)), best], best


def motion_amount(target: np.ndarray, wrist: int) -> np.ndarray:
    rooted = target - target[:, :, wrist : wrist + 1]
    return np.linalg.norm(rooted[:, -1] - rooted[:, 0], axis=-1).mean(axis=-1)


def select_examples(payloads: dict[str, dict[str, np.ndarray]], wrist: int, count: int) -> tuple[list[int], dict[str, np.ndarray], float]:
    target = payloads["gripmodel"]["target"]
    errors: dict[str, np.ndarray] = {}
    for key, payload in payloads.items():
        errors[key], _ = sequence_errors(payload["pred_all"], target)
    control_error = np.minimum.reduce([errors[key] for key in errors if key != "gripmodel"])
    advantage = control_error - errors["gripmodel"]
    motion = motion_amount(target, wrist)
    # Pick the most demanding motion threshold that still admits the requested
    # number of sequences in which GripModel beats every displayed competitor.
    positive_motion = np.sort(motion[advantage > 0])
    if len(positive_motion) < count:
        raise RuntimeError(f"Only found {len(positive_motion)} positive-advantage sequences; need {count}")
    threshold = float(positive_motion[-count])
    candidates = np.flatnonzero((motion >= threshold) & (advantage > 0))
    ranked = sorted(candidates.tolist(), key=lambda index: (advantage[index], motion[index]), reverse=True)
    selected = ranked[:count]
    if len(selected) != count:
        raise RuntimeError(f"Only found {len(selected)} positive-advantage moving sequences; need {count}")
    errors["best_control"] = control_error
    errors["advantage"] = advantage
    errors["motion"] = motion
    return selected, errors, threshold


def rooted(pose: np.ndarray, wrist: int) -> np.ndarray:
    return pose - pose[..., wrist : wrist + 1, :]


def bounds(poses: list[np.ndarray], wrist: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([rooted(pose, wrist).reshape(-1, 3) for pose in poses])
    lower, upper = points.min(axis=0), points.max(axis=0)
    span = max(float((upper - lower).max()), 1e-5)
    center = (lower + upper) / 2
    return center - 0.58 * span, center + 0.58 * span


def draw_hand(
    ax,
    pose: np.ndarray,
    links: tuple[tuple[int, int], ...],
    wrist: int,
    color: str,
    *,
    alpha: float = 1.0,
    lw: float = 2.0,
    size: float = 9,
    zorder: int = 3,
) -> None:
    xyz = rooted(pose, wrist)
    for first, second in links:
        ax.plot(
            xyz[[first, second], 0], xyz[[first, second], 1], xyz[[first, second], 2],
            color=color, alpha=alpha, lw=lw, zorder=zorder,
        )
    ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, s=size, alpha=alpha,
        edgecolors=PLOT_STYLE["joint_edge"], linewidths=0.18, depthshade=False, zorder=zorder,
    )


def draw_sampled_hands(
    ax,
    forecasts: np.ndarray,
    best_index: int,
    links: tuple[tuple[int, int], ...],
    wrist: int,
    best_color: str,
) -> None:
    """Draw every forecast sample lightly, then emphasise its oracle sample."""
    count = len(forecasts)
    colors = CANDIDATE_CMAP(np.linspace(0.0, 1.0, count))
    alphas = np.linspace(PLOT_STYLE["candidate_alpha_min"], PLOT_STYLE["candidate_alpha_max"], count)
    for sample_index in range(count):
        if sample_index == best_index:
            continue
        draw_hand(
            ax, forecasts[sample_index], links, wrist, colors[sample_index],
            alpha=float(alphas[sample_index]), lw=0.45, size=4.5, zorder=1,
        )
    draw_hand(ax, forecasts[best_index], links, wrist, best_color, alpha=1.0, lw=2.5, size=10, zorder=4)


def style_axis(ax, lower: np.ndarray, upper: np.ndarray, *, title: str | None = None) -> None:
    ax.set_xlim(lower[0], upper[0]); ax.set_ylim(lower[1], upper[1]); ax.set_zlim(lower[2], upper[2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False); ax.set_facecolor("white")
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=20, azim=-65)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.pane.set_edgecolor(PLOT_STYLE["axis_edge"])
        if hasattr(axis, "line"):
            axis.line.set_color(PLOT_STYLE["axis_edge"])
            axis.line.set_linewidth(0.35)
    if title:
        ax.set_title(title, fontsize=11, pad=1.5, color=PLOT_STYLE["label_color"], fontweight="semibold")


def render_figure(index: int, payloads: dict[str, dict[str, np.ndarray]], methods: tuple[Method, ...], links: tuple[tuple[int, int], ...], wrist: int, output: Path) -> Path:
    target = payloads["gripmodel"]["target"][index]
    oracle = {method.key: sequence_errors(payloads[method.key]["pred_all"][index : index + 1], target[None])[1][0] for method in methods}
    frames = tuple(np.linspace(0, target.shape[0] - 1, num=min(5, target.shape[0]), dtype=int).tolist())
    # Each figure column is fitted to its ground-truth hand, as in the paper
    # notebook.  This keeps the hand large and readable; diffuse outlier
    # samples may deliberately extend beyond the panel rather than shrinking
    # every hand in the grid.
    frame_bounds = {frame: bounds([target[frame]], wrist) for frame in frames}
    fig, axes = plt.subplots(1 + len(methods), len(frames), figsize=(12.5, 11.5), subplot_kw={"projection": "3d"}, squeeze=False)
    fig.patch.set_facecolor("white")
    for col, frame in enumerate(frames):
        draw_hand(axes[0, col], target[frame], links, wrist, PLOT_STYLE["gt_color"], lw=2.3, size=10)
        style_axis(axes[0, col], *frame_bounds[frame], title=f"t={frame + 1}")
    axes[0, 0].text2D(-0.14, 0.5, "Ground truth", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="center", fontsize=11, fontweight="semibold")
    for row, method in enumerate(methods, start=1):
        forecasts = payloads[method.key]["pred_all"][index]
        for col, frame in enumerate(frames):
            draw_hand(axes[row, col], target[frame], links, wrist, "#BDBDBD", alpha=0.42, lw=1.0, size=5, zorder=2)
            draw_sampled_hands(axes[row, col], forecasts[:, frame], oracle[method.key], links, wrist, method.color)
            style_axis(axes[row, col], *frame_bounds[frame])
        axes[row, 0].text2D(-0.14, 0.5, method.label, transform=axes[row, 0].transAxes, rotation=90, va="center", ha="center", fontsize=10, fontweight="semibold")
    fig.text(0.5, 0.006, "Faint yellow–gray hands: all K=20 forecasts. Colored hand: each method's lowest-MPJPE forecast. Gray: ground truth.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.96, bottom=0.04, hspace=0.01, wspace=0.01)
    path = output / "figures" / f"gripmodel_favorable_example_{index:03d}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def render_video(index: int, payloads: dict[str, dict[str, np.ndarray]], methods: tuple[Method, ...], links: tuple[tuple[int, int], ...], wrist: int, output: Path, fps: int) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for MP4 rendering")
    target = payloads["gripmodel"]["target"][index]
    obs = payloads["gripmodel"]["obs"][index]
    oracle = {m.key: sequence_errors(payloads[m.key]["pred_all"][index : index + 1], target[None])[1][0] for m in methods}
    forecast_samples = {m.key: payloads[m.key]["pred_all"][index] for m in methods}
    lower, upper = bounds([obs[-10:], target, *forecast_samples.values()], wrist)
    fig, axes = plt.subplots(1, len(methods), figsize=(12.0, 3.2), subplot_kw={"projection": "3d"}, squeeze=False)
    fig.patch.set_facecolor("white")
    output_path = output / "videos" / f"gripmodel_favorable_example_{index:03d}.mp4"
    observed = obs[-10:]
    total_frames = len(observed) + len(target)
    writer = FFMpegWriter(fps=fps, codec="h264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"])
    # Avoid calling tight_layout for every video frame: the fixed 3-D grid is
    # unchanged, and repeated layout work dominates rendering time.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.91, wspace=0.03)
    with writer.saving(fig, str(output_path), dpi=160):
        for time in range(total_frames):
            for axis, method in zip(axes[0], methods):
                axis.cla()
                if time < len(observed):
                    draw_hand(axis, observed[time], links, wrist, "#777777", lw=2.2)
                    title = f"{method.label}\nObserved t={time - len(observed) + 1}"
                else:
                    future = time - len(observed)
                    draw_hand(axis, target[future], links, wrist, PLOT_STYLE["gt_color"], alpha=0.42, lw=1.0, size=5, zorder=2)
                    draw_sampled_hands(axis, forecast_samples[method.key][:, future], oracle[method.key], links, wrist, method.color)
                    title = f"{method.label}\nForecast frame {future + 1}/{len(target)}"
                style_axis(axis, lower, upper, title=title)
            writer.grab_frame()
    plt.close(fig)
    return output_path


def write_selection_csv(selected: list[int], errors: dict[str, np.ndarray], output: Path) -> None:
    fields = ["sequence_index", "gripmodel_oracle_mpjpe", "best_control_oracle_mpjpe", "advantage_over_best_control", "future_motion"]
    with (output / "selection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in selected:
            writer.writerow({"sequence_index": index, "gripmodel_oracle_mpjpe": errors["gripmodel"][index], "best_control_oracle_mpjpe": errors["best_control"][index], "advantage_over_best_control": errors["advantage"][index], "future_motion": errors["motion"][index]})


def render_plan(plan_path: Path, output: Path, fps: int) -> None:
    """Render every action/sample in an exact-window qualitative plan."""
    plan = json.loads(plan_path.read_text())
    if int(plan["horizon"]) != 30:
        raise ValueError("The visual plan must use a 30-frame forecast horizon")
    graph = resolve_local_hand_graph_metadata("assembly", (5, 26))
    links, wrist = tuple(graph["links"]), int(graph["wrist_index"])
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "videos").mkdir(parents=True, exist_ok=True)
    for action, spec in plan["actions"].items():
        method_specs = tuple(("gripmodel" if key == "cit_grip_phase_coupled_soft" else key, key) for key in plan["methods"])
        methods = tuple(Method(alias, *DISPLAY[source_key], Path(spec["archives"][source_key])) for alias, source_key in method_specs)
        loaded = {source_key: load(method) for method, (_, source_key) in zip(methods, method_specs)}
        for ordinal, row in enumerate(spec["selected"], start=1):
            payloads = {
                method.key: {name: value[row["indices"][source_key] : row["indices"][source_key] + 1] for name, value in loaded[source_key].items()}
                for method, (_, source_key) in zip(methods, method_specs)
            }
            png = render_figure(0, payloads, methods, links, wrist, output)
            mp4 = render_video(0, payloads, methods, links, wrist, output, fps)
            mode = str(plan.get("selection_mode", "best"))
            stem = f"gripmodel_{str(plan.get('experiment', 'E5')).lower()}_{mode}_{action}_{ordinal:02d}"
            png.rename(output / "figures" / f"{stem}.png")
            png.with_suffix(".pdf").rename(output / "figures" / f"{stem}.pdf")
            mp4.rename(output / "videos" / f"{stem}.mp4")


def main() -> None:
    args = parse_args()
    if args.num_examples < 1:
        raise ValueError("--num-examples must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --force to replace its selected assets")
    if args.force:
        # Remove only files this renderer owns.  Do not erase an output
        # directory wholesale: users may keep paper drafts alongside it.
        for pattern in ("figures/gripmodel_favorable_example_*.*", "videos/gripmodel_favorable_example_*.mp4", "manifest.json", "selection.csv"):
            for path in args.output_dir.glob(pattern):
                path.unlink()
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "videos").mkdir(parents=True, exist_ok=True)
    if args.plan:
        render_plan(args.plan.resolve(), args.output_dir, args.fps)
        print(f"Rendered {json.loads(args.plan.read_text()).get('experiment', 'visual')} plan into {args.output_dir}")
        return
    payloads = {method.key: load(method) for method in DEFAULT_METHODS}
    reference = payloads["gripmodel"]
    for method in DEFAULT_METHODS[1:]:
        for key in ("obs", "target"):
            if not np.array_equal(payloads[method.key][key], reference[key]):
                raise ValueError(f"{method.label} {key} does not match GripModel; refusing invalid comparison")
    graph = resolve_local_hand_graph_metadata("assembly", (5, 26))
    links, wrist = tuple(graph["links"]), int(graph["wrist_index"])
    selected, errors, motion_threshold = select_examples(payloads, wrist, args.num_examples)
    write_selection_csv(selected, errors, args.output_dir)
    figures = [render_figure(index, payloads, DEFAULT_METHODS, links, wrist, args.output_dir) for index in selected]
    videos = [render_video(index, payloads, DEFAULT_METHODS, links, wrist, args.output_dir, args.fps) for index in selected]
    manifest = {
        "dataset": "assembly", "action": "pick", "seed": 0, "num_candidates": int(reference["pred_all"].shape[1]),
        "selection_rule": "Top positive GripModel advantages over the best displayed forecasting-suite competitor, restricted to the highest future-motion threshold retaining the requested number of cases.",
        "comparison_rule": "MPJPE is minimized independently over K=20 trajectories for every displayed method (oracle / best-of-K).",
        "motion_threshold": motion_threshold,
        "methods": [{"key": m.key, "label": m.label, "source": str(m.source.relative_to(ROOT)), "sha256": sha256(m.source)} for m in DEFAULT_METHODS],
        "selected_sequences": selected,
        "figures": [str(path.relative_to(ROOT)) for path in figures],
        "videos": [str(path.relative_to(ROOT)) for path in videos],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
