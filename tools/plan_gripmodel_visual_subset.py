#!/usr/bin/env python3
"""Select and render exact-window, high-motion GripModel visualisations.

Run after a qualitative forecasting suite completes.  It requires the
displayed methods to have saved exactly matching windows; mismatched or
partial archives fail fast instead of silently comparing different motions.
By default it writes the auditable JSON selection plan *and* renders the
corresponding figures and videos.  Use ``--plan-only`` when only the JSON is
needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from render_gripmodel_visuals import render_plan


METHODS = ("cit_grip_phase_coupled_soft", "comusion", "fihard", "stars")
ACTIONS = ("remove", "rotate", "unscrew", "pick", "put", "attempt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_root", type=Path, help="Forecasting-suite output root, containing the experiment directory.")
    parser.add_argument("--experiment", choices=("E2", "E5"), default="E5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--samples-per-action", type=int, default=64,
        help="Number of highest-motion exact common test windows retained per action/dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for figures and videos (default: assets/gripmodel_visualization_<experiment>).",
    )
    parser.add_argument("--fps", type=int, default=6, help="Video playback rate; 6 fps is half-speed.")
    parser.add_argument("--force", action="store_true", help="Replace only GripModel media created by an earlier run.")
    parser.add_argument("--plan-only", action="store_true", help="Write the selection JSON without rendering media.")
    parser.add_argument(
        "--failure-mode",
        action="store_true",
        help="Select the largest GripModel failures against the strongest displayed competitor, rather than favourable cases.",
    )
    return parser.parse_args()


def digest(array: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()


def archive_for(root: Path, experiment: str, method: str, action: str, seed: int) -> Path:
    pattern = f"runs/model_runs/{method}/assembly_{action}_*_{method}_seed{seed}/eval_samples.npz"
    matches = sorted((root / experiment).glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Missing archive for {method}/{action}/seed{seed}")
    if len(matches) > 1:
        print(f"[{method}/{action}] using newest archive: {matches[0]}")
    return matches[0]


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"obs", "target", "pred_all"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} misses {sorted(missing)}")
        return {name: archive[name].copy() for name in required}


def motion(target: np.ndarray) -> np.ndarray:
    # All E2 visual runs use the canonical Assembly 21-joint representation;
    # joint zero is the wrist. Rooting prevents global translation from being
    # mistaken for articulated hand motion.
    rooted = target - target[:, :, :1]
    return np.linalg.norm(rooted[:, -1] - rooted[:, 0], axis=-1).mean(axis=-1)


def oracle_mpjpe(pred_all: np.ndarray, target: np.ndarray) -> float:
    """Lowest candidate MPJPE for one exact forecast window."""
    error = np.linalg.norm(pred_all - target[None], axis=-1).mean(axis=(1, 2))
    return float(error.min())


def prepare_output_dir(output: Path, force: bool, selection_mode: str) -> None:
    """Guard user files while allowing this command's media to be regenerated."""
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(
            f"{output} is not empty; pass --force to replace previously rendered GripModel media"
        )
    if force:
        # Deliberately remove only files whose names are owned by render_plan.
        for pattern in (
            f"figures/gripmodel_e2_{selection_mode}_*.*",
            f"figures/gripmodel_e5_{selection_mode}_*.*",
            f"videos/gripmodel_e2_{selection_mode}_*.mp4",
            f"videos/gripmodel_e5_{selection_mode}_*.mp4",
        ):
            for path in output.glob(pattern):
                path.unlink()


def main() -> None:
    args = parse_args()
    if args.samples_per_action < 1:
        raise ValueError("--samples-per-action must be positive")
    root = args.suite_root.resolve()
    if not (root / args.experiment).is_dir():
        raise FileNotFoundError(f"No {args.experiment} directory under {root}")
    selection_mode = "worst" if args.failure_mode else "best"
    selection_rule = (
        "Exact common windows where GripModel has higher oracle MPJPE than the strongest displayed "
        "competitor, ranked by failure margin and then articulated motion."
        if args.failure_mode
        else "Exact common windows where GripModel has lower oracle MPJPE than every displayed "
        "competitor, ranked by its margin over the strongest competitor and then articulated motion."
    )
    plan: dict[str, object] = {
        "suite_root": str(root), "experiment": args.experiment, "horizon": 30,
        "seed": args.seed, "methods": METHODS, "samples_per_action": args.samples_per_action,
        "selection_mode": selection_mode,
        "selection_rule": selection_rule,
        "actions": {},
    }
    for action in ACTIONS:
        paths = {method: archive_for(root, args.experiment, method, action, args.seed) for method in METHODS}
        payloads = {method: load(path) for method, path in paths.items()}
        reference = payloads[METHODS[0]]
        if reference["target"].shape[1] != 30:
            raise ValueError(f"{action}: expected a 30-frame forecast horizon, got {reference['target'].shape[1]}")
        reference_index = {digest(obs): index for index, obs in enumerate(reference["obs"])}
        common = set(reference_index)
        indexes: dict[str, dict[bytes, int]] = {METHODS[0]: reference_index}
        for method in METHODS[1:]:
            mapping = {digest(obs): index for index, obs in enumerate(payloads[method]["obs"])}
            indexes[method] = mapping
            common &= set(mapping)
        if len(common) < args.samples_per_action:
            raise RuntimeError(f"{action}: only {len(common)} exact common windows; need {args.samples_per_action}")
        # Select either favourable examples (GripModel beats every competitor)
        # or explicit failure cases (it loses to the strongest competitor).
        # In both modes the primary key is the absolute comparison margin and
        # articulated motion breaks ties.
        scored = []
        all_motion = motion(reference["target"])
        for key in common:
            ref_index = reference_index[key]
            target = reference["target"][ref_index]
            errors = {
                method: oracle_mpjpe(payloads[method]["pred_all"][indexes[method][key]], target)
                for method in METHODS
            }
            best_competitor = min(errors[method] for method in METHODS if method != METHODS[0])
            advantage = best_competitor - errors[METHODS[0]]
            rank_margin = -advantage if args.failure_mode else advantage
            if rank_margin > 0:
                scored.append((rank_margin, float(all_motion[ref_index]), key, errors))
        if len(scored) < args.samples_per_action:
            case_name = "failures" if args.failure_mode else "windows where GripModel is better than every competitor"
            raise RuntimeError(
                f"{action}: only {len(scored)} common {case_name}; "
                f"need {args.samples_per_action}"
            )
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = scored[:args.samples_per_action]
        action_rows = []
        for rank_margin, sample_motion, key, errors in selected:
            ref_index = reference_index[key]
            advantage = min(errors[method] for method in METHODS if method != METHODS[0]) - errors[METHODS[0]]
            action_rows.append({
                "reference_index": ref_index,
                "motion": sample_motion,
                "gripmodel_oracle_mpjpe": errors[METHODS[0]],
                "best_competitor_oracle_mpjpe": min(
                    errors[method] for method in METHODS if method != METHODS[0]
                ),
                "gripmodel_advantage": advantage,
                "oracle_mpjpe": errors,
                "indices": {method: indexes[method][key] for method in METHODS},
            })
        plan["actions"][action] = {"archives": {method: str(path) for method, path in paths.items()}, "selected": action_rows}
    suffix = "_worst" if args.failure_mode else ""
    output = root / args.experiment / f"gripmodel_visual_subset{suffix}.json"
    output.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Selection plan: {output}")
    if args.plan_only:
        return
    output_suffix = "_worst" if args.failure_mode else ""
    output_dir = args.output_dir or Path(__file__).resolve().parents[1] / "assets" / f"gripmodel_visualization_{args.experiment.lower()}{output_suffix}"
    prepare_output_dir(output_dir, args.force, selection_mode)
    render_plan(output, output_dir, args.fps)
    print(f"Figures: {output_dir / 'figures'}")
    print(f"Videos: {output_dir / 'videos'}")


if __name__ == "__main__":
    main()
