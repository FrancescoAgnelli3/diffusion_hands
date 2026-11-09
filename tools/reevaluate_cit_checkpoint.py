#!/usr/bin/env python3
"""Run a saved CARD/GRIP checkpoint on a declared full split and cache K predictions."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "splineeqnet"))

from train_best_models import _build_dataset_cfg, _build_train_cfg
from data import build_datasets, get_dataset_metadata, make_loaders
from runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Archived card_best_config.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Metrics JSON")
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument(
        "--eval-samples-path", type=Path,
        help="Optional .npz archive containing observation, target, and predictions for rendering.",
    )
    parser.add_argument("--dataset", default="assembly")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--action-filter", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--eval-subset-indices-path", default="",
        help="JSON list of test-window indices selected before checkpoint evaluation.",
    )
    parser.add_argument(
        "--max-source-files", type=int, default=0,
        help="Source-file cap used by the forecasting suite (0 means no cap).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument(
        "--candidate-batch-size", type=int, default=5,
        help="CIT candidates integrated together per data batch; lower this if GPU memory is limited.",
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Write the candidate cache but defer HumanMAC metrics to the optimized threshold sweep.",
    )
    args = parser.parse_args()
    if args.max_source_files < 0:
        parser.error("--max-source-files must be >= 0")
    if args.max_source_files:
        os.environ["DIFFUSION_HANDS_MAX_SOURCE_FILES"] = str(args.max_source_files)
    else:
        os.environ.pop("DIFFUSION_HANDS_MAX_SOURCE_FILES", None)

    payload = json.loads(args.config.read_text())
    archived = payload.get("card", payload)
    metadata = get_dataset_metadata(args.dataset)
    ds_cfg = _build_dataset_cfg(
        archived, data_dir=args.data_dir, action_filter=args.action_filter,
        dataset_name=args.dataset, wrist_indices=tuple(metadata["default_wrist_indices"]),
        node_count=int(metadata["node_count"]), edge_index=tuple(), adjacency=tuple(),
        window_override=int(archived["input_n"]), horizon_override=int(archived["output_n"]),
        seed_override=args.seed, eval_batch_mult=1,
    )
    ds_cfg.eval_subset_indices_path = str(args.eval_subset_indices_path or "")
    train_cfg = _build_train_cfg(archived, epochs_override=0, model_name="card")
    # Archived result_model_name is only a display label.
    # not a runner backend.  Force the CARD backend so its K-sample branch runs.
    train_cfg.model = "card"
    train_cfg.epochs = 0
    train_cfg.card_diffusion_epochs = 0
    train_cfg.result_model_name = str(archived.get("result_model_name", "card"))
    train_cfg.save_eval_examples = args.eval_samples_path is not None
    if args.eval_samples_path is not None:
        train_cfg.eval_examples_path = str(args.eval_samples_path)
    train, val, test = build_datasets(ds_cfg)
    _, _, test_loader = make_loaders(train, val, test, ds_cfg.batch_size, ds_cfg.seed, ds_cfg.eval_batch_mult)
    metrics = run_experiment(
        ds_cfg, train_cfg, save_root=str(args.output.parent),
        train_loader=test_loader, val_loader=test_loader, test_loader=None,
        load_model_path=str(args.checkpoint), num_candidates=max(1, args.num_candidates),
        card_eval_phase="diffusion", card_eval_collect_all=False, card_eval_oracle_mpjpe=True,
        compute_humanmac_metrics=True, humanmac_multimodal_threshold=args.threshold,
        humanmac_prediction_cache_path=str(args.prediction_cache),
        humanmac_skip_metrics=args.cache_only,
        card_candidate_batch_size=max(1, args.candidate_batch_size),
    )
    if args.cache_only:
        metrics["humanmac_metrics_deferred"] = True
        metrics["humanmac_prediction_cache"] = str(args.prediction_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
