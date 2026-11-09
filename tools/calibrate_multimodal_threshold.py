#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPLINEEQNET = ROOT / "vendor" / "splineeqnet"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SPLINEEQNET))

from common.multimodal_calibration import (
    calibrate_multimodal_threshold,
    cross_sequence_neighbor_coverage,
)
from config import DatasetCfg
from data import build_datasets, get_dataset_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the MMADE/MMFDE radius on training data.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--action-filter", default="")
    parser.add_argument("--input-n", type=int, required=True)
    parser.add_argument("--output-n", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--time-interp", type=int)
    parser.add_argument("--window-norm", type=int)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--max-source-files", type=int, default=0)
    parser.add_argument("--use-official-splits", action="store_true")
    parser.add_argument("--balanced-action-sampling", action="store_true")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--threshold-scale", type=float, default=0.7,
                        help="Historical post-calibration radius scale applied by the suite runner.")
    parser.add_argument("--neighbour-retry-scale", type=float, default=1.1,
                        help="Multiplier applied when the test preflight has no eligible neighbours.")
    parser.add_argument("--max-neighbour-retry-steps", type=int, default=100,
                        help="Maximum 10%%-style radius expansions before reporting an error.")
    parser.add_argument("--max-test-samples", type=int, default=0,
                        help="Optional deterministic cap for test coverage preflight (0 means all windows).")
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metadata = get_dataset_metadata(args.dataset)
    action_filter = "" if args.action_filter.strip().lower() in {"all", "*"} else args.action_filter
    cfg = DatasetCfg(
        data_dir=args.data_dir,
        action_filter=action_filter,
        input_n=args.input_n,
        output_n=args.output_n,
        stride=args.stride,
        time_interp=args.time_interp,
        window_norm=args.window_norm,
        seed=args.data_seed,
        data_seed=args.data_seed,
        dataset=args.dataset,
        node_count=int(metadata["node_count"]),
        max_source_files=args.max_source_files,
        balanced_action_sampling=args.balanced_action_sampling,
        use_official_splits=args.use_official_splits,
    )
    train_dataset, _val_dataset, test_dataset = build_datasets(cfg)

    sample_count = len(train_dataset)
    if args.max_samples > 0 and sample_count > args.max_samples:
        generator = torch.Generator().manual_seed(args.data_seed)
        indices = torch.randperm(sample_count, generator=generator)[: args.max_samples].sort().values
    else:
        indices = torch.arange(sample_count)

    contexts = []
    sequence_ids = []
    for index in indices.tolist():
        observation, _future, _scale, metadata_item = train_dataset[index]
        contexts.append(observation[..., 4:].float())
        sequence_ids.append(torch.as_tensor(metadata_item["sequence_id"]).long())
    context = torch.stack(contexts)
    ids = torch.stack(sequence_ids).reshape(-1)
    threshold, stats = calibrate_multimodal_threshold(
        context,
        ids,
        k=args.k,
        chunk_size=args.chunk_size,
    )
    test_count = len(test_dataset)
    if test_count == 0:
        raise SystemExit("Multimodal preflight failed: the test split contains no windows.")
    if args.max_test_samples > 0 and test_count > args.max_test_samples:
        generator = torch.Generator().manual_seed(args.data_seed)
        test_indices = torch.randperm(test_count, generator=generator)[: args.max_test_samples].sort().values
    else:
        test_indices = torch.arange(test_count)
    test_contexts = []
    test_sequence_ids = []
    for index in test_indices.tolist():
        observation, _future, _scale, metadata_item = test_dataset[index]
        test_contexts.append(observation[..., 4:].float())
        test_sequence_ids.append(torch.as_tensor(metadata_item["sequence_id"]).long())
    if args.threshold_scale <= 0:
        parser.error("--threshold-scale must be positive")
    if args.neighbour_retry_scale <= 1:
        parser.error("--neighbour-retry-scale must be greater than 1")
    if args.max_neighbour_retry_steps < 0:
        parser.error("--max-neighbour-retry-steps must be non-negative")
    test_context = torch.stack(test_contexts)
    test_ids = torch.stack(test_sequence_ids).reshape(-1)
    effective_threshold = threshold * args.threshold_scale
    expansions = 0
    test_coverage = cross_sequence_neighbor_coverage(
        test_context, test_ids, threshold=effective_threshold, chunk_size=args.chunk_size,
    )
    while test_coverage["covered_samples"] == 0 and expansions < args.max_neighbour_retry_steps:
        effective_threshold *= args.neighbour_retry_scale
        expansions += 1
        test_coverage = cross_sequence_neighbor_coverage(
            test_context, test_ids, threshold=effective_threshold, chunk_size=args.chunk_size,
        )
    if test_coverage["covered_samples"] == 0:
        raise SystemExit(
            "Multimodal preflight failed: no held-out test windows have an eligible "
            f"cross-sequence neighbour after {expansions} radius expansions."
        )
    # The suite runner applies threshold_scale after reading this value. Store
    # the adjusted raw radius so every backend receives effective_threshold.
    adjusted_raw_threshold = effective_threshold / args.threshold_scale
    stats["threshold"] = adjusted_raw_threshold
    stats.update(
        {
            "method": "median_cross_sequence_kth_neighbour_with_nonempty_test_retry",
            "dataset": args.dataset,
            "action_filter": action_filter,
            "split": "train",
            "available_training_windows": sample_count,
            "max_samples": args.max_samples,
            "data_seed": args.data_seed,
            "test_windows_checked": int(test_indices.numel()),
            "test_windows_available": test_count,
            "base_threshold": threshold,
            "threshold_scale": args.threshold_scale,
            "effective_threshold": effective_threshold,
            "neighbour_retry_scale": args.neighbour_retry_scale,
            "neighbour_retry_expansions": expansions,
            "test_coverage": test_coverage["coverage"],
            "test_covered_samples": test_coverage["covered_samples"],
            "test_uncovered_samples": test_coverage["uncovered_samples"],
            "test_mean_eligible_neighbours": test_coverage["mean_eligible_neighbours"],
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(
        f"CALIBRATED_MULTIMODAL_THRESHOLD={adjusted_raw_threshold:.9g} "
        f"EFFECTIVE_THRESHOLD={effective_threshold:.9g} "
        f"TEST_COVERAGE={test_coverage['coverage']:.2%} "
        f"EXPANSIONS={expansions}"
    )


if __name__ == "__main__":
    main()
