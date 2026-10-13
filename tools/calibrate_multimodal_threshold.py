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

from common.multimodal_calibration import calibrate_multimodal_threshold
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
    parser.add_argument("--k", type=int, default=5)
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
    train_dataset, _val_dataset, _test_dataset = build_datasets(cfg)

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
    stats.update(
        {
            "method": "median_cross_sequence_kth_neighbour",
            "dataset": args.dataset,
            "action_filter": action_filter,
            "split": "train",
            "available_training_windows": sample_count,
            "max_samples": args.max_samples,
            "data_seed": args.data_seed,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"CALIBRATED_MULTIMODAL_THRESHOLD={threshold:.9g}")


if __name__ == "__main__":
    main()
