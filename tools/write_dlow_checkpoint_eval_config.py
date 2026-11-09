#!/usr/bin/env python3
"""Write the DLow YAML consumed by exp_vae.py for checkpoint-only evaluation."""
import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True,
                        help="Existing DLow YAML whose architecture matches the checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--action-filter", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--input-n", type=int, required=True)
    parser.add_argument("--output-n", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--splineeqnet-root", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.base_config.read_text()) or {}
    config.update({
        "dataset": "assembly", "data_dir": args.data_dir,
        "action_filter": args.action_filter, "seed": args.seed,
        "t_his": args.input_n, "t_pred": args.output_n, "stride": args.stride,
        "splineeqnet_root": args.splineeqnet_root,
        "prediction_cache_path": str(args.prediction_cache),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
