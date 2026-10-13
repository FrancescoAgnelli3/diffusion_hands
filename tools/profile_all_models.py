#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = ROOT / "results" / "model_efficiency.jsonl"
DEFAULT_CSV = ROOT / "results" / "model_efficiency.csv"
DEFAULT_MARKDOWN = ROOT / "results" / "model_efficiency.md"

ALL_MODELS = (
    "card",
    "star_hand",
    "fihard",
    "simlpe_dct",
    "belfusion",
    "comusion",
    "dlow_cvae",
    "humanmac",
    "skeletondiffusion",
    "gsps",
)

CSV_COLUMNS = (
    "model",
    "parameters",
    "trainable_parameters",
    "latency_ms",
    "latency_mean_ms",
    "latency_std_ms",
    "peak_memory_mib",
    "flops",
    "sampling_steps",
    "batch_size",
    "num_candidates",
    "device",
    "gpu",
    "torch_version",
    "warmup",
    "repeats",
    "notes",
    "flop_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch diffusion_hands pipelines and collect efficiency profiles."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment.yaml",
    )
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--enable-all",
        action="store_true",
        help="Enable every model declared in the config. This may train missing models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        help=(
            "Enable and profile only these models. Missing model entries are "
            "loaded from configs/models."
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not launch models; rebuild CSV and Markdown from the JSONL log.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append profiles instead of replacing the JSONL log.",
    )
    return parser.parse_args()


def _profile_config(
    config_path: Path,
    enable_all: bool,
    selected_models: list[str] | None = None,
) -> tuple[Path, Path | None]:
    if not enable_all and not selected_models:
        return config_path, None

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_entries = config.setdefault("models", {})
    model_files = {
        "card": "card.yaml",
        "star_hand": "star_hand.yaml",
        "fihard": "fihard.yaml",
        "simlpe_dct": "simlpe.yaml",
        "belfusion": "belfusion.yaml",
        "comusion": "comusion.yaml",
        "dlow_cvae": "dlow_cvae.yaml",
        "humanmac": "humanmac.yaml",
        "skeletondiffusion": "skeletondiffusion.yaml",
        "gsps": "gsps.yaml",
    }
    requested = set(selected_models or model_files)
    for model_name in requested:
        model_entries.setdefault(
            model_name, {"config_path": f"models/{model_files[model_name]}"}
        )
    for model_name, model_config in model_entries.items():
        if isinstance(model_config, dict):
            model_config["enabled"] = model_name in requested

    descriptor, generated_name = tempfile.mkstemp(
        prefix=".profile_", suffix=".yaml", dir=str(config_path.parent)
    )
    os.close(descriptor)
    generated = Path(generated_name)
    with generated.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return generated, generated


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    # Keep the most recent record for duplicate model names.
    return list({record["model"]: record for record in records}.values())


def _display(value: object, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_outputs(records: list[dict], csv_path: Path, markdown_path: Path) -> None:
    order = {name: index for index, name in enumerate(ALL_MODELS)}
    records.sort(key=lambda record: order.get(record["model"], len(order)))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "| Model | Parameters | Latency (ms) | Peak memory (MiB) | FLOPs | Sampling steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {model} | {parameters} | {latency} | {memory} | {flops} | {steps} |".format(
                model=record["model"],
                parameters=f"{int(record['parameters']):,}",
                latency=_display(record.get("latency_ms"), 3),
                memory=_display(record.get("peak_memory_mib"), 2),
                flops=(
                    f"{int(record['flops']) / 1e9:.3f} GFLOPs"
                    if record.get("flops") is not None
                    else "—"
                ),
                steps=_display(record.get("sampling_steps"), 0),
            )
        )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.jsonl = args.jsonl.expanduser().resolve()
    args.csv = args.csv.expanduser().resolve()
    args.markdown = args.markdown.expanduser().resolve()

    if not args.aggregate_only:
        if not args.append:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
            args.jsonl.unlink(missing_ok=True)

        profile_config, temporary_path = _profile_config(
            args.config, args.enable_all, args.models
        )
        environment = os.environ.copy()
        environment["DIFFUSION_HANDS_PROFILE_PATH"] = str(args.jsonl)
        environment["DIFFUSION_HANDS_PROFILE_WARMUP"] = str(max(0, args.warmup))
        environment["DIFFUSION_HANDS_PROFILE_REPEATS"] = str(max(1, args.repeats))
        python = environment.get(
            "DIFFUSION_HANDS_PYTHON", str(ROOT / ".venv" / "bin" / "python")
        )
        command = [
            python,
            str(ROOT / "tools" / "run_all_models.py"),
            "--config",
            str(profile_config),
        ]
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        if completed.returncode != 0:
            print(
                f"Model launcher exited with status {completed.returncode}; "
                "aggregating any completed profiles.",
                file=sys.stderr,
            )

    records = _read_records(args.jsonl)
    write_outputs(records, args.csv, args.markdown)
    print(f"[PROFILE] JSONL: {args.jsonl}")
    print(f"[PROFILE] CSV: {args.csv}")
    print(f"[PROFILE] Markdown: {args.markdown}")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
