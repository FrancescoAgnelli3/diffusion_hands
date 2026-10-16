#!/usr/bin/env python3
"""Run E1-E5 with a common protocol and isolated result directories."""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_all_models.py"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def _resolve_output_root(raw: str | None, *, resume_latest: bool = False) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    if resume_latest:
        suite_root = ROOT / "results" / "forecasting_suite"
        candidates = sorted(
            (path for path in suite_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if suite_root.is_dir() else []
        if candidates:
            return candidates[0].resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "forecasting_suite" / stamp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", type=Path, default=ROOT / "configs" / "forecasting_suite.yaml")
    parser.add_argument("--base-config", type=Path, help="Override the base experiment config")
    parser.add_argument("--output-root", type=Path, help="Suite output directory; defaults to a timestamped directory")
    parser.add_argument(
        "--gpu-index",
        type=int,
        help="Override the GPU index specified by the suite or base configuration",
    )
    parser.add_argument("--experiments", nargs="+", choices=["E1", "E2", "E3", "E4", "E5"])
    parser.add_argument(
        "--keep-going",
        "--resume-latest",
        dest="resume_latest",
        action="store_true",
        help=(
            "Resume the latest suite directory when --output-root is omitted; "
            "later experiments always run after failures"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Write resolved configs without launching training")
    args = parser.parse_args()

    suite_path = args.suite_config.resolve()
    suite = _load(suite_path)
    base_path = args.base_config.resolve() if args.base_config else (suite_path.parent / str(suite["base_config"])).resolve()
    base = _load(base_path)
    experiments = suite.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError("suite config must contain a non-empty experiments mapping")
    selected = args.experiments or list(experiments)
    sample_cap = int(suite.get("max_samples_per_dataset_action", 0))
    if sample_cap < 0:
        raise ValueError("max_samples_per_dataset_action must be >= 0")
    output_root = _resolve_output_root(
        str(args.output_root) if args.output_root else None,
        resume_latest=bool(args.resume_latest and not args.output_root),
    )
    output_root.mkdir(parents=True, exist_ok=True)

    failures = []
    for name in selected:
        spec = experiments[name]
        if not isinstance(spec, dict):
            raise ValueError(f"{name} must be a mapping")
        input_n = int(spec["input_n"])
        output_n = int(spec["output_n"])
        if input_n <= 0 or output_n <= 0:
            raise ValueError(f"{name}: input_n and output_n must be positive")

        cfg = copy.deepcopy(base)
        cfg["gpu_index"] = (
            int(args.gpu_index)
            if args.gpu_index is not None
            else int(suite.get("gpu_index", cfg.get("gpu_index", 1)))
        )
        cfg["num_candidates"] = int(suite.get("num_candidates", 20))
        cfg["data_seed"] = int(suite.get("data_seed", cfg.get("data_seed", 0)))
        cfg["evaluation_split"] = str(suite.get("evaluation_split", cfg.get("evaluation_split", "test")))
        if sample_cap:
            cfg.setdefault("shared_defaults", {})
            cfg["shared_defaults"]["max_source_files"] = sample_cap
        cfg["max_samples_per_dataset_action"] = sample_cap
        cfg.setdefault("preprocessing", {})
        cfg["preprocessing"]["input_n"] = input_n
        cfg["preprocessing"]["output_n"] = output_n
        # The generated config lives below the result directory, so preserve
        # model YAML resolution against the original base-config directory.
        models = cfg.get("models", {})
        if isinstance(models, dict):
            for model_name, model_entry in models.items():
                if isinstance(model_entry, dict):
                    entry = dict(model_entry)
                    config_path = entry.get("config_path", f"models/{model_name}.yaml")
                    config_path = Path(str(config_path))
                    if not config_path.is_absolute():
                        config_path = (base_path.parent / config_path).resolve()
                    entry["config_path"] = str(config_path)
                    models[model_name] = entry
        experiment_root = output_root / name
        cfg["runtime"] = {
            "output_root": str(experiment_root / "runs"),
            "aggregate_csv": str(experiment_root / "metrics_long.csv"),
            "summary_csv": str(experiment_root / "metrics_mean_std.csv"),
            "per_action_csv": str(experiment_root / "metrics_by_action.csv"),
        }
        cfg["suite_experiment"] = name
        cfg["suite_objective"] = str(spec.get("objective", ""))
        resolved = experiment_root / "resolved_config.yaml"
        _write(resolved, cfg)

        print(
            f"\n=== {name}: past={input_n}, horizon={output_n}, "
            f"K={cfg['num_candidates']}, sample_cap={sample_cap or 'none'} ===",
            flush=True,
        )
        if args.dry_run:
            print(f"[{name}] resolved config: {resolved}", flush=True)
            continue
        command = [sys.executable, str(RUNNER), "--config", str(resolved)]
        result = subprocess.run(command, cwd=str(ROOT), env=os.environ.copy())
        if result.returncode != 0:
            failures.append(name)
            print(f"[{name}] FAILED (exit {result.returncode})", flush=True)
        else:
            print(f"[{name}] complete: {experiment_root / 'metrics_mean_std.csv'}", flush=True)

    print(f"\nSuite output: {output_root}")
    if failures:
        print(f"Failed experiments: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
