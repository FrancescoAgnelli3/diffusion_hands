#!/usr/bin/env python3
"""Fixed-protocol, multi-seed comparison using the existing CARD runner.

No test-time tuning, implicit resume, model-specific training overrides, or
partial-cohort averages advertised as complete comparisons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_all_models import (CANONICAL_METRIC_KEYS, _deep_merge_dicts, _load_yaml,
                            _resolve_action_filters, _resolve_data_roots,
                            _resolve_datasets, _resolve_multimodal_threshold,
                            _resolve_shared_preprocessing, run_card)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def source_manifest():
    files = {}
    for directory in ("common", "vendor/splineeqnet", "tools"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def resolve_plan(path, selected_models=None, seeds=None):
    cfg = _load_yaml(path)
    cfg.setdefault("evaluation_split", "test")
    if cfg["evaluation_split"] not in {"validation", "test"}:
        raise ValueError("evaluation_split must be 'validation' or 'test'")
    base = _load_yaml((path.parent / cfg["shared_model_config"]).resolve())
    defaults = _deep_merge_dicts(base["defaults"], cfg.get("shared_defaults", {}))
    cfg.setdefault("data_seed", 0)
    defaults["data_seed"] = int(cfg["data_seed"])
    train = _deep_merge_dicts(base.get("train", {}), cfg.get("shared_train", {}))
    if train.get("early_stopping", {}).get("enabled", True):
        raise ValueError("A fixed-budget comparison requires early_stopping.enabled=false")
    if defaults.get("use_lr_scheduler", True):
        raise ValueError("Disable loss-dependent LR scheduling for the fixed protocol")
    if not defaults.get("card_freeze_coarse") or not defaults.get("card_cond_use_history"):
        raise ValueError("The controlled protocol requires a frozen common coarse model and history conditioning")
    if (not defaults.get("card_cond_use_coarse") or defaults.get("card_use_latent_variability")
            or defaults.get("card_diffusion_only") or defaults.get("card_use_mamp_condition")
            or defaults.get("card_use_mamp_condition_coarse") or defaults.get("card_dropout", 0) != 0):
        raise ValueError("Use common history/coarse conditioning, zero dropout, and no CARD latent/MAMP extras")
    if int(train.get("epochs", 0)) < 1 or int(defaults.get("card_diffusion_epochs", 0)) < 1:
        raise ValueError("Both training stages require a positive fixed epoch budget")
    if not cfg.get("save_model", False):
        raise ValueError("Research comparisons must retain final checkpoints")
    cfg["_shared_preprocessing"] = _resolve_shared_preprocessing(cfg)
    models = {}
    for name, spec in cfg["models"].items():
        if not spec.get("enabled", True) or (selected_models and name not in selected_models):
            continue
        if set(spec) - {"enabled", "base_model", "proposal_options", "notes"}:
            raise ValueError(f"{name}: model-specific training overrides are forbidden")
        if spec["base_model"] not in {"card", "cit"}:
            raise ValueError(f"Unsupported research model {spec['base_model']}")
        model_defaults = deepcopy(defaults)
        if spec.get("proposal_options"):
            if spec["base_model"] != "cit":
                raise ValueError("Only research proposals accept proposal_options")
            model_defaults["proposal_options"] = spec["proposal_options"]
        models[name] = {"base_model": spec["base_model"], "enabled": True,
                        "save_eval_samples": True, "train": deepcopy(train),
                        "defaults": model_defaults, "notes": spec.get("notes", "")}
    if selected_models and set(selected_models) != set(models):
        raise ValueError("Selected model is missing or disabled")
    if not models:
        raise ValueError("No models selected")
    cfg["models"] = models
    cfg["seeds"] = list(cfg["seeds"] if seeds is None else seeds)
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    cfg["data_roots"] = _resolve_data_roots(cfg)
    return cfg


def aggregate(output, plan):
    rows = [json.loads(p.read_text()) for p in sorted((output / "runs").glob("*/result.json"))]
    fields = ["dataset", "action_filter", "seed", "model", "status", *CANONICAL_METRIC_KEYS,
              "parameters", "coarse_updates", "generative_updates", "nfe_per_candidate",
              "auxiliary_evals_per_candidate",
              "elapsed_seconds", "calibration_passes", "calibration_windows",
              "source_fit_epochs", "source_fit_updates", "source_train_nll", "checkpoint", "error"]
    with (output / "metrics_long.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    paired = []
    baseline_name = ("card" if "card" in plan["models"]
                     else "cit_linear" if "cit_linear" in plan["models"]
                     else None)
    for dataset in _resolve_datasets(plan):
        for action in _resolve_action_filters(plan, dataset):
            for name in plan["models"]:
                selected = [r for r in rows if r["dataset"] == dataset and r["action_filter"] == action and r["model"] == name]
                by_seed = {r["seed"]: r for r in selected if r["status"] == "ok"}
                complete = set(by_seed) == set(plan["seeds"])
                for metric in CANONICAL_METRIC_KEYS:
                    values = [r[metric] for r in by_seed.values() if r.get(metric) is not None and math.isfinite(r[metric])]
                    valid = complete and len(values) == len(plan["seeds"])
                    summary.append({"dataset": dataset, "action_filter": action, "model": name,
                                    "metric": metric, "n": len(values), "expected_n": len(plan["seeds"]),
                                    "complete": valid, "mean": statistics.mean(values) if valid else None,
                                    "std": statistics.stdev(values) if valid and len(values)>1 else None})
                    if baseline_name is not None and name != baseline_name:
                        baseline = {r["seed"]: r for r in rows if r["dataset"] == dataset and r["action_filter"] == action
                                    and r["model"] == baseline_name and r["status"] == "ok"}
                        deltas = [by_seed[s][metric] - baseline[s][metric] for s in plan["seeds"]
                                  if s in by_seed and s in baseline and by_seed[s].get(metric) is not None
                                  and baseline[s].get(metric) is not None]
                        paired.append({"dataset": dataset, "action_filter": action, "model": name,
                                       "metric": metric, "n": len(deltas),
                                       "difference": f"model_minus_{baseline_name}",
                                       "mean_delta": statistics.mean(deltas) if len(deltas)==len(plan["seeds"]) else None,
                                       "baseline": baseline_name})
    for filename, table in (("summary.csv", summary), ("paired_deltas.csv", paired)):
        with (output / filename).open("w") as stream:
            if table:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ablations/cit_grip_phase_next_research.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only:
        if not args.output:
            parser.error("--aggregate-only requires --output")
        aggregate(args.output, json.loads((args.output / "manifest.json").read_text())["plan"])
        return
    plan = resolve_plan(args.config.resolve(), args.models, args.seeds)
    output = (args.output or ROOT / "results" / ("generative_research_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))).resolve()
    if args.dry_run:
        print(yaml.safe_dump({"output": str(output), "resolved_protocol": plan}, sort_keys=False))
        return
    manifest = {"plan": plan, "source_sha256": source_manifest()}
    if (output / "manifest.json").exists():
        if not args.resume:
            raise ValueError("Output already contains a benchmark; use --resume or choose a fresh directory")
        existing = json.loads((output / "manifest.json").read_text())
        if existing["plan"] != plan or existing["source_sha256"] != manifest["source_sha256"]:
            raise ValueError("Resume refused: resolved protocol or implementation changed")
    else:
        if args.resume or (output.exists() and any(output.iterdir())):
            raise ValueError("Resume requires a manifest; a fresh output must be empty")
        manifest["git_diff"] = subprocess.run(["git", "diff", "--binary"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        write_json(output / "manifest.json", manifest)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    if int(plan.get("gpu_index", 0)) < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    failed = False
    tag = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    for dataset in _resolve_datasets(plan):
        data_dir = Path(plan["data_roots"][dataset])
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
        for action in _resolve_action_filters(plan, dataset):
            # train_best_models.py accepts only a numeric threshold.  The
            # research configs intentionally allow ``auto``; calibrate it
            # once per dataset/action and inject the effective value into
            # every model run in this paired cohort.
            action_tag = hashlib.sha256(action.encode()).hexdigest()[:8]
            calibration_id = f"{dataset}_{action_tag}_data{plan['data_seed']}"
            effective_threshold, _ = _resolve_multimodal_threshold(
                plan,
                dataset=dataset,
                data_dir=data_dir,
                action_filter=action,
                run_id=calibration_id,
                out_root=output,
            )
            for seed in plan["seeds"]:
                cohort = f"{dataset}_{hashlib.sha256(action.encode()).hexdigest()[:8]}_s{seed}"
                reference = output / "protocols" / f"{dataset}_{hashlib.sha256(action.encode()).hexdigest()[:8]}_data{plan['data_seed']}.json"
                reference.parent.mkdir(parents=True, exist_ok=True)
                for name in plan["models"]:
                    directory = output / "runs" / f"{cohort}_{name}"
                    directory.mkdir(parents=True, exist_ok=True)
                    result_path = directory / "result.json"
                    if args.resume and result_path.exists():
                        previous = json.loads(result_path.read_text())
                        if previous["status"] == "ok":
                            if (not Path(previous["checkpoint"]).is_file()
                                    or not (Path(previous["checkpoint"]).parent.parent / "eval_samples.npz").is_file()
                                    or not (directory / "training_audit.json").is_file()):
                                raise RuntimeError(f"Completed run has missing artifacts: {directory}")
                            continue
                    cfg = deepcopy(plan)
                    cfg["humanmac_multimodal_threshold"] = effective_threshold
                    cfg["seed"] = seed
                    cfg["runtime"] = {"per_action_csv": str(directory / "per_action.csv"),
                                      "log_path": str(directory / "train.log")}
                    cfg["models"][name]["defaults"].update(protocol_audit_path=str(directory / "data_audit.json"),
                                                          protocol_reference_path=str(reference))
                    write_json(directory / "resolved_config.json", cfg)
                    run_id = f"research_{tag}_{cohort}_{time.time_ns()}"
                    checkpoint = ROOT / "out/diffusion_hands_runs" / name / run_id / "checkpoints/final.pt"
                    row = dict(dataset=dataset, action_filter=action, seed=seed, model=name,
                               status="failed", checkpoint=str(checkpoint))
                    started = time.perf_counter()
                    try:
                        metrics = run_card(name, dataset, data_dir, action, cfg, run_id)
                        audit = json.loads((directory / "training_audit.json").read_text())
                        data_audit = json.loads((directory / "data_audit.json").read_text())
                        n_batches = data_audit["train"]["windows"] // cfg["models"][name]["defaults"]["batch_size"]
                        expected = {"coarse": n_batches * cfg["models"][name]["train"]["epochs"],
                                    "diffusion": n_batches * cfg["models"][name]["defaults"]["card_diffusion_epochs"]}
                        if (n_batches == 0 or audit["updates"] != expected or not checkpoint.is_file()
                                or not (checkpoint.parent.parent / "eval_samples.npz").is_file()):
                            raise RuntimeError("Training budget or checkpoint audit failed")
                        if cfg["models"][name]["base_model"] == "cit":
                            calibration = json.loads((directory / "calibration_audit.json").read_text())
                            proposal = cfg["models"][name]["defaults"].get("proposal_options", {})
                            expected_passes = 2
                            if (calibration["source_split"] != "train" or calibration["passes"] != expected_passes
                                    or calibration["windows_per_pass"] != data_audit["train"]["windows"]):
                                raise RuntimeError("Train-only residual calibration audit failed")
                            row.update(calibration_passes=calibration["passes"],
                                       calibration_windows=calibration["windows_per_pass"])
                        else:
                            row.update(calibration_passes=0, calibration_windows=0)
                        coarse_path = output / "protocols" / f"{cohort}.coarse.json"
                        if coarse_path.exists() and json.loads(coarse_path.read_text()) != audit["coarse_sha256"]:
                            raise RuntimeError("Common coarse model differs between methods")
                        if not coarse_path.exists():
                            write_json(coarse_path, audit["coarse_sha256"])
                        if any(k not in metrics or not math.isfinite(metrics[k]) for k in CANONICAL_METRIC_KEYS):
                            raise RuntimeError(f"Missing/nonfinite canonical metrics: {metrics}")
                        row.update(metrics, status="ok", parameters=audit["parameters"],
                                   coarse_updates=audit["updates"]["coarse"], generative_updates=audit["updates"]["diffusion"])
                        row["nfe_per_candidate"] = cfg["models"][name]["defaults"]["card_ddim_steps"]
                        proposal = cfg["models"][name]["defaults"].get("proposal_options", {})
                        if proposal.get("grip_phase_transport", False):
                            # One cheap clock-network evaluation accompanies each main flow step.
                            row["auxiliary_evals_per_candidate"] = cfg["models"][name]["defaults"]["card_ddim_steps"]
                        else:
                            row["auxiliary_evals_per_candidate"] = 0
                    except Exception as exc:
                        failed = True
                        row["error"] = str(exc)
                        print(f"FAILED {cohort}/{name}: {exc}", flush=True)
                    row["elapsed_seconds"] = time.perf_counter() - started
                    write_json(result_path, row)
                    aggregate(output, plan)
    print(f"Results: {output / 'summary.csv'}")
    if failed:
        raise SystemExit("One or more runs failed; incomplete cohorts are not averaged")


if __name__ == "__main__":
    main()
