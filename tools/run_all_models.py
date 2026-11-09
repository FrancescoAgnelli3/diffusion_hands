#!/usr/bin/env python3
import argparse
import csv
import fcntl
import glob
import json
import math
import os
import statistics
import subprocess
import sys
import time
import tempfile
import shutil
from functools import lru_cache
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.evaluation import (
    CANONICAL_METRIC_KEYS,
    CANONICAL_LONG_HEADER,
    normalize_metrics_dict,
    read_one_row_csv,
)
from common.preprocessing import default_action_filter

VENDOR = ROOT / "vendor"
PYTHON = os.environ.get("DIFFUSION_HANDS_PYTHON", sys.executable)
DEFAULT_DATA_ROOTS: Dict[str, str] = {
    "assembly": "/mnt/pve/Turing-Storage2/AssemblyHands/assembly101-download-scripts/data_our/",
    "h2o": "/mnt/pve/Turing-Storage2/h2o/",
    "bighands": "/mnt/pve/Turing-Storage2/BigHands/BigHand2.2M/data/",
    "fpha": "/mnt/pve/Turing-Storage2/FPHA/data/",
}
DEFAULT_RUNTIME: Dict[str, str] = {
    "output_root": str(ROOT / "results"),
    "aggregate_csv": str(ROOT / "results" / "all_models_metrics_long.csv"),
    "summary_csv": str(ROOT / "results" / "all_models_metrics_summary.csv"),
}

# Hybrid-CVAE deliberately remains an external backend.  Its training script
# owns the history-mode preprocessing, FK constraints, split fingerprints, and
# best-checkpoint selection; importing its modules into this process would make
# those guarantees dependent on whichever ``common`` package Python resolved.
DEFAULT_HYBRID_CVAE_ROOT = ROOT.parent / "hybrid-CVAE-main"

# Preserve the historical calibration protocol: median cross-sequence radius,
# then a conservative fixed scale.  The calibration preflight only expands the
# radius when this would otherwise yield no multimodal test neighbours at all.
AUTO_MULTIMODAL_THRESHOLD_SCALE = 0.7


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


_CIT_DEFAULT_ALIASES = {
    # CIT is executed through the shared CARD-family runner, but these controls
    # belong to CIT's coarse/transport model rather than CARD diffusion.
    "cit_coarse_k_low": "card_k_low",
    "cit_transport_epochs": "card_diffusion_epochs",
    "cit_sampling_steps": "card_ddim_steps",
    "cit_denoiser_heads": "card_denoiser_heads",
    "cit_candidate_batch_size": "card_candidate_batch_size",
}


def _normalize_cit_defaults(defaults: dict) -> dict:
    """Translate public CIT option names to legacy shared-runner keys.

    Existing experiment files may retain the old spellings.  The public CIT
    spelling wins when it overlays a shared CARD-family base configuration.
    """
    normalized = dict(defaults)
    for public_key, legacy_key in _CIT_DEFAULT_ALIASES.items():
        if public_key not in normalized:
            continue
        value = normalized.pop(public_key)
        normalized[legacy_key] = value
    return normalized


def _run(cmd: List[str], cwd: Path, env: Optional[dict] = None, log_path: Optional[str] = None) -> int:
    print(f"[CMD] ({cwd}) {' '.join(cmd)}")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log:
            with subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return int(process.wait())
    p = subprocess.run(cmd, cwd=str(cwd), env=env)
    return int(p.returncode)


def _gpu_subprocess_env(cfg: dict) -> dict:
    env = os.environ.copy()
    gpu_index = int(cfg.get("gpu_index", 0))
    if gpu_index >= 0:
        # Restrict each launched backend to a single physical GPU so vendor-specific
        # device selection code cannot accidentally fall back to GPU 0.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    max_source_files = int(_as_dict(cfg.get("shared_defaults")).get("max_source_files", 0))
    if max_source_files > 0:
        # Every backend ultimately uses SplineEqNet's shared dataset builder.
        # Carry the sample cap out-of-band as well as in generated configs so
        # vendor adapters cannot accidentally omit it.
        env["DIFFUSION_HANDS_MAX_SOURCE_FILES"] = str(max_source_files)
    return env


def _child_gpu_index(cfg: dict) -> int:
    gpu_index = int(cfg.get("gpu_index", 0))
    # Inside a CUDA_VISIBLE_DEVICES-masked subprocess, the requested physical GPU
    # is exposed as local device 0.
    return 0 if gpu_index >= 0 else gpu_index


def _model_run_root(model_name: str, run_id: str) -> Path:
    run_root = ROOT / "out" / "diffusion_hands_runs" / model_name / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _suite_model_run_root(cfg: dict, model_name: str, run_id: str, vendor_name: Optional[str] = None) -> Path:
    """Persistent suite artifacts live only below the configured results root."""
    output_root = Path(str(_as_dict(cfg.get("runtime")).get("output_root", ROOT / "results")))
    kind = "vendor_model_runs" if vendor_name else "model_runs"
    parts = [output_root, kind]
    if vendor_name:
        parts.append(vendor_name)
    parts.extend([model_name, run_id])
    run_root = Path(*parts)
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _maybe_model_run_root(model_name: str, run_id: str, enabled: bool, cfg: Optional[dict] = None) -> Optional[Path]:
    if not enabled:
        return None
    return _suite_model_run_root(cfg, model_name, run_id) if cfg is not None else _model_run_root(model_name, run_id)


def _vendor_model_run_root(vendor_name: str, model_name: str, run_id: str, cfg: Optional[dict] = None) -> Path:
    if cfg is not None:
        return _suite_model_run_root(cfg, model_name, run_id, vendor_name)
    run_root = VENDOR / vendor_name / "out" / "diffusion_hands_runs" / model_name / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _save_eval_samples_enabled(model_cfg: dict) -> bool:
    return bool(model_cfg.get("save_eval_samples", True))


def _save_model_enabled(cfg: dict) -> bool:
    return bool(cfg.get("save_model", False))


def _checkpoint_dir(model_name: str, run_id: str, cfg: dict) -> Optional[Path]:
    if not _save_model_enabled(cfg):
        return None
    ckpt_dir = _suite_model_run_root(cfg, model_name, run_id) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _write_run_manifest(
    *, model_name: str, base_model: str, run_id: str, dataset: str,
    action_filter: str, seed: int, cfg: dict, metrics: Dict[str, object],
) -> None:
    """Record protocol and artifacts required for a later inference-only rerun."""
    roots = [_suite_model_run_root(cfg, model_name, run_id)]
    for vendor_name in ("splineeqnet", "gsps", "belfusion"):
        root = VENDOR / vendor_name / "out" / "diffusion_hands_runs" / model_name / run_id
        if root.exists():
            roots.append(root)
    artifacts = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".pkl", ".ckpt"}:
                artifacts.append(str(path.resolve()))
    payload = {
        "version": 1, "run_id": run_id, "model": model_name, "base_model": base_model,
        "dataset": dataset, "action_filter": action_filter, "seed": seed,
        "data_seed": _data_seed(cfg), "preprocessing": _as_dict(cfg.get("_shared_preprocessing")),
        "num_candidates": int(cfg["num_candidates"]),
        "multimodal_threshold": float(cfg["humanmac_multimodal_threshold"]),
        "execution_mode": "reused_checkpoint" if cfg.get("_reuse_run") else "trained_fallback",
        "source_run": _as_dict(cfg.get("_reuse_run")).get("run_dir"),
        "metrics": metrics, "checkpoint_artifacts": sorted(artifacts),
    }
    suite_run = Path(str(_as_dict(cfg.get("runtime")).get("output_root", ROOT / "results"))) / "checkpoint_runs" / model_name / run_id
    copied = []
    for index, artifact in enumerate(artifacts):
        source = Path(artifact)
        destination = suite_run / "checkpoints" / f"{index:02d}_{source.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(destination.resolve()))
    payload["suite_checkpoint_artifacts"] = copied
    path = _suite_model_run_root(cfg, model_name, run_id) / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    suite_run.mkdir(parents=True, exist_ok=True)
    (suite_run / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )


@lru_cache(maxsize=None)
def _cohort_protocol_sample_paths(cohort: str, seed: int) -> Tuple[Path, ...]:
    candidates: List[Path] = []
    roots = [ROOT / "out" / "diffusion_hands_runs"]
    roots.extend(
        VENDOR / vendor_name / "out" / "diffusion_hands_runs"
        for vendor_name in ("splineeqnet", "gsps", "belfusion")
    )
    for root in roots:
        if root.is_dir():
            for model_dir in root.iterdir():
                if not model_dir.is_dir():
                    continue
                candidates.append(
                    model_dir / f"{cohort}_{model_dir.name}_seed{seed}" / "eval_samples.npz"
                )
    return tuple(candidates)


def _reuse_protocol_sample_paths(run_dir: Path, model_name: str, seed: int) -> List[Path]:
    """Find evaluation caches that prove a legacy run's input/output horizon.

    HumanMAC's legacy checkpoint runs did not retain their own eval cache.  All
    models in a forecasting-suite cohort share the same timestamped run prefix,
    however, so a sibling model's cache is valid protocol evidence for that
    checkpoint.  Requiring such evidence prevents an E2 checkpoint from being
    mistaken for E5 merely because it is the closest earlier timestamp.
    """
    suffix = f"_{model_name}_seed{seed}"
    if not run_dir.name.endswith(suffix):
        return []
    cohort = run_dir.name[:-len(suffix)]
    candidates = [
        run_dir / "eval_samples.npz",
        ROOT / "out" / "diffusion_hands_runs" / model_name / run_dir.name / "eval_samples.npz",
    ]
    candidates.extend(_cohort_protocol_sample_paths(cohort, seed))
    # Preserve discovery order while eliminating duplicates/symlink aliases.
    unique: List[Path] = []
    seen = set()
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _reuse_protocol_matches(
    run_dir: Path, model_name: str, seed: int, input_n: int, output_n: int
) -> bool:
    for sample_path in _reuse_protocol_sample_paths(run_dir, model_name, seed):
        try:
            with np.load(sample_path, allow_pickle=False) as samples:
                observed = (int(samples["obs"].shape[1]), int(samples["target"].shape[1]))
            if observed == (input_n, output_n):
                return True
        except (KeyError, OSError, ValueError):
            continue
    return False


def _resolve_reuse_run(cfg: dict, dataset: str, action_filter: str, seed: int, model_name: str) -> Optional[dict]:
    reuse = _as_dict(cfg.get("checkpoint_reuse"))
    if not reuse.get("enabled"):
        return None
    source = Path(str(reuse.get("source_suite_root", ""))).resolve()
    experiment = str(cfg.get("suite_experiment", ""))
    metrics_path = source / experiment / "metrics_long.csv"
    rows = _read_long_csv(metrics_path)
    # A failed source row can still have a sound checkpoint: for example, an
    # evaluation may have failed because a metric was undefined, after training
    # completed and the checkpoint was archived.  Permit that row *only* in
    # this explicitly requested checkpoint-reuse path; normal resume handling
    # continues to rerun failed rows.
    matches = [r for r in rows if str(r.get("dataset")) == dataset
               and str(r.get("action_filter", "")) == str(action_filter)
               and int(r.get("seed", -1)) == int(seed) and str(r.get("model")) == model_name
               and str(r.get("status", "")).lower() in {"ok", "failed"}]
    if len(matches) != 1:
        print(f"[REUSE] no unique successful source row for {experiment}/{dataset}/{action_filter}/{model_name}/seed{seed}")
        return None
    completed = datetime.strptime(str(matches[0]["timestamp"]), "%Y%m%d_%H%M%S")
    roots = [ROOT / "out" / "diffusion_hands_runs" / model_name]
    roots += [VENDOR / name / "out" / "diffusion_hands_runs" / model_name
              for name in ("splineeqnet", "gsps", "belfusion")]
    candidates = []
    input_n = int(cfg["_shared_preprocessing"]["input_n"])
    output_n = int(cfg["_shared_preprocessing"]["output_n"])
    # A suite may be archived or moved after completion.  Prefer its retained
    # checkpoint_runs copy when present, rather than requiring the ephemeral
    # backend run directories (or the original source suite) to survive.
    archive_root = source / experiment / "runs" / "checkpoint_runs" / model_name
    if archive_root.is_dir():
        for manifest_path in archive_root.glob("*/run_manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                preprocessing = _as_dict(manifest.get("preprocessing"))
                compatible = (
                    str(manifest.get("dataset", "")) == str(dataset)
                    and str(manifest.get("action_filter", "")) == str(action_filter)
                    and int(manifest.get("seed", -1)) == int(seed)
                    and int(preprocessing.get("input_n", -1)) == input_n
                    and int(preprocessing.get("output_n", -1)) == output_n
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if not compatible:
                continue
            run_dir = manifest_path.parent
            artifacts = [
                path for path in run_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".pkl", ".ckpt"}
            ]
            if artifacts:
                candidates.append((completed, run_dir, artifacts))
    # Some suites retain checkpoints in model_runs rather than checkpoint_runs.
    # These manifests are equally self-contained; inspecting them here makes
    # checkpoint-only visual evaluation work with both archive layouts.
    retained_model_root = source / experiment / "runs" / "model_runs" / model_name
    if retained_model_root.is_dir():
        for manifest_path in retained_model_root.glob("*/run_manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                preprocessing = _as_dict(manifest.get("preprocessing"))
                compatible = (
                    str(manifest.get("dataset", "")) == str(dataset)
                    and str(manifest.get("action_filter", "")) == str(action_filter)
                    and int(manifest.get("seed", -1)) == int(seed)
                    and int(preprocessing.get("input_n", -1)) == input_n
                    and int(preprocessing.get("output_n", -1)) == output_n
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if not compatible:
                continue
            run_dir = manifest_path.parent
            artifacts = [
                path for path in run_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".pkl", ".ckpt"}
            ]
            if artifacts:
                candidates.append((completed, run_dir, artifacts))
    prefix = f"{dataset}_{action_filter or 'all'}_"
    suffix = f"_{model_name}_seed{seed}"
    for root in roots:
        for run_dir in root.glob(f"{prefix}*{suffix}") if root.is_dir() else ():
            stamp = run_dir.name[len(prefix):len(prefix) + 15]
            try:
                started = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            if started <= completed:
                if not _reuse_protocol_matches(
                    run_dir, model_name, seed, input_n=input_n, output_n=output_n
                ):
                    continue
                artifacts = [p for p in run_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".pkl", ".ckpt"}]
                if artifacts:
                    candidates.append((started, run_dir, artifacts))
    if not candidates:
        print(f"[REUSE] no checkpoint artifacts for {experiment}/{dataset}/{action_filter}/{model_name}/seed{seed}")
        return None
    started, run_dir, artifacts = max(candidates, key=lambda item: item[0])
    print(f"[REUSE] source run {run_dir}")
    return {"run_dir": str(run_dir), "artifacts": [str(p) for p in artifacts], "source_row": matches[0]}


def _resolve_model_cfg_entry(model_name: str, entry: object, config_dir: Path) -> dict:
    entry_overrides: Dict[str, object] = {}
    if isinstance(entry, str):
        cfg_path = Path(entry)
        if not cfg_path.is_absolute():
            cfg_path = config_dir / cfg_path
        loaded = _load_yaml(cfg_path)
    elif isinstance(entry, dict) and "config_path" in entry:
        cfg_path = Path(str(entry["config_path"]))
        if not cfg_path.is_absolute():
            cfg_path = config_dir / cfg_path
        loaded = _load_yaml(cfg_path)
        entry_overrides = {k: v for k, v in entry.items() if k != "config_path"}
    elif isinstance(entry, dict):
        # Backward-compatible behavior:
        # - inline config dicts are supported directly
        # - lightweight overrides (e.g. enabled: false) default to
        #   configs/models/{model_name}.yaml
        inline_keys = {"model", "train", "eval", "options", "defaults"}
        if inline_keys.intersection(entry.keys()):
            loaded = dict(entry)
        else:
            cfg_path = config_dir / "models" / f"{model_name}.yaml"
            loaded = _load_yaml(cfg_path)
            entry_overrides = dict(entry)
    else:
        raise ValueError(f"Invalid models.{model_name} entry. Expected path string or mapping, got {type(entry)!r}.")

    if not isinstance(loaded, dict):
        raise ValueError(f"Model config for '{model_name}' must be a mapping.")
    declared = str(loaded.get("model", model_name)).strip().lower()
    base_model = str(entry_overrides.get("base_model", declared)).strip().lower()
    if declared != base_model:
        raise ValueError(
            f"Model config mismatch: entry '{model_name}' declares base_model '{base_model}' "
            f"but loaded config model is '{declared}'."
        )
    if declared != model_name and "base_model" not in entry_overrides:
        raise ValueError(
            f"Model config mismatch: key '{model_name}' vs model '{declared}'. "
            "Use 'base_model' when defining an alias entry."
        )
    if entry_overrides:
        loaded = _deep_merge_dicts(dict(loaded), entry_overrides)
    return {
        "model": model_name,
        "base_model": base_model,
        "enabled": bool(loaded.get("enabled", True)),
        "implementation_version": str(loaded.get("implementation_version", "") or ""),
        "train": dict(loaded.get("train", {}) or {}),
        "eval": dict(loaded.get("eval", {}) or {}),
        "options": dict(loaded.get("options", {}) or {}),
        "defaults": dict(loaded.get("defaults", {}) or {}),
        "notes": str(loaded.get("notes", "") or ""),
    }


def _resolve_models_config(cfg: dict, config_path: Path) -> dict:
    model_entries = cfg.get("models", {})
    if not isinstance(model_entries, dict):
        raise ValueError("'models' must be a mapping in experiment config.")
    resolved = {}
    config_dir = config_path.parent
    for model_name, entry in model_entries.items():
        resolved[str(model_name).strip().lower()] = _resolve_model_cfg_entry(
            str(model_name).strip().lower(), entry, config_dir
        )
    cfg["models"] = resolved
    return cfg


def _append_long_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = CANONICAL_LONG_HEADER
    if path.exists():
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_header = list(reader.fieldnames or [])
            if existing_header != header:
                existing_rows = list(reader)
                with open(path, "w", newline="", encoding="utf-8") as wf:
                    ww = csv.DictWriter(wf, fieldnames=header)
                    ww.writeheader()
                    for old_row in existing_rows:
                        ww.writerow({k: old_row.get(k, "") for k in header})
    write_header = not path.exists()
    if path.exists() and path.stat().st_size > 0:
        # Guard against previously interrupted writes leaving the file without a trailing newline.
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) not in (b"\n", b"\r"):
                f.write(b"\n")
    serialized: Dict[str, object] = {}
    for k in header:
        v = row.get(k, "")
        if k in {"APD", "ADE", "FDE", "MMADE", "MMFDE", "CMD", "FID"} and isinstance(v, (int, float)):
            serialized[k] = f"{float(v):.3f}"
        else:
            serialized[k] = v
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(serialized)


def _long_row_key(row: Dict[str, object]) -> Tuple[str, str, int, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("action_filter", "")),
        int(row.get("seed", 0)),
        str(row.get("model", "")),
    )


def _has_all_finite_metrics(row: Dict[str, object]) -> bool:
    """Whether a completed row is safe to retain during automatic resume.

    MMADE/MMFDE are deliberately undefined (NaN) when a held-out window has
    no eligible multimodal neighbour.  They must not cause a successful,
    expensive run to be retrained on resume.
    """
    required = ("APD", "ADE", "FDE", "CMD", "FID")
    try:
        return all(math.isfinite(float(row.get(key, float("nan")))) for key in required)
    except (TypeError, ValueError):
        return False


def _read_long_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _metrics_lock(path: Path):
    """Return an exclusive sidecar lock for a shared aggregate CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = (path.parent / f".{path.name}.lock").open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _release_metrics_lock(handle) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _upsert_long_csv_unlocked(path: Path, row: Dict[str, object]) -> List[Dict[str, object]]:
    """Replace a previous model/seed row; caller must hold its CSV lock."""
    rows: List[Dict[str, object]] = [dict(item) for item in _read_long_csv(path)]
    key = _long_row_key(row)
    rows = [item for item in rows if _long_row_key(item) != key]
    rows.append(dict(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_LONG_HEADER)
        writer.writeheader()
        for item in rows:
            serialized: Dict[str, object] = {}
            for field in CANONICAL_LONG_HEADER:
                value = item.get(field, "")
                if field in {"APD", "ADE", "FDE", "MMADE", "MMFDE", "CMD", "FID"}:
                    try:
                        if value not in (None, ""):
                            value = f"{float(value):.3f}"
                    except (TypeError, ValueError):
                        pass
                serialized[field] = value
            writer.writerow(serialized)
    return rows


def _upsert_long_csv(path: Path, row: Dict[str, object]) -> List[Dict[str, object]]:
    """Replace a row safely when this helper is called in isolation."""
    lock = _metrics_lock(path)
    try:
        return _upsert_long_csv_unlocked(path, row)
    finally:
        _release_metrics_lock(lock)


def _record_result(
    aggregate_csv: Path, summary_csv: Path, row: Dict[str, object], seeds: List[int],
) -> List[Dict[str, object]]:
    """Atomically publish one row and its aggregate summary across suite parents."""
    lock = _metrics_lock(aggregate_csv)
    try:
        rows = _upsert_long_csv_unlocked(aggregate_csv, row)
        _write_summary_csv(summary_csv, rows, seeds)
        return rows
    finally:
        _release_metrics_lock(lock)


def _recover_card_metrics_from_artifacts(
    *,
    aggregate_csv: Path,
    summary_csv: Path,
    seeds: List[int],
) -> int:
    """Recover final CARD metrics written before a parent exited unsuccessfully."""
    rows: List[Dict[str, object]] = [dict(row) for row in _read_long_csv(aggregate_csv)]
    recovered = 0
    for row in rows:
        if str(row.get("status", "")).lower() != "failed":
            continue
        model = str(row.get("model", ""))
        dataset = str(row.get("dataset", ""))
        action = str(row.get("action_filter", ""))
        seed = int(row.get("seed", 0))
        prefix = f"{dataset}_{action}_" if action else f"{dataset}_all_"
        root = aggregate_csv.parent / "runs" / "vendor_model_runs" / "splineeqnet" / model
        candidates = []
        if root.is_dir():
            for run_dir in root.iterdir():
                if not run_dir.is_dir() or not run_dir.name.startswith(prefix):
                    continue
                if not run_dir.name.endswith(f"_{model}_seed{seed}"):
                    continue
                candidates.extend(run_dir.glob("**/card_eval*_w*_h*.csv"))
        if not candidates:
            continue
        metrics_path = max(candidates, key=lambda path: path.stat().st_mtime)
        metrics = normalize_metrics_dict(read_one_row_csv(metrics_path))
        if not metrics:
            continue
        row.update(metrics, status="ok", timestamp=_now())
        note = str(row.get("notes", "")).strip()
        suffix = f"recovered metrics from {metrics_path.relative_to(aggregate_csv.parent)}"
        row["notes"] = f"{note} | {suffix}" if note else suffix
        rows = _upsert_long_csv(aggregate_csv, row)
        recovered += 1
        print(f"[RECOVER] {dataset}:{model}:seed={seed} <- {metrics_path}")
    if recovered:
        _write_summary_csv(summary_csv, rows, seeds)
    return recovered


def _resolve_seeds(cfg: dict) -> List[int]:
    raw = cfg.get("seeds")
    if raw is None:
        raw = [cfg.get("seed", 0)]
    if not isinstance(raw, list) or not raw:
        raise ValueError("'seeds' must be a non-empty list of unique integers.")
    try:
        seeds = [int(seed) for seed in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("'seeds' must contain only integers.") from exc
    if len(seeds) != len(set(seeds)):
        raise ValueError("'seeds' must contain unique values.")
    return seeds


def _data_seed(cfg: dict) -> int:
    return int(cfg.get("data_seed", cfg.get("seed", 0)))


def _write_summary_csv(path: Path, rows: List[Dict[str, object]], seeds: List[int]) -> None:
    """Write one mean/std row per dataset/action/model for complete seed cohorts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        field
        for metric in CANONICAL_METRIC_KEYS
        for field in (f"{metric}_mean", f"{metric}_std")
    ]
    header = [
        "dataset",
        "action_filter",
        "model",
        "successful_seeds",
        "n",
        "expected_n",
        "complete",
        *metric_fields,
        "notes",
    ]
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["action_filter"]), str(row["model"]))
        groups.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    expected = set(seeds)
    for (dataset, action_filter, model), group in sorted(groups.items()):
        successful = {
            int(row["seed"]): row
            for row in group
            if row.get("status") == "ok" and int(row["seed"]) in expected
        }
        cohort_complete = set(successful) == expected
        output: Dict[str, object] = {
            "dataset": dataset,
            "action_filter": action_filter,
            "model": model,
            "successful_seeds": ";".join(str(seed) for seed in seeds if seed in successful),
            "n": len(successful),
            "expected_n": len(seeds),
            "complete": cohort_complete,
            "notes": next((str(row.get("notes", "")) for row in group if row.get("notes")), ""),
        }
        all_metrics_complete = cohort_complete
        for metric in CANONICAL_METRIC_KEYS:
            values: List[float] = []
            if cohort_complete:
                for seed in seeds:
                    try:
                        value = float(successful[seed][metric])
                    except (KeyError, TypeError, ValueError):
                        values = []
                        break
                    if not math.isfinite(value):
                        values = []
                        break
                    values.append(value)
            if len(values) == len(seeds):
                output[f"{metric}_mean"] = f"{statistics.mean(values):.6f}"
                output[f"{metric}_std"] = f"{statistics.stdev(values):.6f}" if len(values) > 1 else "0.000000"
            else:
                all_metrics_complete = False
                output[f"{metric}_mean"] = ""
                output[f"{metric}_std"] = ""
        output["complete"] = all_metrics_complete
        summary_rows.append(output)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(summary_rows)


def _as_dict(obj: object) -> Dict[str, object]:
    return dict(obj) if isinstance(obj, dict) else {}


def _resolve_shared_preprocessing(cfg: dict) -> Dict[str, object]:
    raw = _as_dict(cfg.get("preprocessing"))

    def _int(key: str, default: int) -> int:
        value = raw.get(key, default)
        try:
            out = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"preprocessing.{key} must be an integer, got {value!r}")
        if out <= 0:
            raise ValueError(f"preprocessing.{key} must be > 0, got {out}")
        return out

    def _int_or_none(key: str, default: Optional[int]) -> Optional[int]:
        value = raw.get(key, default)
        if value in (None, "", "None", "null"):
            return None
        try:
            out = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"preprocessing.{key} must be an integer or null, got {value!r}")
        if out <= 0:
            raise ValueError(f"preprocessing.{key} must be > 0 when set, got {out}")
        return out

    return {
        "input_n": _int("input_n", 70),
        "output_n": _int("output_n", 30),
        "stride": _int("stride", 5),
        "time_interp": _int_or_none("time_interp", None),
        "window_norm": _int_or_none("window_norm", None),
        "eval_batch_mult": _int("eval_batch_mult", 1),
    }


def _hydra_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        if v == "":
            return "''"
        return v
    return json.dumps(v, separators=(",", ":"))


def _flatten_hydra(prefix: str, obj: Dict[str, object]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for k, v in obj.items():
        if str(k) == "defaults":
            continue
        if isinstance(v, str) and (v == "" or "${" in v):
            # Keep empty/interpolated values in the underlying Hydra config;
            # passing them as CLI overrides is error-prone.
            continue
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten_hydra(key, v))
        else:
            out.append((key, _hydra_value(v)))
    return out


def _resolve_data_roots(cfg: dict) -> Dict[str, str]:
    roots = dict(DEFAULT_DATA_ROOTS)
    user_roots = cfg.get("data_roots")
    if user_roots is None:
        return roots
    if not isinstance(user_roots, dict):
        raise ValueError("'data_roots' must be a mapping when provided in experiment config.")
    for k, v in user_roots.items():
        key = str(k).strip().lower()
        if not key:
            continue
        roots[key] = str(v)
    return roots


def _resolve_evaluation_subset_path(cfg: dict, dataset: str, action_filter: str) -> str:
    """Return the cohort-specific, precomputed evaluation window list.

    The indices refer to the unwrapped test dataset produced by SplineEqNet's
    shared preprocessing.  Resolving the path here (rather than in individual
    vendor adapters) guarantees every backend sees exactly the same 64 windows.
    """
    subset = _as_dict(cfg.get("evaluation_subset"))
    if not bool(subset.get("enabled", False)):
        return ""
    root = subset.get("manifest_root")
    if not root:
        raise ValueError("evaluation_subset.enabled requires evaluation_subset.manifest_root")
    root_path = Path(str(root))
    if not root_path.is_absolute():
        root_path = ROOT / root_path
    action_tag = str(action_filter).strip() or "all"
    path = root_path / f"{str(dataset).lower()}__{action_tag}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing mobile-window manifest for {dataset}/{action_tag}: {path}. "
            "Run tools/build_mobile_eval_manifests.py first."
        )
    return str(path)


def _checkpoint_only_requested(cfg: dict) -> bool:
    reuse = _as_dict(cfg.get("checkpoint_reuse"))
    return bool(reuse.get("enabled")) and not bool(reuse.get("fallback_to_training", True))


def _resolve_runtime(cfg: dict) -> Dict[str, str]:
    runtime = dict(DEFAULT_RUNTIME)
    user_runtime = cfg.get("runtime")
    if user_runtime is None:
        return runtime
    if not isinstance(user_runtime, dict):
        raise ValueError("'runtime' must be a mapping when provided in experiment config.")
    for k, v in user_runtime.items():
        runtime[str(k)] = str(v)
    return runtime


def run_card(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "splineeqnet"
    mcfg = cfg.get("models", {}).get(model_name, {})
    base_model = str(mcfg.get("base_model", model_name)).strip().lower()
    target_model = "simlpe_dct" if base_model == "simlpe_dct" else "card"
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    samples_root = _maybe_model_run_root(model_name, run_id, save_eval_samples, cfg)
    checkpoint_dir = _checkpoint_dir(model_name, run_id, cfg)
    best_cfg = deepcopy(_as_dict(mcfg.get("defaults")))
    if not best_cfg:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")
    if base_model == "cit":
        best_cfg = _normalize_cit_defaults(best_cfg)
    best_cfg["model"] = target_model
    if target_model == "card":
        best_cfg["card_variant"] = base_model
    best_cfg["result_model_name"] = model_name
    best_cfg["data_seed"] = _data_seed(cfg)
    best_cfg["eval_subset_indices_path"] = str(
        _as_dict(cfg.get("evaluation_subset")).get("indices_path", "") or ""
    )
    best_cfg["input_n"] = int(pp["input_n"])
    best_cfg["output_n"] = int(pp["output_n"])
    # CARD-family models use a low-frequency DCT subspace.  The configured
    # default (80) is valid for the 100-frame baseline, but not for shorter
    # horizon-suite windows. Reserve ten coefficients for the high-frequency
    # diffusion stage and make the effective value explicit in the child
    # config instead of failing deep inside the model constructor.
    total_frames = int(pp["input_n"]) + int(pp["output_n"])
    reserve = max(1, int(cfg.get("card_k_low_reserve", 10)))
    max_k_low = max(1, total_frames - reserve)
    configured_k_low = int(best_cfg.get("card_k_low", max_k_low))
    if configured_k_low > max_k_low:
        print(
            f"[{model_name}] card_k_low={configured_k_low} exceeds total_frames={total_frames}; "
            f"using card_k_low={max_k_low}"
        )
        best_cfg["card_k_low"] = max_k_low
    best_cfg["stride"] = int(pp["stride"])
    best_cfg["time_interp"] = pp.get("time_interp")
    best_cfg["window_norm"] = pp.get("window_norm")
    epochs = mcfg.get("train", {}).get("epochs")
    if epochs is not None:
        best_cfg["epochs"] = int(epochs)
    es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if es_cfg:
        if "enabled" in es_cfg:
            best_cfg["early_stopping_enabled"] = bool(es_cfg.get("enabled"))
        if "patience" in es_cfg:
            best_cfg["early_stopping_patience"] = int(es_cfg.get("patience"))
        if "min_delta" in es_cfg:
            best_cfg["early_stopping_min_delta"] = float(es_cfg.get("min_delta"))
        if "warmup" in es_cfg:
            best_cfg["early_stopping_warmup"] = int(es_cfg.get("warmup"))
        if "monitor" in es_cfg:
            best_cfg["early_stopping_monitor"] = str(es_cfg.get("monitor"))

    eval_examples_path = samples_root / "eval_samples.npz" if samples_root is not None else None
    with tempfile.TemporaryDirectory(prefix="card_") as td:
        temp_root = Path(td)
        run_root = (
            _vendor_model_run_root("splineeqnet", model_name, run_id, cfg)
            if save_eval_samples
            else temp_root / "run"
        )
        run_root.mkdir(parents=True, exist_ok=True)
        best_json = run_root / "card_best_config.json"
        out_eval_base = run_root / "card_eval.csv"
        save_root = run_root / "checkpoints"
        per_action_cfg = _as_dict(cfg.get("runtime")).get(
            "per_action_csv", "./results/all_models_by_action.csv"
        )
        per_action_path = Path(str(per_action_cfg))
        if not per_action_path.is_absolute():
            per_action_path = ROOT / per_action_path
        best_cfg["per_action_metrics_path"] = str(per_action_path)
        if samples_root is not None:
            best_cfg["explainability_output_dir"] = str(samples_root / "explainability")
        # ``train_best_models.py`` parses this argument as a float.  Most
        # orchestrated runs resolve ``auto`` once per action before reaching
        # this function, but keep direct callers safe as well.
        threshold = cfg.get("humanmac_multimodal_threshold", 0.5)
        if isinstance(threshold, str) and threshold.strip().lower() == "auto":
            runtime_cfg = _as_dict(cfg.get("runtime"))
            per_action = Path(str(runtime_cfg.get(
                "per_action_csv", ROOT / "results" / "all_models_by_action.csv"
            )))
            if not per_action.is_absolute():
                per_action = ROOT / per_action
            threshold, _ = _resolve_multimodal_threshold(
                cfg, dataset=dataset, data_dir=data_dir,
                action_filter=action_filter, run_id=run_id,
                out_root=per_action.parent,
            )
        best_cfg["humanmac_multimodal_threshold"] = float(threshold)
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump({"card": best_cfg}, f, indent=2)
        reuse_run = _as_dict(cfg.get("_reuse_run"))
        if reuse_run:
            # Retained suite archives prefix copied files (for example,
            # ``00_final.pt``), whereas native training runs use ``final.pt``.
            # Both name forms denote the same terminal checkpoint.
            checkpoints = [
                Path(p) for p in reuse_run.get("artifacts", [])
                if Path(p).name == "final.pt" or Path(p).name.endswith("_final.pt")
            ]
            source_name = Path(str(reuse_run["run_dir"])).name
            reuse_source = Path(str(_as_dict(cfg.get("checkpoint_reuse")).get("source_suite_root", "")))
            archived_cfg_candidates = [
                reuse_source / str(cfg.get("suite_experiment", "")) / "runs" / "vendor_model_runs"
                / "splineeqnet" / model_name / source_name / "card_best_config.json",
                VENDOR / "splineeqnet" / "out" / "diffusion_hands_runs" / model_name
                / source_name / "card_best_config.json",
                # The generated current config has the same architecture and
                # protocol fields; it is a safe final fallback for retained
                # checkpoints whose auxiliary config was not archived.
                best_json,
            ]
            archived_cfg = next((path for path in archived_cfg_candidates if path.is_file()), None)
            if len(checkpoints) == 1 and archived_cfg is not None:
                destination = _suite_model_run_root(cfg, model_name, run_id) / "checkpoints" / "final.pt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(checkpoints[0], destination)
                metrics_json = run_root / "checkpoint_eval_metrics.json"
                prediction_cache = run_root / "prediction_cache.pt"
                rc = _run([
                    PYTHON, str(ROOT / "tools" / "reevaluate_cit_checkpoint.py"),
                    "--config", str(archived_cfg), "--checkpoint", str(destination),
                    "--output", str(metrics_json), "--prediction-cache", str(prediction_cache),
                    *(["--eval-samples-path", str(samples_root / "eval_samples.npz")]
                      if samples_root is not None else []),
                    "--dataset", dataset, "--data-dir", str(data_dir),
                    "--action-filter", str(action_filter), "--seed", str(cfg["seed"]),
                    "--max-source-files", str(int(_as_dict(cfg.get("shared_defaults")).get("max_source_files", 0))),
                    "--eval-subset-indices-path", str(best_cfg["eval_subset_indices_path"]),
                    "--threshold", str(cfg["humanmac_multimodal_threshold"]),
                    "--num-candidates", str(cfg["num_candidates"]),
                    "--candidate-batch-size", str(max(1, int(best_cfg.get("card_candidate_batch_size", 1)))),
                ], cwd=ROOT, env=_gpu_subprocess_env(cfg))
                if rc == 0:
                    return normalize_metrics_dict(json.loads(metrics_json.read_text()))
                raise RuntimeError(f"{model_name} checkpoint evaluation failed")
            if _checkpoint_only_requested(cfg):
                raise RuntimeError(
                    f"{model_name} checkpoint-only evaluation requires one usable final checkpoint and config."
                )
        rc = _run(
            [
                PYTHON,
                "train_best_models.py",
                "--model",
                target_model,
                "--dataset",
                str(dataset),
                "--data-dir",
                str(data_dir),
                "--action-filter",
                str(action_filter),
                "--eval-split",
                str(cfg.get("evaluation_split", "test")),
                "--seed",
                str(cfg["seed"]),
                "--window",
                str(int(pp["input_n"])),
                "--horizon",
                str(int(pp["output_n"])),
                "--eval-batch-mult",
                str(int(pp["eval_batch_mult"])),
                "--best-json",
                str(best_json),
                "--num-candidates",
                str(cfg["num_candidates"]),
                "--candidate-batch-size",
                str(max(1, int(best_cfg.get("card_candidate_batch_size", 1)))),
                "--humanmac-multimodal-threshold",
                str(cfg["humanmac_multimodal_threshold"]),
                *(["--epochs", str(int(epochs))] if epochs is not None else []),
                *(["--save-eval-examples"] if save_eval_samples else []),
                *(["--eval-examples-path", str(eval_examples_path)] if eval_examples_path is not None else []),
                "--results-csv",
                str(out_eval_base),
                "--save-root",
                str(save_root),
                *(["--final-checkpoint-path", str(checkpoint_dir / "final.pt")] if checkpoint_dir is not None else []),
            ],
            cwd=wd,
            env=_gpu_subprocess_env(cfg),
            log_path=_as_dict(cfg.get("runtime")).get("log_path"),
        )
        if rc != 0:
            # A wrapper can report failure after the child has already written
            # final evaluation metrics. Preserve that completed experiment.
            completed = sorted(run_root.glob("**/card_eval*_w*_h*.csv"), key=os.path.getmtime)
            if completed:
                metrics = normalize_metrics_dict(read_one_row_csv(completed[-1]))
                if metrics:
                    print(
                        f"[{model_name}] backend exited {rc}, but recovered completed metrics "
                        f"from {completed[-1]}",
                        flush=True,
                    )
                    return metrics
            raise RuntimeError(
                f"{model_name} training failed (backend={target_model})"
            )

        out_candidates = sorted(run_root.glob("**/card_eval*_w*_h*.csv"), key=os.path.getmtime)
        if not out_candidates:
            raise RuntimeError(f"{target_model} did not produce any metrics CSV under {run_root}")
        row = read_one_row_csv(out_candidates[-1])

    return normalize_metrics_dict(row)


def run_belfusion(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "belfusion"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    samples_root = _maybe_model_run_root(model_name, run_id, save_eval_samples, cfg)
    run_root = _vendor_model_run_root("belfusion", model_name, run_id, cfg)
    template = deepcopy(_as_dict(mcfg.get("defaults")))
    if not template:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")

    template["seed"] = int(cfg["seed"])
    template.setdefault("data", {})
    template["data"]["dataset"] = str(dataset).lower()
    template["data"]["data_dir"] = str(data_dir)
    template["data"]["action_filter"] = str(action_filter)
    template["data"]["input_n"] = int(pp["input_n"])
    template["data"]["output_n"] = int(pp["output_n"])
    template["data"]["stride"] = int(pp["stride"])
    template["data"]["time_interp"] = pp.get("time_interp")
    template["data"]["window_norm"] = pp.get("window_norm")

    train_epochs = mcfg.get("train", {}).get("epochs")
    if train_epochs is not None:
        template.setdefault("train", {})
        template["train"]["epochs"] = int(train_epochs)
    es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if es_cfg:
        template.setdefault("train", {})
        if "enabled" in es_cfg:
            template["train"]["early_stopping_enabled"] = bool(es_cfg.get("enabled"))
        if "patience" in es_cfg:
            template["train"]["early_stopping_patience"] = int(es_cfg.get("patience"))
        if "min_delta" in es_cfg:
            template["train"]["early_stopping_min_delta"] = float(es_cfg.get("min_delta"))
        if "warmup" in es_cfg:
            template["train"]["early_stopping_warmup"] = int(es_cfg.get("warmup"))
        if "monitor" in es_cfg:
            template["train"]["early_stopping_monitor"] = str(es_cfg.get("monitor"))

    template.setdefault("eval", {})
    template["eval"]["num_candidates"] = int(cfg["num_candidates"])
    template["eval"]["multimodal_threshold"] = float(cfg["humanmac_multimodal_threshold"])

    with tempfile.TemporaryDirectory(prefix="belfusion_") as td:
        td_path = Path(td)
        cfg_path = td_path / "belfusion.yaml"
        out_root = run_root
        out_eval = out_root / "eval_stats.csv"
        template.setdefault("runtime", {})
        template["runtime"]["output_dir"] = str(out_root)
        template["runtime"]["metrics_csv"] = str(out_eval)
        reuse_run = _as_dict(cfg.get("_reuse_run"))
        if reuse_run:
            source_root = Path(str(reuse_run["run_dir"]))
            for source_path in (Path(p) for p in reuse_run.get("artifacts", [])):
                relative = source_path.relative_to(source_root)
                destination = out_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
            reused = [out_root / Path(p).relative_to(source_root) for p in reuse_run.get("artifacts", []) if Path(p).name == "final.pt"]
            if len(reused) == 1:
                template["runtime"]["reuse_checkpoint_path"] = str(reused[0])
        template["runtime"]["checkpoint_path"] = str(out_root / "checkpoints" / "final.pt")
        if samples_root is not None:
            template["runtime"]["eval_samples_path"] = str(samples_root / "eval_samples.npz")
        _dump_yaml(cfg_path, template)

        cmd = [
            PYTHON,
            "run_belfusion.py",
            "--config",
            str(cfg_path),
            "--seed",
            str(cfg["seed"]),
            "--data-dir",
            str(data_dir),
            "--action-filter",
            str(action_filter),
            "--num-candidates",
            str(cfg["num_candidates"]),
            "--multimodal-threshold",
            str(cfg["humanmac_multimodal_threshold"]),
        ]
        rc = _run(cmd, cwd=wd, env=_gpu_subprocess_env(cfg))
        if rc != 0:
            raise RuntimeError("BeLFusion run failed")

        row = read_one_row_csv(out_eval)
        return normalize_metrics_dict(row)


def run_comusion(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "comusion"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    run_root = _maybe_model_run_root(model_name, run_id, save_eval_samples, cfg)
    checkpoint_dir = _checkpoint_dir(model_name, run_id, cfg)
    cfg_id = f"dh_{run_id}_comusion"
    cfg_path = wd / "cfg" / f"{cfg_id}.yml"
    template = deepcopy(_as_dict(mcfg.get("defaults")))
    if not template:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")
    template["t_his"] = int(pp["input_n"])
    template["t_pred"] = int(pp["output_n"])
    # CoMusion's DCT basis must not exceed the temporal length available to
    # the current experiment.  Infer it from the shared preprocessing so a
    # 30+10 window uses dct_dim=40, while the 50+50 benchmark uses 100.
    total_frames = int(pp["input_n"]) + int(pp["output_n"])
    template.setdefault("model_specs", {})["dct_dim"] = total_frames
    template["data_specs"]["dataset"] = str(dataset).lower()
    template["data_specs"]["data_dir"] = str(data_dir)
    template["data_specs"]["action_filter"] = (
        "" if str(action_filter).strip().lower() in {"all", "*"} else str(action_filter)
    )
    template["data_specs"]["stride"] = int(pp["stride"])
    template["data_specs"]["time_interp"] = pp.get("time_interp")
    template["data_specs"]["window_norm"] = pp.get("window_norm")
    template["data_specs"]["eval_batch_mult"] = int(pp["eval_batch_mult"])
    template["data_specs"]["splineeqnet_root"] = str(VENDOR / "splineeqnet")
    template["data_specs"]["eval_subset_indices_path"] = str(_as_dict(cfg.get("evaluation_subset")).get("indices_path", "") or "")
    # Keep CoMusion's multimodal grouping consistent with the shared
    # experiment-level evaluation threshold used by the other runners.
    template["data_specs"]["multimodal_threshold"] = float(cfg["humanmac_multimodal_threshold"])
    template["data_specs"]["humanmac_multimodal_threshold"] = float(cfg["humanmac_multimodal_threshold"])
    template["eval_sample_num"] = int(cfg["num_candidates"])
    template.setdefault("diff_specs", {})
    if template["diff_specs"].get("div_k") is None:
        template["diff_specs"]["div_k"] = int(cfg["num_candidates"])
    else:
        template["diff_specs"]["div_k"] = int(template["diff_specs"]["div_k"])
    train_epochs = mcfg.get("train", {}).get("epochs")
    if train_epochs is not None:
        tepoch = int(train_epochs)
        template["learn_specs"]["train_epoch"] = tepoch
        template["learn_specs"]["num_epoch_fix_lr"] = min(tepoch, int(template["learn_specs"].get("num_epoch_fix_lr", tepoch)))
    es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if es_cfg:
        template.setdefault("learn_specs", {})
        if "enabled" in es_cfg:
            template["learn_specs"]["early_stopping_enabled"] = bool(es_cfg.get("enabled"))
        if "patience" in es_cfg:
            template["learn_specs"]["early_stopping_patience"] = int(es_cfg.get("patience"))
        if "min_delta" in es_cfg:
            template["learn_specs"]["early_stopping_min_delta"] = float(es_cfg.get("min_delta"))
        if "warmup" in es_cfg:
            template["learn_specs"]["early_stopping_warmup"] = int(es_cfg.get("warmup"))
        if "monitor" in es_cfg:
            template["learn_specs"]["early_stopping_monitor"] = str(es_cfg.get("monitor"))
    template["logging_specs"]["model_id"] = cfg_id
    template["logging_specs"]["model_path"] = f"./results/{cfg_id}"
    template["logging_specs"]["final_checkpoint_path"] = str(checkpoint_dir / "final.pt") if checkpoint_dir is not None else ""
    reuse_run = _as_dict(cfg.get("_reuse_run"))
    if reuse_run:
        candidates = [
            Path(p) for p in reuse_run.get("artifacts", [])
            if Path(p).name == "final.pt" or Path(p).name.endswith("_final.pt")
        ]
        if len(candidates) == 1:
            destination = _suite_model_run_root(cfg, model_name, run_id) / "checkpoints" / "final.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidates[0], destination)
            template["logging_specs"]["reuse_checkpoint_path"] = str(destination)
        elif _checkpoint_only_requested(cfg):
            raise RuntimeError("CoMusion checkpoint-only evaluation requires exactly one final checkpoint.")
    per_action_cfg = _as_dict(cfg.get("runtime")).get(
        "per_action_csv", "./results/all_models_by_action.csv"
    )
    per_action_path = Path(str(per_action_cfg))
    if not per_action_path.is_absolute():
        per_action_path = ROOT / per_action_path
    template["logging_specs"]["per_action_metrics_path"] = str(per_action_path)
    template["logging_specs"]["result_model_name"] = str(model_name)
    if run_root is not None:
        template["eval_samples_path"] = str(run_root / "eval_samples.npz")
    _dump_yaml(cfg_path, template)

    try:
        rc = _run(
            [
                PYTHON,
                "train.py",
                "--cfg",
                cfg_id,
                "--seed",
                str(cfg["seed"]),
                "--gpu_index",
                str(_child_gpu_index(cfg)),
            ],
            cwd=wd,
            env=_gpu_subprocess_env(cfg),
        )
        if rc != 0:
            raise RuntimeError("CoMusion run failed")

        stats_path = wd / "results" / cfg_id / "results" / "eval_stats.csv"
        row = read_one_row_csv(stats_path)
        return normalize_metrics_dict(row)
    finally:
        try:
            cfg_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            shutil.rmtree(wd / "results" / cfg_id, ignore_errors=True)
        except OSError:
            pass


def run_dlow_cvae(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "dlow"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    run_root = _maybe_model_run_root(model_name, run_id, save_eval_samples, cfg)
    checkpoint_dir = _checkpoint_dir(model_name, run_id, cfg)
    cfg_id = f"dh_{run_id}_dlow_cvae"
    cfg_path = wd / "motion_pred" / "cfg" / f"{cfg_id}.yml"
    template = deepcopy(_as_dict(mcfg.get("defaults")))
    if not template:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")
    template["dataset"] = str(dataset).lower()
    template["data_dir"] = str(data_dir)
    template["action_filter"] = str(action_filter)
    template["eval_subset_indices_path"] = str(
        _as_dict(cfg.get("evaluation_subset")).get("indices_path", "") or ""
    )
    template["t_his"] = int(pp["input_n"])
    template["t_pred"] = int(pp["output_n"])
    template["stride"] = int(pp["stride"])
    template["time_interp"] = pp.get("time_interp")
    template["window_norm"] = pp.get("window_norm")
    template["splineeqnet_root"] = str(VENDOR / "splineeqnet")
    template["seed"] = int(cfg["seed"])
    template["data_seed"] = int(_data_seed(cfg))
    template["nk"] = int(cfg["num_candidates"])
    train_epochs = mcfg.get("train", {}).get("epochs")
    if train_epochs is not None:
        vae_epoch = int(train_epochs)
        template["num_vae_epoch"] = vae_epoch
        template["num_vae_epoch_fix"] = min(vae_epoch, int(template.get("num_vae_epoch_fix", vae_epoch)))
    es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if es_cfg:
        if "enabled" in es_cfg:
            template["early_stopping_enabled"] = bool(es_cfg.get("enabled"))
        if "patience" in es_cfg:
            template["early_stopping_patience"] = int(es_cfg.get("patience"))
        if "min_delta" in es_cfg:
            template["early_stopping_min_delta"] = float(es_cfg.get("min_delta"))
        if "warmup" in es_cfg:
            template["early_stopping_warmup"] = int(es_cfg.get("warmup"))
        if "monitor" in es_cfg:
            template["early_stopping_monitor"] = str(es_cfg.get("monitor"))
    save_interval = mcfg.get("train", {}).get("save_model_interval")
    if save_interval is not None:
        template["save_model_interval"] = int(save_interval)
    else:
        template["save_model_interval"] = 0
    if run_root is not None:
        template["eval_samples_path"] = str(run_root / "eval_samples.npz")
    template["final_checkpoint_path"] = str(checkpoint_dir / "final.pt") if checkpoint_dir is not None else ""
    _dump_yaml(cfg_path, template)

    reuse_run = _as_dict(cfg.get("_reuse_run"))
    reused_checkpoint = None
    if reuse_run:
        candidates = [Path(p) for p in reuse_run.get("artifacts", []) if Path(p).name == "final.pt"]
        if len(candidates) == 1:
            reused_checkpoint = _suite_model_run_root(cfg, model_name, run_id) / "checkpoints" / "final.pt"
            reused_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidates[0], reused_checkpoint)

    rc = _run(
        [
            PYTHON,
            "motion_pred/exp_vae.py",
            "--cfg",
            cfg_id,
            "--seed",
            str(cfg["seed"]),
            "--gpu_index",
            str(_child_gpu_index(cfg)),
            *( ["--mode", "eval", "--checkpoint", str(reused_checkpoint)] if reused_checkpoint else ["--eval_after_train"] ),
            "--eval_sample_num",
            str(cfg["num_candidates"]),
            "--multimodal_threshold",
            str(cfg["humanmac_multimodal_threshold"]),
        ],
        cwd=wd,
        env=_gpu_subprocess_env(cfg),
    )
    if rc != 0:
        raise RuntimeError("DLow CVAE training failed")

    stats_csv = wd / "results" / cfg_id / "results" / "stats_1.csv"
    out: Dict[str, float] = {}
    with open(stats_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metric = row.get("Metric", "")
            if metric and "vae" in row and row["vae"]:
                try:
                    out[metric] = float(row["vae"])
                except ValueError:
                    pass
    try:
        cfg_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        shutil.rmtree(wd / "results" / cfg_id, ignore_errors=True)
    except OSError:
        pass
    return normalize_metrics_dict(out)


def run_humanmac(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "humanmac"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    run_root = _maybe_model_run_root(model_name, run_id, save_eval_samples, cfg)
    checkpoint_dir = _checkpoint_dir(model_name, run_id, cfg)
    cfg_id = f"dh_{run_id}_humanmac"
    cfg_path = wd / "cfg" / f"{cfg_id}.yml"
    template = deepcopy(_as_dict(mcfg.get("defaults")))
    if not template:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")
    template["dataset"] = str(dataset).lower()
    template["data_dir"] = str(data_dir)
    template["action_filter"] = str(action_filter)
    template["t_his"] = int(pp["input_n"])
    template["t_pred"] = int(pp["output_n"])
    template["time_interp"] = pp.get("time_interp")
    template["window_norm"] = int(pp["window_norm"]) if pp.get("window_norm") is not None else int(pp["input_n"])
    template["stride"] = int(pp["stride"])
    template["splineeqnet_root"] = str(VENDOR / "splineeqnet")
    template["num_candidates"] = int(cfg["num_candidates"])
    template["mpjpe_best_of_k"] = int(cfg["num_candidates"])
    train_epochs = mcfg.get("train", {}).get("epochs")
    if train_epochs is not None:
        template["num_epoch"] = int(train_epochs)
    es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if es_cfg:
        if "enabled" in es_cfg:
            template["early_stopping_enabled"] = bool(es_cfg.get("enabled"))
        if "patience" in es_cfg:
            template["early_stopping_patience"] = int(es_cfg.get("patience"))
        if "min_delta" in es_cfg:
            template["early_stopping_min_delta"] = float(es_cfg.get("min_delta"))
        if "warmup" in es_cfg:
            template["early_stopping_warmup"] = int(es_cfg.get("warmup"))
        if "monitor" in es_cfg:
            template["early_stopping_monitor"] = str(es_cfg.get("monitor"))
    _dump_yaml(cfg_path, template)

    num_epoch = int(template.get("num_epoch", 200))
    es_enabled = bool(es_cfg.get("enabled", False)) if es_cfg else False
    es_patience = int(es_cfg.get("patience", 20)) if es_cfg else 20
    es_min_delta = float(es_cfg.get("min_delta", 1e-4)) if es_cfg else 1e-4
    es_warmup = int(es_cfg.get("warmup", 0)) if es_cfg else 0
    es_monitor = str(es_cfg.get("monitor", "train_loss")) if es_cfg else "train_loss"
    validate_last_only = False if (es_enabled and es_monitor.strip().lower() == "val_loss") else True
    reuse_run = _as_dict(cfg.get("_reuse_run"))
    reused_checkpoint = None
    if reuse_run:
        candidates = [Path(p) for p in reuse_run.get("artifacts", []) if Path(p).name == "final.pt"]
        if len(candidates) == 1:
            reused_checkpoint = _suite_model_run_root(cfg, model_name, run_id) / "checkpoints" / "final.pt"
            reused_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidates[0], reused_checkpoint)
    run_mode = "eval" if reused_checkpoint else "train"
    # HumanMAC uses ``inference/`` whenever Config is constructed with
    # ``test=True`` (all eval-mode checkpoint runs); fresh training writes to
    # ``results/``.  Looking in the wrong root made completed evaluations look
    # like failures and discarded their metrics.
    result_root = _humanmac_result_root(wd, run_mode)
    result_dirs_before = {
        path.resolve() for path in result_root.iterdir()
        if path.is_dir()
    } if result_root.is_dir() else set()
    rc = _run(
        [
            PYTHON,
            "main_comp.py",
            "--cfg",
            cfg_id,
            "--mode",
            run_mode,
            *(["--ckpt", str(reused_checkpoint)] if reused_checkpoint else []),
            "--seed",
            str(cfg["seed"]),
            "--split_seed",
            str(_data_seed(cfg)),
            "--num_candidates",
            str(cfg["num_candidates"]),
            "--multimodal_threshold",
            str(cfg["humanmac_multimodal_threshold"]),
            "--data_dir",
            str(data_dir),
            "--action_filter",
            str(action_filter),
            *(["--eval_samples_path", str(run_root / "eval_samples.npz")] if run_root is not None else []),
            "--num_epoch",
            str(num_epoch),
            "--save_model_interval",
            "0",
            "--save_metrics_interval",
            "1",
            "--validate_last_epoch_only",
            "false" if not validate_last_only else "true",
            "--early_stopping_enabled",
            "true" if es_enabled else "false",
            "--early_stopping_patience",
            str(es_patience),
            "--early_stopping_min_delta",
            str(es_min_delta),
            "--early_stopping_warmup",
            str(es_warmup),
            "--early_stopping_monitor",
            str(es_monitor),
            *(["--final_checkpoint_path", str(checkpoint_dir / "final.pt")] if checkpoint_dir is not None else []),
        ],
        cwd=wd,
        env=_gpu_subprocess_env(cfg),
    )
    if rc != 0:
        raise RuntimeError("HumanMAC run failed")

    # Prefer the directory created by this invocation.  This is robust to
    # HumanMAC's historical eval_* naming and avoids accidentally reading an
    # older run with the same configuration id.
    candidates = sorted(
        [
            path for path in result_root.iterdir()
            if path.is_dir() and path.resolve() not in result_dirs_before
        ] if result_root.is_dir() else [],
        key=os.path.getmtime,
    )
    if not candidates:
        candidates = sorted(result_root.glob(f"{cfg_id}_*"), key=os.path.getmtime)
    if not candidates:
        raise RuntimeError("HumanMAC result directory not found")
    latest = candidates[-1]
    stats_latest = latest / "results" / "stats_latest.csv"
    if not stats_latest.exists():
        raise RuntimeError(f"HumanMAC training metrics not found in {latest / 'results'}")

    out: Dict[str, float] = {}
    with open(stats_latest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metric = row.get("Metric", "")
            val = row.get("HumanMAC", "")
            if metric and val:
                try:
                    out[metric] = float(val)
                except ValueError:
                    pass
    try:
        cfg_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        shutil.rmtree(latest, ignore_errors=True)
    except OSError:
        pass
    return normalize_metrics_dict(out)


def _humanmac_result_root(workdir: Path, mode: str) -> Path:
    """Return the root selected by HumanMAC's ``Config(test=...)`` branch."""
    return Path(workdir) / ("inference" if str(mode).lower() == "eval" else "results")


def run_skeletondiffusion(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "skeletondiffusion"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    samples_root = _maybe_model_run_root(model_name, run_id, save_eval_samples, cfg)
    checkpoint_dir = _checkpoint_dir(model_name, run_id, cfg)
    run_root = wd / "out" / "diffusion_hands_runs" / f"{model_name}_{run_id}"
    if run_root.exists():
        shutil.rmtree(run_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        train_cfg = mcfg.get("train", {})
        defaults = _as_dict(mcfg.get("defaults"))
        if not defaults:
            raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")
        auto_defaults = _as_dict(defaults.get("train_autoencoder"))
        diff_defaults = _as_dict(defaults.get("train_diffusion"))
        eval_defaults = _as_dict(defaults.get("eval"))
        auto_cfg = train_cfg.get("autoencoder", {}) if isinstance(train_cfg.get("autoencoder", {}), dict) else {}
        diff_cfg = train_cfg.get("diffusion", {}) if isinstance(train_cfg.get("diffusion", {}), dict) else {}
        auto_epochs = auto_cfg.get("epochs")
        auto_iters = auto_cfg.get("iter_per_epoch")
        diff_epochs = diff_cfg.get("epochs")
        diff_iters = diff_cfg.get("iter_per_epoch")
        diff_model = str(diff_cfg.get("model", "skeleton_diffusion"))
        auto_es_cfg = _as_dict(auto_cfg.get("early_stopping"))
        diff_es_cfg = _as_dict(diff_cfg.get("early_stopping"))
        fps = int(_as_dict(auto_defaults.get("dataset")).get("fps", 10))
        hist_sec = float(pp["input_n"]) / float(fps)
        pred_sec = float(pp["output_n"]) / float(fps)
        time_interp_val = _hydra_value(pp.get("time_interp"))
        window_norm_val = _hydra_value(pp.get("window_norm"))
        dataset_lower = str(dataset).lower()
        wrist_indices_by_dataset = {
            "assembly": "[5,26]",
            "h2o": "[0,21]",
            "bighands": "[0]",
            "fpha": "[0]",
        }
        wrist_indices_hydra = wrist_indices_by_dataset.get(dataset_lower, "[5,26]")

        autoenc_out = run_root / "autoencoder"
        diff_out = run_root / "diffusion"
        reuse_run = _as_dict(cfg.get("_reuse_run"))
        if reuse_run:
            reuse_artifacts = [Path(p) for p in reuse_run.get("artifacts", [])]
            auto_saved = [p for p in reuse_artifacts if p.name == "autoencoder_final.pt"]
            diff_saved = [p for p in reuse_artifacts if p.name == "diffusion_final.pt"]
            if len(auto_saved) == 1 and len(diff_saved) == 1:
                (autoenc_out / "checkpoints").mkdir(parents=True, exist_ok=True)
                (diff_out / "checkpoints").mkdir(parents=True, exist_ok=True)
                shutil.copy2(auto_saved[0], autoenc_out / "checkpoints" / "checkpoint_0.pt")
                shutil.copy2(diff_saved[0], diff_out / "checkpoints" / "checkpoint_0.pt")
                auto_epochs = 0
                diff_epochs = 0

        # Autoencoder (original protocol first stage).
        auto_cmd = [PYTHON, "train_autoencoder.py"]
        auto_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("", _as_dict(auto_defaults.get("config")))])
        auto_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("task", _as_dict(auto_defaults.get("task")))])
        auto_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("dataset", _as_dict(auto_defaults.get("dataset")))])
        auto_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("model", _as_dict(auto_defaults.get("model")))])
        auto_cmd.extend(
            [
                "dataset=assembly",
                "task=assembly_hmp",
                f"dataset.assembly_dataset_name={dataset}",
                f"dataset.assembly_splineeqnet_root={VENDOR / 'splineeqnet'}",
                f"dataset.assembly_data_dir={data_dir}",
                f"dataset.assembly_action_filter={action_filter}",
                f"dataset.assembly_wrist_indices={wrist_indices_hydra}",
                f"++dataset.assembly_data_seed={_data_seed(cfg)}",
                f"task.history_sec={hist_sec}",
                f"task.prediction_horizon_sec={pred_sec}",
                f"dataset.assembly_time_interp={time_interp_val}",
                f"dataset.assembly_window_norm={window_norm_val}",
                f"dataset.data_loader_train.stride={int(pp['stride'])}",
                f"dataset.data_loader_train_eval.stride={int(pp['stride'])}",
                f"dataset.data_loader_valid.stride={int(pp['stride'])}",
                f"output_log_path={autoenc_out}",
                f"model.seed={cfg['seed']}",
                *(["model.num_epochs=%d" % int(auto_epochs)] if auto_epochs is not None else []),
                *(["model.num_iter_perepoch=%d" % int(auto_iters)] if auto_iters is not None else []),
                *(["++model.early_stopping_enabled=%s" % ("true" if bool(auto_es_cfg.get("enabled", False)) else "false")] if auto_es_cfg else []),
                *(["++model.early_stopping_patience=%d" % int(auto_es_cfg.get("patience", 20))] if auto_es_cfg else []),
                *(["++model.early_stopping_min_delta=%s" % float(auto_es_cfg.get("min_delta", 1e-4))] if auto_es_cfg else []),
                *(["++model.early_stopping_warmup=%d" % int(auto_es_cfg.get("warmup", 0))] if auto_es_cfg else []),
                *(["++model.early_stopping_monitor=%s" % str(auto_es_cfg.get("monitor", "train_loss"))] if auto_es_cfg else []),
            ]
        )
        rc = _run(auto_cmd, cwd=wd, env=_gpu_subprocess_env(cfg))
        if rc != 0:
            raise RuntimeError("skeletondiffusion autoencoder training failed")

        auto_ckpts = sorted((autoenc_out / "checkpoints").glob("checkpoint_*.pt"), key=os.path.getmtime)
        if not auto_ckpts:
            raise RuntimeError("skeletondiffusion autoencoder checkpoint not found")
        auto_ckpt = auto_ckpts[-1]
        if checkpoint_dir is not None:
            shutil.copy2(auto_ckpt, checkpoint_dir / "autoencoder_final.pt")

        # Diffusion (original protocol second stage).
        diff_cmd = [PYTHON, "train_diffusion.py"]
        diff_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("", _as_dict(diff_defaults.get("config")))])
        diff_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("model", _as_dict(diff_defaults.get("model")))])
        diff_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("cov_matrix", _as_dict(diff_defaults.get("cov_matrix")))])
        diff_cmd.extend(
            [
                f"model={diff_model}",
                f"model.pretrained_autoencoder_path={auto_ckpt}",
                f"output_log_path={diff_out}",
                f"model.seed={cfg['seed']}",
                *(["model.num_epochs=%d" % int(diff_epochs)] if diff_epochs is not None else []),
                *(["model.num_iter_perepoch=%d" % int(diff_iters)] if diff_iters is not None else []),
                *(["++model.early_stopping_enabled=%s" % ("true" if bool(diff_es_cfg.get("enabled", False)) else "false")] if diff_es_cfg else []),
                *(["++model.early_stopping_patience=%d" % int(diff_es_cfg.get("patience", 20))] if diff_es_cfg else []),
                *(["++model.early_stopping_min_delta=%s" % float(diff_es_cfg.get("min_delta", 1e-4))] if diff_es_cfg else []),
                *(["++model.early_stopping_warmup=%d" % int(diff_es_cfg.get("warmup", 0))] if diff_es_cfg else []),
                *(["++model.early_stopping_monitor=%s" % str(diff_es_cfg.get("monitor", "train_loss"))] if diff_es_cfg else []),
            ]
        )
        rc = _run(diff_cmd, cwd=wd, env=_gpu_subprocess_env(cfg))
        if rc != 0:
            raise RuntimeError("skeletondiffusion diffusion training failed")

        diff_ckpts = sorted((diff_out / "checkpoints").glob("checkpoint_*.pt"), key=os.path.getmtime)
        if not diff_ckpts:
            raise RuntimeError("skeletondiffusion diffusion checkpoint not found")
        diff_ckpt = diff_ckpts[-1]
        if checkpoint_dir is not None:
            shutil.copy2(diff_ckpt, checkpoint_dir / "diffusion_final.pt")

        # Eval (keeps test split, as requested).
        eval_cmd = [PYTHON, "eval.py"]
        eval_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("", _as_dict(eval_defaults.get("config")))])
        eval_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("task", _as_dict(eval_defaults.get("task")))])
        eval_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("method_specs", _as_dict(eval_defaults.get("method_specs")))])
        eval_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("dataset", _as_dict(eval_defaults.get("dataset")))])
        eval_cmd.extend(
            [
                "dataset=assembly",
                "task=assembly_hmp",
                "dataset_split=test",
                "assembly_split_strategy=test",
                f"assembly_dataset_name={dataset}",
                f"dataset.assembly_dataset_name={dataset}",
                f"checkpoint_path={diff_ckpt}",
                "if_use_splineeqnet_assembly_pipeline=True",
                f"assembly_splineeqnet_root={VENDOR / 'splineeqnet'}",
                f"dataset.assembly_splineeqnet_root={VENDOR / 'splineeqnet'}",
                f"assembly_data_dir={data_dir}",
                f"dataset.assembly_data_dir={data_dir}",
                f"assembly_action_filter={action_filter}",
                f"dataset.assembly_action_filter={action_filter}",
                f"dataset.assembly_wrist_indices={wrist_indices_hydra}",
                f"++assembly_data_seed={_data_seed(cfg)}",
                f"task.history_sec={hist_sec}",
                f"task.prediction_horizon_sec={pred_sec}",
                f"obs_length={int(pp['input_n'])}",
                f"pred_length={int(pp['output_n'])}",
                f"assembly_stride={int(pp['stride'])}",
                f"assembly_time_interp={time_interp_val}",
                f"dataset.assembly_time_interp={time_interp_val}",
                f"assembly_window_norm={window_norm_val}",
                f"dataset.assembly_window_norm={window_norm_val}",
                f"assembly_eval_batch_mult={int(pp['eval_batch_mult'])}",
                f"num_samples={cfg['num_candidates']}",
                f"num_candidates={cfg['num_candidates']}",
                f"humanmac_multimodal_threshold={cfg['humanmac_multimodal_threshold']}",
                f"seed={cfg['seed']}",
                *([f"++eval_samples_path={samples_root / 'eval_samples.npz'}"] if samples_root is not None else []),
            ]
        )
        rc = _run(eval_cmd, cwd=wd, env=_gpu_subprocess_env(cfg))
        if rc != 0:
            raise RuntimeError("skeletondiffusion eval failed")

        eval_yamls = sorted(glob.glob(str(diff_out / "**" / f"results_*_{dataset}.yaml"), recursive=True), key=os.path.getmtime)
        if not eval_yamls:
            raise RuntimeError("skeletondiffusion eval yaml not found")
        with open(eval_yamls[-1], "r", encoding="utf-8") as f:
            row = yaml.safe_load(f)
        return normalize_metrics_dict({k: float(v) for k, v in row.items() if isinstance(v, (int, float))})
    finally:
        try:
            shutil.rmtree(run_root, ignore_errors=True)
        except OSError:
            pass


def run_hybrid_cvae(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    """Run the upstream Hybrid-CVAE architecture under the suite protocol.

    The architecture remains upstream, but its inputs and benchmark protocol
    are the same shared SplineEqNet pipeline used by the other models.
    """
    mcfg = cfg.get("models", {}).get(model_name, {})
    defaults = deepcopy(_as_dict(mcfg.get("defaults")))
    options = _as_dict(mcfg.get("options"))
    source_root = Path(str(options.get("source_root", DEFAULT_HYBRID_CVAE_ROOT))).expanduser()
    model_source = source_root / "models" / "hybrid_hyperbolic_cvae.py"
    if not model_source.is_file():
        raise FileNotFoundError(
            f"Hybrid-CVAE model source not found: {model_source}. "
            f"Set models.{model_name}.options.source_root."
        )

    pp = _as_dict(cfg.get("_shared_preprocessing"))
    # The adapter writes native model checkpoints into the standard suite run
    # directory, while all data comes from the shared loader.
    persistent = _save_model_enabled(cfg) or _save_eval_samples_enabled(mcfg)
    temporary: Optional[tempfile.TemporaryDirectory] = None
    if persistent:
        run_root = _suite_model_run_root(cfg, model_name, run_id)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="hybrid_cvae_")
        run_root = Path(temporary.name) / "run"
    run_root.mkdir(parents=True, exist_ok=True)

    try:
        train_cfg = _as_dict(mcfg.get("train"))
        epochs = int(train_cfg.get("epochs", defaults.get("epochs", 40)))
        cmd = [
            PYTHON, str(ROOT / "tools" / "train_hybrid_cvae.py"),
            "--source-root", str(source_root),
            "--data-dir", str(data_dir),
            "--dataset", str(dataset),
            "--action-filter", str(action_filter),
            "--out-dir", str(run_root),
            "--input-n", str(int(pp["input_n"])),
            "--output-n", str(int(pp["output_n"])),
            "--stride", str(int(pp["stride"])),
            "--hidden", str(int(defaults.get("hidden", 128))),
            "--latent", str(int(defaults.get("latent", 32))),
            "--geometry", str(defaults.get("geometry", "hyperbolic")),
            "--curvature", str(float(defaults.get("curvature", 1.0))),
            "--hierarchy-weight", str(float(defaults.get("hierarchy_weight", 0.001))),
            "--components", str(int(defaults.get("components", 1))),
            "--beta", str(float(defaults.get("beta", 1e-3))),
            "--velocity-weight", str(float(defaults.get("velocity_weight", 0.1))),
            "--warmup", str(int(defaults.get("warmup", 10))),
            "--epochs", str(epochs),
            "--patience", str(int(defaults.get("patience", 7))),
            "--batch-size", str(int(defaults.get("batch_size", 64))),
            "--lr", str(float(defaults.get("lr", 3e-4))),
            "--seed", str(int(cfg["seed"])),
            "--num-candidates", str(int(cfg["num_candidates"])),
            "--threshold", str(float(cfg["humanmac_multimodal_threshold"])),
            "--data-seed", str(_data_seed(cfg)),
            "--eval-batch-mult", str(int(pp["eval_batch_mult"])),
            "--device", "cpu" if _child_gpu_index(cfg) < 0 else "cuda:0",
        ]
        if pp.get("time_interp") is not None:
            cmd.extend(["--time-interp", str(int(pp["time_interp"]))])
        if pp.get("window_norm") is not None:
            cmd.extend(["--window-norm", str(int(pp["window_norm"]))])
        if _save_eval_samples_enabled(mcfg):
            cmd.extend(["--save-eval-samples", str(run_root / "eval_samples.npz")])

        rc = _run(
            cmd, cwd=source_root, env=_gpu_subprocess_env(cfg),
            log_path=_as_dict(cfg.get("runtime")).get("log_path"),
        )
        if rc != 0:
            raise RuntimeError(f"{model_name} training failed (Hybrid-CVAE backend)")
        metrics_path = run_root / "metrics_test.json"
        if not metrics_path.is_file():
            raise RuntimeError(f"Hybrid-CVAE did not write test metrics: {metrics_path}")
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        return normalize_metrics_dict(raw)
    finally:
        if temporary is not None:
            temporary.cleanup()


def run_gsps(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "gsps"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    samples_root = _maybe_model_run_root(model_name, run_id, save_eval_samples, cfg)
    run_root = _vendor_model_run_root("gsps", model_name, run_id, cfg)
    template = deepcopy(_as_dict(mcfg.get("defaults")))
    if not template:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")

    template["seed"] = int(cfg["seed"])
    template["gpu_index"] = int(_child_gpu_index(cfg))
    template["dataset"] = str(dataset).lower()
    template["data_dir"] = str(data_dir)
    template["action_filter"] = str(action_filter)
    template["splineeqnet_root"] = str(VENDOR / "splineeqnet")

    template.setdefault("preprocessing", {})
    template["preprocessing"]["input_n"] = int(pp["input_n"])
    template["preprocessing"]["output_n"] = int(pp["output_n"])
    template["preprocessing"]["stride"] = int(pp["stride"])
    template["preprocessing"]["time_interp"] = pp.get("time_interp")
    template["preprocessing"]["window_norm"] = pp.get("window_norm")
    template["preprocessing"]["eval_batch_mult"] = int(pp["eval_batch_mult"])

    template.setdefault("model", {})
    template["model"]["nk"] = int(cfg["num_candidates"])
    train_epochs = mcfg.get("train", {}).get("epochs")
    if train_epochs is not None:
        template.setdefault("train", {})
        template["train"]["epochs"] = int(train_epochs)
    train_batch = mcfg.get("train", {}).get("batch_size")
    if train_batch is not None:
        template.setdefault("train", {})
        template["train"]["batch_size"] = int(train_batch)
    train_es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if train_es_cfg:
        template.setdefault("train", {})
        if "enabled" in train_es_cfg:
            template["train"]["early_stopping_enabled"] = bool(train_es_cfg.get("enabled"))
        if "patience" in train_es_cfg:
            template["train"]["early_stopping_patience"] = int(train_es_cfg.get("patience"))
        if "min_delta" in train_es_cfg:
            template["train"]["early_stopping_min_delta"] = float(train_es_cfg.get("min_delta"))
        if "warmup" in train_es_cfg:
            template["train"]["early_stopping_warmup"] = int(train_es_cfg.get("warmup"))
        if "monitor" in train_es_cfg:
            template["train"]["early_stopping_monitor"] = str(train_es_cfg.get("monitor"))
    prior_es_cfg = _as_dict(_as_dict(mcfg.get("prior")).get("early_stopping"))
    if prior_es_cfg:
        template.setdefault("prior", {})
        if "enabled" in prior_es_cfg:
            template["prior"]["early_stopping_enabled"] = bool(prior_es_cfg.get("enabled"))
        if "patience" in prior_es_cfg:
            template["prior"]["early_stopping_patience"] = int(prior_es_cfg.get("patience"))
        if "min_delta" in prior_es_cfg:
            template["prior"]["early_stopping_min_delta"] = float(prior_es_cfg.get("min_delta"))
        if "warmup" in prior_es_cfg:
            template["prior"]["early_stopping_warmup"] = int(prior_es_cfg.get("warmup"))
        if "monitor" in prior_es_cfg:
            template["prior"]["early_stopping_monitor"] = str(prior_es_cfg.get("monitor"))

    template.setdefault("eval", {})
    template["eval"]["multimodal_threshold"] = float(cfg["humanmac_multimodal_threshold"])

    with tempfile.TemporaryDirectory(prefix="gsps_") as td:
        td_path = Path(td)
        cfg_path = td_path / "gsps.yaml"
        out_root = run_root
        out_eval = out_root / "eval_stats.csv"
        template.setdefault("runtime", {})
        template["runtime"]["output_dir"] = str(out_root)
        template["runtime"]["metrics_csv"] = str(out_eval)
        template["runtime"]["checkpoint_only"] = _checkpoint_only_requested(cfg)
        reuse_run = _as_dict(cfg.get("_reuse_run"))
        if reuse_run:
            source_root = Path(str(reuse_run["run_dir"]))
            for source_path in (Path(p) for p in reuse_run.get("artifacts", [])):
                destination = out_root / source_path.relative_to(source_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
        if samples_root is not None:
            template["runtime"]["eval_samples_path"] = str(samples_root / "eval_samples.npz")
        _dump_yaml(cfg_path, template)

        rc = _run([PYTHON, "run_gsps.py", "--config", str(cfg_path)], cwd=wd, env=_gpu_subprocess_env(cfg))
        if rc != 0:
            raise RuntimeError("GSPS run failed")

        row = read_one_row_csv(out_eval)
        return normalize_metrics_dict(row)


def run_literature_motion_model(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    """Run a shared-data adapter for an upstream motion-prediction method."""
    mcfg = cfg.get("models", {}).get(model_name, {})
    method = str(mcfg.get("base_model", model_name)).lower()
    if method not in {"cacheflow", "stars"}:
        raise ValueError(f"Unsupported literature adapter: {method}")
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    run_root = _suite_model_run_root(cfg, model_name, run_id)
    template = {
        "method": method,
        "seed": int(cfg["seed"]),
        "data": {
            "dataset": str(dataset).lower(), "data_dir": str(data_dir),
            "action_filter": str(action_filter), "input_n": int(pp["input_n"]),
            "output_n": int(pp["output_n"]), "stride": int(pp["stride"]),
            "time_interp": pp.get("time_interp"), "window_norm": pp.get("window_norm"),
            "data_seed": _data_seed(cfg),
            "n_nodes": 21, "splineeqnet_root": str(VENDOR / "splineeqnet"),
        },
        "model": deepcopy(_as_dict(mcfg.get("defaults"))),
        "train": deepcopy(_as_dict(mcfg.get("train"))),
        "eval": {
            "num_candidates": int(cfg["num_candidates"]),
            "multimodal_threshold": float(cfg["humanmac_multimodal_threshold"]),
            "batch_size": int(_as_dict(mcfg.get("defaults")).get("eval_batch_size", 128)),
        },
        "runtime": {
            "output_dir": str(run_root), "metrics_csv": str(run_root / "eval_stats.csv"),
            "eval_samples_path": str(run_root / "eval_samples.npz"),
            "eval_subset_indices_path": str(_as_dict(cfg.get("evaluation_subset")).get("indices_path", "") or ""),
            "checkpoint_only": _checkpoint_only_requested(cfg),
        },
    }
    # The shared adapter accepts the concise top-level training fields used by
    # the model YAMLs while retaining the same early-stop semantics.
    train = template["train"]
    early = _as_dict(train.pop("early_stopping", {}))
    if early:
        train.setdefault("patience", int(early.get("patience", 12)))
        train.setdefault("min_delta", float(early.get("min_delta", 1e-4)))
    reuse_run = _as_dict(cfg.get("_reuse_run"))
    if reuse_run:
        candidates = [
            Path(path) for path in reuse_run.get("artifacts", [])
            if Path(path).suffix.lower() in {".pt", ".pth", ".ckpt"}
        ]
        # The adapter produces one terminal checkpoint. Prefer that stable
        # name, while accepting archived names such as ``00_final.pt``.
        candidates = [
            path for path in candidates
            if path.name == "final.pt" or path.name.endswith("_final.pt")
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"{method} checkpoint reuse requires exactly one final checkpoint; found {len(candidates)}"
            )
        template["runtime"]["reuse_checkpoint_path"] = str(candidates[0])
    with tempfile.TemporaryDirectory(prefix=f"{method}_") as td:
        cfg_path = Path(td) / f"{method}.yaml"
        _dump_yaml(cfg_path, template)
        rc = _run(
            [PYTHON, str(ROOT / "tools" / "train_literature_motion_model.py"), "--config", str(cfg_path)],
            cwd=ROOT, env=_gpu_subprocess_env(cfg),
        )
        if rc != 0:
            raise RuntimeError(f"{method} adapter run failed")
    return normalize_metrics_dict(read_one_row_csv(run_root / "eval_stats.csv"))


def _resolve_datasets(cfg: dict) -> List[str]:
    datasets_raw = cfg.get("datasets")
    if datasets_raw is not None:
        if not isinstance(datasets_raw, list) or not datasets_raw:
            raise ValueError("'datasets' must be a non-empty list when provided.")
        out = [str(d).strip().lower() for d in datasets_raw if str(d).strip()]
    else:
        single = str(cfg.get("dataset", "")).strip().lower()
        if not single:
            raise ValueError("Missing 'dataset' (single value) or 'datasets' (non-empty list).")
        out = [single]

    deduped: List[str] = []
    seen = set()
    for d in out:
        if d not in seen:
            deduped.append(d)
            seen.add(d)
    return deduped


def _resolve_action_filters(cfg: dict, dataset: str) -> List[str]:
    # Apply action_filter only to assembly; keep other datasets unfiltered.
    if dataset != "assembly":
        return [""]

    raw = cfg.get("action_filter", "")
    if isinstance(raw, list):
        if not raw:
            raise ValueError("'action_filter' must be a non-empty list when provided as a list.")
        out = [str(action).strip() for action in raw if str(action).strip()]
        if not out:
            raise ValueError("'action_filter' list must contain at least one non-empty action.")
    else:
        configured_action_filter = "" if raw is None else str(raw).strip()
        if configured_action_filter.lower() in {"all", "*"}:
            out = ["all"]
        else:
            out = [default_action_filter(dataset, configured_action_filter)]

    deduped: List[str] = []
    seen = set()
    for action_filter in out:
        if action_filter not in seen:
            deduped.append(action_filter)
            seen.add(action_filter)
    return deduped


def _cleanup_legacy_result_artifacts(out_root: Path) -> None:
    # Remove legacy skeletondiffusion runs that were previously written under
    # diffusion_hands/results by older versions of this orchestrator.
    for p in out_root.glob("skeletondiffusion_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def _resolve_multimodal_threshold(
    cfg: dict,
    *,
    dataset: str,
    data_dir: Path,
    action_filter: str,
    run_id: str,
    out_root: Path,
) -> Tuple[float, Optional[Path]]:
    configured = cfg.get("humanmac_multimodal_threshold", 0.5)
    if not (isinstance(configured, str) and configured.strip().lower() == "auto"):
        threshold = float(configured)
        if threshold <= 0:
            raise ValueError("humanmac_multimodal_threshold must be positive or 'auto'")
        return threshold, None

    calibration = _as_dict(cfg.get("humanmac_multimodal_calibration"))
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    output_path = out_root / "multimodal_calibration" / f"{run_id}.json"
    cmd = [
        PYTHON,
        str(ROOT / "tools" / "calibrate_multimodal_threshold.py"),
        "--dataset", dataset,
        "--data-dir", str(data_dir),
        "--action-filter", action_filter,
        "--input-n", str(int(pp["input_n"])),
        "--output-n", str(int(pp["output_n"])),
        "--stride", str(int(pp["stride"])),
        "--data-seed", str(int(cfg.get("data_seed", 0) or 0)),
        "--max-source-files", str(int(_as_dict(cfg.get("shared_defaults")).get("max_source_files", 0))),
        "--k", str(int(calibration.get("k", 1))),
        "--threshold-scale", str(AUTO_MULTIMODAL_THRESHOLD_SCALE),
        "--neighbour-retry-scale", str(float(calibration.get("neighbour_retry_scale", 1.1))),
        "--max-neighbour-retry-steps", str(int(calibration.get("max_neighbour_retry_steps", 100))),
        "--max-test-samples", str(int(calibration.get("max_test_samples", 0))),
        "--max-samples", str(int(calibration.get("max_samples", 4096))),
        "--chunk-size", str(int(calibration.get("chunk_size", 128))),
        "--output", str(output_path),
    ]
    if pp.get("time_interp") is not None:
        cmd.extend(["--time-interp", str(int(pp["time_interp"]))])
    if pp.get("window_norm") is not None:
        cmd.extend(["--window-norm", str(int(pp["window_norm"]))])
    shared_defaults = _as_dict(cfg.get("shared_defaults"))
    if bool(shared_defaults.get("balanced_action_sampling", False)):
        cmd.append("--balanced-action-sampling")
    if bool(shared_defaults.get("use_official_splits", False)):
        cmd.append("--use-official-splits")

    rc = _run(cmd, cwd=ROOT, env=_gpu_subprocess_env(cfg))
    if rc != 0:
        raise RuntimeError("automatic multimodal-threshold calibration failed")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    raw_threshold = float(payload["threshold"])
    threshold = raw_threshold * AUTO_MULTIMODAL_THRESHOLD_SCALE
    payload["raw_threshold"] = raw_threshold
    payload["threshold_scale"] = AUTO_MULTIMODAL_THRESHOLD_SCALE
    payload["effective_threshold"] = threshold
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[MULTIMODAL CALIBRATION] raw_threshold={raw_threshold:.6g} "
        f"scale={AUTO_MULTIMODAL_THRESHOLD_SCALE:.4g} "
        f"effective_threshold={threshold:.6g} "
        f"k={payload['k']} samples={payload['num_samples']} manifest={output_path}"
    )
    return threshold, output_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all diffusion_hands models sequentially.")
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    ap.add_argument(
        "--recover-existing-metrics",
        action="store_true",
        help="Ingest completed CARD backend artifacts for rows previously marked failed, then exit.",
    )
    ap.add_argument(
        "--force-rerun",
        action="store_true",
        help="Run configured models even when their successful metric rows already exist.",
    )
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = _load_yaml(config_path)
    cfg = _resolve_models_config(cfg, config_path)
    cfg["_shared_preprocessing"] = _resolve_shared_preprocessing(cfg)
    cfg.setdefault("humanmac_multimodal_threshold", 0.5)

    data_roots = _resolve_data_roots(cfg)
    datasets = _resolve_datasets(cfg)
    unknown = [d for d in datasets if d not in data_roots]
    if unknown:
        raise ValueError(f"Dataset(s) missing in data_roots: {unknown}")


    runtime = _resolve_runtime(cfg)
    out_root = Path(runtime["output_root"])
    aggregate_csv = Path(runtime["aggregate_csv"])
    summary_csv = Path(runtime["summary_csv"])
    seeds = _resolve_seeds(cfg)
    # Resume from any existing aggregate: successful rows are immutable and
    # skipped; failed rows are rerun and replaced in-place.
    invocation_rows: List[Dict[str, object]] = [dict(row) for row in _read_long_csv(aggregate_csv)]
    existing_rows = {
        _long_row_key(row): row
        for row in invocation_rows
    }
    if args.recover_existing_metrics:
        recovered = _recover_card_metrics_from_artifacts(
            aggregate_csv=aggregate_csv,
            summary_csv=summary_csv,
            seeds=seeds,
        )
        print(f"[RECOVER] recovered {recovered} row(s)")
        return
    had_failures = False

    out_root.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_result_artifacts(out_root)
    _write_summary_csv(summary_csv, invocation_rows, seeds)
    print(f"[PREPROCESSING] shared={cfg['_shared_preprocessing']}")

    runners = {
        "card": run_card,
        "star_hand": run_card,
        "fihard": run_card,
        "cit": run_card,
        "simlpe_dct": run_card,
        "belfusion": run_belfusion,
        "comusion": run_comusion,
        "dlow_cvae": run_dlow_cvae,
        "humanmac": run_humanmac,
        "skeletondiffusion": run_skeletondiffusion,
        "hybrid_cvae": run_hybrid_cvae,
        "gsps": run_gsps,
        "cacheflow": run_literature_motion_model,
        "stars": run_literature_motion_model,
    }
    for dataset in datasets:
        data_dir = Path(data_roots[dataset])
        for action_filter in _resolve_action_filters(cfg, dataset):
            cohort_id = f"{dataset}_{action_filter or 'all'}_{_now()}"

            threshold, _calibration_manifest = _resolve_multimodal_threshold(
                cfg,
                dataset=dataset,
                data_dir=data_dir,
                action_filter=action_filter,
                run_id=cohort_id,
                out_root=out_root,
            )
            dataset_cfg = deepcopy(cfg)
            dataset_cfg["humanmac_multimodal_threshold"] = threshold
            subset_path = _resolve_evaluation_subset_path(dataset_cfg, dataset, action_filter)
            if subset_path:
                dataset_cfg.setdefault("evaluation_subset", {})["indices_path"] = subset_path

            print(
                f"[DATASET] starting dataset={dataset} action_filter={action_filter!r} "
                f"data_dir={data_dir} humanmac_threshold={threshold:.6g}"
            )

            for model_name, model_cfg in dataset_cfg.get("models", {}).items():
                if not bool(model_cfg.get("enabled", False)):
                    continue
                supported_datasets = _as_dict(model_cfg.get("options")).get("supported_datasets")
                if supported_datasets is not None:
                    supported = {str(name).strip().lower() for name in supported_datasets}
                    if str(dataset).lower() not in supported:
                        print(
                            f"[{dataset}:{model_name}] SKIP unsupported dataset "
                            f"(supported: {sorted(supported)})",
                            flush=True,
                        )
                        continue
                base_model = str(model_cfg.get("base_model", model_name)).strip().lower()
                fn = runners.get(base_model)
                if fn is None:
                    raise ValueError(
                        f"Unsupported base_model '{base_model}' for models.{model_name}. "
                        f"Supported models: {sorted(runners)}"
                    )
                for seed in seeds:
                    row_key = (str(dataset), str(action_filter), int(seed), str(model_name))
                    previous = existing_rows.get(row_key)
                    implementation_version = str(model_cfg.get("implementation_version", "") or "")
                    revision_marker = f"[implementation:{implementation_version}]"
                    outdated_implementation = (
                        bool(implementation_version)
                        and revision_marker not in str(previous.get("notes", ""))
                    ) if previous is not None else False
                    missing_metrics = previous is not None and not _has_all_finite_metrics(previous)
                    if (not args.force_rerun and previous is not None and str(previous.get("status", "")).lower() == "ok"
                            and not outdated_implementation and not missing_metrics):
                        print(
                            f"[{dataset}:{model_name}:seed={seed}] SKIP existing successful run",
                            flush=True,
                        )
                        continue
                    if previous is not None and outdated_implementation:
                        print(
                            f"[{dataset}:{model_name}:seed={seed}] RERUN implementation revision "
                            f"{implementation_version}", flush=True,
                        )
                    elif previous is not None and missing_metrics:
                        print(
                            f"[{dataset}:{model_name}:seed={seed}] RERUN missing/non-finite metrics",
                            flush=True,
                        )

                    seed_cfg = deepcopy(dataset_cfg)
                    seed_cfg["seed"] = seed
                    seed_cfg["_reuse_run"] = _resolve_reuse_run(
                        seed_cfg, dataset, action_filter, seed, model_name
                    )
                    reuse_cfg = _as_dict(seed_cfg.get("checkpoint_reuse"))
                    if (reuse_cfg.get("enabled") and not bool(reuse_cfg.get("fallback_to_training", True))
                            and not seed_cfg["_reuse_run"]):
                        raise RuntimeError(
                            f"Checkpoint-only evaluation requested, but no compatible checkpoint was found for "
                            f"{dataset}/{action_filter or 'all'}/{model_name}/seed{seed}."
                        )
                    run_id = f"{cohort_id}_{model_name}_seed{seed}"
                    row: Dict[str, object] = {
                        "timestamp": _now(),
                        "dataset": dataset,
                        "action_filter": action_filter,
                        "seed": seed,
                        "model": model_name,
                        "status": "ok",
                        "notes": (
                            str(model_cfg.get("notes", "") or "")
                            + (f" {revision_marker}" if implementation_version else "")
                        ).strip(),
                        **{k: float("nan") for k in CANONICAL_METRIC_KEYS},
                    }
                    try:
                        metrics = fn(model_name, dataset, data_dir, action_filter, seed_cfg, run_id)
                        row.update(metrics)
                        _write_run_manifest(
                            model_name=model_name,
                            base_model=base_model,
                            run_id=run_id,
                            dataset=dataset,
                            action_filter=action_filter,
                            seed=seed,
                            cfg=seed_cfg,
                            metrics=metrics,
                        )
                        print(f"[{dataset}:{model_name}:seed={seed}] metrics={metrics}")
                    except Exception as exc:
                        had_failures = True
                        row["status"] = "failed"
                        prefix = str(model_cfg.get("notes", "") or "").strip()
                        row["notes"] = f"{prefix} | {exc}" if prefix else str(exc)
                        print(f"[{dataset}:{model_name}:seed={seed}] FAILED: {exc}")
                    invocation_rows = _record_result(aggregate_csv, summary_csv, row, seeds)
                    existing_rows[row_key] = row

    print(f"[DONE] aggregate metrics csv: {aggregate_csv}")
    print(f"[DONE] three-seed mean/std csv: {summary_csv}")
    if had_failures:
        raise SystemExit("One or more runs failed; incomplete cohorts were not averaged.")


if __name__ == "__main__":
    main()
