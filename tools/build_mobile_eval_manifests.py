#!/usr/bin/env python3
"""Select the most dynamic test windows before a forecasting-suite evaluation.

The generated JSON files contain indices into the *unfiltered* shared test
dataset.  `run_all_models.py` passes the appropriate file to each backend,
which wraps its test dataset before inference.  Consequently training remains
unchanged while checkpoint evaluation is cheap and exactly comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLINE_ROOT = ROOT / "vendor" / "splineeqnet"
DEFAULT_DATA_ROOTS = {
    "assembly": "/mnt/pve/Turing-Storage2/AssemblyHands/assembly101-download-scripts/data_our/",
    "h2o": "/mnt/pve/Turing-Storage2/h2o/",
    "bighands": "/mnt/pve/Turing-Storage2/BigHands/BigHand2.2M/data/",
    "fpha": "/mnt/pve/Turing-Storage2/FPHA/data/",
}
for location in (str(ROOT), str(SPLINE_ROOT)):
    if location not in sys.path:
        sys.path.insert(0, location)

from config import DatasetCfg  # noqa: E402
from data import build_datasets, get_dataset_metadata  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _actions(cfg: Dict[str, Any], dataset: str) -> Iterable[str]:
    if dataset != "assembly":
        return ("",)
    raw = cfg.get("action_filter", "")
    if isinstance(raw, list):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    value = str(raw or "").strip()
    return ("",) if value.lower() in {"", "all", "*"} else (value,)


def _motion_score(future: torch.Tensor) -> float:
    """Mean articulated displacement across the 30-frame forecast horizon.

    Root translation is explicitly removed so windows are selected for hand
    articulation, not camera/global motion.  The max-over-time displacement
    favors sequences that remain visibly active over the long video horizon.
    """
    xyz = future.detach().cpu().numpy()
    if xyz.ndim != 3 or xyz.shape[-1] < 3:
        raise ValueError(f"Expected future [T,J,3], got {xyz.shape}")
    xyz = xyz[..., :3]
    articulated = xyz - xyz[:, :1, :]
    displacement = np.linalg.norm(articulated - articulated[0:1], axis=-1)
    return float(displacement.max(axis=0).mean())


def _make_ds_cfg(cfg: Dict[str, Any], dataset: str, action: str, data_dir: str) -> DatasetCfg:
    pp = dict(cfg.get("preprocessing") or {})
    shared = dict(cfg.get("shared_defaults") or {})
    metadata = get_dataset_metadata(dataset)
    return DatasetCfg(
        data_dir=data_dir,
        action_filter=action,
        input_n=int(pp.get("input_n", 30)),
        output_n=int(pp.get("output_n", 30)),
        stride=int(pp.get("stride", 1)),
        time_interp=pp.get("time_interp"),
        window_norm=pp.get("window_norm"),
        batch_size=int(shared.get("batch_size", 128)),
        eval_batch_mult=int(pp.get("eval_batch_mult", 1)),
        seed=int(cfg.get("seed", 0)),
        data_seed=int(cfg.get("data_seed", cfg.get("seed", 0))),
        wrist_indices=tuple(int(i) for i in metadata.get("default_wrist_indices", ())),
        dataset=dataset,
        node_count=int(metadata["node_count"]),
        edge_index=tuple(metadata.get("edge_index", ())),
        adjacency=tuple(metadata.get("adjacency", ())),
        # Do not cap source files: these indices must be valid for the full
        # test split that every backend reconstructs at evaluation time.
        max_source_files=0,
        balanced_action_sampling=False,
        use_official_splits=bool(shared.get("use_official_splits", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Visual E2 experiment YAML with data_roots.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Rebuild manifests that already exist.")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    cfg = _load_yaml(args.config)
    roots = dict(DEFAULT_DATA_ROOTS)
    configured_roots = cfg.get("data_roots", {})
    if not isinstance(configured_roots, dict):
        parser.error("data_roots must be a mapping when supplied")
    roots.update({str(key).lower(): str(value) for key, value in configured_roots.items()})
    subset = dict(cfg.get("evaluation_subset") or {})
    output_dir = args.output_dir or Path(str(subset.get("manifest_root", "results/gripmodel_visual_e2/mobile_eval_manifests")))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [str(d).lower() for d in cfg.get("datasets", ["assembly"])]
    records: List[Dict[str, Any]] = []
    for dataset in datasets:
        if dataset not in roots:
            raise ValueError(f"Missing data_roots.{dataset}")
        for action in _actions(cfg, dataset):
            action_tag = action or "all"
            path = output_dir / f"{dataset}__{action_tag}.json"
            if path.is_file() and not args.force:
                selected = json.loads(path.read_text(encoding="utf-8"))
                records.append({
                    "dataset": dataset, "action_filter": action, "path": path.name,
                    "selected": len(selected), "reused": True,
                })
                print(f"[{dataset}/{action_tag}] reused {len(selected)} windows -> {path}")
                continue
            ds_cfg = _make_ds_cfg(cfg, dataset, action, str(roots[dataset]))
            _, _, test_dataset = build_datasets(ds_cfg)
            scored = [(index, _motion_score(test_dataset[index][1])) for index in range(len(test_dataset))]
            scored.sort(key=lambda item: (-item[1], item[0]))
            selected = [index for index, _ in scored[: min(args.top_k, len(scored))]]
            path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
            records.append({
                "dataset": dataset, "action_filter": action, "path": path.name,
                "num_test_windows": len(test_dataset), "selected": len(selected),
                "score_min": scored[len(selected) - 1][1], "score_max": scored[0][1],
            })
            print(f"[{dataset}/{action_tag}] selected {len(selected)}/{len(test_dataset)} windows -> {path}")

    manifest = {"top_k": args.top_k, "config": str(args.config), "cohorts": records}
    encoded = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
