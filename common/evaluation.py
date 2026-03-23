from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Mapping

from common.metrics import (
    compute_all_metrics_single,
    humanmac_metrics,
    humanmac_metrics_prefixed,
    splineeqnet_diffusion_batch_eval,
)


CANONICAL_METRIC_KEYS: List[str] = ["MPJPE", "MPJPE_norm", "APD", "ADE", "FDE", "MMADE", "MMFDE"]
CANONICAL_LONG_HEADER: List[str] = [
    "timestamp",
    "dataset",
    "action_filter",
    "model",
    "status",
    *CANONICAL_METRIC_KEYS,
    "notes",
]

METRIC_ALIASES = {
    "MPJPE": ["MPJPE", "test_mpjpe_best", "validation_mpjpe_best", "mpjpe"],
    "MPJPE_norm": ["MPJPE_norm", "test_mpjpe_norm_best", "validation_mpjpe_norm_best", "mpjpe_norm", "MPJPE_NORM"],
    "APD": ["APD", "test_humanmac_apd_best", "validation_humanmac_apd_best"],
    "ADE": ["ADE", "test_humanmac_ade_best", "validation_humanmac_ade_best"],
    "FDE": ["FDE", "test_humanmac_fde_best", "validation_humanmac_fde_best"],
    "MMADE": ["MMADE", "test_humanmac_mmade_best", "validation_humanmac_mmade_best"],
    "MMFDE": ["MMFDE", "test_humanmac_mmfde_best", "validation_humanmac_mmfde_best"],
}


def read_one_row_csv(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV has no rows: {path}")
    row = rows[0]
    out: Dict[str, float] = {}
    for k, v in row.items():
        if v is None or v == "":
            continue
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def normalize_metrics_dict(src: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for target, candidates in METRIC_ALIASES.items():
        for cand in candidates:
            if cand in src:
                out[target] = float(src[cand])
                break
    return out
