#!/usr/bin/env python3
"""Threshold sensitivity analysis from fixed DLow/GRIP full-test prediction caches.

The script never samples a model.  It refuses comparisons with different targets,
contexts, or sequence IDs, preventing a threshold crossover from mixing cohorts.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.metrics import distributional_motion_metrics


def _digest(value):
    if value is None:
        return None
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def load_cache(path: Path) -> Dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    required = {"pred_candidates", "gt_future", "conditioning_context"}
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    pred, gt, context = (cache["pred_candidates"].float(), cache["gt_future"].float(),
                         cache["conditioning_context"].float())
    # A reference GRIP cache retains global 21-joint coordinates, whereas
    # DLow is trained on wrist-relative non-root joints.  Canonicalize both
    # representations before validating or calculating metrics.
    if pred.shape[-2:] == (21, 3):
        pred = (pred[..., 1:, :] - pred[..., :1, :]).contiguous()
    if gt.shape[-2:] == (21, 3):
        gt = (gt[..., 1:, :] - gt[..., :1, :]).contiguous()
    if context.shape[-2:] == (21, 3):
        context = (context[..., 1:, :] - context[..., :1, :]).contiguous()
    sequence_ids = cache.get("sequence_ids")
    if sequence_ids is not None:
        sequence_ids = sequence_ids.long().view(-1)
    if pred.ndim != 5 or gt.ndim != 4 or context.ndim != 4:
        raise ValueError(f"{path}: expected [K,N,T,J,3], [N,T,J,3], [N,H,J,3]")
    if pred.shape[1] != gt.shape[0] or gt.shape[0] != context.shape[0]:
        raise ValueError(f"{path}: inconsistent sample dimension")
    return {"path": str(path), "pred": pred, "gt": gt, "context": context,
            "sequence_ids": sequence_ids, "metadata": cache.get("metadata", {})}


def check_same_cohort(dlow: Dict[str, Any], grip: Dict[str, Any]) -> Dict[str, Any]:
    fingerprints = {}
    for key in ("gt", "context", "sequence_ids"):
        left, right = dlow[key], grip[key]
        if (left is None) != (right is None):
            raise ValueError(f"Cohort mismatch: {key} exists for only one model")
        if left is not None and (tuple(left.shape) != tuple(right.shape) or not torch.equal(left, right)):
            raise ValueError(f"Cohort mismatch: {key} differs; do not compare these caches")
        fingerprints[key] = _digest(left)
    return fingerprints


def parse_thresholds(spec: str):
    values = sorted({float(value.strip()) for value in spec.split(",") if value.strip()})
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("thresholds must be a nonempty list of finite values >= 0")
    return values


class ThresholdSweepEvaluator:
    """Reuse work which is invariant across a threshold grid.

    ``humanmac_metrics`` is intentionally convenient for one-off evaluation, but
    each invocation builds the same O(N^2) context-distance matrix and
    recalculates APD/ADE/FDE/CMD/FID.  A sensitivity sweep only changes the
    multimodal-neighbour mask, so doing that work for every grid point is
    needlessly expensive.
    """

    def __init__(
        self,
        cache: Dict[str, Any],
        pairwise_context_distances: torch.Tensor | None = None,
    ):
        self.pred = cache["pred"].reshape(
            cache["pred"].shape[0], cache["pred"].shape[1], cache["pred"].shape[2], -1
        ).cpu()
        self.gt = cache["gt"].reshape(cache["gt"].shape[0], cache["gt"].shape[1], -1).cpu()
        context = cache["context"].reshape(cache["context"].shape[0], -1).cpu()
        self.sequence_ids = cache["sequence_ids"]
        if self.sequence_ids is not None:
            self.sequence_ids = self.sequence_ids.reshape(-1).cpu()

        self.fixed_metrics = self._fixed_metrics(cache["pred"], cache["gt"])
        self.pairwise_context_distances = (
            pairwise_context_distances
            if pairwise_context_distances is not None
            else torch.cdist(context, context)
        )

    @staticmethod
    def _fixed_metrics(pred_candidates: torch.Tensor, gt_future: torch.Tensor) -> Dict[str, float]:
        """Calculate the HumanMAC metrics that do not use neighbour groups."""
        pred = pred_candidates.reshape(
            pred_candidates.shape[0], pred_candidates.shape[1], pred_candidates.shape[2], -1
        ).cpu()
        gt = gt_future.reshape(gt_future.shape[0], gt_future.shape[1], -1).cpu()
        apd_total = ade_total = fde_total = 0.0
        for sample_idx in range(gt.shape[0]):
            pred_sample = pred[:, sample_idx]
            if pred_sample.shape[0] > 1:
                apd_total += float(torch.pdist(pred_sample.reshape(pred_sample.shape[0], -1)).mean().item())
            target_dist = torch.linalg.norm(pred_sample - gt[sample_idx : sample_idx + 1], dim=2)
            ade_total += float(target_dist.mean(dim=1).min(dim=0).values.item())
            fde_total += float(target_dist[:, -1].min(dim=0).values.item())
        n = max(1, int(gt.shape[0]))
        metrics = {"APD": apd_total / n, "ADE": ade_total / n, "FDE": fde_total / n}
        metrics.update(distributional_motion_metrics(pred_candidates, gt_future))
        return metrics

    def metrics_for_thresholds(self, thresholds: list[float]) -> Dict[float, Dict[str, float]]:
        """Compute every threshold from one exact pairwise-error traversal.

        For a source window i and candidate target j, MMADE/MMFDE need only
        the best candidate error.  Those errors do not depend on the radius;
        the radius merely selects which j values enter each row average.
        """
        thresholds = list(thresholds)
        totals = {threshold: [0.0, 0.0, 0] for threshold in thresholds}
        n = self.gt.shape[0]
        source_chunk = 64
        target_chunk = 512
        time_steps = self.gt.shape[1]

        for source_start in range(0, n, source_chunk):
            source_end = min(source_start + source_chunk, n)
            source_size = source_end - source_start
            best_ade = torch.full((source_size, n), float("inf"), dtype=self.pred.dtype)
            best_fde = torch.full_like(best_ade, float("inf"))
            for target_start in range(0, n, target_chunk):
                target_end = min(target_start + target_chunk, n)
                target_size = target_end - target_start
                ade_by_candidate = torch.zeros(
                    (self.pred.shape[0], source_size, target_size), dtype=self.pred.dtype
                )
                for time_idx in range(time_steps):
                    # cdist is batched over candidates, avoiding a large
                    # [K, source, target, time, coordinate] materialization.
                    candidates = self.pred[:, source_start:source_end, time_idx]
                    targets = self.gt[target_start:target_end, time_idx].unsqueeze(0).expand(
                        self.pred.shape[0], -1, -1
                    )
                    ade_by_candidate += torch.cdist(candidates, targets)
                best_ade[:, target_start:target_end] = (
                    ade_by_candidate.div(time_steps).min(dim=0).values
                )
                candidates = self.pred[:, source_start:source_end, -1]
                targets = self.gt[target_start:target_end, -1].unsqueeze(0).expand(
                    self.pred.shape[0], -1, -1
                )
                best_fde[:, target_start:target_end] = torch.cdist(candidates, targets).min(dim=0).values

            context_distances = self.pairwise_context_distances[source_start:source_end]
            compatible = torch.ones_like(context_distances, dtype=torch.bool)
            row_indices = torch.arange(source_start, source_end)
            compatible[torch.arange(source_size), row_indices] = False
            if self.sequence_ids is not None:
                compatible &= self.sequence_ids[source_start:source_end, None] != self.sequence_ids[None, :]
            for threshold in thresholds:
                mask = compatible & (context_distances < threshold)
                counts = mask.sum(dim=1)
                active = counts > 0
                if not active.any():
                    continue
                counts_f = counts[active].to(dtype=self.pred.dtype)
                ade_rows = (best_ade[active] * mask[active]).sum(dim=1) / counts_f
                fde_rows = (best_fde[active] * mask[active]).sum(dim=1) / counts_f
                totals[threshold][0] += float(ade_rows.sum().item())
                totals[threshold][1] += float(fde_rows.sum().item())
                totals[threshold][2] += int(active.sum().item())

        results = {}
        for threshold, (mmade_total, mmfde_total, sample_count) in totals.items():
            metrics = dict(self.fixed_metrics)
            metrics["MMADE"] = mmade_total / sample_count if sample_count else float("nan")
            metrics["MMFDE"] = mmfde_total / sample_count if sample_count else float("nan")
            results[threshold] = metrics
        return results


def evaluate(name, cache, threshold, metrics: Dict[str, float]):
    return {"model": name, "threshold": threshold, "samples": int(cache["gt"].shape[0]),
            "candidates": int(cache["pred"].shape[0]), **{key: float(value) for key, value in metrics.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlow-cache", type=Path, required=True)
    parser.add_argument("--grip-cache", type=Path, required=True)
    parser.add_argument("--thresholds", required=True, help="Example: 0,0.5,1,2,4,6,8,10")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dlow, grip = load_cache(args.dlow_cache), load_cache(args.grip_cache)
    fingerprints = check_same_cohort(dlow, grip)
    thresholds = parse_thresholds(args.thresholds)
    # The cohort validation above establishes that both caches use identical
    # targets/contexts/sequence IDs.  Each model still needs its own fixed
    # prediction metrics, while its threshold grid reuses one distance matrix.
    shared_context = dlow["context"].reshape(dlow["context"].shape[0], -1).cpu()
    shared_distances = torch.cdist(shared_context, shared_context)
    dlow_evaluator = ThresholdSweepEvaluator(dlow, shared_distances)
    grip_evaluator = ThresholdSweepEvaluator(grip, shared_distances)
    dlow_metrics = dlow_evaluator.metrics_for_thresholds(thresholds)
    grip_metrics = grip_evaluator.metrics_for_thresholds(thresholds)
    rows = [evaluate("dlow", dlow, threshold, dlow_metrics[threshold]) for threshold in thresholds]
    rows += [evaluate("grip", grip, threshold, grip_metrics[threshold]) for threshold in thresholds]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "threshold_sensitivity_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    indexed = {(row["model"], row["threshold"]): row for row in rows}
    summary = []
    for threshold in thresholds:
        dlow_row, grip_row = indexed[("dlow", threshold)], indexed[("grip", threshold)]
        summary.append({"threshold": threshold,
                        "grip_better_mmade": grip_row["MMADE"] < dlow_row["MMADE"],
                        "grip_better_mmfde": grip_row["MMFDE"] < dlow_row["MMFDE"],
                        "delta_mmade_grip_minus_dlow": grip_row["MMADE"] - dlow_row["MMADE"],
                        "delta_mmfde_grip_minus_dlow": grip_row["MMFDE"] - dlow_row["MMFDE"]})
    with (args.output_dir / "threshold_sensitivity_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    manifest = {"thresholds": thresholds, "dlow_cache": dlow["path"], "grip_cache": grip["path"],
                "cohort_fingerprints": fingerprints, "dlow_metadata": dlow["metadata"],
                "grip_metadata": grip["metadata"], "comparison_validated": True}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
