import math
import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/splineeqnet"))

from common.metrics import humanmac_metrics
from common.multimodal_calibration import (
    calibrate_multimodal_threshold,
    cross_sequence_neighbor_coverage,
)
from datasets import AssemblyDataset

_SWEEP_SPEC = importlib.util.spec_from_file_location(
    "sweep_multimodal_threshold", ROOT / "tools" / "sweep_multimodal_threshold.py"
)
assert _SWEEP_SPEC is not None and _SWEEP_SPEC.loader is not None
_SWEEP_MODULE = importlib.util.module_from_spec(_SWEEP_SPEC)
_SWEEP_SPEC.loader.exec_module(_SWEEP_MODULE)
ThresholdSweepEvaluator = _SWEEP_MODULE.ThresholdSweepEvaluator


class MultimodalNeighborTests(unittest.TestCase):
    def test_threshold_calibration_defaults_to_first_cross_sequence_neighbour(self):
        context = torch.tensor([0.0, 0.1, 2.0, 4.0, 8.0]).reshape(5, 1, 1, 1)
        sequence_ids = torch.tensor([0, 0, 1, 2, 3])

        threshold, stats = calibrate_multimodal_threshold(
            context,
            sequence_ids,
            chunk_size=2,
            device=torch.device("cpu"),
        )

        self.assertAlmostEqual(threshold, 2.0, places=5)
        self.assertEqual(stats["k"], 1)

    def test_threshold_calibration_uses_cross_sequence_kth_neighbours(self):
        context = torch.tensor([0.0, 0.1, 2.0, 4.0, 8.0]).reshape(5, 1, 1, 1)
        sequence_ids = torch.tensor([0, 0, 1, 2, 3])

        threshold, stats = calibrate_multimodal_threshold(
            context,
            sequence_ids,
            k=2,
            chunk_size=2,
            device=torch.device("cpu"),
        )

        # Cross-sequence second-neighbour distances are [4, 3.9, 2, 3.9, 6].
        self.assertAlmostEqual(threshold, 3.9, places=5)
        self.assertEqual(stats["num_samples"], 5)
        self.assertEqual(stats["num_sequences"], 4)

    def test_threshold_calibration_rejects_insufficient_external_neighbours(self):
        with self.assertRaisesRegex(ValueError, "fewer than k=2"):
            calibrate_multimodal_threshold(
                torch.zeros(3, 1),
                torch.tensor([0, 0, 1]),
                k=2,
                device=torch.device("cpu"),
            )

    def test_coverage_excludes_same_sequence_windows(self):
        context = torch.tensor([0.0, 0.0, 5.0]).reshape(3, 1, 1, 1)
        coverage = cross_sequence_neighbor_coverage(
            context, torch.tensor([0, 0, 1]), threshold=1.0, device=torch.device("cpu")
        )

        self.assertEqual(coverage["covered_samples"], 0)
        self.assertEqual(coverage["coverage"], 0.0)

    def test_query_itself_is_not_a_multimodal_neighbor(self):
        context = torch.zeros(2, 2, 1, 3)
        target = torch.zeros(2, 2, 1, 3)
        target[1, ..., 0] = 10
        predictions = target.unsqueeze(0)

        metrics = humanmac_metrics(predictions, target, context, threshold=1)

        self.assertEqual(metrics["ADE"], 0)
        self.assertEqual(metrics["FDE"], 0)
        self.assertAlmostEqual(metrics["MMADE"], 10)
        self.assertAlmostEqual(metrics["MMFDE"], 10)

    def test_stride_windows_from_same_sequence_are_excluded(self):
        context = torch.zeros(3, 2, 1, 3)
        target = torch.zeros(3, 2, 1, 3)
        target[1, ..., 0] = 100
        target[2, ..., 0] = 10
        predictions = target.unsqueeze(0)

        metrics = humanmac_metrics(
            predictions,
            target,
            context,
            threshold=1,
            sequence_ids=torch.tensor([4, 4, 9]),
        )

        self.assertAlmostEqual(metrics["MMADE"], 50)
        self.assertAlmostEqual(metrics["MMFDE"], 50)

    def test_metric_retries_radius_when_evaluation_subset_has_no_neighbors(self):
        context = torch.tensor([0.0, 10.0]).reshape(2, 1, 1, 1)
        target = torch.zeros(2, 1, 1, 3)
        predictions = target.unsqueeze(0)

        metrics = humanmac_metrics(predictions, target, context, threshold=1)

        self.assertEqual(metrics["MMADE"], 0)
        self.assertEqual(metrics["MMFDE"], 0)

    def test_threshold_sweep_reuses_work_without_changing_metrics(self):
        torch.manual_seed(7)
        cache = {
            "pred": torch.randn(3, 5, 2, 2, 3),
            "gt": torch.randn(5, 2, 2, 3),
            "context": torch.tensor([0.0, 0.2, 1.0, 1.1, 4.0]).reshape(5, 1, 1, 1),
            "sequence_ids": torch.tensor([0, 0, 1, 2, 3]),
        }
        evaluator = ThresholdSweepEvaluator(cache)
        thresholds = (0.0, 0.5, 2.0, 10.0)
        actual_by_threshold = evaluator.metrics_for_thresholds(thresholds)
        for threshold in thresholds:
            expected = humanmac_metrics(
                cache["pred"], cache["gt"], cache["context"], threshold=threshold,
                sequence_ids=cache["sequence_ids"],
            )
            actual = actual_by_threshold[threshold]
            for key, expected_value in expected.items():
                if math.isnan(expected_value):
                    self.assertTrue(math.isnan(actual[key]), key)
                else:
                    self.assertAlmostEqual(actual[key], expected_value, places=6, msg=key)

    def test_sliding_window_dataset_exposes_source_sequence_identity(self):
        sequences = [
            np.zeros((8, 2, 7), dtype=np.float32),
            np.ones((8, 2, 7), dtype=np.float32),
        ]
        dataset = AssemblyDataset(sequences, input_n=2, output_n=2, stride=2)

        ids = [int(dataset[i][3]["sequence_id"]) for i in range(len(dataset))]

        self.assertEqual(ids, [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
