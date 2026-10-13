import unittest
from unittest import mock
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/comusion"))

from utils import metrics as comusion_metrics
from vendor.skeletondiffusion.src.data.loaders.assembly import (
    AssemblyDataset as SkeletonAssemblyDataset,
)


class CoMusionMetricCompatibilityTests(unittest.TestCase):
    def test_sequence_ids_are_forwarded_to_shared_metrics(self):
        sequence_ids = torch.tensor([3, 7])
        with mock.patch.object(
            comusion_metrics,
            "_common_splineeqnet_diffusion_batch_eval",
            return_value={"humanmac": {}},
        ) as shared_eval:
            result = comusion_metrics.SPLINEEQNET_DIFFUSION_BATCH_EVAL(
                torch.zeros(1, 2, 1, 3),
                torch.zeros(2, 1, 3),
                torch.zeros(2, 1, 1, 3),
                torch.ones(2),
                threshold=0.25,
                sequence_ids=sequence_ids,
            )

        self.assertEqual(result, {"humanmac": {}})
        self.assertIs(shared_eval.call_args.kwargs["sequence_ids"], sequence_ids)


class SkeletonDiffusionDatasetCompatibilityTests(unittest.TestCase):
    def test_adapter_accepts_shared_dataset_aux_metadata(self):
        adapter = object.__new__(SkeletonAssemblyDataset)
        adapter.dataset = [
            (
                torch.zeros(2, 21, 7),
                torch.ones(1, 21, 7),
                torch.tensor(2.5),
                {"sequence_id": torch.tensor(4)},
            )
        ]
        adapter.if_noisy_obs = False
        adapter.in_eval = True
        adapter.obs_length = 2
        adapter.skeleton = mock.Mock()
        adapter.skeleton.tranform_to_input_space.side_effect = lambda value: value
        adapter.skeleton.adj_matrix = torch.eye(22)

        obs, pred, extra = adapter[0]

        self.assertEqual(tuple(obs.shape), (2, 22, 3))
        self.assertEqual(tuple(pred.shape), (1, 22, 3))
        self.assertEqual(extra["norm_factor"], 2.5)


if __name__ == "__main__":
    unittest.main()
