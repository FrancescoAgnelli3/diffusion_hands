"""Geometry, optimization, sampling, and protocol tests for grip-phase CIT."""
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/splineeqnet"))
sys.path.insert(0, str(ROOT / "tools"))

from models.causal_innovation_transport import (
    ASSEMBLY_PROJECTION_EDGES, CausalInnovationConfig, CausalInnovationForecaster,
)
from models.hand_specific_transports import from_canonical, grip_phase, to_canonical
from common.dataset_graphs import resolve_local_hand_graph_metadata
from run_generative_research import resolve_plan


META = {"wrist_index": 5, "edges": list(ASSEMBLY_PROJECTION_EDGES)}


def hand_sequence(batch, frames):
    result = torch.zeros(batch, frames, 21, 3)
    for b in range(batch):
        for t in range(frames):
            positions = {5: torch.zeros(3)}
            for e, (parent, child) in enumerate(ASSEMBLY_PROJECTION_EDGES):
                angle = 0.03*t+0.02*e
                direction = torch.tensor([1.0, angle, 0.1*((e % 2)*2-1)])
                direction = direction/direction.norm()
                positions[child] = positions[parent]+(0.4+0.02*e+0.01*b)*direction
            result[b, t] = torch.stack([positions[i] for i in range(21)])
    return result


class CITNextDirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(211)

    def model(self, metadata=META, **options):
        args = dict(input_length=4, pred_length=5, num_nodes=21, hidden_dim=16,
                    num_layers=1, k_low=4, denoiser_dim=16, denoiser_depth=1,
                    denoiser_heads=4, ddim_steps=3, cond_use_history=True,
                    cond_use_coarse=True, freeze_coarse=True, propagation="linear",
                    terminal_kinematic_projection=True)
        args.update(options)
        return CausalInnovationForecaster(CausalInnovationConfig(**args), metadata)

    def loader(self):
        history = hand_sequence(5, 4)
        future = hand_sequence(5, 5)
        prefix = torch.zeros(5, 4, 21, 4)
        future_prefix = torch.zeros(5, 5, 21, 4)
        return DataLoader(TensorDataset(torch.cat((prefix, history), -1),
                                       torch.cat((future_prefix, future), -1)),
                          batch_size=2, shuffle=True, drop_last=True,
                          generator=torch.Generator().manual_seed(17))

    def test_grip_phase_and_resampling_contracts(self):
        history, future = hand_sequence(2, 4), hand_sequence(2, 5)
        phase, logits = grip_phase(history, future)
        soft_phase, soft_logits = grip_phase(history, future, strength=0.5)
        self.assertTrue(torch.all(phase[:, 1:] > phase[:, :-1]))
        torch.testing.assert_close(phase[:, -1], torch.ones(2))
        torch.testing.assert_close(logits.mean(1), torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(soft_logits, logits)
        uniform = torch.arange(1, 6, dtype=phase.dtype)[None]/5
        torch.testing.assert_close(soft_phase, 0.5*uniform+0.5*phase)

        # A curve linear in its own phase is invariant to canonicalize/decanonicalize.
        initial = history[:, -1]
        direction = torch.randn_like(initial)
        physical = initial[:, None]+phase[:, :, None, None]*direction[:, None]
        canonical = to_canonical(physical, phase, initial)
        restored = from_canonical(canonical, phase, initial)
        torch.testing.assert_close(restored, physical, atol=2e-6, rtol=2e-6)

    def test_clock_conditioning_preserves_common_field_initialization(self):
        torch.manual_seed(73)
        independent = self.model(grip_phase_transport=True, grip_clock_hidden_dim=16)
        torch.manual_seed(73)
        coupled = self.model(grip_phase_transport=True, grip_clock_hidden_dim=16,
                             grip_clock_conditioning=True)
        coupled_parameters = dict(coupled.diffusion.denoiser.named_parameters())
        for name, parameter in independent.diffusion.denoiser.named_parameters():
            if name.startswith("latent_") or name.startswith("proj_mamp"):
                continue
            torch.testing.assert_close(parameter, coupled_parameters[name])

    def test_each_grip_variant_calibrates_trains_samples_and_preserves_lengths(self):
        variants = (
            {"grip_phase_transport": True, "grip_clock_hidden_dim": 16},
            {"grip_phase_transport": True, "grip_clock_hidden_dim": 16,
             "grip_phase_strength": 0.5, "grip_clock_conditioning": True},
            {"grip_phase_transport": True, "grip_clock_hidden_dim": 16,
             "grip_clock_conditioning": True, "grip_joint_shared_noise": True,
             "grip_joint_noise_strength": 0.35},
        )
        loader = self.loader()
        raw_history, raw_future = loader.dataset.tensors
        history, future = raw_history[:2, ..., 4:], raw_future[:2, ..., 4:]
        for options in variants:
            with self.subTest(options=options):
                model = self.model(**options)
                report = model.calibrate_reference(loader)
                self.assertEqual(report["passes"], 2)
                self.assertEqual(report["windows_per_pass"], 5)
                loss, _, metrics = model.diffusion_loss(history, future, return_metrics=True)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                gradients = [p.grad for p in model.diffusion.parameters() if p.grad is not None]
                self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
                if options.get("grip_phase_transport"):
                    self.assertIn("grip_clock_loss", metrics)
                if options.get("grip_clock_conditioning"):
                    self.assertIsNotNone(model.diffusion.denoiser.latent_gate.grad)
                model.eval()
                # The shared benchmark evaluates inside inference_mode.
                with torch.inference_mode():
                    prediction = model.predict(history, seed=23)
                self.assertTrue(torch.isfinite(prediction).all())
                predicted_lengths = model.diffusion.observed_bone_lengths(prediction)
                observed_lengths = model.diffusion.observed_bone_lengths(history)
                torch.testing.assert_close(predicted_lengths, observed_lengths,
                                           atol=2e-5, rtol=2e-5)

    def test_terminal_projection_uses_each_dataset_hand_graph(self):
        for dataset in ("h2o", "bighands", "fpha"):
            with self.subTest(dataset=dataset):
                metadata = resolve_local_hand_graph_metadata(dataset)
                model = self.model(
                    metadata={
                        "wrist_index": metadata["wrist_index"],
                        "edges": metadata["links"],
                    },
                    grip_phase_transport=True,
                    grip_clock_hidden_dim=16,
                )
                transport = model.diffusion
                self.assertEqual(len(transport.projection_edges), 20)

                history = torch.randn(2, 4, 21, 3)
                history[:, :, transport.wrist_index] = 0
                pose = torch.randn(2, 5, 21, 3)
                projected = transport.terminal_projection(pose, history)
                expected_lengths = transport.observed_bone_lengths(history)
                projected_vectors = (
                    projected.index_select(-2, transport.projection_children)
                    - projected.index_select(-2, transport.projection_parents)
                )
                actual_lengths = projected_vectors.norm(dim=-1)

                torch.testing.assert_close(
                    actual_lengths,
                    expected_lengths[:, None].expand_as(actual_lengths),
                    atol=2e-5,
                    rtol=2e-5,
                )
                torch.testing.assert_close(
                    projected[:, :, transport.wrist_index],
                    torch.zeros_like(projected[:, :, transport.wrist_index]),
                )

    def test_invalid_combinations_and_full_protocol(self):
        for options in (
            {"grip_phase_transport": True, "grip_clock_loss_weight": 0},
            {"grip_clock_conditioning": True},
            {"grip_phase_transport": True, "grip_phase_strength": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model(**options)
        proposal = resolve_plan(ROOT / "configs/ablations/cit_next_directions_research.yaml")
        next_phase = resolve_plan(ROOT / "configs/ablations/cit_grip_phase_next_research.yaml")
        joint = resolve_plan(ROOT / "configs/ablations/cit_joint_clock_research.yaml")
        self.assertEqual(proposal["seeds"], [0, 1, 2])
        self.assertEqual(proposal["gpu_index"], 1)
        self.assertEqual(set(proposal["models"]), {"cit_grip_phase"})
        self.assertEqual(next_phase["seeds"], [0, 1, 2])
        self.assertEqual(next_phase["gpu_index"], 0)
        self.assertEqual(set(next_phase["models"]), {
            "cit_grip_phase_soft", "cit_grip_phase_coupled",
            "cit_grip_phase_coupled_soft",
        })
        self.assertEqual(joint["gpu_index"], 0)
        self.assertEqual(set(joint["models"]), {
            "cit_linear", "cit_grip_phase", "cit_grip_joint",
        })
        self.assertTrue(joint["models"]["cit_grip_joint"]["defaults"]["proposal_options"]
                        ["grip_joint_shared_noise"])


if __name__ == "__main__":
    unittest.main()
