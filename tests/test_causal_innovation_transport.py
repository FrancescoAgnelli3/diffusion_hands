"""Numerical and integration contracts for the CIT proposal (stdlib unittest)."""
import copy
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/splineeqnet"))
sys.path.insert(0, str(ROOT / "tools"))
from models.causal_innovation_transport import CausalInnovationConfig, CausalInnovationForecaster
from run_generative_research import resolve_plan


class CausalInnovationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)

    def model(self, **options):
        cfg = CausalInnovationConfig(input_length=4, pred_length=5, num_nodes=5,
                                    hidden_dim=16, num_layers=1, k_low=4, denoiser_dim=16,
                                    denoiser_depth=1, denoiser_heads=4, ddim_steps=2,
                                    cond_use_history=True, freeze_coarse=True, **options)
        return CausalInnovationForecaster(cfg, {"wrist_index": 2})

    def loader(self, count=5):
        h, y = torch.randn(count, 4, 5, 7), torch.randn(count, 5, 5, 7)
        return DataLoader(TensorDataset(h, y), batch_size=2, shuffle=True, drop_last=True,
                          generator=torch.Generator().manual_seed(17))

    def test_chart_inverse_jvp_and_path_velocity(self):
        transport = self.model().diffusion.double()
        transport.persistence.fill_(0.85)
        transport.innovation_scale.uniform_(0.1, 0.5)
        u, du = torch.randn(2, 5, 4, 3, dtype=torch.float64), torch.randn(2, 5, 4, 3, dtype=torch.float64)
        x, velocity = transport.decode_with_jvp(u, du)
        self.assertTrue(torch.allclose(transport.encode(x), u, atol=1e-12))
        self.assertTrue(torch.allclose(transport.inverse_jvp(x, velocity), du, atol=1e-12))
        _, autograd_velocity = torch.autograd.functional.jvp(transport.decode, u, du)
        self.assertTrue(torch.allclose(velocity, autograd_velocity, atol=1e-12))
        eps = 1e-5
        finite_difference = (transport.decode(u + eps*du) - transport.decode(u - eps*du)) / (2*eps)
        self.assertTrue(torch.allclose(velocity, finite_difference, atol=1e-9))

    def test_calibration_includes_partial_batch_preserves_weights_and_rng(self):
        model, loader = self.model(), self.loader()
        original = copy.deepcopy(model.coarse.state_dict())
        rng, sampler_rng = torch.random.get_rng_state(), loader.generator.get_state()
        report = model.calibrate_reference(loader)
        self.assertEqual(report["source_split"], "train")
        self.assertEqual(report["passes"], 2)
        self.assertEqual(report["windows_per_pass"], 5)
        self.assertEqual(int(model.diffusion.calibration_windows), 5)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(sampler_rng, loader.generator.get_state()))
        self.assertTrue(all(torch.equal(value, model.coarse.state_dict()[key]) for key, value in original.items()))
        h, y = (t[..., 4:] for t in loader.dataset.tensors)
        with torch.no_grad():
            residual = model.diffusion.select(y - model.coarse(h)).double()
            radius = residual.square().sum(-1).mean((0, 1)).sqrt().unsqueeze(-1)
            self.assertTrue(torch.allclose(model.diffusion.radius.double(), radius, atol=1e-6))
        with self.assertRaisesRegex(RuntimeError, "already calibrated"):
            model.calibrate_reference(loader)

    def test_uncalibrated_empty_and_nonfinite_data_rejected(self):
        model = self.model()
        h, y = torch.randn(2, 4, 5, 3), torch.randn(2, 5, 5, 3)
        with self.assertRaisesRegex(RuntimeError, "uncalibrated"):
            model.diffusion_loss(h, y)
        with self.assertRaisesRegex(ValueError, "Empty training"):
            model.calibrate_reference(DataLoader(TensorDataset(torch.empty(0,4,5,7), torch.empty(0,5,5,7)), batch_size=2))
        loader = self.loader()
        loader.dataset.tensors[1][0, 0, 0, 4] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            model.calibrate_reference(loader)
        self.assertTrue(model.training)
        self.assertFalse(bool(model.diffusion.calibrated))

    def test_zero_residual_calibration_keeps_chart_invertible(self):
        model, loader = self.model(), self.loader()
        h, y = loader.dataset.tensors
        with torch.no_grad():
            y[..., 4:] = model.coarse(h[..., 4:])
        model.calibrate_reference(loader)
        transport = model.diffusion
        self.assertTrue(torch.all(transport.radius >= model.cfg.scale_floor))
        self.assertTrue(torch.all(transport.innovation_scale >= model.cfg.scale_floor))
        u = torch.randn(2, 5, 4, 3)
        self.assertTrue(torch.allclose(transport.encode(transport.decode(u)), u, atol=1e-6))

    def test_loss_gradients_checkpoint_seed_and_sampling_budget(self):
        h, y = torch.randn(2,4,5,3), torch.randn(2,5,5,3)
        for mode in ("linear",):
            with self.subTest(mode=mode):
                model = self.model(propagation=mode)
                model.calibrate_reference(self.loader())
                loss, _, metrics = model.diffusion_loss(h, y, return_metrics=True)
                loss.backward()
                gradients = [p.grad for p in model.diffusion.parameters() if p.grad is not None]
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
                self.assertTrue(all(p.grad is None for p in model.coarse.parameters()))
                self.assertIn("total_loss", metrics)
                twin = self.model(propagation=mode)
                twin.load_state_dict(copy.deepcopy(model.state_dict()))
                model.eval(); twin.eval()
                calls = []
                handle = model.diffusion.denoiser.register_forward_hook(lambda *args: calls.append(1))
                rng = torch.random.get_rng_state()
                pred = model.predict(h, seed=12)
                handle.remove()
                self.assertTrue(torch.isfinite(pred).all())
                self.assertEqual(len(calls), model.cfg.ddim_steps)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                self.assertTrue(torch.equal(pred, twin.predict(h, seed=12)))
                self.assertFalse(torch.allclose(pred, model.predict(h, seed=13)))
                self.assertTrue(torch.equal(pred[:,:,2], torch.zeros_like(pred[:,:,2])))

    def test_fixed_target_can_be_fitted(self):
        model, loader = self.model(), self.loader()
        model.calibrate_reference(loader)
        h, y = (t[:2, ..., 4:] for t in loader.dataset.tensors)
        optimizer = torch.optim.Adam(model.diffusion.parameters(), lr=0.01)
        losses = []
        for _ in range(35):
            # Fix the target probability path to test optimization, not accuracy.
            torch.manual_seed(55)
            optimizer.zero_grad()
            loss, _ = model.diffusion_loss(h, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], 0.25*losses[0])

    def test_benchmark_protocol_preserves_requested_action_and_budgets(self):
        plan = resolve_plan(ROOT/"configs/ablations/cit_grip_phase_next_research.yaml")
        self.assertEqual(plan["action_filter"], "pick_up_screwd")
        self.assertEqual(plan["seeds"], [0, 1, 2])
        self.assertEqual(set(plan["models"]), {
            "cit_grip_phase_soft", "cit_grip_phase_coupled",
            "cit_grip_phase_coupled_soft",
        })


if __name__ == "__main__":
    unittest.main()
