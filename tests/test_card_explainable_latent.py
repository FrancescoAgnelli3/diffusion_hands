from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SPLINEEQNET = ROOT / "vendor" / "splineeqnet"
if str(SPLINEEQNET) not in sys.path:
    sys.path.insert(0, str(SPLINEEQNET))

from models.card import CardConfig, CardForecaster, _gradient_reverse  # noqa: E402
from explainability import generate_latent_explainability_artifacts  # noqa: E402


def _model(num_actions: int = 3) -> CardForecaster:
    cfg = CardConfig(
        input_length=4,
        pred_length=3,
        num_nodes=2,
        hidden_dim=8,
        num_layers=1,
        k_low=4,
        diffusion_steps=4,
        ddim_steps=2,
        isotropic_noise=True,
        spatial_anisotropy=False,
        temporal_anisotropy=False,
        denoiser_dim=16,
        denoiser_depth=1,
        denoiser_heads=4,
        use_latent_variability=True,
        latent_dim=12,
        latent_encoder_hidden_dim=16,
        num_actions=num_actions,
        explainable_latent_decomposition=True,
        latent_action_dim=4,
        latent_style_dim=4,
        latent_residual_dim=4,
        freeze_coarse=False,
    )
    return CardForecaster(
        cfg,
        metadata={
            "wrist_index": 0,
            "edges": (),
            "action_class_weights": torch.ones(num_actions),
        },
    )


def test_gradient_reversal_changes_only_gradient_sign() -> None:
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    reversed_value = _gradient_reverse(value, 0.25)
    assert torch.equal(reversed_value, value)
    reversed_value.sum().backward()
    assert torch.allclose(value.grad, torch.full_like(value, -0.25))


def test_partition_dimensions_must_sum_to_latent_dim() -> None:
    cfg = _model().cfg
    cfg.latent_residual_dim = 3
    try:
        CardForecaster(cfg, metadata={"wrist_index": 0, "edges": ()})
    except ValueError as exc:
        assert "sum to latent_dim" in str(exc)
    else:
        raise AssertionError("Invalid latent partition was accepted")


def test_explainable_prior_and_loss_partition() -> None:
    model = _model()
    history = torch.randn(3, 4, 2, 3)
    coarse = torch.randn(3, 3, 2, 3)
    residual = torch.randn(3, 3, 2, 3)
    action = torch.tensor([0, 1, 2])
    latent, metrics = model._posterior_latent(
        residual_gt=residual,
        history=history,
        coarse_future=coarse,
        action_id=action,
        grl_strength=0.5,
    )
    assert latent.shape == (3, 12)
    assert set(model.latent_slices) == {"action", "style", "residual"}
    assert torch.allclose(metrics["kl_loss"], metrics["kl_structured"] + metrics["kl_residual"])
    assert metrics["action_loss"].ndim == 0
    assert metrics["adversarial_loss"].ndim == 0

    _sample, prior_mu, prior_logvar = model.sample_prior_latent(
        history, action_id=action, coarse_future=coarse, seed=3
    )
    residual_slice = model.latent_slices["residual"]
    assert torch.count_nonzero(prior_mu[:, residual_slice]) == 0
    assert torch.count_nonzero(prior_logvar[:, residual_slice]) == 0
    prior_blocks = model.prior_latent_blocks(
        history, action_id=action, coarse_future=coarse, seed=3
    )
    posterior_blocks = model.posterior_latent_blocks(
        residual_gt=residual,
        history=history,
        coarse_future=coarse,
        action_id=action,
    )
    assert {name: block["z"].shape[-1] for name, block in prior_blocks.items()} == {
        "action": 4,
        "style": 4,
        "residual": 4,
    }
    assert set(posterior_blocks) == {"action", "style", "residual"}

    total, _coarse, objective_metrics = model.diffusion_loss(
        history,
        residual + coarse,
        coarse_future=coarse,
        action_id=action,
        beta_kl_current=0.002,
        grl_strength=0.5,
        return_metrics=True,
    )
    expected = (
        objective_metrics["diffusion_loss"]
        + 0.002 * objective_metrics["kl_loss"]
        + 0.1 * objective_metrics["action_loss"]
        + 0.1 * objective_metrics["adversarial_loss"]
    )
    assert torch.allclose(total.detach(), expected, atol=1e-6)


def test_one_action_fallback_omits_auxiliary_objective() -> None:
    model = _model(num_actions=1)
    history = torch.randn(2, 4, 2, 3)
    coarse = torch.randn(2, 3, 2, 3)
    future = torch.randn(2, 3, 2, 3)
    total, _coarse, metrics = model.diffusion_loss(
        history,
        future,
        coarse_future=coarse,
        action_id=torch.zeros(2, dtype=torch.long),
        beta_kl_current=0.002,
        grl_strength=1.0,
        return_metrics=True,
    )
    expected = metrics["diffusion_loss"] + 0.002 * metrics["kl_loss"]
    assert torch.allclose(total.detach(), expected, atol=1e-6)


def test_shared_noise_and_latent_are_reproducible() -> None:
    model = _model().eval()
    history = torch.randn(1, 4, 2, 3)
    coarse = torch.randn(1, 3, 2, 3)
    latent = torch.randn(1, 12)
    noise = torch.randn(1, 3, model.diffusion.free_feature_dim)
    first = model.predict_with_latent(
        history, latent, coarse_future=coarse, initial_noise=noise, steps=2
    )
    second = model.predict_with_latent(
        history, latent, coarse_future=coarse, initial_noise=noise, steps=2
    )
    assert torch.equal(first, second)


def test_explainability_artifact_schema() -> None:
    model = _model().eval()

    class TinyDataset:
        id_to_action = ["action_zero"]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            del index
            inp = torch.randn(4, 2, 7)
            target = torch.randn(3, 2, 7)
            return inp, target, torch.tensor(1.0), {"action_id": torch.tensor(0)}

    with tempfile.TemporaryDirectory() as temp_dir:
        result = generate_latent_explainability_artifacts(
            model,
            SimpleNamespace(dataset=TinyDataset()),
            device=torch.device("cpu"),
            output_dir=temp_dir,
            ddim_steps=1,
        )
        assert result["actions"] == 1
        assert (Path(temp_dir) / "manifest.json").exists()
        assert (Path(temp_dir) / "latent_sensitivity.csv").exists()
        assert len(list((Path(temp_dir) / "traversals").glob("*.npz"))) == 1
