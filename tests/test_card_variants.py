from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPLINEEQNET = ROOT / "vendor" / "splineeqnet"
if str(SPLINEEQNET) not in sys.path:
    sys.path.insert(0, str(SPLINEEQNET))

from models.card import CardConfig, CardForecaster  # noqa: E402
from models.fihard import CardConfig as FiHardConfig  # noqa: E402
from models.fihard import CardForecaster as FiHardForecaster  # noqa: E402
from models.star_hand import STARHandConfig, STARHandForecaster  # noqa: E402


def _common_config() -> dict[str, object]:
    return {
        "input_length": 4,
        "pred_length": 4,
        "num_nodes": 5,
        "hidden_dim": 16,
        "num_layers": 1,
        "k_low": 4,
        "diffusion_steps": 4,
        "ddim_steps": 2,
        "denoiser_dim": 16,
        "denoiser_depth": 1,
        "denoiser_heads": 4,
        "spatial_anisotropy": True,
        "node_covariance_type": "laplacian_heat_kernel",
        "cond_use_history": False,
        "cond_use_coarse": True,
        "freeze_coarse": False,
    }


def _metadata() -> dict[str, object]:
    return {
        "wrist_index": 0,
        "edges": ((0, 1), (1, 2), (2, 3), (3, 4)),
    }


def test_card_uses_identity_temporal_factor() -> None:
    model = CardForecaster(
        CardConfig(
            **_common_config(),
            temporal_anisotropy=False,
            temporal_operator_type="identity",
        ),
        _metadata(),
    )

    assert not model.diffusion.temporal_anisotropy
    assert torch.equal(model.diffusion.temporal_basis, torch.eye(4))
    assert torch.equal(model.diffusion.temporal_basis_t, torch.eye(4))


def test_all_card_variants_support_shared_forecasting_api() -> None:
    history = torch.randn(2, 4, 5, 3)
    future = torch.randn(2, 4, 5, 3)
    variants = (
        CardForecaster(
            CardConfig(
                **_common_config(),
                temporal_anisotropy=False,
                temporal_operator_type="identity",
            ),
            _metadata(),
        ),
        STARHandForecaster(
            STARHandConfig(
                **_common_config(),
                temporal_anisotropy=True,
                temporal_operator_type="velocity",
                temporal_anisotropy_q=0.25,
            ),
            _metadata(),
        ),
        FiHardForecaster(
            FiHardConfig(
                **_common_config(),
                temporal_anisotropy=True,
                temporal_operator_type="velocity",
                temporal_anisotropy_q=0.125,
            ),
            _metadata(),
        ),
    )

    for model in variants:
        coarse = model.coarse(history)
        loss, _, metrics = model.diffusion_loss(
            history,
            future,
            coarse_future=coarse,
            return_metrics=True,
        )
        model.eval()
        prediction = model.predict(history, coarse_future=coarse, deterministic=True)

        assert prediction.shape == future.shape
        assert torch.isfinite(loss)
        assert torch.isfinite(prediction).all()
        assert "diffusion_loss" in metrics
