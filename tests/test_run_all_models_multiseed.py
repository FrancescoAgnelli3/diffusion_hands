import csv

from common.evaluation import CANONICAL_METRIC_KEYS
from tools.run_all_models import _normalize_cit_defaults, _resolve_seeds, _write_summary_csv


def _row(seed, status="ok"):
    return {
        "dataset": "assembly",
        "action_filter": "pick_up_screwd",
        "seed": seed,
        "model": "example",
        "status": status,
        "notes": "",
        **{metric: float(seed + 1) for metric in CANONICAL_METRIC_KEYS},
    }


def test_resolve_seeds_supports_multi_seed_and_legacy_single_seed():
    assert _resolve_seeds({"seeds": [0, 1, 2]}) == [0, 1, 2]
    assert _resolve_seeds({"seed": 7}) == [7]


def test_cit_public_defaults_normalize_to_the_legacy_runner_values():
    defaults = {
        "cit_coarse_k_low": 80,
        "cit_transport_epochs": 200,
        "cit_sampling_steps": 25,
        "cit_denoiser_heads": 4,
    }
    assert _normalize_cit_defaults(defaults) == {
        "card_k_low": 80,
        "card_diffusion_epochs": 200,
        "card_ddim_steps": 25,
        "card_denoiser_heads": 4,
    }
    assert defaults["cit_sampling_steps"] == 25  # Normalization is non-mutating.
    assert _normalize_cit_defaults({"card_ddim_steps": 25}) == {"card_ddim_steps": 25}
    assert _normalize_cit_defaults({
        "card_ddim_steps": 50,
        "cit_sampling_steps": 25,
    }) == {"card_ddim_steps": 25}


def test_summary_contains_mean_and_sample_std_for_complete_cohort(tmp_path):
    output = tmp_path / "summary.csv"
    _write_summary_csv(output, [_row(0), _row(1), _row(2)], [0, 1, 2])

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["complete"] == "True"
    assert row["n"] == "3"
    assert row["ADE_mean"] == "2.000000"
    assert row["ADE_std"] == "1.000000"


def test_summary_does_not_average_incomplete_cohort(tmp_path):
    output = tmp_path / "summary.csv"
    _write_summary_csv(output, [_row(0), _row(1, status="failed"), _row(2)], [0, 1, 2])

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["complete"] == "False"
    assert row["n"] == "2"
    assert row["ADE_mean"] == ""
    assert row["ADE_std"] == ""
