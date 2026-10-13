from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "bayesian_signed_rank_test.py"
SPEC = importlib.util.spec_from_file_location("bayesian_signed_rank_test", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_load_complete_four_model_metric_cache(tmp_path: Path) -> None:
    combined = pd.DataFrame({"sample_idx": [0, 1, 2]})
    for model_name in (*TOOL.CARD_VARIANTS, TOOL.REFERENCE_MODEL):
        for metric_idx, metric in enumerate(TOOL.METRICS):
            combined[f"{model_name}__{metric}"] = [
                float(metric_idx),
                float(metric_idx + 1),
                float(metric_idx + 2),
            ]
    combined.to_csv(tmp_path / "all_sample_metrics.csv", index=False)

    variant_tables, reference_table = TOOL.load_metrics_tables_from_csv(tmp_path)

    assert tuple(variant_tables) == TOOL.CARD_VARIANTS
    assert len(reference_table) == 3
    assert list(variant_tables["star_hand"].columns) == ["sample_idx", *TOOL.METRICS]


def test_all_unique_model_pairs_are_compared() -> None:
    assert TOOL.PAIRWISE_COMPARISONS == (
        ("card", "star_hand"),
        ("card", "fihard"),
        ("card", "comusion"),
        ("star_hand", "fihard"),
        ("star_hand", "comusion"),
        ("fihard", "comusion"),
    )


def test_incomplete_legacy_metric_cache_is_rejected(tmp_path: Path) -> None:
    combined = pd.DataFrame({"sample_idx": [0, 1]})
    for model_name in ("card", TOOL.REFERENCE_MODEL):
        for metric in TOOL.METRICS:
            combined[f"{model_name}__{metric}"] = [0.0, 1.0]
    combined.to_csv(tmp_path / "all_sample_metrics.csv", index=False)

    try:
        TOOL.load_metrics_tables_from_csv(tmp_path)
    except RuntimeError as exc:
        assert "star_hand__ADE" in str(exc)
        assert "fihard__FID" in str(exc)
    else:
        raise AssertionError("Incomplete legacy metrics cache was accepted")
