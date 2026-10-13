from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLINEEQNET = ROOT / "vendor" / "splineeqnet"
if str(SPLINEEQNET) not in sys.path:
    sys.path.insert(0, str(SPLINEEQNET))

from data import (  # noqa: E402
    extract_action_name_from_filename,
    extract_assembly_split_from_filename,
    select_balanced_action_files,
)


def _files() -> list[str]:
    files = []
    for action in ("pick_up_screw", "rotate_partial_toy", "put_down_screwdriver"):
        for prefix in ("tr", "va", "te"):
            for index in range(8):
                files.append(
                    f"/dataset/{prefix}{index}_a1_v1_n1_lenPose140_lenRGB70_{action}.npy"
                )
    return files


def test_global_balanced_cap_is_deterministic_and_preserves_split_coverage() -> None:
    selected = select_balanced_action_files(
        _files(), dataset="assembly", max_files=18, seed=7, preserve_official_splits=True
    )
    repeated = select_balanced_action_files(
        _files(), dataset="assembly", max_files=18, seed=7, preserve_official_splits=True
    )
    assert selected == repeated
    assert len(selected) == 18

    actions = {extract_action_name_from_filename(path, "assembly") for path in selected}
    assert len(actions) == 3
    for action in actions:
        action_files = [p for p in selected if extract_action_name_from_filename(p, "assembly") == action]
        assert {extract_assembly_split_from_filename(path) for path in action_files} == {
            "train", "val", "test"
        }
        assert len(action_files) == 6
