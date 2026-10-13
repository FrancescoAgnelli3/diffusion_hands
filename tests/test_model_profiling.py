import json

import torch

from common.model_profiling import (
    PROFILE_PATH_ENV,
    PROFILE_REPEATS_ENV,
    PROFILE_WARMUP_ENV,
    _PROFILED_KEYS,
    profile_once,
)


def test_profile_once_writes_one_record(monkeypatch, tmp_path):
    output = tmp_path / "profiles.jsonl"
    monkeypatch.setenv(PROFILE_PATH_ENV, str(output))
    monkeypatch.setenv(PROFILE_WARMUP_ENV, "0")
    monkeypatch.setenv(PROFILE_REPEATS_ENV, "2")
    _PROFILED_KEYS.clear()

    model = torch.nn.Linear(4, 3)
    sample = torch.ones(1, 4)

    first = profile_once(
        key="tiny",
        model_name="tiny",
        modules=[model],
        sample_fn=lambda: model(sample),
        sampling_steps=1,
        batch_size=1,
        num_candidates=1,
    )
    second = profile_once(
        key="tiny",
        model_name="tiny",
        modules=[model],
        sample_fn=lambda: model(sample),
        sampling_steps=1,
        batch_size=1,
        num_candidates=1,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert first is not None
    assert second is None
    assert len(records) == 1
    assert records[0]["parameters"] == 15
    assert records[0]["flops"] == 24
    assert records[0]["sampling_steps"] == 1
