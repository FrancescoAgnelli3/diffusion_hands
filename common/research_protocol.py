"""Fail-closed data and training audits for controlled research comparisons."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, value in tensors:
        array = value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else np.asarray(value)
        digest.update(str(name).encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def audit_datasets(datasets, path, reference_path=None):
    result = {}
    seen = set()
    for name, dataset in zip(("train", "val", "test"), datasets):
        files = [str(Path(f).resolve()) for f in dataset.source_files]
        if seen.intersection(files) or len(files) != len(set(files)):
            raise ValueError("Repeated source files or overlap between source-file splits")
        seen.update(files)
        tensors = [(f"sequence_{i}", value) for i, value in enumerate(dataset.sequences)]
        tensors += [(f"norm_{i}", value) for i, value in enumerate(dataset.norm_factors)]
        tensors += [("windows", np.asarray(dataset.index, dtype=np.int64))]
        if dataset.window_action_ids is not None:
            tensors.append(("actions", dataset.window_action_ids))
        result[name] = {"source_files": files, "windows": len(dataset),
                        "sha256": tensor_digest(tensors), "action_to_id": dataset.action_to_id}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    if reference_path:
        reference = Path(reference_path)
        if reference.exists():
            if json.loads(reference.read_text()) != result:
                raise RuntimeError("Dataset protocol mismatch: transformed sequences/windows/splits differ from reference")
        else:
            # Launcher is sequential; exclusive creation prevents silently replacing a reference.
            with reference.open("x") as stream:
                json.dump(result, stream, indent=2)
    return result


def audit_training(model, path, updates, config, parameter_count):
    parameter_count = sum(p.numel() for p in model.parameters()
                          if not isinstance(p, torch.nn.parameter.UninitializedParameter))
    report = {"updates": updates, "parameters": parameter_count,
              "coarse_sha256": tensor_digest(sorted(model.coarse.state_dict().items())),
              "seed": config["seed"], "variant": config["card_variant"],
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "device": str(next(model.parameters()).device)}
    Path(path).with_name("training_audit.json").write_text(json.dumps(report, indent=2))
