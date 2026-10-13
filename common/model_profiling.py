from __future__ import annotations

import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from torch import nn
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
from torch.utils.flop_counter import FlopCounterMode


PROFILE_PATH_ENV = "DIFFUSION_HANDS_PROFILE_PATH"
PROFILE_WARMUP_ENV = "DIFFUSION_HANDS_PROFILE_WARMUP"
PROFILE_REPEATS_ENV = "DIFFUSION_HANDS_PROFILE_REPEATS"

_PROFILED_KEYS: set[str] = set()


def profiling_enabled() -> bool:
    return bool(os.environ.get(PROFILE_PATH_ENV, "").strip())


def profiling_pending(key: str) -> bool:
    return profiling_enabled() and key not in _PROFILED_KEYS


def _unique_parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    unique: dict[int, nn.Parameter] = {}
    for module in modules:
        for parameter in module.parameters():
            # Optional disabled branches can retain LazyLinear parameters that
            # never participate in this configured inference path.
            if isinstance(parameter, UninitializedParameter):
                continue
            unique[id(parameter)] = parameter
    return list(unique.values())


def _module_storage_bytes(modules: Iterable[nn.Module]) -> int:
    tensors: dict[int, torch.Tensor] = {}
    for module in modules:
        for tensor in (*module.parameters(), *module.buffers()):
            if isinstance(tensor, (UninitializedParameter, UninitializedBuffer)):
                continue
            tensors[id(tensor)] = tensor
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _first_device(modules: Iterable[nn.Module]) -> torch.device:
    for module in modules:
        parameter = next(module.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(module.buffers(), None)
        if buffer is not None:
            return buffer.device
    return torch.device("cpu")


def _append_jsonl(record: dict) -> None:
    output_path = Path(os.environ[PROFILE_PATH_ENV]).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def profile_once(
    *,
    key: str,
    model_name: str,
    modules: Iterable[nn.Module],
    sample_fn: Callable[[], object],
    sampling_steps: Optional[int],
    batch_size: int,
    num_candidates: int,
    notes: str = "",
) -> Optional[dict]:
    """Profile one complete sampling call once per process and model key.

    The caller should make ``sample_fn`` operate on a single real test example.
    The normal evaluation call remains separate so profiling cannot change its
    predictions or tensor shapes.
    """
    if not profiling_enabled() or key in _PROFILED_KEYS:
        return None
    _PROFILED_KEYS.add(key)

    modules = list(modules)
    if not modules:
        raise ValueError("profile_once requires at least one inference module")
    for module in modules:
        module.eval()

    warmup = max(0, int(os.environ.get(PROFILE_WARMUP_ENV, "5")))
    repeats = max(1, int(os.environ.get(PROFILE_REPEATS_ENV, "30")))
    parameters = _unique_parameters(modules)
    module_storage_bytes = _module_storage_bytes(modules)
    device = _first_device(modules)
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )

    flops: Optional[int] = None
    flop_error = ""
    peak_memory_mib: Optional[float] = None
    latencies_ms: list[float] = []

    with torch.inference_mode():
        for _ in range(warmup):
            output = sample_fn()
            del output
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        try:
            with FlopCounterMode(display=False) as counter:
                output = sample_fn()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            flops = int(counter.get_total_flops())
            del output
        except Exception as exc:  # Some vendor-specific operators lack flop rules.
            flop_error = f"{type(exc).__name__}: {exc}"

        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            baseline_allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            output = sample_fn()
            torch.cuda.synchronize(device)
            sampling_allocation = max(
                0,
                torch.cuda.max_memory_allocated(device) - baseline_allocated,
            )
            # Vendor training scripts often retain optimizer state during their
            # final evaluation. Count model storage plus allocations made by the
            # profiled B=1 call instead of unrelated process allocations.
            peak_memory_mib = float(
                (module_storage_bytes + sampling_allocation) / (1024**2)
            )
            del output

            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = sample_fn()
                end.record()
                torch.cuda.synchronize(device)
                latencies_ms.append(float(start.elapsed_time(end)))
                del output
        else:
            for _ in range(repeats):
                start_time = time.perf_counter()
                output = sample_fn()
                latencies_ms.append((time.perf_counter() - start_time) * 1000.0)
                del output

    torch.random.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, device)

    record = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model": model_name,
        "parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "latency_ms": float(statistics.median(latencies_ms)),
        "latency_mean_ms": float(statistics.mean(latencies_ms)),
        "latency_std_ms": (
            float(statistics.stdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0
        ),
        "peak_memory_mib": peak_memory_mib,
        "flops": flops,
        "sampling_steps": sampling_steps,
        "batch_size": int(batch_size),
        "num_candidates": int(num_candidates),
        "warmup": warmup,
        "repeats": repeats,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else ""
        ),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "notes": notes,
        "flop_error": flop_error,
    }
    _append_jsonl(record)
    print(
        "[PROFILE] "
        f"{model_name}: params={total_parameters:,}, "
        f"latency={record['latency_ms']:.3f} ms, "
        f"peak={peak_memory_mib if peak_memory_mib is not None else 'N/A'} MiB, "
        f"flops={flops if flops is not None else 'N/A'}, steps={sampling_steps}"
    )
    return record
