from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def calibrate_multimodal_threshold(
    conditioning_context: torch.Tensor,
    sequence_ids: torch.Tensor,
    *,
    k: int = 5,
    chunk_size: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[float, Dict[str, float]]:
    """Calibrate a fixed radius from cross-sequence training neighbours.

    The returned radius is the median distance to each sample's k-th nearest
    eligible neighbour. Windows from the same source sequence are ineligible.
    """
    context = torch.as_tensor(conditioning_context, dtype=torch.float32)
    ids = torch.as_tensor(sequence_ids, dtype=torch.long).reshape(-1)
    if context.ndim < 2:
        raise ValueError("conditioning_context must have a sample dimension and features")
    if context.shape[0] != ids.numel():
        raise ValueError("sequence_ids must contain one id per conditioning sample")
    if context.shape[0] < 2:
        raise ValueError("at least two conditioning samples are required for calibration")
    if int(k) < 1:
        raise ValueError("k must be at least 1")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be at least 1")

    _unique_ids, inverse_ids = torch.unique(ids, return_inverse=True)
    counts_by_sequence = torch.bincount(inverse_ids)
    eligible_counts = int(context.shape[0]) - counts_by_sequence[inverse_ids]
    if torch.any(eligible_counts < int(k)):
        bad = int((eligible_counts < int(k)).sum())
        raise ValueError(
            f"{bad} training samples have fewer than k={int(k)} cross-sequence neighbours"
        )

    work_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flat = context.reshape(context.shape[0], -1).to(work_device)
    ids_device = ids.to(work_device)
    kth_distances = []
    for start in range(0, flat.shape[0], int(chunk_size)):
        query = flat[start : start + int(chunk_size)]
        distances = torch.cdist(query, flat)
        query_ids = ids_device[start : start + query.shape[0]]
        distances.masked_fill_(query_ids[:, None] == ids_device[None, :], float("inf"))
        kth_distances.append(distances.kthvalue(int(k), dim=1).values.cpu())

    kth = torch.cat(kth_distances)
    threshold = float(kth.median().item())
    if not torch.isfinite(torch.tensor(threshold)) or threshold <= 0:
        raise ValueError(f"calibration produced an invalid threshold: {threshold}")
    stats = {
        "threshold": threshold,
        "num_samples": int(context.shape[0]),
        "num_sequences": int(torch.unique(ids).numel()),
        "k": int(k),
        "kth_distance_mean": float(kth.mean().item()),
        "kth_distance_median": threshold,
        "kth_distance_min": float(kth.min().item()),
        "kth_distance_max": float(kth.max().item()),
    }
    return threshold, stats
