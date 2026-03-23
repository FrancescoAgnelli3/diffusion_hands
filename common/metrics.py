from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray]


def _to_tensor(x: TensorLike, *, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if device is not None:
            return x.to(device=device, dtype=dtype)
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def time_slice(array: torch.Tensor, t0: int, t: int, axis: int) -> torch.Tensor:
    if t == -1:
        return torch.index_select(
            array, axis, torch.arange(t0, array.shape[axis], device=array.device, dtype=torch.int32)
        )
    return torch.index_select(array, axis, torch.arange(t0, t, device=array.device, dtype=torch.int32))


def apd(pred: torch.Tensor, target: torch.Tensor, *args, t0: int = 0, t: int = -1) -> torch.Tensor:
    del target, args
    pred = time_slice(pred, t0, t, 2)
    batch_size, n_samples = pred.shape[:2]
    if n_samples == 1:
        return torch.zeros((batch_size,), device=pred.device)
    arr = pred.reshape(batch_size, n_samples, -1)
    dist = torch.cdist(arr, arr)
    idx = torch.triu_indices(n_samples, n_samples, offset=1, device=pred.device)
    vals = dist[:, idx[0], idx[1]]
    return vals.mean(dim=1)


def ade(pred: torch.Tensor, target: torch.Tensor, *args, t0: int = 0, t: int = -1) -> torch.Tensor:
    del args
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    target = target.reshape((batch_size, 1, seq_length, -1))
    diff = pred - target
    dist = torch.linalg.norm(diff, dim=-1).mean(dim=-1)
    return dist.min(dim=-1).values


def fde(pred: torch.Tensor, target: torch.Tensor, *args, t0: int = 0, t: int = -1) -> torch.Tensor:
    del args
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    target = target.reshape((batch_size, 1, seq_length, -1))
    diff = pred - target
    dist = torch.linalg.norm(diff, dim=-1)[:, :, -1]
    return dist.min(dim=-1).values


def mmade(
    pred: torch.Tensor,
    target: torch.Tensor,
    gt_multi: Sequence[TensorLike],
    *args,
    t0: int = 0,
    t: int = -1,
) -> torch.Tensor:
    del target, args
    pred = time_slice(pred, t0, t, 2)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size,), dtype=pred.dtype, device=pred.device)
    for i in range(batch_size):
        gt_i = _to_tensor(gt_multi[i], dtype=pred.dtype, device=pred.device)
        n_gts = gt_i.shape[0]
        if n_gts == 1:
            results[i] = float("nan")
            continue
        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)
        gt = time_slice(gt_i, t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(1)
        dist = torch.linalg.norm(p - gt, dim=-1).mean(dim=-1)
        results[i] = dist.min(dim=-1).values.mean()
    return results


def mmfde(
    pred: torch.Tensor,
    target: torch.Tensor,
    gt_multi: Sequence[TensorLike],
    *args,
    t0: int = 0,
    t: int = -1,
) -> torch.Tensor:
    del target, args
    pred = time_slice(pred, t0, t, 2)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size,), dtype=pred.dtype, device=pred.device)
    for i in range(batch_size):
        gt_i = _to_tensor(gt_multi[i], dtype=pred.dtype, device=pred.device)
        n_gts = gt_i.shape[0]
        if n_gts == 1:
            results[i] = float("nan")
            continue
        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)
        gt = time_slice(gt_i, t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(1)
        dist = torch.linalg.norm(p - gt, dim=-1)[:, :, -1]
        results[i] = dist.min(dim=-1).values.mean()
    return results


def humanmac_metrics(
    pred_candidates: torch.Tensor,
    gt_future: torch.Tensor,
    start_pose: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if pred_candidates.numel() == 0 or gt_future.numel() == 0 or start_pose.numel() == 0:
        return {"APD": float("nan"), "ADE": float("nan"), "FDE": float("nan"), "MMADE": float("nan"), "MMFDE": float("nan")}

    pred_flat = pred_candidates.reshape(
        pred_candidates.shape[0], pred_candidates.shape[1], pred_candidates.shape[2], -1
    ).cpu()
    gt_flat = gt_future.reshape(gt_future.shape[0], gt_future.shape[1], -1).cpu()
    start_flat = start_pose.reshape(start_pose.shape[0], -1).cpu()

    pairwise = torch.cdist(start_flat, start_flat)
    apd_total = 0.0
    ade_total = 0.0
    fde_total = 0.0
    mmade_total = 0.0
    mmfde_total = 0.0
    num_samples = int(gt_flat.shape[0])

    for sample_idx in range(num_samples):
        pred_sample = pred_flat[:, sample_idx, :, :]
        gt_sample = gt_flat[sample_idx : sample_idx + 1]
        group_idx = torch.nonzero(pairwise[sample_idx] < float(threshold), as_tuple=False).view(-1)
        gt_multi = gt_flat[group_idx]

        if pred_sample.shape[0] <= 1:
            apd_val = pred_sample.new_tensor(0.0)
        else:
            apd_val = torch.pdist(pred_sample.reshape(pred_sample.shape[0], -1)).mean()

        gt_multi_gt = torch.cat([gt_multi, gt_sample], dim=0)
        dist = torch.linalg.norm(pred_sample[:, None, :, :] - gt_multi_gt[None, :, :, :], dim=3)

        mmfde_val = dist[:, :-1, -1].min(dim=0).values.mean()
        mmade_val = dist[:, :-1].mean(dim=2).min(dim=0).values.mean()
        ade_val = dist[:, -1].mean(dim=1).min(dim=0).values.mean()
        fde_val = dist[:, -1, -1].min(dim=0).values.mean()

        apd_total += float(apd_val.item())
        ade_total += float(ade_val.item())
        fde_total += float(fde_val.item())
        mmade_total += float(mmade_val.item())
        mmfde_total += float(mmfde_val.item())

    denom = max(1, num_samples)
    return {
        "APD": apd_total / denom,
        "ADE": ade_total / denom,
        "FDE": fde_total / denom,
        "MMADE": mmade_total / denom,
        "MMFDE": mmfde_total / denom,
    }


def humanmac_metrics_prefixed(
    pred_candidates: torch.Tensor,
    gt_future: torch.Tensor,
    start_pose: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    m = humanmac_metrics(pred_candidates, gt_future, start_pose, threshold=threshold)
    return {
        "humanmac_apd": m["APD"],
        "humanmac_ade": m["ADE"],
        "humanmac_fde": m["FDE"],
        "humanmac_mmade": m["MMADE"],
        "humanmac_mmfde": m["MMFDE"],
    }


def splineeqnet_diffusion_batch_eval(
    pred_candidates: torch.Tensor,
    gt_future: torch.Tensor,
    start_pose: torch.Tensor,
    norm_factor: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> Dict[str, Union[torch.Tensor, Dict[str, float]]]:
    if pred_candidates.dim() != 4 or gt_future.dim() != 3:
        raise ValueError("Expected pred_candidates[K,B,T,NC] and gt_future[B,T,NC].")
    if pred_candidates.shape[3] % 3 != 0:
        raise ValueError("Last dim (NC) must be divisible by 3.")

    pred_bktc = pred_candidates.permute(1, 0, 2, 3).contiguous()
    b, k, t, nc = pred_bktc.shape
    n = nc // 3
    pred_bktn3 = pred_bktc.reshape(b, k, t, n, 3)
    gt_btn3 = gt_future.reshape(b, t, n, 3)

    oracle_terms = torch.norm(pred_bktn3 - gt_btn3.unsqueeze(1), dim=-1)
    oracle_mpjpe = oracle_terms.mean(dim=(2, 3))
    best_idx = oracle_mpjpe.argmin(dim=1)
    batch_selector = torch.arange(b, device=best_idx.device)
    recons_btnc = pred_bktc[batch_selector, best_idx]

    recons_btn3 = recons_btnc.reshape(b, t, n, 3)
    mpjpe_terms = torch.norm(recons_btn3 - gt_btn3, dim=-1)
    per_sample_mpjpe = mpjpe_terms.mean(dim=(1, 2))
    per_sample_mpjpe_norm = per_sample_mpjpe * norm_factor.view(-1).to(per_sample_mpjpe.device)

    humanmac = humanmac_metrics(pred_candidates, gt_future, start_pose, threshold=threshold)
    return {
        "per_sample_mpjpe": per_sample_mpjpe,
        "per_sample_mpjpe_norm": per_sample_mpjpe_norm,
        "humanmac": humanmac,
    }


def compute_all_metrics_single(
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt_multi: TensorLike,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = _to_tensor(pred, dtype=torch.float32, device=pred.device if isinstance(pred, torch.Tensor) else None)
    gt = _to_tensor(gt, dtype=torch.float32, device=pred.device)
    gt_multi_t = _to_tensor(gt_multi, dtype=torch.float32, device=pred.device)

    if pred.shape[0] == 1:
        diversity = pred.new_tensor(0.0)
    else:
        diversity = torch.pdist(pred.reshape(pred.shape[0], -1)).mean()

    gt_multi_gt = torch.cat([gt_multi_t, gt], dim=0)[None, ...]
    pred_b = pred[:, None, ...]
    dist = torch.linalg.norm(pred_b - gt_multi_gt, dim=3)

    mmfde = dist[:, :-1, -1].min(dim=0).values.mean()
    mmade = dist[:, :-1].mean(dim=2).min(dim=0).values.mean()
    ade_v = dist[:, -1].mean(dim=1).min(dim=0).values.mean()
    fde_v = dist[:, -1, -1].min(dim=0).values.mean()
    return diversity, ade_v, fde_v, mmade, mmfde

