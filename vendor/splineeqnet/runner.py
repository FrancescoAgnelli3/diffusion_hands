import hashlib
import math
import os
import random
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DatasetCfg, TrainCfg
from train_utils import train


def set_global_seed(seed: int) -> None:
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int % (2**32))
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


def _prepare_wandb_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, bool):
            payload[key] = float(value)
            continue
        if isinstance(value, (int, float)):
            val = float(value)
            if math.isfinite(val):
                payload[key] = val
    return payload


def run_experiment(
    ds: DatasetCfg,
    train_cfg: TrainCfg,
    *,
    save_root: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    log_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_prefix: Optional[str] = None,
    save_best_model: bool = False,
    best_model_path_override: Optional[str] = None,
    load_model_path: Optional[str] = None,
    twostage_eval_phase: Optional[str] = None,
    twostage_eval_best_of_k: Optional[int] = None,
    twostage_eval_collect_all: bool = False,
    twostage_eval_oracle_mpjpe: bool = False,
    compute_humanmac_metrics: bool = False,
    humanmac_num_candidates: int = 50,
    humanmac_multimodal_threshold: float = 0.5,
) -> Optional[Dict[str, float]]:
    set_global_seed(ds.seed)

    model_key = str(train_cfg.model).strip().lower()
    if model_key not in {"simlpe_dct", "twostage_dct_diffusion"}:
        raise ValueError(
            f"Unsupported model '{train_cfg.model}'. Available models in this pruned repo: "
            "simlpe_dct, twostage_dct_diffusion"
        )

    tag = (
        f"dataset={ds.dataset}__model={train_cfg.model}__hs={train_cfg.hidden_size}__gl={train_cfg.gru_layers}__"
        f"lr={train_cfg.lr}__gc={train_cfg.gradient_clip}__bs={ds.batch_size}__"
        f"in={ds.input_n}__out={ds.output_n}__st={ds.stride}__seed={ds.seed}"
    )
    if train_cfg.dct_keep_coeffs is not None:
        tag += f"__dctk={int(train_cfg.dct_keep_coeffs)}"
    if train_cfg.simlpe_norm_axis is not None:
        tag += f"__simnorm={train_cfg.simlpe_norm_axis}"
    if train_cfg.simlpe_use_norm is not None:
        tag += f"__simusenorm={int(train_cfg.simlpe_use_norm)}"
    if train_cfg.simlpe_use_spatial_fc_only is not None:
        tag += f"__simspfc={int(train_cfg.simlpe_use_spatial_fc_only)}"
    if train_cfg.simlpe_mix_spatial_temporal is not None:
        tag += f"__simmix={int(train_cfg.simlpe_mix_spatial_temporal)}"
    if train_cfg.simlpe_add_last_offset is not None:
        tag += f"__simoffset={int(train_cfg.simlpe_add_last_offset)}"
    if train_cfg.twostage_diffusion_epochs is not None:
        tag += f"__twepochs={int(train_cfg.twostage_diffusion_epochs)}"
    if train_cfg.twostage_k_low is not None:
        tag += f"__twk={int(train_cfg.twostage_k_low)}"
    if train_cfg.twostage_diffusion_steps is not None:
        tag += f"__twsteps={int(train_cfg.twostage_diffusion_steps)}"
    if train_cfg.twostage_ddim_steps is not None:
        tag += f"__twddim={int(train_cfg.twostage_ddim_steps)}"

    config = {
        "batch_size": int(ds.batch_size),
        "input_n": int(ds.input_n),
        "output_n": int(ds.output_n),
        "learning_rate": float(train_cfg.lr),
        "train_epoches": int(train_cfg.epochs),
        "hidden_size": int(train_cfg.hidden_size),
        "gru_layers": int(train_cfg.gru_layers),
        "gradient_clip": float(train_cfg.gradient_clip),
        "node_num": int(ds.node_count),
        "use_space": bool(train_cfg.use_space),
        "velocity_loss_weight": float(train_cfg.velocity_loss_weight),
        "save_eval_examples": bool(train_cfg.save_eval_examples),
        "save_eval_examples_all_k": bool(train_cfg.save_eval_examples_all_k),
        "log_gcn_stats": False,
        "log_wandb": bool(log_wandb),
        "dataset": ds.dataset,
        "action_filter": ds.action_filter,
        "edge_index": tuple(ds.edge_index),
        "adjacency": tuple(tuple(int(val) for val in row) for row in ds.adjacency) if ds.adjacency else tuple(),
    }

    if save_best_model:
        examples_models_dir = os.path.join(os.path.dirname(__file__), "examples", "models")
        os.makedirs(examples_models_dir, exist_ok=True)
        if best_model_path_override:
            config["best_model_path"] = str(best_model_path_override)
        else:
            action_segment = (ds.action_filter or "all").replace(os.sep, "_").replace(" ", "_")
            model_file = f"{train_cfg.model}_{ds.dataset}_{action_segment}.pt"
            config["best_model_path"] = os.path.join(examples_models_dir, model_file)
        config["best_model_tag"] = tag
    if load_model_path:
        config["load_model_path"] = str(load_model_path)
    if twostage_eval_phase:
        config["twostage_eval_phase"] = str(twostage_eval_phase)
    if twostage_eval_best_of_k is not None:
        config["twostage_eval_best_of_k"] = max(1, int(twostage_eval_best_of_k))
    if twostage_eval_collect_all:
        config["twostage_eval_collect_all"] = True
    if twostage_eval_oracle_mpjpe:
        config["twostage_eval_oracle_mpjpe"] = True
    if compute_humanmac_metrics:
        config["compute_humanmac_metrics"] = True
        config["humanmac_num_candidates"] = max(1, int(humanmac_num_candidates))
        config["humanmac_multimodal_threshold"] = float(humanmac_multimodal_threshold)

    if train_cfg.save_coarse_model:
        examples_models_dir = os.path.join(os.path.dirname(__file__), "examples", "models")
        os.makedirs(examples_models_dir, exist_ok=True)
        action_segment = (ds.action_filter or "all").replace(os.sep, "_").replace(" ", "_")
        tag_hash = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:12]
        model_file = f"{train_cfg.model}_coarse_{ds.dataset}_{action_segment}_{tag_hash}.pt"
        config["coarse_model_path"] = os.path.join(examples_models_dir, model_file)
        config["coarse_model_tag"] = tag
        config["save_coarse_model"] = True

    if train_cfg.dct_keep_coeffs is not None:
        config["dct_keep_coeffs"] = int(train_cfg.dct_keep_coeffs)
    if train_cfg.simlpe_norm_axis is not None:
        config["simlpe_norm_axis"] = str(train_cfg.simlpe_norm_axis)
    if train_cfg.simlpe_use_norm is not None:
        config["simlpe_use_norm"] = bool(train_cfg.simlpe_use_norm)
    if train_cfg.simlpe_use_spatial_fc_only is not None:
        config["simlpe_spatial_fc_only"] = bool(train_cfg.simlpe_use_spatial_fc_only)
    if train_cfg.simlpe_mix_spatial_temporal is not None:
        config["simlpe_mix_spatial_temporal"] = bool(train_cfg.simlpe_mix_spatial_temporal)
    if train_cfg.simlpe_add_last_offset is not None:
        config["simlpe_add_last_offset"] = bool(train_cfg.simlpe_add_last_offset)

    twostage_fields = [
        ("twostage_diffusion_epochs", train_cfg.twostage_diffusion_epochs, int),
        ("twostage_k_low", train_cfg.twostage_k_low, int),
        ("twostage_diffusion_steps", train_cfg.twostage_diffusion_steps, int),
        ("twostage_ddim_steps", train_cfg.twostage_ddim_steps, int),
        ("twostage_denoiser_dim", train_cfg.twostage_denoiser_dim, int),
        ("twostage_denoiser_depth", train_cfg.twostage_denoiser_depth, int),
        ("twostage_denoiser_heads", train_cfg.twostage_denoiser_heads, int),
        ("twostage_dropout", train_cfg.twostage_dropout, float),
        ("twostage_freeze_coarse", train_cfg.twostage_freeze_coarse, bool),
        ("twostage_cond_use_history", train_cfg.twostage_cond_use_history, bool),
        ("twostage_cond_use_coarse", train_cfg.twostage_cond_use_coarse, bool),
        ("twostage_use_mamp_condition", train_cfg.twostage_use_mamp_condition, bool),
        ("twostage_use_mamp_condition_coarse", train_cfg.twostage_use_mamp_condition_coarse, bool),
        ("twostage_mamp_checkpoint", train_cfg.twostage_mamp_checkpoint, str),
        ("twostage_mamp_config", train_cfg.twostage_mamp_config, str),
        ("twostage_mamp_repo_root", train_cfg.twostage_mamp_repo_root, str),
        ("twostage_mamp_mask_ratio", train_cfg.twostage_mamp_mask_ratio, float),
        ("twostage_mamp_motion_aware_tau", train_cfg.twostage_mamp_motion_aware_tau, float),
        ("twostage_mpjpe_weight", train_cfg.twostage_mpjpe_weight, float),
    ]
    for key, value, caster in twostage_fields:
        if value is not None:
            config[key] = caster(value)

    cfg_dict = asdict(train_cfg)
    cfg_dict["dataset"] = asdict(ds)

    run = None
    if log_wandb:
        try:
            import wandb

            project = wandb_project or os.environ.get("WANDB_PROJECT") or "pose-forecasting"
            run_name = (
                f"{wandb_run_prefix}-{tag}" if wandb_run_prefix else tag
            )
            run = wandb.init(
                project=project,
                entity=wandb_entity,
                name=run_name,
                config={"train": cfg_dict, "runtime": config},
                reinit=True,
            )
        except Exception as exc:
            print(f"[wandb] init skipped: {exc}")
            run = None

    try:
        out = train(
            config,
            epochs=train_cfg.epochs,
            lr=train_cfg.lr,
            bone_loss_weight=train_cfg.bone_loss_weight,
            model=train_cfg.model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            log_wandb=bool(run is not None),
            wandb_run=run,
        )
    finally:
        if run is not None:
            try:
                wandb_payload = _prepare_wandb_metrics(out or {}) if isinstance(out, dict) else {}
                if wandb_payload:
                    run.log(wandb_payload)
            except Exception:
                pass
            run.finish()

    if out is None:
        return None
    out = dict(out)
    out["run_tag"] = tag
    out["elapsed_seconds"] = float(out.get("elapsed_seconds", 0.0))
    return out
