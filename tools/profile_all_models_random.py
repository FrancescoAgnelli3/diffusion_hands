#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from common.model_profiling import (
    PROFILE_PATH_ENV,
    PROFILE_REPEATS_ENV,
    PROFILE_WARMUP_ENV,
    _PROFILED_KEYS,
    profile_once,
)
from profile_all_models import write_outputs
from bayesian_signed_rank_test import (
    REPO_ROOT,
    VENDOR_SPLINE,
    load_yaml,
    prepend_sys_path,
    purge_vendor_modules,
)


CARD_VARIANTS = ("card", "star_hand", "fihard")
VENDOR_DLOW = REPO_ROOT / "vendor" / "dlow"
VENDOR_BELFUSION = REPO_ROOT / "vendor" / "belfusion"
VENDOR_COMUSION = REPO_ROOT / "vendor" / "comusion"
VENDOR_HUMANMAC = REPO_ROOT / "vendor" / "humanmac"
VENDOR_GSPS = REPO_ROOT / "vendor" / "gsps"
VENDOR_SKELETONDIFFUSION = REPO_ROOT / "vendor" / "skeletondiffusion"
DEFAULT_JSONL = REPO_ROOT / "results" / "all_model_efficiency.jsonl"
DEFAULT_CSV = REPO_ROOT / "results" / "all_model_efficiency.csv"
DEFAULT_MARKDOWN = REPO_ROOT / "results" / "all_model_efficiency.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile every randomly initialized model on one synthetic input "
            "batch; no data or checkpoints are loaded."
        )
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=REPO_ROOT / "configs" / "train_card_variants_vs_comusion.yaml",
    )
    parser.add_argument("--dataset", default="assembly")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def load_variant(
    model_name: str,
    ds_cfg,
    device: torch.device,
):
    purge_vendor_modules("config", "data", "datasets", "models", "runner", "train_utils")
    with prepend_sys_path(VENDOR_SPLINE):
        from data import resolve_card_hand_graph_metadata

        module_name = {
            "card": "models.card",
            "star_hand": "models.star_hand",
            "fihard": "models.fihard",
        }[model_name]
        model_module = importlib.import_module(module_name)
        config_type = model_module.CardConfig
        forecaster_type = model_module.CardForecaster
        model_cfg = load_yaml(
            REPO_ROOT / "configs" / "models" / f"{model_name}.yaml"
        )["defaults"]
        config_kwargs = dict(
            input_length=int(ds_cfg.input_n),
            pred_length=int(ds_cfg.output_n),
            num_nodes=int(ds_cfg.node_count),
            hidden_dim=int(model_cfg["hidden_size"]),
            num_layers=int(model_cfg["gru_layers"]),
            k_low=int(model_cfg["card_k_low"]),
            diffusion_steps=int(model_cfg["card_diffusion_steps"]),
            ddim_steps=int(model_cfg["card_ddim_steps"]),
            isotropic_noise=bool(model_cfg["card_isotropic_noise"]),
            spatial_anisotropy=bool(model_cfg["card_spatial_anisotropy"]),
            beta_matrix_power=float(model_cfg["card_beta_matrix_power"]),
            beta_matrix_min_rate=float(model_cfg["card_beta_matrix_min_rate"]),
            beta_matrix_max_rate=float(model_cfg["card_beta_matrix_max_rate"]),
            temporal_anisotropy=bool(model_cfg["card_temporal_anisotropy"]),
            temporal_anisotropy_q=float(model_cfg["card_temporal_anisotropy_q"]),
            temporal_operator_type=str(model_cfg["card_temporal_operator_type"]),
            temporal_operator_spectral_transform=str(
                model_cfg["card_temporal_operator_spectral_transform"]
            ),
            temporal_velocity_weight=float(model_cfg["card_temporal_velocity_weight"]),
            temporal_acceleration_weight=float(model_cfg["card_temporal_acceleration_weight"]),
            temporal_jerk_weight=float(model_cfg["card_temporal_jerk_weight"]),
            temporal_anisotropy_learned_from_history=bool(
                model_cfg["card_temporal_anisotropy_learned_from_history"]
            ),
            temporal_anisotropy_history_dim=int(
                model_cfg["card_temporal_anisotropy_history_dim"]
            ),
            temporal_anisotropy_delta_max_abs=float(
                model_cfg["card_temporal_anisotropy_delta_max_abs"]
            ),
            node_covariance_type=str(model_cfg["card_node_covariance_type"]),
            mobility_palm_var=float(model_cfg["card_mobility_palm_var"]),
            mobility_depth1_var=float(model_cfg["card_mobility_depth1_var"]),
            mobility_depth2_var=float(model_cfg["card_mobility_depth2_var"]),
            mobility_depth3plus_var=float(model_cfg["card_mobility_depth3plus_var"]),
            dhalf_gamma=float(model_cfg["card_dhalf_gamma"]),
            learnable_dhalf=bool(model_cfg["card_learnable_dhalf"]),
            graph_laplacian_alpha=float(model_cfg["card_graph_laplacian_alpha"]),
            graph_laplacian_beta=float(model_cfg["card_graph_laplacian_beta"]),
            graph_laplacian_normalized=bool(
                model_cfg.get("card_graph_laplacian_normalized", True)
            ),
            denoiser_dim=int(model_cfg["card_denoiser_dim"]),
            denoiser_depth=int(model_cfg["card_denoiser_depth"]),
            denoiser_heads=int(model_cfg["card_denoiser_heads"]),
            dropout=float(model_cfg["card_dropout"]),
            freeze_coarse=bool(model_cfg["card_freeze_coarse"]),
            cond_use_history=bool(model_cfg["card_cond_use_history"]),
            cond_use_coarse=bool(model_cfg["card_cond_use_coarse"]),
            allow_no_conditioning=bool(model_cfg["card_allow_no_conditioning"]),
            coarse_target_lowpass_only=bool(model_cfg["card_coarse_target_lowpass_only"]),
            diffusion_only=bool(model_cfg["card_diffusion_only"]),
            simlpe_use_norm=bool(model_cfg["simlpe_use_norm"]),
            simlpe_spatial_fc_only=bool(model_cfg["simlpe_use_spatial_fc_only"]),
            simlpe_mix_spatial_temporal=bool(model_cfg["simlpe_mix_spatial_temporal"]),
            simlpe_norm_axis=str(model_cfg["simlpe_norm_axis"]),
            simlpe_add_last_offset=bool(model_cfg["simlpe_add_last_offset"]),
        )
        supported_fields = {field.name for field in fields(config_type)}
        variant_cfg = config_type(
            **{key: value for key, value in config_kwargs.items() if key in supported_fields}
        )
        graph_meta = resolve_card_hand_graph_metadata(
            ds_cfg.dataset, tuple(int(idx) for idx in ds_cfg.wrist_indices)
        )
        model = forecaster_type(
            variant_cfg,
            metadata={
                "wrist_index": int(graph_meta["wrist_index"]),
                "edges": tuple((int(i), int(j)) for i, j in graph_meta["links"]),
                "empirical_feature_covariance": None,
            },
        ).to(device)
    model.eval()
    return model, variant_cfg


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be non-negative and --repeats at least 1")

    args.experiment_config = args.experiment_config.expanduser().resolve()
    args.jsonl = args.jsonl.expanduser().resolve()
    args.csv = args.csv.expanduser().resolve()
    args.markdown = args.markdown.expanduser().resolve()
    experiment_cfg = load_yaml(args.experiment_config)
    preprocessing = dict(experiment_cfg.get("preprocessing", {}))
    num_candidates = int(
        experiment_cfg.get("num_candidates", 10)
        if args.num_candidates is None
        else args.num_candidates
    )
    if num_candidates < 1:
        raise ValueError("--num-candidates must be at least 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; random-model profiling requires a visible GPU")

    purge_vendor_modules("config", "data", "datasets", "models", "runner", "train_utils")
    with prepend_sys_path(VENDOR_SPLINE):
        from data import get_dataset_metadata

        metadata = get_dataset_metadata(args.dataset)
    ds_cfg = SimpleNamespace(
        dataset=str(args.dataset),
        input_n=int(preprocessing.get("input_n", 70)),
        output_n=int(preprocessing.get("output_n", 30)),
        node_count=int(metadata.get("node_count", 21)),
        wrist_indices=tuple(int(index) for index in metadata.get("default_wrist_indices", ())),
    )
    random_generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    batch_input = torch.randn(
        args.batch_size,
        ds_cfg.input_n,
        ds_cfg.node_count,
        3,
        generator=random_generator,
        dtype=torch.float32,
    ).to(device)

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    # A comprehensive run is intentionally fresh; never mix previous rows.
    args.jsonl.unlink(missing_ok=True)
    os.environ[PROFILE_PATH_ENV] = str(args.jsonl)
    os.environ[PROFILE_WARMUP_ENV] = str(args.warmup)
    os.environ[PROFILE_REPEATS_ENV] = str(args.repeats)
    _PROFILED_KEYS.clear()

    for model_name in CARD_VARIANTS:
        torch.manual_seed(int(args.seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.seed))
        model, variant_cfg = load_variant(model_name, ds_cfg, device)

        def sample(model=model, variant_cfg=variant_cfg):
            coarse = (
                model._zero_coarse_future(batch_input)
                if variant_cfg.diffusion_only
                else model.coarse(batch_input)
            )
            candidates = [
                model.predict(
                    batch_input,
                    mamp_feat=None,
                    coarse_future=coarse,
                    deterministic=False,
                    seed=candidate_index,
                )
                for candidate_index in range(num_candidates)
            ]
            return torch.stack(candidates, dim=1)

        # The forecasters contain lazy layers whose shapes are fixed by their
        # first input. Materialize them before profile_once counts parameters.
        with torch.inference_mode():
            initialized_output = sample()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            del initialized_output

        profile_once(
            key=f"{model_name}_random_sampling",
            model_name=model_name,
            modules=[model],
            sample_fn=sample,
            sampling_steps=int(variant_cfg.ddim_steps),
            batch_size=int(batch_input.shape[0]),
            num_candidates=num_candidates,
            notes="Random initialization and synthetic batch; K serial DDIM trajectories plus one coarse forecast",
        )
        del sample, model, variant_cfg
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # DLow-CVAE always constructs its configured latent bank (nk=10) even when
    # one forecast is requested, matching the baseline's evaluation hook.
    purge_vendor_modules("models", "utils", "motion_pred")
    with prepend_sys_path(VENDOR_DLOW):
        from models.motion_pred import get_dlow_model, get_vae_model

        dlow_defaults = load_yaml(
            REPO_ROOT / "configs" / "models" / "dlow_cvae.yaml"
        )["defaults"]
        dlow_cfg = SimpleNamespace(
            nz=int(dlow_defaults.get("nz", 128)),
            nk=10,
            t_pred=int(ds_cfg.output_n),
            vae_specs=dict(dlow_defaults.get("vae_specs", {})),
            dlow_specs={
                "model_name": "NFDiag",
                "rnn_type": "gru",
                "nh_mlp": [1024, 512],
                "x_birnn": False,
            },
        )
        trajectory_dim = (int(ds_cfg.node_count) - 1) * 3
        vae = get_vae_model(dlow_cfg, trajectory_dim).to(device).eval()
        dlow = get_dlow_model(dlow_cfg, trajectory_dim).to(device).eval()
    dlow_history = batch_input[:, :, 1:, :].reshape(
        args.batch_size, ds_cfg.input_n, trajectory_dim
    ).permute(1, 0, 2).contiguous()

    def sample_dlow():
        if num_candidates > dlow_cfg.nk:
            raise ValueError(
                f"DLow supports at most {dlow_cfg.nk} candidates, got {num_candidates}"
            )
        latent_bank = dlow.sample(
            dlow_history,
            z=torch.zeros(
                (dlow_history.shape[1], dlow_cfg.nz),
                device=device,
                dtype=dlow_history.dtype,
            ),
        ).view(dlow_history.shape[1], dlow_cfg.nk, -1)
        selected_latent = latent_bank[:, :num_candidates, :].reshape(
            -1, dlow_cfg.nz
        )
        repeated_history = dlow_history.repeat_interleave(num_candidates, dim=1)
        return vae.decode(repeated_history, selected_latent)

    profile_once(
        key="dlow_cvae_random_sampling",
        model_name="dlow_cvae",
        modules=[dlow, vae],
        sample_fn=sample_dlow,
        sampling_steps=1,
        batch_size=int(args.batch_size),
        num_candidates=num_candidates,
        notes="Random initialization and synthetic batch; DLow latent generator plus CVAE decoder",
    )
    del sample_dlow, dlow, vae
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Belfusion: complete single-pass conditional VAE sampler.
    with prepend_sys_path(VENDOR_BELFUSION):
        from run_belfusion import BelFusionCVAE

        belfusion_cfg = load_yaml(
            REPO_ROOT / "configs" / "models" / "belfusion.yaml"
        )["defaults"]
        belfusion = BelFusionCVAE(
            obs_dim=ds_cfg.input_n * ds_cfg.node_count * 3,
            fut_dim=ds_cfg.output_n * ds_cfg.node_count * 3,
            hidden_dim=int(belfusion_cfg["model"]["hidden_dim"]),
            latent_dim=int(belfusion_cfg["model"]["latent_dim"]),
        ).to(device).eval()
    belfusion_input = batch_input.reshape(args.batch_size, -1)
    profile_once(
        key="belfusion_random_sampling",
        model_name="belfusion",
        modules=[belfusion],
        sample_fn=lambda: belfusion.sample(belfusion_input, num_samples=num_candidates),
        sampling_steps=1,
        batch_size=args.batch_size,
        num_candidates=num_candidates,
        notes="Random initialization and synthetic batch; single-pass CVAE sampling",
    )
    del belfusion
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # siMLPe: deterministic DCT-domain forecast (one output regardless of K).
    purge_vendor_modules("models", "config", "data", "datasets")
    with prepend_sys_path(VENDOR_SPLINE):
        from models.simlpe_dct import SiMLPeDCTConfig, SiMLPeDCTForecaster

        simlpe_cfg = load_yaml(REPO_ROOT / "configs" / "models" / "simlpe.yaml")["defaults"]
        simlpe = SiMLPeDCTForecaster(
            SiMLPeDCTConfig(
                input_length=ds_cfg.input_n,
                pred_length=ds_cfg.output_n,
                num_nodes=ds_cfg.node_count,
                hidden_dim=int(simlpe_cfg["hidden_size"]),
                num_layers=int(simlpe_cfg["gru_layers"]),
                use_norm=bool(simlpe_cfg["simlpe_use_norm"]),
                use_spatial_fc_only=bool(simlpe_cfg["simlpe_use_spatial_fc_only"]),
                mix_spatial_temporal=bool(simlpe_cfg["simlpe_mix_spatial_temporal"]),
                norm_axis=str(simlpe_cfg["simlpe_norm_axis"]),
                add_last_offset=bool(simlpe_cfg["simlpe_add_last_offset"]),
                dct_components=simlpe_cfg.get("dct_components"),
            )
        ).to(device).eval()
    profile_once(
        key="simlpe_random_sampling",
        model_name="simlpe",
        modules=[simlpe],
        sample_fn=lambda: simlpe(batch_input),
        sampling_steps=1,
        batch_size=args.batch_size,
        num_candidates=1,
        notes="Random initialization and synthetic batch; deterministic single-pass forecast",
    )
    del simlpe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # CoMusion: complete configured ancestral diffusion chain.
    purge_vendor_modules("models", "utils", "data_utils")
    with prepend_sys_path(VENDOR_COMUSION):
        from models.CoMusion import CoMusion
        from models.GaussianDiffusion import GaussianDiffusion

        comusion_yaml = load_yaml(
            REPO_ROOT / "configs" / "models" / "comusion.yaml"
        )["defaults"]
        cm = comusion_yaml["model_specs"]
        cd = comusion_yaml["diff_specs"]
        core = CoMusion(
            input_n=ds_cfg.input_n,
            output_n=ds_cfg.output_n,
            dct_dim=int(cm["dct_dim"]),
            gcn_dim=int(cm["gcn_dim"]),
            gcn_drop=float(cm["gcn_drop"]),
            inner_stage=int(cm["inner_stage"]),
            outer_stage=int(cm["outer_stage"]),
            node_n=ds_cfg.node_count * 3,
            trans_dim=int(cm["trans_dim"]),
            trans_drop=float(cm["trans_drop"]),
            trans_ff_dim=int(cm["trans_ff_dim"]),
            trans_num_heads=int(cm["trans_num_heads"]),
            trans_num_layers=int(cm["trans_num_layers"]),
        )
        comusion = GaussianDiffusion(
            core,
            SimpleNamespace(),
            future_motion_size=(ds_cfg.output_n, ds_cfg.node_count * 3),
            timesteps=int(cd["diffuse_steps"]),
            loss_type=str(cd["loss_type"]),
            objective=str(cd["objective"]),
            beta_schedule=str(cd["beta_schedule"]),
            history_weight=1.0,
            future_weight=1.0,
        ).to(device).eval()
    comusion_history = batch_input.reshape(args.batch_size, ds_cfg.input_n, -1)
    comusion_history = comusion_history.repeat_interleave(num_candidates, dim=0)
    profile_once(
        key="comusion_random_sampling",
        model_name="comusion",
        modules=[comusion],
        sample_fn=lambda: comusion.sample(
            comusion_history,
            None,
            batch_size=comusion_history.shape[0],
            clip_denoised=False,
            uncond=True,
        ),
        sampling_steps=int(comusion.num_timesteps),
        batch_size=args.batch_size,
        num_candidates=num_candidates,
        notes="Random initialization and synthetic batch; complete ancestral diffusion chain",
    )
    del comusion, core
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # HumanMAC: K complete configured DDIM trajectories.
    purge_vendor_modules("models", "utils", "data_loader", "config")
    with prepend_sys_path(VENDOR_HUMANMAC):
        from utils.script import create_model_and_diffusion, sample_preprocessing
        from utils.util import generate_pad, get_dct_matrix

        human_yaml = load_yaml(
            REPO_ROOT / "configs" / "models" / "humanmac.yaml"
        )["defaults"]
        human_cfg = SimpleNamespace(
            device=device,
            joint_num=ds_cfg.node_count - 1,
            n_pre=int(human_yaml["n_pre"]),
            num_layers=int(human_yaml["num_layers"]),
            num_heads=int(human_yaml["num_heads"]),
            latent_dims=int(human_yaml["latent_dims"]),
            dropout=float(human_yaml["dropout"]),
            noise_steps=int(human_yaml["noise_steps"]),
            padding=str(human_yaml["padding"]),
            Complete=bool(human_yaml["Complete"]),
            ddim_timesteps=int(human_yaml["ddim_timesteps"]),
            scheduler=str(human_yaml["scheduler"]),
            mod_test=float(human_yaml["mod_test"]),
            t_his=ds_cfg.input_n,
            t_pred=ds_cfg.output_n,
        )
        human_cfg.idx_pad, human_cfg.zero_index = generate_pad(
            human_cfg.padding, human_cfg.t_his, human_cfg.t_pred
        )
        human_cfg.dct_m_all, human_cfg.idct_m_all = get_dct_matrix(
            human_cfg.t_his + human_cfg.t_pred
        )
        human_cfg.dct_m_all = human_cfg.dct_m_all.float().to(device)
        human_cfg.idct_m_all = human_cfg.idct_m_all.float().to(device)
        human_model, human_diffusion = create_model_and_diffusion(human_cfg)
    human_history = batch_input[:, :, 1:, :].reshape(
        args.batch_size, ds_cfg.input_n, -1
    )
    human_mode, human_dct, human_dct_condition = sample_preprocessing(
        human_history, human_cfg, mode="metrics"
    )

    def sample_humanmac():
        return torch.stack(
            [
                human_diffusion.sample_ddim(
                    human_model, human_dct, human_dct_condition, human_mode
                )
                for _ in range(num_candidates)
            ],
            dim=1,
        )

    profile_once(
        key="humanmac_random_sampling",
        model_name="humanmac",
        modules=[human_model],
        sample_fn=sample_humanmac,
        sampling_steps=int(human_diffusion.ddim_timesteps),
        batch_size=args.batch_size,
        num_candidates=num_candidates,
        notes="Random initialization and synthetic batch; K complete DDIM trajectories",
    )
    del sample_humanmac, human_model, human_diffusion
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # GSPS: full generator and inverse-DCT reconstruction.
    purge_vendor_modules("models", "utils")
    with prepend_sys_path(VENDOR_GSPS):
        from models.motion_pred_ours import get_model as get_gsps_model
        from run_gsps import _sample_predictions as sample_gsps_predictions
        from utils.util import get_dct_matrix as get_gsps_dct

        gsps_yaml = load_yaml(REPO_ROOT / "configs" / "models" / "gsps.yaml")
        gm = gsps_yaml["defaults"]["model"]
        n_pre = int(gm["n_pre"])
        nz = int(gm["nz"])
        parts = [list(map(int, part)) for part in gm["parts"]]
        gsps_cfg = SimpleNamespace(
            n_pre=n_pre,
            nf_specs={
                "hidden_dim": int(gm["hidden_dim"]),
                "num_stage": int(gm["num_stage"]),
                "parts": parts,
                "nz": nz,
                "num_flow_layer": int(gm["num_flow_layer"]),
            },
        )
        gsps, _pose_prior = get_gsps_model(
            gsps_cfg, traj_dim=ds_cfg.node_count - 1, model_type="h36m"
        )
        gsps = gsps.to(device).eval()
        gsps_dct, gsps_idct = get_gsps_dct(ds_cfg.input_n + ds_cfg.output_n)
        gsps_dct = gsps_dct.float().to(device)
        gsps_idct = gsps_idct.float().to(device)
    gsps_history = batch_input[:, :, 1:, :]
    gsps_padded = torch.cat(
        [
            gsps_history,
            gsps_history[:, -1:].expand(-1, ds_cfg.output_n, -1, -1),
        ],
        dim=1,
    ).reshape(args.batch_size, ds_cfg.input_n + ds_cfg.output_n, -1)
    gsps_input = torch.matmul(gsps_dct[:n_pre], gsps_padded).transpose(1, 2)
    gsps_input = gsps_input.reshape(
        args.batch_size, ds_cfg.node_count - 1, 3, n_pre
    ).reshape(args.batch_size, ds_cfg.node_count - 1, -1)

    profile_once(
        key="gsps_random_sampling",
        model_name="gsps",
        modules=[gsps],
        sample_fn=lambda: sample_gsps_predictions(
            gsps,
            inp_dct=gsps_input,
            idct_m=gsps_idct,
            t_his=ds_cfg.input_n,
            t_pred=ds_cfg.output_n,
            nk=num_candidates,
            nz=nz,
            n_parts=len(parts),
            seed=int(args.seed),
            cartesian=False,
            return_total=False,
        ),
        sampling_steps=1,
        batch_size=args.batch_size,
        num_candidates=num_candidates,
        notes="Random initialization and synthetic batch; single-pass GSPS generator",
    )
    del gsps, _pose_prior
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # SkeletonDiffusion: random graph autoencoder, latent diffusion, and decoder.
    purge_vendor_modules("models", "utils", "src")
    with prepend_sys_path(VENDOR_SKELETONDIFFUSION):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from src.core import DiffusionManager
        from src.data import create_skeleton
        from src.inference_utils import create_model as create_skeleton_model

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        skeleton_config_root = VENDOR_SKELETONDIFFUSION / "configs"
        with initialize_config_dir(
            config_dir=str((skeleton_config_root / "config_train_autoencoder").resolve()),
            version_base="1.3",
        ):
            auto_cfg = compose(
                config_name="config", overrides=["task=assembly_hmp"]
            )
        with initialize_config_dir(
            config_dir=str((skeleton_config_root / "config_train_diffusion").resolve()),
            version_base="1.3",
        ):
            diffusion_cfg = compose(config_name="config")
        OmegaConf.set_struct(auto_cfg, False)
        skeleton_cfg = OmegaConf.to_container(
            OmegaConf.merge(auto_cfg, diffusion_cfg), resolve=True
        )
        for section in ("model", "task", "dataset", "cov_matrix"):
            if section in skeleton_cfg:
                skeleton_cfg.update(skeleton_cfg.pop(section))
        skeleton_cfg["device"] = device
        skeleton_cfg["obs_length"] = ds_cfg.input_n
        skeleton_cfg["pred_length"] = ds_cfg.output_n
        skeleton_cfg["prediction_horizon"] = ds_cfg.output_n
        skeleton_cfg["prediction_horizon_train"] = ds_cfg.output_n
        skeleton_cfg["prediction_horizon_eval"] = ds_cfg.output_n

        skeleton = create_skeleton(**skeleton_cfg)
        skeleton_autoencoder = create_skeleton_model(
            node_types=skeleton.nodes_type_id,
            num_nodes=skeleton.num_nodes,
            **skeleton_cfg,
            **skeleton_cfg["autoenc_arch"],
        ).to(device).eval()
        skeleton_manager = DiffusionManager(
            skeleton=skeleton,
            num_nodes=skeleton.num_nodes,
            node_types=skeleton.nodes_type_id,
            **skeleton_cfg,
        )
        skeleton_diffusion = skeleton_manager.get_diffusion().to(device).eval()

    def sample_skeletondiffusion():
        past = skeleton_autoencoder.get_past_embedding(batch_input)
        repeated_past = past.repeat_interleave(num_candidates, dim=0)
        latent, _ = skeleton_diffusion.sample(
            batch_size=args.batch_size * num_candidates,
            x_cond=repeated_past,
        )
        repeated_input = batch_input.repeat_interleave(num_candidates, dim=0)
        return skeleton_autoencoder.decode(
            repeated_input,
            latent,
            repeated_past,
            ph=ds_cfg.output_n,
        ).reshape(
            args.batch_size,
            num_candidates,
            ds_cfg.output_n,
            ds_cfg.node_count,
            3,
        )

    profile_once(
        key="skeletondiffusion_random_sampling",
        model_name="skeletondiffusion",
        modules=[skeleton_autoencoder, skeleton_diffusion],
        sample_fn=sample_skeletondiffusion,
        sampling_steps=int(skeleton_diffusion.sampling_timesteps),
        batch_size=args.batch_size,
        num_candidates=num_candidates,
        notes="Random initialization and synthetic batch; latent diffusion plus autoencoder decoder",
    )
    del sample_skeletondiffusion, skeleton_autoencoder, skeleton_diffusion
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from profile_all_models import _read_records

    records = _read_records(args.jsonl)
    expected_models = {
        "card", "star_hand", "fihard", "dlow_cvae", "belfusion",
        "simlpe", "comusion", "humanmac", "gsps", "skeletondiffusion",
    }
    recorded_models = {str(record.get("model", "")) for record in records}
    missing_models = sorted(expected_models - recorded_models)
    if missing_models:
        raise RuntimeError(
            "Comprehensive profile is incomplete; missing: "
            + ", ".join(missing_models)
        )
    write_outputs(records, args.csv, args.markdown)
    print(f"[PROFILE] JSONL: {args.jsonl}")
    print(f"[PROFILE] CSV: {args.csv}")
    print(f"[PROFILE] Markdown: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
