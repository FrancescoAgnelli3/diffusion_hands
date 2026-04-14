import copy
import math
import os
import sys
import time
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = ["train"]

# Keep hard dependencies minimal for twostage-only runtime.
from models.simlpe_dct import SiMLPeDCTConfig, SiMLPeDCTForecaster
from models.two_stage_dct_diffusion import TwoStageDCTDiffusionConfig, TwoStageDCTDiffusionForecaster

import utils as U

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from common.evaluation import humanmac_metrics_prefixed as _common_humanmac_metrics_prefixed  # type: ignore

# Minimal hand graph helpers kept for twostage runtime compatibility.
_HAND_BONE_LINK_1 = []
_HAND_BONE_LINK_2 = []


def _HAND_BUILD_UNDIRECTED(node_num: int, links):
    graph = {i: [] for i in range(node_num)}
    for a, b in links:
        if 0 <= a < node_num and 0 <= b < node_num:
            graph[a].append(b)
            graph[b].append(a)
    return graph


def _HAND_BUILD_TREE(node_num: int, links, wrists_hint=None):
    wrists = list(wrists_hint or [])
    return None, None, wrists, None


def _compute_humanmac_metrics(
    pred_candidates: torch.Tensor,
    gt_future: torch.Tensor,
    start_pose: torch.Tensor,
    *,
    threshold: float,
) -> Dict[str, float]:
    return _common_humanmac_metrics_prefixed(
        pred_candidates=pred_candidates,
        gt_future=gt_future,
        start_pose=start_pose,
        threshold=threshold,
    )

def _print_model_parameters(model_or_modules, title: str = "Model") -> int:
    """Calculate and print the number of trainable parameters.

    Accepts a single `nn.Module` or an iterable of modules. Returns the total
    count of trainable parameters.
    """
    def count_params(m) -> int:
        total = 0
        for p in m.parameters():
            if not getattr(p, "requires_grad", False):
                continue
            try:
                total += int(p.numel())
            except Exception:
                # Skip uninitialized lazy parameters; they initialize on first forward pass.
                continue
        return total

    if isinstance(model_or_modules, (list, tuple)):
        total = sum(count_params(m) for m in model_or_modules)
    else:
        total = count_params(model_or_modules)

    # Human-friendly formatting
    if total >= 1_000_000:
        pretty = f"{total/1_000_000:.2f}M"
    elif total >= 1_000:
        pretty = f"{total/1_000:.2f}K"
    else:
        pretty = str(total)

    print(f"{title} parameters: {total} ({pretty})")
    return total

def train(
    config: dict,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    bone_loss_weight: float = 0.0,
    model: str = "mpt",
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    log_wandb: bool = False,
    wandb_run: Optional[Any] = None,
):
    device = U.device()

    raw_model_name = str(model or "").strip()
    model_key = raw_model_name.lower()
    if not model_key:
        model_key = "twostage_dct_diffusion"
    if model_key not in {"simlpe_dct", "twostage_dct_diffusion"}:
        raise ValueError(
            f"Unsupported model '{model_key}'. Available models in this pruned repository: "
            "simlpe_dct, twostage_dct_diffusion."
        )
    model = model_key

    log_wandb = bool(log_wandb or config.get("log_wandb", False))
    wandb_handle = wandb_run
    if log_wandb:
        try:
            import wandb  # type: ignore
        except ImportError:
            print("Weights & Biases requested for logging but package is unavailable; disabling logging.")
            log_wandb = False
        else:
            if wandb_handle is None:
                wandb_handle = wandb.run
            if wandb_handle is None:
                print("Weights & Biases logging requested without an active run; call wandb.init() before train(). Disabling logging.")
                log_wandb = False

    def _maybe_add_metric(bucket: Dict[str, float], key: str, value: Optional[float]) -> None:
        if value is None:
            return
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(scalar):
            bucket[key] = scalar

    save_eval_examples = bool(config.get("save_eval_examples", False))
    save_eval_examples_all_k = bool(config.get("save_eval_examples_all_k", False))
    save_coarse_model = bool(config.get("save_coarse_model", False))
    maybe_save_coarse_model = None
    params_count = 0
    node_num = int(config.get("node_num", 21))
    input_n = int(config.get("input_n", 90))
    output_n = int(config.get("output_n", 42))
    hidden_size = int(config.get("hidden_size", 128))
    t_layers = int(config.get("t_layers", 2))
    gnn_layers = int(config.get("gnn_layers", 2))
    batch_size = int(config.get("batch_size", 16))
    space = bool(config.get("use_space", True))
    gru_layers = int(config.get("gru_layers", 4))
    train_epochs = int(epochs if epochs is not None else config.get("train_epoches", 50))
    if model == "eqmotion" or model == "pgbig":
        epochs = 50
    if model == "estag":
        epochs =5
    learning_rate = float(lr if lr is not None else config.get("learning_rate", 1e-3))
    twostage_diffusion_epochs = int(config.get("twostage_diffusion_epochs", 0))
    twostage_eval_best_of_k = max(1, int(config.get("twostage_eval_best_of_k", 1)))
    twostage_mamp_checkpoint = str(config.get("twostage_mamp_checkpoint", "")).strip()
    twostage_use_mamp_condition = bool(config.get("twostage_use_mamp_condition", bool(twostage_mamp_checkpoint)))
    twostage_use_mamp_condition_coarse = bool(config.get("twostage_use_mamp_condition_coarse", False))
    twostage_use_any_mamp_condition = bool(twostage_use_mamp_condition or twostage_use_mamp_condition_coarse)
    twostage_mamp_mask_ratio = float(config.get("twostage_mamp_mask_ratio", 0.0))
    twostage_mamp_motion_aware_tau = float(config.get("twostage_mamp_motion_aware_tau", 0.80))
    twostage_mamp_repo_root = str(config.get("twostage_mamp_repo_root", "/home/agnelli/projects/MAMP")).strip()
    dct_keep_coeffs_cfg = config.get("dct_keep_coeffs")
    dct_keep_coeffs = int(dct_keep_coeffs_cfg) if dct_keep_coeffs_cfg is not None else None
    dual_prefix = "SFgDNet" if model == "SFgDNet" else "SplineEqNet" if model == "SplineEqNet" else None
    mask_save_path = config.get(f"{dual_prefix}_mask_path") if dual_prefix else None
    mask_logging_enabled = bool(mask_save_path) and dual_prefix is not None
    mask_epoch_history: List[torch.Tensor] = []
    gate_save_path = config.get(f"{dual_prefix}_gate_path") if dual_prefix else None
    gate_logging_enabled = bool(gate_save_path) and dual_prefix is not None
    gate_param_save_path = config.get(f"{dual_prefix}_gate_param_path") if dual_prefix else None
    gate_param_logging_enabled = bool(gate_param_save_path) and dual_prefix is not None
    gate_epoch_history: List[torch.Tensor] = []
    gate_param_snapshot: Optional[Dict[str, torch.Tensor]] = None
    velocity_loss_weight = float(config.get("velocity_loss_weight", 0.0))
    negpearson_weight = float(config.get("SplineEqNet_negpearson_weight", 0.0))

    adjacency_cfg = config.get("adjacency", ())
    adjacency_tensor = None
    if adjacency_cfg:
        adjacency_tensor = torch.tensor(adjacency_cfg, dtype=torch.float32)
    else:
        adjacency_tensor = torch.eye(node_num, dtype=torch.float32)
    if adjacency_tensor.ndim != 2:
        raise ValueError("Adjacency matrix must be a 2D square array.")
    if adjacency_tensor.shape[0] != node_num or adjacency_tensor.shape[1] != node_num:
        adj_resized = torch.zeros((node_num, node_num), dtype=torch.float32)
        rows = min(node_num, adjacency_tensor.shape[0])
        cols = min(node_num, adjacency_tensor.shape[1])
        adj_resized[:rows, :cols] = adjacency_tensor[:rows, :cols]
        adjacency_tensor = adj_resized
    adjacency_tensor = adjacency_tensor.clone().detach()
    adjacency_tensor = 0.5 * (adjacency_tensor + adjacency_tensor.transpose(0, 1))
    diag_indices = torch.arange(node_num)
    adjacency_tensor[diag_indices, diag_indices] = 1.0

    edge_index_cfg = config.get("edge_index", ())
    edge_pairs: List[Tuple[int, int]] = []
    for pair in edge_index_cfg:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            i, j = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if 0 <= i < node_num and 0 <= j < node_num:
            edge_pairs.append((i, j))
    if not edge_pairs:
        nz = adjacency_tensor.nonzero(as_tuple=False)
        edge_pairs = [(int(i), int(j)) for i, j in nz]

    wrists_hint_cfg = config.get("wrists_hint")
    wrists_hint: Optional[List[int]] = None
    if wrists_hint_cfg is not None:
        if isinstance(wrists_hint_cfg, str):
            raw_tokens = wrists_hint_cfg.replace(";", ",").split(",")
            parsed: List[int] = []
            for token in raw_tokens:
                token = token.strip()
                if not token:
                    continue
                try:
                    parsed.append(int(float(token)))
                except ValueError:
                    continue
            filtered = [w for w in parsed if 0 <= w < node_num]
            wrists_hint = filtered if filtered else None
        elif isinstance(wrists_hint_cfg, (list, tuple)):
            parsed: List[int] = []
            for token in wrists_hint_cfg:
                try:
                    parsed.append(int(float(token)))
                except (ValueError, TypeError):
                    continue
            filtered = [w for w in parsed if 0 <= w < node_num]
            wrists_hint = filtered if filtered else None

    wrist_links = _HAND_BONE_LINK_1 + _HAND_BONE_LINK_2
    try:
        wrist_graph = _HAND_BUILD_UNDIRECTED(node_num, wrist_links)
    except Exception:
        wrist_graph = {i: [] for i in range(node_num)}
    try:
        _, _, wrists_auto, _ = _HAND_BUILD_TREE(node_num, wrist_links, wrists_hint)
    except Exception:
        wrists_auto = []
    wrists_candidates: List[int] = []
    if wrists_hint:
        wrists_candidates.extend(wrists_hint)
    wrists_candidates.extend(int(w) for w in wrists_auto if 0 <= int(w) < node_num)

    seen_wrists = set()
    ordered_wrists: List[int] = []
    for wrist_id in wrists_candidates:
        if wrist_id in seen_wrists:
            continue
        seen_wrists.add(wrist_id)
        ordered_wrists.append(wrist_id)

    fallback_wrists = [idx for idx in (5, 26) if idx < node_num]
    if not ordered_wrists:
        ordered_wrists = fallback_wrists or [0]

    node2wrist_map: List[int] = [-1] * node_num
    for root in ordered_wrists:
        stack = [root]
        while stack:
            node = stack.pop()
            if node2wrist_map[node] != -1:
                continue
            node2wrist_map[node] = root
            for neigh in wrist_graph.get(node, []):
                if 0 <= neigh < node_num and node2wrist_map[neigh] == -1:
                    stack.append(neigh)
    for idx, owner in enumerate(node2wrist_map):
        if owner == -1:
            node2wrist_map[idx] = ordered_wrists[0]
    node2wrist_tensor = torch.tensor(node2wrist_map, dtype=torch.long, device=device)

    def _delta_to_absolute(pred: torch.Tensor, last_frame: torch.Tensor) -> torch.Tensor:
        pred = pred.clone()
        pred[:, 0, :] = pred[:, 0, :] + last_frame
        return torch.cumsum(pred, dim=1)

    def _velocity_magnitude(future_seq: torch.Tensor, last_observed: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep velocity magnitudes for future sequence.

        Args:
            future_seq: Tensor shaped (B, T_out, N, 3) with predicted positions.
            last_observed: Tensor shaped (B, N, 3) representing final observed pose.

        Returns:
            Tensor shaped (B, T_out, N) containing velocity magnitudes.
        """
        if future_seq.ndim != 4 or last_observed.ndim != 3:
            raise ValueError("Invalid shapes for velocity magnitude computation.")
        prev = torch.cat([last_observed.unsqueeze(1), future_seq[:, :-1]], dim=1)
        diffs = future_seq - prev
        return torch.norm(diffs, dim=-1)

    loader = train_loader

    if edge_pairs:
        edge_index_tensor_device = torch.tensor(edge_pairs, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index_tensor_device = torch.zeros((2, 0), dtype=torch.long, device=device)
    graph_edges_no_self = [(i, j) for (i, j) in edge_pairs if i != j]
    if graph_edges_no_self:
        edges_t = torch.tensor(graph_edges_no_self, dtype=torch.long, device=device)
    else:
        edges_t = torch.zeros((0, 2), dtype=torch.long, device=device)

    # Align model batch_size with the actual train loader to avoid GRU hidden-state mismatches
    if hasattr(loader, "batch_size") and loader.batch_size is not None:
        try:
            batch_size = int(loader.batch_size)
        except Exception:
            pass

    # Build models and optimizer
    simlpe = None
    sfgdnet_model = None
    SplineEqNet_model = None
    physmamba_model = None
    params: List[torch.nn.Parameter] = []
    eqmotion_model = None
    trajdep = None
    pgbig_model = None
    motion_mixer = None
    factorized_loss = None
    twostage_model = None
    twostage_phase: Optional[str] = None
    twostage_diffusion_lr = float(learning_rate) * 0.1
    twostage_train_coarse_in_diffusion = False
    twostage_diffusion_coarse_warmup_epochs = 0
    twostage_coarse_diffusion_group_added = False
    mamp_encoder = None
    mamp_num_frames = input_n
    contrastive_model = None
    contrastive_path = str(config.get("contrastive_score_path", "") or "")
    contrastive_use = bool(config.get("contrastive_score_use", True))

    def _load_contrastive_model():
        nonlocal contrastive_model
        if not contrastive_use:
            return None
        if contrastive_model is not None:
            return contrastive_model

        dataset_name = str(config.get("dataset", "") or "").strip()
        action_segment = str(config.get("action_filter", "") or "all")
        action_segment = action_segment.replace(os.sep, "_").replace(" ", "_")
        default_name = f"contrastive_score_{dataset_name}_{action_segment}.pt" if dataset_name else "contrastive_score.pt"
        default_path = os.path.join(os.path.dirname(__file__), "examples", "models", default_name)
        load_path = contrastive_path or default_path

        if not os.path.exists(load_path):
            raise RuntimeError(
                "Contrastive scorer checkpoint not found. "
                f"Expected at {load_path}. "
                "Provide it via config['contrastive_score_path'] or train the model with train_contrastive_score.py."
            )

        checkpoint = torch.load(load_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state = checkpoint["model_state"]
            dataset_cfg = checkpoint.get("dataset_cfg", {}) or {}
            n_joints = int(dataset_cfg.get("node_count", 21))
        else:
            state = checkpoint
            n_joints = int(config.get("node_num", 21))

        embed_dim = None
        for key in ("x_enc.proj.3.weight", "x_enc.proj.2.weight", "x_enc.proj.1.weight"):
            weight = state.get(key)
            if weight is not None and hasattr(weight, "shape"):
                embed_dim = int(weight.shape[0])
                break
        if embed_dim is None:
            raise RuntimeError("Unable to infer contrastive embed_dim from checkpoint state dict.")

        from models.contrastive.contrastive_score import ContrastiveScorer

        contrastive_model = ContrastiveScorer(n_joints=n_joints, embed_dim=embed_dim, dropout=0.0, share_encoders=False)
        missing, unexpected = contrastive_model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Warn] Contrastive scorer missing keys: {missing}")
        if unexpected:
            print(f"[Warn] Contrastive scorer unexpected keys: {unexpected}")
        contrastive_model = contrastive_model.to(device)
        contrastive_model.eval()
        return contrastive_model
    def _parse_fixed_centers_cfg(raw_value: Any) -> Optional[Tuple[float, float]]:
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            tokens = [tok.strip() for tok in raw_value.replace("|", ",").split(",") if tok.strip()]
            if len(tokens) >= 2:
                return float(tokens[0]), float(tokens[1])
        elif isinstance(raw_value, (list, tuple)):
            if len(raw_value) >= 2:
                return float(raw_value[0]), float(raw_value[1])
        return None

    def _build_dual_branch_model(
        prefix: str,
        config_cls,
        forecaster_cls,
        label: str,
    ) -> torch.nn.Module:
        nonlocal params, params_count, optimizer
        slow_layers = config.get("simlpe_slow_layers")
        fast_layers = config.get("simlpe_fast_layers")
        force_mlp_padding_cfg = config.get(f"{prefix}_force_mlp_padding")
        force_mlp_padding_flag = True if force_mlp_padding_cfg is None else bool(force_mlp_padding_cfg)
        padding_mode_cfg = config.get(f"{prefix}_padding_mode")
        padding_predictor_cfg = config.get(f"{prefix}_padding_predictor")
        padding_full_horizon_cfg = config.get(f"{prefix}_padding_full_horizon")
        learn_lambda_cfg = config.get(f"{prefix}_learn_lambda")
        learn_centers_cfg = config.get(f"{prefix}_learn_centers")
        learn_lambda_flag = True if learn_lambda_cfg is None else bool(learn_lambda_cfg)
        learn_centers_flag = True if learn_centers_cfg is None else bool(learn_centers_cfg)
        fixed_lambda_cfg = config.get(f"{prefix}_fixed_lambda")
        fixed_lambda_value = None if fixed_lambda_cfg is None else float(fixed_lambda_cfg)
        fixed_centers_cfg = config.get(f"{prefix}_fixed_centers")
        fixed_centers_value = _parse_fixed_centers_cfg(fixed_centers_cfg)
        gate_basis_degree_cfg = config.get(f"{prefix}_gate_basis_degree")
        gate_basis_size_cfg = config.get(f"{prefix}_gate_basis_size")
        num_gates_cfg = config.get(f"{prefix}_num_gates")
        num_layers_cfg = config.get(f"{prefix}_num_layers")
        routing_cfg = config.get(f"{prefix}_routing")
        sim_cfg = config_cls(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            use_norm=bool(config.get("simlpe_use_norm", True)),
            use_spatial_fc_only=bool(config.get("simlpe_spatial_fc_only", False)),
            mix_spatial_temporal=bool(config.get("simlpe_mix_spatial_temporal", False)),
            norm_axis=str(config.get("simlpe_norm_axis", "spatial")),
            add_last_offset=bool(config.get("simlpe_add_last_offset", True)),
            dct_components=dct_keep_coeffs,
            mask_temperature=float(config.get("simlpe_mask_temperature", 1)),
            slow_num_layers=None if slow_layers is None else int(slow_layers),
            fast_num_layers=None if fast_layers is None else int(fast_layers),
            mid_gate_max=float(config.get("simlpe_mid_gate_max", 0.9)),
            force_mlp_padding=force_mlp_padding_flag,
            padding_mode=str(padding_mode_cfg) if padding_mode_cfg is not None else None,
            padding_predictor=str(padding_predictor_cfg) if padding_predictor_cfg is not None else "mlp",
            padding_full_horizon=bool(padding_full_horizon_cfg) if padding_full_horizon_cfg is not None else False,
            learn_lambda=learn_lambda_flag,
            learn_centers=learn_centers_flag,
            fixed_lambda=fixed_lambda_value,
            fixed_centers=fixed_centers_value,
        )
        if gate_basis_degree_cfg is not None:
            setattr(sim_cfg, "gate_basis_degree", int(gate_basis_degree_cfg))
        if gate_basis_size_cfg is not None:
            setattr(sim_cfg, "gate_basis_size", int(gate_basis_size_cfg))
        if num_gates_cfg is not None:
            setattr(sim_cfg, "num_gates", int(num_gates_cfg))
        if num_layers_cfg is not None:
            setattr(sim_cfg, "num_layers", int(num_layers_cfg))
        if routing_cfg is not None:
            setattr(sim_cfg, "routing", str(routing_cfg))
        model_instance = forecaster_cls(sim_cfg).to(device)
        if bool(config.get(f"{prefix}_skip_dct", False)):
            model_instance.set_skip_dct(True)
        fixed_mask_cfg = config.get(f"{prefix}_fixed_mask")
        if fixed_mask_cfg is not None:
            model_instance.set_fixed_mask(float(fixed_mask_cfg))
        fixed_mid_gate_cfg = config.get(f"{prefix}_fixed_mid_gate")
        if fixed_mid_gate_cfg is not None:
            model_instance.set_fixed_mid_gate(float(fixed_mid_gate_cfg))
        if fixed_lambda_value is not None:
            model_instance.set_fixed_lambda(fixed_lambda_value, freeze=not learn_lambda_flag)
        if fixed_centers_value is not None:
            model_instance.set_fixed_centers(fixed_centers_value[0], fixed_centers_value[1], freeze=not learn_centers_flag)
        disable_slow = bool(config.get(f"{prefix}_disable_slow_branch", False))
        disable_fast = bool(config.get(f"{prefix}_disable_fast_branch", False))
        if disable_slow or disable_fast:
            model_instance.set_branch_usage(not disable_slow, not disable_fast)
        if bool(config.get(f"{prefix}_freeze_gate", False)):
            model_instance.freeze_frequency_gate(True)
        params = list(model_instance.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(model_instance, title=label)
        return model_instance

    if model == "mpt":
        gen_x, gen_y, gen_z, gen_v = _build_mpt_generators(
            input_n - 1,
            hidden_size,
            output_n,
            node_num,
            batch_size,
            device,
            adjacency=adjacency_tensor,
            edge_index=tuple(edge_pairs),
            space=space,
            gru_layers=gru_layers,
        )
        params = list(gen_x.parameters()) + list(gen_y.parameters()) + list(gen_z.parameters()) + list(gen_v.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        # Print parameter counts for each generator and total
        params_count = _print_model_parameters([gen_x, gen_y, gen_z, gen_v], title="MPT Generators (x,y,z,v)")
    elif model in ("hisrep", "hisrep_hand", "hisrep_rigidwrist"):
        # HisRep variants: AttModel-based forecasters operating on wrist/body sequences
        if model == "hisrep_rigidwrist":
            dct_keep = config.get("dct_keep_coeffs")
            dct_n = int(dct_keep) if dct_keep is not None else 10
            wrist_cfg = hisrep_rigidwrist_module.WristOnlyConfig(
                num_nodes=node_num,
                input_length=input_n,
                pred_length=output_n,
                hidden_dim=hidden_size,
                num_stages=gru_layers,
                dct_n=dct_n,
                wrists_hint=config.get("wrists_hint"),
            )
            hisrep = hisrep_rigidwrist_module.WristRigidForecaster(wrist_cfg).to(device)
            tag = "WristRigidForecaster"
        else:
            module = hisrep_module if model == "hisrep" else hisrep_handmodule
            hisrep_kwargs = {
                "num_nodes": node_num,
                "input_length": input_n,
                "pred_length": output_n,
                "hidden_dim": hidden_size,
                "num_stages": gru_layers,
            }
            wrists_hint = config.get("wrists_hint")
            if wrists_hint is not None:
                hisrep_kwargs["wrists_hint"] = wrists_hint
            if model == "hisrep_hand":
                dct_override = config.get("hisrep_handdct_n")
                dropout_cfg = config.get("hisrep_handdropout")
                if dropout_cfg is not None:
                    hisrep_kwargs["dropout"] = float(dropout_cfg)
                token_hidden_cfg = config.get("hisrep_handtoken_hidden")
                if token_hidden_cfg is not None:
                    hisrep_kwargs["token_hidden"] = int(token_hidden_cfg)
                channel_hidden_cfg = config.get("hisrep_handchannel_hidden")
                if channel_hidden_cfg is not None:
                    hisrep_kwargs["channel_hidden"] = int(channel_hidden_cfg)
                gcn_hidden_cfg = config.get("hisrep_handgcn_hidden")
                if gcn_hidden_cfg is not None:
                    hisrep_kwargs["gcn_hidden"] = int(gcn_hidden_cfg)
                input_scale_cfg = config.get("hisrep_handinput_scale")
                if input_scale_cfg is not None:
                    hisrep_kwargs["input_scale"] = float(input_scale_cfg)
            else:
                dct_override = config.get("hisrep_dct_n")
            if dct_override is not None:
                hisrep_kwargs["dct_n"] = int(dct_override)
            his_cfg = module.HisRepConfig(**hisrep_kwargs)
            hisrep = module.HisRepForecaster(his_cfg).to(device)
            tag = "HisRepForecaster" if model == "hisrep" else "HisRepHandForecaster"
        params = list(hisrep.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(hisrep, title=tag)
        if model == "hisrep_hand":
            factorized_loss = hisrep_handmodule.FactorizedLoss().to(device)
            hand_mask = torch.zeros(node_num, device=device)
            hand_mask[hisrep.distal_idx] = 1.0
            hisrep_handdir_mask = hand_mask
    elif model == "trajdep":
        traj_cfg = TrajDepConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            num_stages=max(1, gnn_layers),
            dropout=float(config.get("trajdep_dropout", 0.5)),
            dct_components=int(config.get("trajdep_dct", min(input_n + output_n, 35))),
        )
        trajdep = TrajDepForecaster(traj_cfg).to(device)
        params = list(trajdep.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(trajdep, title="TrajDepGCN")
    elif model == "pgbig":
        cfg = PGBIGConfig(
            input_n=input_n,
            output_n=output_n,
            in_features=node_num * 3,
            d_model=hidden_size,
            dct_n=min(int(config.get("pgbig_dct", input_n + output_n)), input_n + output_n),
            drop_out=float(config.get("pgbig_dropout", 0.3)),
            encoder_layers=max(1, int(config.get("pgbig_encoder_layers", 1))),
            decoder_layers=max(1, int(config.get("pgbig_decoder_layers", 2))),
        )
        pgbig_model = PGBIGMultiStageModel(cfg).to(device)
        params = list(pgbig_model.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(pgbig_model, title="PGBIGMultiStageModel")
    elif model == "motionmixer":
        mixer_blocks = int(config.get("mixer_blocks", 4))
        mixer_tokens_dim = int(config.get("mixer_tokens_dim", hidden_size))
        mixer_channels_dim = int(config.get("mixer_channels_dim", hidden_size))
        mixer_regularization = float(config.get("mixer_regularization", 0.0))
        mixer_use_se = bool(config.get("mixer_use_se", True))
        mixer_activation = str(config.get("mixer_activation", "gelu"))
        mixer_block_type = str(config.get("mixer_block_type", "normal"))
        mixer_reduction = int(config.get("mixer_reduction", 4))
        mixer_use_max_pool = bool(config.get("mixer_use_max_pooling", False))
        motion_mixer = MlpMixer(
            num_classes=node_num * 3,
            num_blocks=mixer_blocks,
            hidden_dim=hidden_size,
            tokens_mlp_dim=mixer_tokens_dim,
            channels_mlp_dim=mixer_channels_dim,
            seq_len=input_n,
            pred_len=output_n,
            activation=mixer_activation,
            mlp_block_type=mixer_block_type,
            regularization=mixer_regularization,
            input_size=node_num * 3,
            reduction=mixer_reduction,
            use_max_pooling=mixer_use_max_pool,
            use_se=mixer_use_se,
        ).to(device)
        params = list(motion_mixer.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(motion_mixer, title="MotionMixer")
    elif model == "eqmotion":
        eq_cfg = EqMotionConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            hid_channel=hidden_size,
        )
        eqmotion_model = EqMotionForecaster(eq_cfg).to(device)
        params = list(eqmotion_model.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(eqmotion_model, title="EqMotionForecaster")
    elif model == "estag":
        # Lazy import to keep deps optional
        from models.model import ESTAG  # wrapper points to Assembly implementation
        estag = ESTAG(
            num_past=input_n,
            num_future=output_n,
            in_node_nf=1,
            hidden_nf=hidden_size,
            fft=True,
            eat=True,
            device=str(device),
            n_layers=t_layers,
            n_nodes=node_num,
            nodes_att_dim=0,
            coords_weight=1.0,
            tempo=True,
            filter=True,
        )
        params = list(estag.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(estag, title="ESTAG")
    elif model == "simlpe":
        sim_cfg = SiMLPeConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            num_layers=gru_layers,
            use_norm=bool(config.get("simlpe_use_norm", True)),
            use_spatial_fc_only=bool(config.get("simlpe_spatial_fc_only", False)),
            mix_spatial_temporal=bool(config.get("simlpe_mix_spatial_temporal", False)),
            norm_axis=str(config.get("simlpe_norm_axis", "spatial")),
            add_last_offset=bool(config.get("simlpe_add_last_offset", True)),
        )
        simlpe = SiMLPeForecaster(sim_cfg).to(device)
        params = list(simlpe.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(simlpe, title="SiMLPeForecaster")
    elif model == "simlpe_dct":
        sim_cfg = SiMLPeDCTConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            num_layers=gru_layers,
            use_norm=bool(config.get("simlpe_use_norm", True)),
            use_spatial_fc_only=bool(config.get("simlpe_spatial_fc_only", False)),
            mix_spatial_temporal=bool(config.get("simlpe_mix_spatial_temporal", False)),
            norm_axis=str(config.get("simlpe_norm_axis", "spatial")),
            add_last_offset=bool(config.get("simlpe_add_last_offset", True)),
            dct_components=dct_keep_coeffs,
        )
        simlpe_dct = SiMLPeDCTForecaster(sim_cfg).to(device)
        params = list(simlpe_dct.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(simlpe_dct, title="SiMLPeDCTForecaster")

    elif model == "twostage_dct_diffusion":
        # Two-stage: coarse low-band DCT predictor + conditional diffusion over high bands.
        k_low = int(config.get("twostage_k_low", 16))
        diff_steps = int(config.get("twostage_diffusion_steps", 100))
        ddim_steps = int(config.get("twostage_ddim_steps", 50))
        d_model = int(config.get("twostage_denoiser_dim", 256))
        depth = int(config.get("twostage_denoiser_depth", 6))
        n_heads = int(config.get("twostage_denoiser_heads", 8))
        p_drop = float(config.get("twostage_dropout", 0.0))
        diffusion_loss_type = str(config.get("twostage_diffusion_loss_type", "mahalanobis_mse"))
        freeze_coarse = bool(config.get("twostage_freeze_coarse", True))
        # If coarse is frozen, keep it frozen during diffusion.
        # Only allow diffusion-stage coarse updates when freeze_coarse is false.
        twostage_train_coarse_in_diffusion = not bool(freeze_coarse)
        twostage_diffusion_coarse_warmup_epochs = max(
            0, int(config.get("twostage_diffusion_coarse_warmup_epochs", 10))
        )
        twostage_diffusion_lr = float(learning_rate) * 0.1
        cond_use_history = bool(config.get("twostage_cond_use_history", True))
        cond_use_coarse = bool(config.get("twostage_cond_use_coarse", True))
        allow_no_conditioning = bool(config.get("twostage_allow_no_conditioning", False))
        coarse_target_lowpass_only = bool(config.get("twostage_coarse_target_lowpass_only", False))
        mobility_palm_var = float(config.get("twostage_mobility_palm_var", 0.15))
        mobility_depth1_var = float(config.get("twostage_mobility_depth1_var", 0.35))
        mobility_depth2_var = float(config.get("twostage_mobility_depth2_var", 0.70))
        mobility_depth3plus_var = float(config.get("twostage_mobility_depth3plus_var", 1.00))
        graph_edge_strength = float(config.get("twostage_graph_edge_strength", 0.22))
        graph_two_hop_strength = float(config.get("twostage_graph_two_hop_strength", 0.05))
        graph_laplacian_strength = float(config.get("twostage_graph_laplacian_strength", 0.5))
        graph_laplacian_tau = float(config.get("twostage_graph_laplacian_tau", 1.0))
        covariance_jitter = float(config.get("twostage_covariance_jitter", 1e-4))
        twostage_use_mamp_condition = bool(config.get("twostage_use_mamp_condition", False))
        twostage_use_mamp_condition_coarse = bool(config.get("twostage_use_mamp_condition_coarse", False))
        twostage_use_any_mamp_condition = (
            twostage_use_mamp_condition or twostage_use_mamp_condition_coarse
        )
        if not (
            cond_use_history
            or cond_use_coarse
            or twostage_use_any_mamp_condition
            or allow_no_conditioning
        ):
            raise ValueError(
                "twostage has no conditioning source enabled. "
                "Set twostage_allow_no_conditioning=true to run a true unconditional mode."
            )
        twostage_wrist_cfg = config.get("twostage_wrist_index")
        if twostage_wrist_cfg is None:
            twostage_wrist_index = int(ordered_wrists[0]) if ordered_wrists else 0
        else:
            twostage_wrist_index = int(twostage_wrist_cfg)
        if not (0 <= twostage_wrist_index < node_num):
            raise ValueError(
                f"twostage_wrist_index must be in [0, {node_num - 1}], got {twostage_wrist_index}"
            )

        raw_twostage_links = config.get("twostage_links", ())
        twostage_links: List[Tuple[int, int]] = []
        if isinstance(raw_twostage_links, (list, tuple)):
            for pair in raw_twostage_links:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                try:
                    i, j = int(pair[0]), int(pair[1])
                except (TypeError, ValueError):
                    continue
                if 0 <= i < node_num and 0 <= j < node_num and i != j:
                    if i > j:
                        i, j = j, i
                    twostage_links.append((i, j))
        if not twostage_links:
            twostage_links = []
            for i, j in edge_pairs:
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                twostage_links.append((int(i), int(j)))
        twostage_links = sorted(set(twostage_links))
        if not twostage_links:
            raise ValueError(
                "twostage requires non-empty hand links. "
                "Provide config['twostage_links'] or ensure edge_index/adjacency includes hand edges."
            )

        tw_cfg = TwoStageDCTDiffusionConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            num_layers=gru_layers,
            k_low=k_low,
            diffusion_steps=diff_steps,
            ddim_steps=ddim_steps,
            denoiser_dim=d_model,
            denoiser_depth=depth,
            denoiser_heads=n_heads,
            dropout=p_drop,
            freeze_coarse=freeze_coarse,
            cond_use_history=cond_use_history,
            cond_use_coarse=cond_use_coarse,
            allow_no_conditioning=allow_no_conditioning,
            coarse_target_lowpass_only=coarse_target_lowpass_only,
            diffusion_loss_type=diffusion_loss_type,
            mobility_palm_var=mobility_palm_var,
            mobility_depth1_var=mobility_depth1_var,
            mobility_depth2_var=mobility_depth2_var,
            mobility_depth3plus_var=mobility_depth3plus_var,
            graph_edge_strength=graph_edge_strength,
            graph_two_hop_strength=graph_two_hop_strength,
            graph_laplacian_strength=graph_laplacian_strength,
            graph_laplacian_tau=graph_laplacian_tau,
            covariance_jitter=covariance_jitter,
            simlpe_use_norm=bool(config.get("simlpe_use_norm", True)),
            simlpe_spatial_fc_only=bool(config.get("simlpe_spatial_fc_only", False)),
            simlpe_mix_spatial_temporal=bool(config.get("simlpe_mix_spatial_temporal", False)),
            simlpe_norm_axis=str(config.get("simlpe_norm_axis", "spatial")),
            simlpe_add_last_offset=bool(config.get("simlpe_add_last_offset", True)),
        )
        twostage_model = TwoStageDCTDiffusionForecaster(
            tw_cfg,
            metadata={
                "wrist_index": int(twostage_wrist_index),
                "edges": tuple((int(i), int(j)) for i, j in twostage_links),
            },
        ).to(device)
        if twostage_use_any_mamp_condition:
            if not twostage_mamp_checkpoint:
                raise ValueError(
                    "MAMP conditioning requires twostage_mamp_checkpoint in config."
                )
            if twostage_mamp_repo_root:
                mamp_repo_abs = os.path.abspath(twostage_mamp_repo_root)
                if mamp_repo_abs not in sys.path:
                    sys.path.insert(0, mamp_repo_abs)
            try:
                from model_mamp.transformer import Transformer as MAMPTransformer  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"Failed to import MAMP Transformer: {exc}") from exc

            mamp_model_args: Optional[Dict[str, Any]] = None
            twostage_mamp_config = str(config.get("twostage_mamp_config", "")).strip()
            if twostage_mamp_config:
                try:
                    import yaml  # type: ignore

                    with open(twostage_mamp_config, "r", encoding="utf-8") as f:
                        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
                    if isinstance(yaml_cfg, dict) and isinstance(yaml_cfg.get("model_args"), dict):
                        mamp_model_args = dict(yaml_cfg["model_args"])
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to parse twostage_mamp_config={twostage_mamp_config}: {exc}"
                    ) from exc

            if mamp_model_args is None:
                mamp_model_args = {
                    "dim_in": 3,
                    "dim_feat": int(config.get("twostage_mamp_dim_feat", 256)),
                    "decoder_dim_feat": int(config.get("twostage_mamp_decoder_dim_feat", 256)),
                    "depth": int(config.get("twostage_mamp_depth", 8)),
                    "decoder_depth": int(config.get("twostage_mamp_decoder_depth", 5)),
                    "num_heads": int(config.get("twostage_mamp_num_heads", 8)),
                    "mlp_ratio": float(config.get("twostage_mamp_mlp_ratio", 4.0)),
                    "num_frames": int(config.get("twostage_mamp_num_frames", input_n)),
                    "num_joints": int(config.get("twostage_mamp_num_joints", node_num)),
                    "patch_size": int(config.get("twostage_mamp_patch_size", 1)),
                    "t_patch_size": int(config.get("twostage_mamp_t_patch_size", 1)),
                    "qkv_bias": True,
                    "qk_scale": None,
                    "drop_rate": 0.0,
                    "attn_drop_rate": 0.0,
                    "drop_path_rate": 0.0,
                    "norm_skes_loss": True,
                }
            mamp_num_frames = int(mamp_model_args.get("num_frames", input_n))
            if twostage_use_mamp_condition and mamp_num_frames != input_n:
                raise ValueError(
                    f"MAMP num_frames ({mamp_model_args.get('num_frames')}) must match input_n ({input_n}) "
                    "when twostage_use_mamp_condition=True."
                )
            if int(mamp_model_args.get("num_joints", node_num)) != node_num:
                raise ValueError(
                    f"MAMP num_joints ({mamp_model_args.get('num_joints')}) must match node_num ({node_num})."
                )

            mamp_encoder = MAMPTransformer(**mamp_model_args).to(device)
            try:
                ckpt = torch.load(twostage_mamp_checkpoint, map_location=device, weights_only=True)
            except Exception:
                ckpt = torch.load(twostage_mamp_checkpoint, map_location=device, weights_only=False)
            mamp_state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = mamp_encoder.load_state_dict(mamp_state, strict=False)
            if missing:
                print(f"[Warning] MAMP checkpoint missing keys: {len(missing)}")
            if unexpected:
                print(f"[Warning] MAMP checkpoint unexpected keys: {len(unexpected)}")
            for p in mamp_encoder.parameters():
                p.requires_grad = False
            mamp_encoder.eval()
            print(f"[Info] Loaded frozen MAMP encoder from {twostage_mamp_checkpoint}")
        coarse_model_path = str(config.get("coarse_model_path", "")).strip() if save_coarse_model else ""
        coarse_model_saved = False

        def _maybe_save_coarse_model() -> None:
            nonlocal coarse_model_saved
            if not coarse_model_path or coarse_model_saved:
                return
            if os.path.exists(coarse_model_path):
                coarse_model_saved = True
                return
            os.makedirs(os.path.dirname(coarse_model_path), exist_ok=True)
            torch.save(
                {
                    "coarse_state": twostage_model.coarse.state_dict(),
                    "metadata": {
                        "tag": config.get("coarse_model_tag"),
                        "model": model,
                        "dataset": config.get("dataset"),
                    },
                },
                coarse_model_path,
            )
            coarse_model_saved = True
            print(f"[Info] Saved coarse model to {coarse_model_path}")

        maybe_save_coarse_model = _maybe_save_coarse_model

        if coarse_model_path and os.path.exists(coarse_model_path):
            try:
                payload = torch.load(coarse_model_path, map_location=device, weights_only=False)
                if isinstance(payload, dict) and "coarse_state" in payload:
                    state = payload["coarse_state"]
                else:
                    state = payload
                missing, unexpected = twostage_model.coarse.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"[Warning] Coarse model load had missing={len(missing)} unexpected={len(unexpected)} keys.")
                coarse_model_saved = True
                print(f"[Info] Loaded coarse model from {coarse_model_path}")
            except Exception as exc:
                print(f"[Warning] Failed to load coarse model from {coarse_model_path}: {exc}")

        def _set_twostage_phase(phase: str) -> List[torch.nn.Parameter]:
            nonlocal twostage_phase
            if phase == "coarse":
                for p in twostage_model.coarse.parameters():
                    p.requires_grad = True
                for p in twostage_model.diffusion.parameters():
                    p.requires_grad = False
                twostage_phase = "coarse"
                return [p for p in twostage_model.coarse.parameters() if p.requires_grad]
            if phase == "diffusion":
                for p in twostage_model.coarse.parameters():
                    p.requires_grad = False
                for p in twostage_model.diffusion.parameters():
                    p.requires_grad = True
                twostage_phase = "diffusion"
                return [p for p in twostage_model.diffusion.parameters() if p.requires_grad]
            raise ValueError(f"Unknown twostage phase: {phase}")

        params = _set_twostage_phase("coarse")
        eval_phase = str(config.get("twostage_eval_phase", "")).strip().lower()
        if eval_phase in {"coarse", "diffusion"} and train_epochs <= 0:
            params = _set_twostage_phase(eval_phase)
        init_lr = twostage_diffusion_lr if twostage_phase == "diffusion" else learning_rate
        optimizer = torch.optim.Adam(params, lr=init_lr)
        params_count = _print_model_parameters(twostage_model, title="TwoStageDCTDiffusionForecaster")
    elif model == "simlpe_wave":
        wavelet_num_scales = config.get("simlpe_wave_num_scales")
        wavelet_name = config.get("simlpe_wave_wavelet", "db2")
        wavelet_mode = config.get("simlpe_wave_wavelet_mode", "symmetric")
        padding_predictor = config.get("simlpe_wave_padding_predictor", "mlp")
        if padding_predictor is None:
            padding_predictor = "mlp"
        sim_cfg = SiMLPeWaveletConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            num_layers=gru_layers,
            use_norm=bool(config.get("simlpe_wave_use_norm", True)),
            norm_axis=str(config.get("simlpe_wave_norm_axis", "all")),
            add_last_offset=bool(config.get("simlpe_wave_add_last_offset", False)),
            padding_mode=config.get("simlpe_wave_padding_mode", None),
            padding_predictor=str(padding_predictor),
            padding_full_horizon=bool(config.get("simlpe_wave_padding_full_horizon", False)),
            num_scales=int(wavelet_num_scales) if wavelet_num_scales is not None else None,
            wavelet=str(wavelet_name),
            wavelet_mode=str(wavelet_mode),
        )
        simlpe_wave = SiMLPeWaveletForecaster(sim_cfg).to(device)
        params = list(simlpe_wave.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(simlpe_wave, title="SiMLPeWaveletForecaster")
    elif model == "physmamba":
        if HandForecastConfig is None or HandMotionForecasterResidual is None:
            raise RuntimeError(
                "physmamba dependencies are not available (missing mamba_ssm). "
                "Install optional physmamba deps or use another model."
            )
        phys_cfg = HandForecastConfig(
            n_landmarks=node_num,
            coord_dim=3,
            t_out=output_n,
            n_layers=int(config.get("t_layers", 2)),
            d_state=int(config.get("physmamba_d_state", 16)),
            d_conv=int(config.get("physmamba_d_conv", 4)),
            expand=int(config.get("physmamba_expand", 2)),
            dropout=float(config.get("physmamba_dropout", 0.1)),
            decoder_conv_kernel=int(config.get("physmamba_decoder_kernel", 5)),
        )
        physmamba_model = HandMotionForecasterResidual(phys_cfg).to(device)
        params = list(physmamba_model.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(physmamba_model, title="HandMotionForecasterResidual")
    elif model == "SFgDNet":
        sfgdnet_model = _build_dual_branch_model(
            prefix="SFgDNet",
            config_cls=SFgDNetConfig,
            forecaster_cls=SFgDNetForecaster,
            label="SFgDNetForecaster",
        )
    elif model == "SplineEqNet":
        SplineEqNet_model = _build_dual_branch_model(
            prefix="SplineEqNet",
            config_cls=SplineEqNetConfig,
            forecaster_cls=SplineEqNetForecaster,
            label="SplineEqNetForecaster",
        )
    elif model == "mamba":
        from models.baselines import TimeSpaceModel
        gtype = "gcn"
        temporal_type = "mamba"
        tl = t_layers
        gl = gnn_layers
        bas = TimeSpaceModel(
            input_size=3,
            n_nodes=node_num,
            horizon=output_n,
            hidden_size=hidden_size,
            t_layers=tl,
            gnn_layers=gl,
            gnn_type=gtype,
            temporal_type=temporal_type,
        ).to(device)
        params = list(bas.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(bas, title=f"TimeSpaceModel[{model}]")
    elif model == "cde":
        from models.cde import MultiNeuralCDE
        cde = MultiNeuralCDE(input_channels=3, hidden_channels=hidden_size, output_channels=3, horizon=output_n, fast=False).to(device)
        params = list(cde.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(cde, title="MultiNeuralCDE")
    elif model == "hstgcnn":
        from models.HSTGCN import hstgcnn
        hst = hstgcnn(input_feat=3, output_feat=3, seq_len=input_n, pred_seq_len=output_n).to(device)
        params = list(hst.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(hst, title="HSTGCN")
    elif model == "gwnet":
        from models.wavenet import gwnet
        gwn = gwnet(device, node_num, in_dim=3, out_dim=3, pred_seq_len=output_n).to(device)
        params = list(gwn.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(gwn, title="GWNet")
    else:
        gen_x = gen_y = gen_z = gen_v = None
        params = []
        optimizer = None

    model_modules: Dict[str, torch.nn.Module] = {}
    if model == "mpt" and all(m is not None for m in (gen_x, gen_y, gen_z, gen_v)):
        model_modules = {
            "gen_x": gen_x,
            "gen_y": gen_y,
            "gen_z": gen_z,
            "gen_v": gen_v,
        }
    elif model in ("hisrep", "hisrep_hand", "hisrep_rigidwrist") and 'hisrep' in locals():
        model_modules = {"hisrep": hisrep}
    elif model == "trajdep" and 'trajdep' in locals():
        model_modules = {"trajdep": trajdep}
    elif model == "pgbig" and 'pgbig_model' in locals():
        model_modules = {"pgbig_model": pgbig_model}
    elif model == "motionmixer" and 'motion_mixer' in locals():
        model_modules = {"motion_mixer": motion_mixer}
    elif model == "eqmotion" and 'eqmotion_model' in locals():
        model_modules = {"eqmotion_model": eqmotion_model}
    elif model == "simlpe" and 'simlpe' in locals():
        model_modules = {"simlpe": simlpe}
    elif model == "simlpe_dct" and 'simlpe_dct' in locals():
        model_modules = {"simlpe_dct": simlpe_dct}
    elif model == "twostage_dct_diffusion" and twostage_model is not None:
        model_modules = {"twostage": twostage_model}
    elif model == "simlpe_wave" and 'simlpe_wave' in locals():
        model_modules = {"simlpe_wave": simlpe_wave}
    elif model == "physmamba" and physmamba_model is not None:
        model_modules = {"physmamba": physmamba_model}
    elif model == "SFgDNet" and sfgdnet_model is not None:
        model_modules = {"SFgDNet": sfgdnet_model}
    elif model == "SplineEqNet" and SplineEqNet_model is not None:
        model_modules = {"SplineEqNet": SplineEqNet_model}
    elif model == "mamba" and 'bas' in locals():
        model_modules = {"bas": bas}
    elif model == "cde" and 'cde' in locals():
        model_modules = {"cde": cde}
    elif model == "hstgcnn" and 'hst' in locals():
        model_modules = {"hst": hst}
    elif model == "gwnet" and 'gwn' in locals():
        model_modules = {"gwn": gwn}
    elif model == "estag" and 'estag' in locals():
        model_modules = {"estag": estag}

    def _set_mode(training: bool) -> None:
        for module in model_modules.values():
            if training:
                module.train()
            else:
                module.eval()

    def _capture_state() -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        if not model_modules:
            return None
        return {name: copy.deepcopy(module.state_dict()) for name, module in model_modules.items()}

    def _load_state(state: Optional[Dict[str, Dict[str, torch.Tensor]]]) -> None:
        if not state:
            return
        for name, module in model_modules.items():
            if name in state:
                module.load_state_dict(state[name])

    def _get_active_dual_model() -> Optional[torch.nn.Module]:
        if model == "SFgDNet":
            return sfgdnet_model
        if model == "SplineEqNet":
            return SplineEqNet_model
        return None

    def _resize_mamp_time(coords_3d: torch.Tensor) -> torch.Tensor:
        if int(coords_3d.shape[1]) == int(mamp_num_frames):
            return coords_3d
        B, T, N, C = coords_3d.shape
        if C != 3:
            raise ValueError(f"Expected 3 coordinate channels, got {C}.")
        flat = coords_3d.reshape(B, T, N * C).transpose(1, 2).contiguous()  # (B, F, T)
        flat = F.interpolate(flat, size=int(mamp_num_frames), mode="linear", align_corners=False)
        return flat.transpose(1, 2).reshape(B, int(mamp_num_frames), N, C).contiguous()

    def _compute_mamp_feat(coords_3d: torch.Tensor) -> Optional[torch.Tensor]:
        if mamp_encoder is None:
            return None
        coords_3d = _resize_mamp_time(coords_3d)
        with torch.no_grad():
            latent, _mask, _ids_restore = mamp_encoder.forward_encoder(
                coords_3d,
                mask_ratio=twostage_mamp_mask_ratio,
                motion_aware_tau=twostage_mamp_motion_aware_tau,
            )
        return latent.mean(dim=1).detach()

    def _merge_mamp_features(
        hist_feat: Optional[torch.Tensor],
        coarse_feat: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if hist_feat is None:
            return coarse_feat
        if coarse_feat is None:
            return hist_feat
        if int(hist_feat.shape[0]) != int(coarse_feat.shape[0]):
            raise ValueError(
                f"MAMP batch mismatch while merging features: {hist_feat.shape[0]} vs {coarse_feat.shape[0]}"
            )
        return torch.stack([hist_feat, coarse_feat], dim=1).contiguous()

    def _set_dataset_aux_feature(ds: Any, feature_key: str, values: torch.Tensor) -> None:
        values_cpu = values.detach().cpu().float().contiguous()
        existing = getattr(ds, "aux_features", None)
        if isinstance(existing, dict):
            merged = dict(existing)
            merged[feature_key] = values_cpu
            setattr(ds, "aux_features", merged)
            return
        if feature_key == "history":
            if hasattr(ds, "set_aux_features"):
                ds.set_aux_features(values_cpu)
            else:
                setattr(ds, "aux_features", values_cpu)
            return
        merged = {}
        if torch.is_tensor(existing):
            merged["history"] = existing.detach().cpu().float().contiguous()
        merged[feature_key] = values_cpu
        setattr(ds, "aux_features", merged)

    def _precompute_mamp_features_for_dataset(
        ds: Any,
        split_name: str,
        *,
        feature_key: str,
        source: str,
    ) -> None:
        if ds is None or mamp_encoder is None:
            return
        if not hasattr(ds, "__len__") or len(ds) <= 0:
            return
        pre_bs = max(1, int(config.get("twostage_mamp_precompute_batch_size", batch_size)))
        tmp_loader = DataLoader(ds, batch_size=pre_bs, shuffle=False, drop_last=False)
        cached_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in tmp_loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    raise ValueError(f"Unexpected batch format while precomputing MAMP features for {split_name}.")
                inp = batch[0].to(device).float()
                in_3d = inp[:, :, :, 4:]
                if source == "history":
                    feat = _compute_mamp_feat(in_3d)
                elif source == "coarse":
                    if twostage_model is None:
                        raise RuntimeError("Two-stage model is not available while precomputing coarse MAMP features.")
                    coarse_3d = twostage_model.coarse(in_3d)
                    feat = _compute_mamp_feat(coarse_3d)
                else:
                    raise ValueError(f"Unknown MAMP precompute source: {source}")
                if feat is None:
                    raise RuntimeError("MAMP encoder is not available during precompute.")
                cached_chunks.append(feat.cpu())
        if not cached_chunks:
            return
        cached = torch.cat(cached_chunks, dim=0).float().contiguous()
        if int(cached.shape[0]) != len(ds):
            raise RuntimeError(
                f"Precomputed MAMP features mismatch for {split_name}: got {cached.shape[0]}, expected {len(ds)}."
            )
        _set_dataset_aux_feature(ds, feature_key, cached)
        print(f"[Info] Precomputed MAMP {feature_key} features for {split_name}: {tuple(cached.shape)}")

    def _extract_cached_mamp_features(aux_payload: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        hist_feat: Optional[torch.Tensor] = None
        coarse_feat: Optional[torch.Tensor] = None
        if torch.is_tensor(aux_payload):
            hist_feat = aux_payload
        elif isinstance(aux_payload, dict):
            h = aux_payload.get("history")
            c = aux_payload.get("coarse")
            if torch.is_tensor(h):
                hist_feat = h
            if torch.is_tensor(c):
                coarse_feat = c
        return hist_feat, coarse_feat

    load_model_path = str(config.get("load_model_path", "")).strip()
    if load_model_path:
        if os.path.exists(load_model_path):
            try:
                try:
                    payload = torch.load(load_model_path, map_location=device, weights_only=True)
                except Exception:
                    payload = torch.load(load_model_path, map_location=device, weights_only=False)
                if isinstance(payload, dict):
                    if "best_model_state" in payload:
                        state = payload["best_model_state"]
                    elif "model_state" in payload:
                        state = payload["model_state"]
                    else:
                        state = payload
                else:
                    state = payload
                _load_state(state)
                print(f"[Info] Loaded model checkpoint from {load_model_path}")
            except Exception as exc:
                print(f"[Warning] Failed to load checkpoint from {load_model_path}: {exc}")
        else:
            print(f"[Warning] Checkpoint not found at {load_model_path}")

    if model == "twostage_dct_diffusion" and twostage_use_mamp_condition and mamp_encoder is not None:
        _precompute_mamp_features_for_dataset(
            getattr(loader, "dataset", None), "train", feature_key="history", source="history"
        )
        _precompute_mamp_features_for_dataset(
            getattr(val_loader, "dataset", None), "val", feature_key="history", source="history"
        )
        _precompute_mamp_features_for_dataset(
            getattr(test_loader, "dataset", None), "test", feature_key="history", source="history"
        )

    def _evaluate_loader(
        eval_loader: Optional[DataLoader],
        *,
        collect_examples: bool = False,
        restore_train: bool = True,
        save_examples: bool = False,
        ) -> Optional[Dict[str, float]]:
        if eval_loader is None:
            return None
        total_loss = 0.0
        total_loss_norm = 0.0
        total_score = 0.0
        total_velocity_mae = 0.0
        total_samples = 0
        collect_for_saving = bool(collect_examples and save_examples)
        compute_humanmac_metrics = bool(config.get("compute_humanmac_metrics", False))
        use_oracle_mpjpe = bool(config.get("twostage_eval_oracle_mpjpe", False))
        collected_preds: List[torch.Tensor] = []
        collected_tgts: List[torch.Tensor] = []
        collected_allk_preds: List[torch.Tensor] = []
        humanmac_pred_batches: List[torch.Tensor] = []
        humanmac_tgt_batches: List[torch.Tensor] = []
        humanmac_start_pose_batches: List[torch.Tensor] = []
        collect_twostage_tries = (
            bool(config.get("twostage_eval_collect_all", False))
            and model == "twostage_dct_diffusion"
            and twostage_phase != "coarse"
            and twostage_eval_best_of_k > 1
        )
        if save_eval_examples_all_k and not (
            model == "twostage_dct_diffusion" and twostage_phase != "coarse" and twostage_eval_best_of_k > 1
        ):
            print("[Info] save_eval_examples_all_k is only supported for twostage_dct_diffusion with best_of_k > 1.")
        per_try_loss: Optional[List[float]] = None
        per_try_loss_norm: Optional[List[float]] = None
        if collect_twostage_tries:
            per_try_loss = [0.0 for _ in range(twostage_eval_best_of_k)]
            per_try_loss_norm = [0.0 for _ in range(twostage_eval_best_of_k)]

        _set_mode(False)
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_loader):
                    if isinstance(batch, (list, tuple)):
                        if len(batch) < 2:
                            raise ValueError("Loader batch must provide input and target tensors.")
                        inp, out = batch[0], batch[1]
                        norm_factor = batch[2] if len(batch) > 2 else None
                        cached_aux = batch[3] if len(batch) > 3 else None
                        cached_mamp_feat, cached_mamp_feat_coarse = _extract_cached_mamp_features(cached_aux)
                    else:
                        raise TypeError("Expected batch from loader to be a tuple/list.")

                    inp = inp.to(device).float()
                    out = out.to(device).float()
                    if norm_factor is not None:
                        norm_factor_t = norm_factor.to(device=device).float().view(-1)
                    else:
                        norm_factor_t = torch.ones(inp.size(0), device=device)

                    in_angle = inp[:, 1:, :, :3]
                    in_vel = inp[:, 1:, :, 3].permute(0, 2, 1)
                    tgt_3d = out[:, :, :, 4:]
                    tgt_vel = out[:, :, :, 3]
                    in_3d = inp[:, :, :, 4:]
                    in_ang_x = in_angle[:, :, :, 0].permute(0, 2, 1).contiguous()
                    in_ang_y = in_angle[:, :, :, 1].permute(0, 2, 1).contiguous()
                    in_ang_z = in_angle[:, :, :, 2].permute(0, 2, 1).contiguous()
                    B, T_out, N = tgt_3d.shape[0], tgt_3d.shape[1], tgt_3d.shape[2]
                    humanmac_candidates_batch: Optional[torch.Tensor] = None
                    twostage_mamp_feat = None
                    coarse_future_for_condition = None
                    if model == "twostage_dct_diffusion" and twostage_phase != "coarse":
                        # Cache coarse prediction once per eval batch and reuse for all sampled candidates.
                        coarse_future_for_condition = twostage_model.coarse(in_3d)
                        twostage_mamp_hist_feat = None
                        twostage_mamp_coarse_feat = None
                        if twostage_use_mamp_condition:
                            if cached_mamp_feat is not None:
                                twostage_mamp_hist_feat = cached_mamp_feat.to(device=device).float()
                            else:
                                twostage_mamp_hist_feat = _compute_mamp_feat(in_3d)
                        if twostage_use_mamp_condition_coarse:
                            if cached_mamp_feat_coarse is not None:
                                twostage_mamp_coarse_feat = cached_mamp_feat_coarse.to(device=device).float()
                            else:
                                twostage_mamp_coarse_feat = _compute_mamp_feat(coarse_future_for_condition)
                        twostage_mamp_feat = _merge_mamp_features(twostage_mamp_hist_feat, twostage_mamp_coarse_feat)

                    if model == "mpt":
                        out_v, _ = gen_v(in_vel, hidden_size)
                        out_x, _ = gen_x(in_ang_x, hidden_size)
                        out_y, _ = gen_y(in_ang_y, hidden_size)
                        out_z, _ = gen_z(in_ang_z, hidden_size)
                        pred_v = out_v.view(B, N, output_n).permute(0, 2, 1)
                        ang_x = out_x.view(B, N, output_n).permute(0, 2, 1)
                        ang_y = out_y.view(B, N, output_n).permute(0, 2, 1)
                        ang_z = out_z.view(B, N, output_n).permute(0, 2, 1)
                        pred_angles = torch.stack([ang_x, ang_y, ang_z], dim=-1)
                        init_pose = in_3d[:, -1, :, :]
                        recons = U.reconstruct_sequence(pred_v, pred_angles, init_pose, node_num)
                    elif model in ("hisrep", "hisrep_hand", "hisrep_rigidwrist"):
                        recons = hisrep(in_3d)
                    elif model == "eqmotion":
                        recons = eqmotion_model(in_3d)
                    elif model == "estag":
                        x_hist = in_3d.permute(1, 0, 2, 3).reshape(input_n, B * N, 3)
                        h = torch.ones(B * N, 1, device=device)
                        x_hat = estag(h, x_hist, edge_index_tensor_device)
                        recons = x_hat.view(T_out, B, N, 3).permute(1, 0, 2, 3)
                    elif model == "simlpe":
                        recons = simlpe(in_3d)
                    elif model == "simlpe_dct":
                        recons = simlpe_dct(in_3d)

                    elif model == "twostage_dct_diffusion":
                        if twostage_phase == "coarse":
                            recons = twostage_model.coarse(in_3d)
                        else:
                            selection_k = max(1, int(twostage_eval_best_of_k))
                            humanmac_k = (
                                max(1, int(config.get("humanmac_num_candidates", 50)))
                                if compute_humanmac_metrics
                                else 1
                            )
                            if selection_k <= 1 and humanmac_k <= 1:
                                recons = twostage_model.predict(
                                    in_3d,
                                    mamp_feat=twostage_mamp_feat,
                                    coarse_future=coarse_future_for_condition,
                                    deterministic=True,
                                )
                                if compute_humanmac_metrics:
                                    humanmac_candidates_batch = recons.unsqueeze(0)
                            else:
                                all_recons: List[torch.Tensor] = []
                                all_scores: List[torch.Tensor] = []
                                total_candidate_samples = max(selection_k, humanmac_k)
                                for sample_idx in range(total_candidate_samples):
                                    sample_seed = int(batch_idx * 1000003 + sample_idx)
                                    sampled_recons, sampled_score = twostage_model.predict(
                                        in_3d,
                                        mamp_feat=twostage_mamp_feat,
                                        coarse_future=coarse_future_for_condition,
                                        deterministic=False,
                                        seed=sample_seed,
                                        return_score=True,
                                    )
                                    all_recons.append(sampled_recons)
                                    all_scores.append(sampled_score)
                                    if (
                                        sample_idx < selection_k
                                        and collect_twostage_tries
                                        and per_try_loss is not None
                                        and per_try_loss_norm is not None
                                    ):
                                        mpjpe_terms_try = torch.norm(sampled_recons - tgt_3d, dim=-1)
                                        per_sample_mpjpe_try = mpjpe_terms_try.mean(dim=(1, 2))
                                        per_sample_mpjpe_norm_try = per_sample_mpjpe_try * norm_factor_t
                                        per_try_loss[sample_idx] += float(per_sample_mpjpe_try.sum().item())
                                        per_try_loss_norm[sample_idx] += float(per_sample_mpjpe_norm_try.sum().item())
                                if not all_recons:
                                    raise RuntimeError("Failed to generate diffusion samples during evaluation.")
                                if compute_humanmac_metrics:
                                    humanmac_candidates_batch = torch.stack(all_recons[:humanmac_k], dim=0)
                                if selection_k <= 1:
                                    recons = twostage_model.predict(
                                        in_3d,
                                        mamp_feat=twostage_mamp_feat,
                                        coarse_future=coarse_future_for_condition,
                                        deterministic=True,
                                    )
                                else:
                                    stacked_recons = torch.stack(all_recons[:selection_k], dim=0)  # (K, B, T_out, N, 3)
                                    if use_oracle_mpjpe:
                                        oracle_terms = torch.norm(stacked_recons - tgt_3d.unsqueeze(0), dim=-1)
                                        oracle_mpjpe = oracle_terms.mean(dim=(2, 3))  # (K, B)
                                        best_idx = oracle_mpjpe.argmin(dim=0)
                                        batch_selector = torch.arange(stacked_recons.size(1), device=stacked_recons.device)
                                        recons = stacked_recons[best_idx, batch_selector]
                                    else:
                                        contrastive_scorer = _load_contrastive_model()
                                        if contrastive_scorer is None:
                                            raise RuntimeError("Contrastive scorer requested but not available.")
                                        from models.contrastive.contrastive_score import rank_diffusion_samples
                                        y_candidates = stacked_recons.permute(1, 0, 2, 3, 4)  # (B, K, T_out, N, 3)
                                        recons, _ = rank_diffusion_samples(
                                            contrastive_scorer,
                                            in_3d,
                                            y_candidates,
                                            device=str(device),
                                        )
                                if collect_for_saving and save_eval_examples_all_k:
                                    if selection_k > 1:
                                        collected_allk_preds.append(stacked_recons[:, 0].detach().cpu())
                    elif model == "simlpe_wave":
                        recons = simlpe_wave(in_3d)
                    elif model == "physmamba":
                        recons = physmamba_model(in_3d)
                    elif model in ("SFgDNet", "SplineEqNet"):
                        dual_model = _get_active_dual_model()
                        if dual_model is None:
                            raise RuntimeError(f"{model} model is not initialized.")
                        recons = dual_model(in_3d)
                    elif model == "pgbig":
                        seq_flat = in_3d.reshape(B, input_n, -1)
                        last_pose = seq_flat[:, -1:, :]
                        padded = torch.cat([seq_flat, last_pose.repeat(1, output_n, 1)], dim=1)
                        pred_full = pgbig_model(padded)[0]
                        recons = pred_full[:, -output_n:, :].reshape(B, T_out, N, 3)
                    elif model == "motionmixer":
                        seq_flat = in_3d.reshape(B, input_n, pose_dim)
                        tgt_flat = tgt_3d.reshape(B, T_out, pose_dim)
                        if mixer_input_scale != 1.0:
                            seq_scaled = seq_flat / mixer_input_scale
                            tgt_scaled = tgt_flat / mixer_input_scale
                        else:
                            seq_scaled = seq_flat
                            tgt_scaled = tgt_flat
                        if mixer_use_delta:
                            sequences_all = torch.cat([seq_scaled, tgt_scaled], dim=1)
                            diffs = sequences_all[:, 1:, :] - sequences_all[:, :-1, :]
                            diffs = torch.cat([diffs[:, 0:1, :], diffs], dim=1)
                            mixer_in = diffs[:, :input_n, :]
                            pred_scaled = motion_mixer(mixer_in)
                            pred_scaled = _delta_to_absolute(pred_scaled, seq_scaled[:, -1, :])
                        else:
                            mixer_in = seq_scaled
                            pred_scaled = motion_mixer(mixer_in)
                        pred_flat = pred_scaled * mixer_input_scale if mixer_input_scale != 1.0 else pred_scaled
                        recons = pred_flat.view(B, T_out, N, 3)
                    elif model == "trajdep":
                        recons = trajdep(in_3d)
                    elif model == "mamba":
                        recons = bas(in_3d, edge_index_tensor_device)
                    elif model == "cde":
                        recons = cde(in_3d, edge_index_tensor_device)
                    elif model == "hstgcnn":
                        recons = hst(in_3d, edge_index_tensor_device)
                    elif model == "gwnet":
                        recons = gwn(in_3d, edge_index_tensor_device)
                    else:
                        init_pose = in_3d[:, -1, :, :]
                        recons = init_pose.unsqueeze(1).expand(B, T_out, N, 3)

                    if compute_humanmac_metrics:
                        if humanmac_candidates_batch is None:
                            humanmac_candidates_batch = recons.unsqueeze(0)
                        humanmac_pred_batches.append(humanmac_candidates_batch.detach().cpu())
                        humanmac_tgt_batches.append(tgt_3d.detach().cpu())
                        humanmac_start_pose_batches.append(in_3d[:, -1, :, :].detach().cpu())

                    mpjpe_terms = torch.norm(recons - tgt_3d, dim=-1)
                    per_sample_mpjpe = mpjpe_terms.mean(dim=(1, 2))
                    per_sample_mpjpe_norm = per_sample_mpjpe * norm_factor_t

                    total_loss += float(per_sample_mpjpe.sum().item())
                    total_loss_norm += float(per_sample_mpjpe_norm.sum().item())
                    last_pose_eval = in_3d[:, -1, :, :]
                    pred_vel_mag_eval = _velocity_magnitude(recons, last_pose_eval)
                    vel_mae_eval = (pred_vel_mag_eval - tgt_vel).abs().mean(dim=(1, 2))
                    total_velocity_mae += float(vel_mae_eval.sum().item())
                    score_val = U.weighted_joint_loss(joint_weights, recons, tgt_3d, metric='mae')
                    total_score += float(score_val.item())
                    total_samples += inp.size(0)

                    if collect_for_saving:
                        collected_preds.append(recons[0].detach().cpu())
                        collected_tgts.append(tgt_3d[0].detach().cpu())
        finally:
            if restore_train:
                _set_mode(True)

        if total_samples == 0:
            return {
                "mpjpe": float("nan"),
                "mpjpe_norm": float("nan"),
                "weighted_mae": float("nan"),
                "samples": 0,
            }

        avg_loss = total_loss / total_samples
        avg_loss_norm = total_loss_norm / total_samples
        avg_score = total_score / max(1, total_samples)
        avg_velocity_mae = total_velocity_mae / max(1, total_samples)

        if collect_for_saving and collected_preds and collected_tgts:
            examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
            os.makedirs(examples_dir, exist_ok=True)
            merged_pred = torch.stack(collected_preds, dim=0)
            merged_tgt = torch.stack(collected_tgts, dim=0)
            torch.save(merged_pred, os.path.join(examples_dir, f"{model}_pred.pt"))
            torch.save(merged_tgt, os.path.join(examples_dir, f"{model}_target.pt"))
            if collected_allk_preds:
                merged_pred_all_k = torch.stack(collected_allk_preds, dim=1)  # (K, B, T, N, 3)
                torch.save(merged_pred_all_k, os.path.join(examples_dir, f"{model}_pred_all_k.pt"))

        metrics = {
            "mpjpe": float(avg_loss),
            "mpjpe_norm": float(avg_loss_norm),
            "weighted_mae": float(avg_score),
            "velocity_mae": float(avg_velocity_mae),
            "samples": float(total_samples),
        }
        if compute_humanmac_metrics and humanmac_pred_batches and humanmac_tgt_batches and humanmac_start_pose_batches:
            humanmac_metrics = _compute_humanmac_metrics(
                torch.cat(humanmac_pred_batches, dim=1),
                torch.cat(humanmac_tgt_batches, dim=0),
                torch.cat(humanmac_start_pose_batches, dim=0),
                threshold=float(config.get("humanmac_multimodal_threshold", 0.5)),
            )
            metrics.update(humanmac_metrics)
        if collect_twostage_tries and per_try_loss is not None and per_try_loss_norm is not None:
            per_try_mpjpe = [val / max(1, total_samples) for val in per_try_loss]
            per_try_mpjpe_norm = [val / max(1, total_samples) for val in per_try_loss_norm]
            metrics["mpjpe_by_try"] = per_try_mpjpe
            metrics["mpjpe_norm_by_try"] = per_try_mpjpe_norm
        return metrics

    # Optional LR scheduler that reduces LR on plateau of the monitored loss
    use_lr_scheduler = bool(config.get("use_lr_scheduler", True))
    scheduler = None
    if optimizer is not None and use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(config.get("lr_factor", 0.5)),
            patience=int(config.get("lr_patience", 5)),
            threshold=float(config.get("lr_threshold", 1e-4)),
            cooldown=int(config.get("lr_cooldown", 0)),
            min_lr=float(config.get("lr_min", 0.0))
        )

    # Compute joint weights for scoring (not used in backprop)
    joint_weights = U.compute_node_motion_weights(loader, device=device, normalize='sum')

    n_train_windows = len(getattr(loader, "dataset", []))
    print(f"Training on {n_train_windows} windows")
    print(f"Config: input_n={input_n}, output_n={output_n}, hidden_size={hidden_size}, batch_size={batch_size}, model={model}")

    pose_dim = node_num * 3
    mixer_use_delta = bool(config.get("mixer_use_delta", True))
    mixer_input_scale = float(config.get("mixer_input_scale", 1.0))

    best_test_mpjpe = None
    best_test_mpjpe_norm = None
    best_test_epoch = None
    best_test_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    last_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    best_test_velocity_mae = None

    best_train_mpjpe = None
    best_train_mpjpe_norm = None
    best_train_epoch = None
    best_train_velocity_mae = None

    # Optional: log initial SemSkeConv parameter norms to assess magnitude
    if bool(config.get("log_gcn_stats", False)) and model == "mpt":
        layers = U.find_semskeconvs(gen_x, gen_y, gen_z, gen_v)
        U.print_semskeconv_stats(U.semskeconv_stats(layers), prefix="[GCN][epoch 000]")

    total_epochs = train_epochs
    if model == "twostage_dct_diffusion":
        total_epochs = train_epochs + max(0, twostage_diffusion_epochs)
    early_stop_enabled = bool(config.get("early_stopping_enabled", False))
    early_stop_patience = max(1, int(config.get("early_stopping_patience", 20)))
    early_stop_min_delta = float(config.get("early_stopping_min_delta", 1e-4))
    early_stop_warmup = max(0, int(config.get("early_stopping_warmup", 0)))
    early_stop_monitor = str(config.get("early_stopping_monitor", "auto")).strip().lower()
    early_stop_reset_on_phase_change = bool(config.get("early_stopping_reset_on_phase_change", True))
    early_stop_best: Optional[float] = None
    early_stop_bad_epochs = 0
    # Epoch index (1-based) where the current early-stop stage starts.
    # Warmup is applied relative to this boundary.
    early_stop_stage_start_epoch = 1

    for epoch in range(1, total_epochs + 1):
        if model == "twostage_dct_diffusion" and twostage_phase == "coarse":
            if twostage_diffusion_epochs > 0 and epoch == train_epochs + 1:
                if maybe_save_coarse_model is not None:
                    maybe_save_coarse_model()
                params = _set_twostage_phase("diffusion")
                optimizer = torch.optim.Adam(params, lr=twostage_diffusion_lr)
                twostage_coarse_diffusion_group_added = False
                if use_lr_scheduler:
                    scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=float(config.get("lr_factor", 0.5)),
                        patience=int(config.get("lr_patience", 5)),
                        threshold=float(config.get("lr_threshold", 1e-4)),
                        cooldown=int(config.get("lr_cooldown", 0)),
                        min_lr=float(config.get("lr_min", 0.0))
                    )
                if twostage_use_mamp_condition_coarse and mamp_encoder is not None:
                    _set_mode(False)
                    _precompute_mamp_features_for_dataset(
                        getattr(loader, "dataset", None), "train", feature_key="coarse", source="coarse"
                    )
                    _precompute_mamp_features_for_dataset(
                        getattr(val_loader, "dataset", None), "val", feature_key="coarse", source="coarse"
                    )
                    _precompute_mamp_features_for_dataset(
                        getattr(test_loader, "dataset", None), "test", feature_key="coarse", source="coarse"
                    )
                if early_stop_reset_on_phase_change:
                    early_stop_best = None
                    early_stop_bad_epochs = 0
                    early_stop_stage_start_epoch = epoch
                print(f"[Info] Switched to twostage diffusion training for {twostage_diffusion_epochs} epochs.")
        if (
            model == "twostage_dct_diffusion"
            and twostage_phase == "diffusion"
            and twostage_train_coarse_in_diffusion
            and not twostage_coarse_diffusion_group_added
        ):
            diffusion_stage_epoch = epoch - train_epochs
            if diffusion_stage_epoch > twostage_diffusion_coarse_warmup_epochs:
                coarse_params = [p for p in twostage_model.coarse.parameters()]
                for p in coarse_params:
                    p.requires_grad = True
                coarse_params = [p for p in coarse_params if p.requires_grad]
                if coarse_params:
                    if optimizer is None:
                        raise RuntimeError("Optimizer is not initialized for twostage diffusion phase.")
                    coarse_lr = twostage_diffusion_lr * 0.1
                    optimizer.add_param_group({"params": coarse_params, "lr": coarse_lr})
                    twostage_coarse_diffusion_group_added = True
                    print(
                        f"[Info] Unfroze coarse module during diffusion at epoch {epoch} "
                        f"(diffusion_epoch={diffusion_stage_epoch}, warmup={twostage_diffusion_coarse_warmup_epochs}, "
                        f"coarse_lr={coarse_lr:.6g}, diffusion_lr={twostage_diffusion_lr:.6g})."
                    )
        _set_mode(True)
        running = 0.0
        score_running = 0.0
        mpjpe_running = 0.0
        mpjpe_norm_running = 0.0
        vel_running = 0.0
        diffusion_running = 0.0
        n_seen = 0
        t0 = time.time()
        epoch_masks: List[torch.Tensor] = [] if mask_logging_enabled else []
        epoch_gate_slow: List[torch.Tensor] = [] if gate_logging_enabled else []
        epoch_gate_fast: List[torch.Tensor] = [] if gate_logging_enabled else []

        for it, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError("Loader batch must provide input and target tensors.")
                inp, out = batch[0], batch[1]
                norm_factor = batch[2] if len(batch) > 2 else None
                cached_aux = batch[3] if len(batch) > 3 else None
                cached_mamp_feat, cached_mamp_feat_coarse = _extract_cached_mamp_features(cached_aux)
            else:
                raise TypeError("Expected batch from loader to be a tuple/list.")
            # Shapes: inp (B, T_in, N, 7), out (B, T_out, N, 7)
            inp = inp.to(device).float()
            out = out.to(device).float()
            if norm_factor is not None:
                norm_factor_t = norm_factor.to(device=device).float().view(-1)
            else:
                norm_factor_t = torch.ones(inp.size(0), device=device)

            # Prepare inputs/targets as in prediction_model.py
            # Drop first frame for velocity/angle inputs
            in_angle = inp[:, 1:, :, :3]
            in_vel = inp[:, 1:, :, 3].permute(0, 2, 1)  # (B, N, T_in-1)

            tgt_angle = out[:, :, :, :3]
            tgt_vel = out[:, :, :, 3]
            in_3d = inp[:, :, :, 4:]
            tgt_3d = out[:, :, :, 4:]

            # Build per-axis inputs
            in_ang_x = in_angle[:, :, :, 0].permute(0, 2, 1).contiguous()  # (B, N, T_in-1)
            in_ang_y = in_angle[:, :, :, 1].permute(0, 2, 1).contiguous()
            in_ang_z = in_angle[:, :, :, 2].permute(0, 2, 1).contiguous()

            B, T_out, N = tgt_3d.shape[0], tgt_3d.shape[1], tgt_3d.shape[2]
            aux = None
            diffusion_loss = None

            if model == "mpt":
                optimizer.zero_grad()

                # One-shot horizon prediction: decode full sequences of v and angles
                out_v, _ = gen_v(in_vel, hidden_size)
                out_x, _ = gen_x(in_ang_x, hidden_size)
                out_y, _ = gen_y(in_ang_y, hidden_size)
                out_z, _ = gen_z(in_ang_z, hidden_size)

                # Reshape to (B, T_out, N) and (B, T_out, N, 3)
                pred_v = out_v.view(B, N, output_n).permute(0, 2, 1)
                ang_x = out_x.view(B, N, output_n).permute(0, 2, 1)
                ang_y = out_y.view(B, N, output_n).permute(0, 2, 1)
                ang_z = out_z.view(B, N, output_n).permute(0, 2, 1)
                pred_angles = torch.stack([ang_x, ang_y, ang_z], dim=-1)

                # Reconstruct 3D trajectory from last observed pose
                init_pose = in_3d[:, -1, :, :]
                recons = U.reconstruct_sequence(pred_v, pred_angles, init_pose, node_num)
            elif model == "estag":
                optimizer.zero_grad()
                x_hist = in_3d.permute(1, 0, 2, 3).reshape(input_n, B * N, 3)
                h = torch.ones(B * N, 1, device=device)
                x_hat = estag(h, x_hist, edge_index_tensor_device)
                recons = x_hat.view(T_out, B, N, 3).permute(1, 0, 2, 3)
            elif model == "simlpe":
                optimizer.zero_grad()
                recons = simlpe(in_3d)
            elif model == "simlpe_dct":
                optimizer.zero_grad()
                recons = simlpe_dct(in_3d)

            elif model == "twostage_dct_diffusion":
                optimizer.zero_grad()
                if twostage_phase == "diffusion":
                    twostage_mamp_hist_feat = None
                    twostage_mamp_coarse_feat = None
                    coarse_grad_active = bool(
                        twostage_train_coarse_in_diffusion and twostage_coarse_diffusion_group_added
                    )
                    # Cache coarse prediction once per train batch during diffusion phase.
                    if coarse_grad_active:
                        coarse_future_for_condition = twostage_model.coarse(in_3d)
                    else:
                        with torch.no_grad():
                            coarse_future_for_condition = twostage_model.coarse(in_3d)
                    if twostage_use_mamp_condition:
                        if cached_mamp_feat is not None:
                            twostage_mamp_hist_feat = cached_mamp_feat.to(device=device).float()
                        else:
                            twostage_mamp_hist_feat = _compute_mamp_feat(in_3d)
                    if twostage_use_mamp_condition_coarse:
                        # Coarse-conditioned MAMP branch is always computed without gradients.
                        coarse_for_mamp = coarse_future_for_condition.detach()
                        if cached_mamp_feat_coarse is not None:
                            twostage_mamp_coarse_feat = cached_mamp_feat_coarse.to(device=device).float()
                        else:
                            twostage_mamp_coarse_feat = _compute_mamp_feat(coarse_for_mamp)
                    twostage_mamp_feat = _merge_mamp_features(twostage_mamp_hist_feat, twostage_mamp_coarse_feat)
                    diffusion_loss, coarse_pred = twostage_model.diffusion_loss(
                        in_3d,
                        tgt_3d,
                        mamp_feat=twostage_mamp_feat,
                        coarse_future=coarse_future_for_condition,
                        allow_coarse_grad=coarse_grad_active,
                    )
                    recons = coarse_pred
                else:
                    recons = twostage_model.coarse(in_3d)
            elif model == "simlpe_wave":
                optimizer.zero_grad()
                recons = simlpe_wave(in_3d)
            elif model == "physmamba":
                optimizer.zero_grad()
                recons = physmamba_model(in_3d)
            elif model in ("SFgDNet", "SplineEqNet"):
                optimizer.zero_grad()
                dual_model = _get_active_dual_model()
                if dual_model is None:
                    raise RuntimeError(f"{model} model is not initialized.")
                recons = dual_model(in_3d)
            elif model == "pgbig":
                optimizer.zero_grad()
                seq_flat = in_3d.reshape(B, input_n, -1)
                last_pose = seq_flat[:, -1:, :]
                padded = torch.cat([seq_flat, last_pose.repeat(1, output_n, 1)], dim=1)
                stage_outputs = pgbig_model(padded)
                pred_full = stage_outputs[0]
                pred_future = pred_full[:, -output_n:, :].reshape(B, T_out, N, 3)
                recons = pred_future
            elif model == "motionmixer":
                optimizer.zero_grad()
                seq_flat = in_3d.reshape(B, input_n, pose_dim)
                tgt_flat = tgt_3d.reshape(B, T_out, pose_dim)
                if mixer_input_scale != 1.0:
                    seq_scaled = seq_flat / mixer_input_scale
                    tgt_scaled = tgt_flat / mixer_input_scale
                else:
                    seq_scaled = seq_flat
                    tgt_scaled = tgt_flat
                if mixer_use_delta:
                    sequences_all = torch.cat([seq_scaled, tgt_scaled], dim=1)
                    diffs = sequences_all[:, 1:, :] - sequences_all[:, :-1, :]
                    diffs = torch.cat([diffs[:, 0:1, :], diffs], dim=1)
                    mixer_in = diffs[:, :input_n, :]
                    pred_scaled = motion_mixer(mixer_in)
                    pred_scaled = _delta_to_absolute(pred_scaled, seq_scaled[:, -1, :])
                else:
                    mixer_in = seq_scaled
                    pred_scaled = motion_mixer(mixer_in)
                pred_flat = pred_scaled * mixer_input_scale if mixer_input_scale != 1.0 else pred_scaled
                recons = pred_flat.view(B, T_out, N, 3)
            elif model == "trajdep":
                optimizer.zero_grad()
                recons = trajdep(in_3d)
            elif model in ("hisrep", "hisrep_hand", "hisrep_rigidwrist"):
                optimizer.zero_grad()
                aux = None
                diffusion_loss = None
                if model == "hisrep_hand":
                    recons, aux = hisrep(in_3d, return_aux=True)
                else:
                    recons = hisrep(in_3d)
            elif model == "eqmotion":
                optimizer.zero_grad()
                recons = eqmotion_model(in_3d)
            elif model == "mamba":
                optimizer.zero_grad()
                recons = bas(in_3d, edge_index_tensor_device)
            elif model == "cde":
                optimizer.zero_grad()
                recons = cde(in_3d, edge_index_tensor_device)
            elif model == "hstgcnn":
                optimizer.zero_grad()
                recons = hst(in_3d, edge_index_tensor_device)
            elif model == "gwnet":
                optimizer.zero_grad()
                recons = gwn(in_3d, edge_index_tensor_device)
            elif model == "last":
                init_pose = in_3d[:, -1, :, :]
                recons = init_pose.unsqueeze(1).expand(B, T_out, N, 3)
            else:
                init_pose = in_3d[:, -1, :, :]
                recons = init_pose.unsqueeze(1).expand(B, T_out, N, 3)

            if gate_logging_enabled:
                dual_model = _get_active_dual_model()
                gate_slow_tensor = getattr(dual_model, "last_gate_slow", None) if dual_model is not None else None
                gate_fast_tensor = getattr(dual_model, "last_gate_fast", None) if dual_model is not None else None
                if gate_slow_tensor is not None and gate_fast_tensor is not None:
                    epoch_gate_slow.append(gate_slow_tensor.view(-1).detach().cpu())
                    epoch_gate_fast.append(gate_fast_tensor.view(-1).detach().cpu())
            if gate_param_logging_enabled:
                params_dict = getattr(dual_model, "last_gate_params", None) if dual_model is not None else None
                if isinstance(params_dict, dict):
                    gate_param_snapshot = {
                        key: value.detach().clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
                        for key, value in params_dict.items()
                    }

            if mask_logging_enabled:
                mask_tensor = getattr(dual_model, "last_mask", None) if dual_model is not None else None
                if mask_tensor is not None:
                    epoch_masks.append(mask_tensor.view(-1).detach().cpu())

            # Loss: MPJPE + bone length regularization

            # Base supervised loss (MPJPE).
            mpjpe_terms = torch.norm(recons - tgt_3d, dim=-1)  # (B, T_out, N)
            per_sample_mpjpe = mpjpe_terms.mean(dim=(1, 2))
            per_sample_mpjpe_norm = per_sample_mpjpe * norm_factor_t
            mpjpe_loss = per_sample_mpjpe.mean()
            mpjpe_running += float(per_sample_mpjpe.sum().item())
            mpjpe_norm_running += float(per_sample_mpjpe_norm.sum().item())

            apply_supervised_losses = not (model == "twostage_dct_diffusion" and twostage_phase == "diffusion")
            supervised_tgt_3d = tgt_3d
            if (
                model == "twostage_dct_diffusion"
                and twostage_phase == "coarse"
                and coarse_target_lowpass_only
            ):
                with torch.no_grad():
                    supervised_tgt_3d = twostage_model.coarse.lowpass_future_target(in_3d, tgt_3d)
            loss = mpjpe_loss

            if model == "hisrep_hand":
                gt_dirs, _ = hisrep_handmodule.compute_dirs_lens(tgt_3d, hisrep.parents)
                loss = factorized_loss(
                    pred_xyz=recons,
                    gt_xyz=tgt_3d,
                    pred_dirs=aux["pred_dirs_full"],
                    gt_dirs=gt_dirs,
                    dir_mask=hisrep_handdir_mask,
                    pred_wrists=aux["pred_wrists"],
                    delta_lengths=aux.get("delta_lengths"))
            else:
                if model == "twostage_dct_diffusion" and twostage_phase == "diffusion":
                    if diffusion_loss is None:
                        raise RuntimeError("Expected diffusion_loss to be set for twostage_dct_diffusion.")
                    loss = diffusion_loss
                else:
                    # Optional bone-length loss on hand skeleton
                    if bone_loss_weight > 0:
                        supervised_mpjpe_loss = torch.norm(recons - supervised_tgt_3d, dim=-1).mean()
                        bl_loss = U.bone_length_loss_edges(recons, supervised_tgt_3d, edges_t)
                        loss = supervised_mpjpe_loss + float(bone_loss_weight) * bl_loss
                    else:
                        loss = torch.norm(recons - supervised_tgt_3d, dim=-1).mean()
            
            if model == "SplineEqNet":
                reg_model = _get_active_dual_model()
                lambda_l1 = config.get("SplineEqNet_lambda_l1")
                lambda_div = config.get("SplineEqNet_lambda_div")
                if negpearson_weight > 0:
                    neg_pearson = U.neg_pearson_loss(recons, tgt_3d)
                    loss = loss + negpearson_weight * neg_pearson
                if reg_model is not None:
                    if lambda_l1 not in (None, 0, 0.0):
                        loss = loss + reg_model.gate_usage_l1(float(lambda_l1))
                    if lambda_div not in (None, 0, 0.0):
                        loss = loss + reg_model.gate_diversity_loss(float(lambda_div))
            
            last_observed_pose = in_3d[:, -1, :, :]
            pred_velocity_mag = _velocity_magnitude(recons, last_observed_pose)
            tgt_velocity_mag = _velocity_magnitude(supervised_tgt_3d, last_observed_pose)
            vel_mae_per_sample = (pred_velocity_mag - tgt_velocity_mag).abs().mean(dim=(1, 2))
            vel_loss = vel_mae_per_sample.mean()
            if apply_supervised_losses and velocity_loss_weight > 0:
                loss = loss + velocity_loss_weight * vel_loss
            vel_running += float(vel_mae_per_sample.sum().item())
            if diffusion_loss is not None:
                diffusion_running += float(diffusion_loss.item()) * inp.size(0)

            # Abort immediately if loss becomes NaN/Inf during any batch
            if not torch.isfinite(loss):
                print("[NaN detected] Loss is NaN/Inf; aborting training and returning NaN metrics.")
                return {
                    "test_mpjpe": float('nan'),
                    "test_weighted_mae": float('nan'),
                }

            if model in ("mpt", "estag", "simlpe", "simlpe_dct", "twostage_dct_diffusion", "simlpe_wave", "physmamba", "SFgDNet", "SplineEqNet", "trajdep", "hisrep", "hisrep_hand", "hisrep_rigidwrist", "eqmotion", "pgbig", "motionmixer", "mamba", "cde", "hstgcnn", "gwnet"):
                loss.backward()
                max_norm = float(config.get("gradient_clip", 5.0))
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
                optimizer.step()

            running += float(loss.item()) * inp.size(0)
            # Scoring loss (weighted joint MAE) as in test_prediction.py (not used for backprop)
            with torch.no_grad():
                score_val = U.weighted_joint_loss(joint_weights, recons, tgt_3d, metric='mae')
                score_running += float(score_val.item())
            n_seen += inp.size(0)

        if mask_logging_enabled and epoch_masks:
            epoch_stack = torch.stack(epoch_masks, dim=0)
            mask_epoch_history.append(epoch_stack.mean(dim=0))
        if gate_logging_enabled and epoch_gate_slow and epoch_gate_fast:
            slow_mean = torch.stack(epoch_gate_slow, dim=0)
            fast_mean = torch.stack(epoch_gate_fast, dim=0)
            gate_epoch_history.append(torch.stack([slow_mean, fast_mean], dim=0))

        epoch_loss = running / max(1, n_seen)
        dt = time.time() - t0
        score_avg = score_running / max(1, n_seen)
        mpjpe_avg = mpjpe_running / max(1, n_seen)
        mpjpe_norm_avg = mpjpe_norm_running / max(1, n_seen)
        vel_avg = vel_running / max(1, n_seen)
        diffusion_avg = diffusion_running / max(1, n_seen)
        objective_train_loss = diffusion_avg if (model == "twostage_dct_diffusion" and twostage_phase == "diffusion") else epoch_loss
        vel_clause = f"| vel_mae={vel_avg:.6f} "
        phase_clause = ""
        if model == "twostage_dct_diffusion":
            phase_clause = f"| phase={twostage_phase} | diff_loss={diffusion_avg:.6f}"
        print(
            f"[Epoch {epoch:03d}] loss={objective_train_loss:.6f} | mpjpe={mpjpe_avg:.6f} | "
            f"mpjpe_norm={mpjpe_norm_avg:.6f}{vel_clause}"
            f"{phase_clause} | score(mae-wj)={score_avg:.6f} | time={dt:.1f}s | lr={optimizer.param_groups[0]['lr'] if optimizer is not None else 'n/a'}"
        )

        if best_train_mpjpe is None or mpjpe_avg < best_train_mpjpe:
            best_train_mpjpe = mpjpe_avg
            best_train_mpjpe_norm = mpjpe_norm_avg
            best_train_epoch = epoch
        if best_train_velocity_mae is None or vel_avg < best_train_velocity_mae:
            best_train_velocity_mae = vel_avg

        if bool(config.get("log_gcn_stats", False)) and model == "mpt":
            layers = U.find_semskeconvs(gen_x, gen_y, gen_z, gen_v)
            U.print_semskeconv_stats(U.semskeconv_stats(layers), prefix=f"[GCN][epoch {epoch:03d}]")

    # -----------------
    # Test/Evaluation
    # -----------------
        avg_loss = None
        avg_loss_norm = None
        avg_score = None
        avg_velocity_mae = None
        total_samples = 0
        test_metrics = None
        if model != "twostage_dct_diffusion" and test_loader is not None:
            test_metrics = _evaluate_loader(test_loader, collect_examples=False, save_examples=save_eval_examples)
        if test_metrics is not None:
            avg_loss = float(test_metrics["mpjpe"])
            avg_loss_norm = float(test_metrics["mpjpe_norm"])
            avg_score = float(test_metrics["weighted_mae"])
            avg_velocity_mae = float(test_metrics.get("velocity_mae", float("nan")))
            total_samples = int(test_metrics["samples"])
            vel_print = "nan"
            if math.isfinite(avg_velocity_mae):
                vel_print = f"{avg_velocity_mae:.6f}"
            print(
                f"[Test] loss={avg_loss:.6f} | score(mae-wj)={avg_score:.6f} | loss(norm)={avg_loss_norm:.6f} | "
                f"vel_mae={vel_print} | samples={total_samples}"
            )

            if best_test_mpjpe is None or avg_loss < best_test_mpjpe:
                best_test_mpjpe = avg_loss
                best_test_mpjpe_norm = avg_loss_norm
                best_test_epoch = epoch
                best_test_state = _capture_state()
                print(f"[Info] New best test MPJPE {best_test_mpjpe:.6f} at epoch {epoch}")

        if log_wandb and wandb_handle is not None:
            wandb_metrics: Dict[str, float] = {}
            _maybe_add_metric(wandb_metrics, "epoch", float(epoch))
            _maybe_add_metric(wandb_metrics, "train/loss", objective_train_loss)
            _maybe_add_metric(wandb_metrics, "train/mpjpe", mpjpe_avg)
            _maybe_add_metric(wandb_metrics, "train/mpjpe_norm", mpjpe_norm_avg)
            _maybe_add_metric(wandb_metrics, "train/weighted_mae", score_avg)
            _maybe_add_metric(wandb_metrics, "train/velocity_mae", vel_avg)
            if model == "twostage_dct_diffusion":
                _maybe_add_metric(wandb_metrics, "train/twostage_phase", 0.0 if twostage_phase == "coarse" else 1.0)
                _maybe_add_metric(wandb_metrics, "train/diffusion_loss", diffusion_avg)
            _maybe_add_metric(wandb_metrics, "train/time_s", dt)
            lr_value = optimizer.param_groups[0]['lr'] if optimizer is not None else None
            _maybe_add_metric(wandb_metrics, "train/lr", lr_value)
            if avg_loss is not None:
                _maybe_add_metric(wandb_metrics, "test/loss", avg_loss)
                _maybe_add_metric(wandb_metrics, "test/loss_norm", avg_loss_norm)
                _maybe_add_metric(wandb_metrics, "test/weighted_mae", avg_score)
                if avg_velocity_mae is not None:
                    _maybe_add_metric(wandb_metrics, "test/velocity_mae", avg_velocity_mae)
                _maybe_add_metric(wandb_metrics, "test/samples", float(total_samples))
            if wandb_metrics:
                wandb_handle.log(wandb_metrics, step=epoch)

        # Step LR scheduler based on validation loss if enabled
        if avg_loss is not None:
            if scheduler is not None:
                # Prefer validation loss when available
                scheduler.step(float(avg_loss))
        else:
            # Step LR scheduler based on training loss if no validation loader
            if scheduler is not None:
                scheduler.step(float(objective_train_loss))

        if early_stop_enabled:
            if early_stop_monitor == "test_mpjpe":
                monitored = avg_loss
            elif early_stop_monitor == "train_mpjpe":
                monitored = mpjpe_avg
            elif early_stop_monitor in {"train_loss", "loss"}:
                monitored = objective_train_loss
            else:
                monitored = avg_loss if avg_loss is not None else objective_train_loss
            if monitored is None or not math.isfinite(float(monitored)):
                print("[EarlyStop] Monitored value is not finite; skipping early-stop check for this epoch.")
            else:
                early_stop_stage_epoch = epoch - early_stop_stage_start_epoch + 1
                if early_stop_stage_epoch > early_stop_warmup:
                    monitored_f = float(monitored)
                    improved = early_stop_best is None or (monitored_f < (float(early_stop_best) - early_stop_min_delta))
                    if improved:
                        early_stop_best = monitored_f
                        early_stop_bad_epochs = 0
                    else:
                        early_stop_bad_epochs += 1
                        if early_stop_bad_epochs >= early_stop_patience:
                            if model == "twostage_dct_diffusion" and twostage_phase == "coarse" and twostage_diffusion_epochs > 0:
                                train_epochs = epoch
                                early_stop_best = None
                                early_stop_bad_epochs = 0
                                print(
                                    f"[EarlyStop] Coarse stage stopped at epoch {epoch}; "
                                    "switching to twostage diffusion stage."
                                )
                            else:
                                print(
                                    f"[EarlyStop] Stopping at epoch {epoch} "
                                    f"(monitor={early_stop_monitor or 'auto'}, best={early_stop_best:.6f}, "
                                    f"current={monitored_f:.6f}, patience={early_stop_patience}, min_delta={early_stop_min_delta})."
                                )
                                break

    last_state = _capture_state()

    if mask_logging_enabled and mask_epoch_history:
        mask_matrix = torch.stack(mask_epoch_history, dim=0)
        torch.save(mask_matrix, mask_save_path)
    if gate_logging_enabled and gate_epoch_history:
        gate_matrix = torch.stack(gate_epoch_history, dim=0)
        torch.save(gate_matrix, gate_save_path)
    if gate_param_logging_enabled and gate_param_snapshot is not None:
        torch.save(gate_param_snapshot, gate_param_save_path)
    if maybe_save_coarse_model is not None:
        maybe_save_coarse_model()

    val_metrics_best = None
    if val_loader is not None:
        state_snapshot = _capture_state()
        if best_test_state is not None:
            _load_state(best_test_state)
        val_metrics_best = _evaluate_loader(val_loader, collect_examples=True, restore_train=False, save_examples=save_eval_examples)
        if state_snapshot is not None:
            _load_state(state_snapshot)

    if val_metrics_best is not None:
        val_mpjpe_best = float(val_metrics_best["mpjpe"])
        val_mpjpe_norm_best = float(val_metrics_best["mpjpe_norm"])
        val_weighted_mae_best = float(val_metrics_best["weighted_mae"])
        val_velocity_mae_best = float(val_metrics_best.get("velocity_mae", float("nan")))
        val_humanmac_apd_best = float(val_metrics_best.get("humanmac_apd", float("nan")))
        val_humanmac_ade_best = float(val_metrics_best.get("humanmac_ade", float("nan")))
        val_humanmac_fde_best = float(val_metrics_best.get("humanmac_fde", float("nan")))
        val_humanmac_mmade_best = float(val_metrics_best.get("humanmac_mmade", float("nan")))
        val_humanmac_mmfde_best = float(val_metrics_best.get("humanmac_mmfde", float("nan")))
        val_humanmac_cmd_best = float(val_metrics_best.get("humanmac_cmd", float("nan")))
        val_humanmac_fid_best = float(val_metrics_best.get("humanmac_fid", float("nan")))
        val_samples = int(val_metrics_best["samples"])
        ref_epoch = best_test_epoch if best_test_epoch is not None else "final"
        print(
            f"[Validation @ epoch {ref_epoch}] mpjpe={val_mpjpe_best:.6f} | "
            f"mpjpe_norm={val_mpjpe_norm_best:.6f} | "
            f"score(mae-wj)={val_weighted_mae_best:.6f} | vel_mae={val_velocity_mae_best:.6f} | samples={val_samples}"
        )
        if bool(config.get("compute_humanmac_metrics", False)):
            print(
                f"[Validation HumanMAC @ epoch {ref_epoch}] APD={val_humanmac_apd_best:.6f} | "
                f"ADE={val_humanmac_ade_best:.6f} | FDE={val_humanmac_fde_best:.6f} | "
                f"MMADE={val_humanmac_mmade_best:.6f} | MMFDE={val_humanmac_mmfde_best:.6f} | "
                f"CMD={val_humanmac_cmd_best:.6f} | FID={val_humanmac_fid_best:.6f}"
            )
    else:
        val_mpjpe_best = None
        val_mpjpe_norm_best = None
        val_weighted_mae_best = None
        val_velocity_mae_best = None
        val_humanmac_apd_best = None
        val_humanmac_ade_best = None
        val_humanmac_fde_best = None
        val_humanmac_mmade_best = None
        val_humanmac_mmfde_best = None
        val_humanmac_cmd_best = None
        val_humanmac_fid_best = None
        val_samples = 0

    best_model_path = config.get("best_model_path")
    if best_model_path:
        if best_test_state is not None:
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(
                {
                    "best_model_state": best_test_state,
                    "metadata": {
                        "tag": config.get("best_model_tag"),
                        "model": model,
                        "dataset": config.get("dataset"),
                    },
                },
                best_model_path,
            )
            print(f"Saved best model checkpoint to {best_model_path}")
        elif model == "twostage_dct_diffusion" and last_state is not None:
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(
                {
                    "best_model_state": last_state,
                    "metadata": {
                        "tag": config.get("best_model_tag"),
                        "model": model,
                        "dataset": config.get("dataset"),
                        "note": "last_state_after_diffusion",
                    },
                },
                best_model_path,
            )
            print(f"Saved last twostage checkpoint to {best_model_path}")
        else:
            print(f"Best model checkpoint was requested but no best state was captured for {best_model_path}")

    return {
        "train_mpjpe_best": float(best_train_mpjpe) if best_train_mpjpe is not None else None,
        "train_mpjpe_norm_best": float(best_train_mpjpe_norm) if best_train_mpjpe_norm is not None else None,
        "train_best_epoch": int(best_train_epoch) if best_train_epoch is not None else None,
        "train_mpjpe_best": float(best_train_mpjpe) if best_train_mpjpe is not None else None,
        "train_mpjpe_norm_best": float(best_train_mpjpe_norm) if best_train_mpjpe_norm is not None else None,
        "validation_mpjpe_best": val_mpjpe_best if val_loader is not None else None,
        "validation_mpjpe_norm_best": val_mpjpe_norm_best if val_loader is not None else None,
        "validation_humanmac_apd_best": val_humanmac_apd_best if val_loader is not None else None,
        "validation_humanmac_ade_best": val_humanmac_ade_best if val_loader is not None else None,
        "validation_humanmac_fde_best": val_humanmac_fde_best if val_loader is not None else None,
        "validation_humanmac_mmade_best": val_humanmac_mmade_best if val_loader is not None else None,
        "validation_humanmac_mmfde_best": val_humanmac_mmfde_best if val_loader is not None else None,
        "validation_humanmac_cmd_best": val_humanmac_cmd_best if val_loader is not None else None,
        "validation_humanmac_fid_best": val_humanmac_fid_best if val_loader is not None else None,
        "validation_samples": float(val_samples) if val_loader is not None else None,
        "validation_mpjpe_by_try": (
            val_metrics_best.get("mpjpe_by_try") if val_metrics_best is not None else None
        ),
        "validation_mpjpe_norm_by_try": (
            val_metrics_best.get("mpjpe_norm_by_try") if val_metrics_best is not None else None
        ),
        "test_mpjpe_best": best_test_mpjpe if test_loader is not None else None,
        "test_mpjpe_norm_best": best_test_mpjpe_norm if test_loader is not None else None,
        "params": params_count,
    }
