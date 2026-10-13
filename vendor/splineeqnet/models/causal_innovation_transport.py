"""Linear Causal Innovation Transport and grip-phase ablations.

The CARD-shaped interface is an adapter for the shared experiment infrastructure.
See docs/research/causal_innovation_transport.md for the model and its controls.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.card import (CardConfig, CoarseDCTForecaster,
                         MRTimeTransformerDenoiser, _assert_shape)
from models.hand_specific_transports import (
    GripJointNoiseCoupler, GripPhaseClock, from_canonical, grip_phase, to_canonical,
)


# AssemblyHands' rooted palm-to-tip tree.  The order is topological: every
# parent has already been reconstructed when its child is visited.
ASSEMBLY_WRIST = 5
ASSEMBLY_PROJECTION_EDGES = (
    (5, 6), (6, 7), (7, 0),
    (5, 8), (8, 9), (9, 10), (10, 1),
    (5, 11), (11, 12), (12, 13), (13, 2),
    (5, 14), (14, 15), (15, 16), (16, 3),
    (5, 17), (17, 18), (18, 19), (19, 4),
    (5, 20),
)


def _rooted_projection_edges(num_nodes, wrist_index, edges):
    """Build a deterministic wrist-rooted projection tree from a hand graph."""
    num_nodes = int(num_nodes)
    wrist_index = int(wrist_index)
    adjacency = {idx: set() for idx in range(num_nodes)}
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            continue
        first, second = int(edge[0]), int(edge[1])
        if (0 <= first < num_nodes and 0 <= second < num_nodes
                and first != second):
            adjacency[first].add(second)
            adjacency[second].add(first)

    visited = {wrist_index}
    queue = deque([wrist_index])
    rooted_edges = []
    while queue:
        parent = queue.popleft()
        for child in sorted(adjacency[parent]):
            if child in visited:
                continue
            visited.add(child)
            queue.append(child)
            rooted_edges.append((parent, child))

    if len(visited) != num_nodes:
        missing = sorted(set(range(num_nodes)) - visited)
        raise ValueError(
            "Terminal kinematic projection requires a connected hand graph; "
            f"nodes unreachable from wrist_index={wrist_index}: {missing}"
        )
    return tuple(rooted_edges)


@dataclass
class CausalInnovationConfig(CardConfig):
    propagation: str = "linear"
    max_persistence: float = 0.98
    scale_floor: float = 1e-4
    terminal_kinematic_projection: bool = False
    grip_phase_transport: bool = False
    grip_clock_hidden_dim: int = 128
    grip_clock_loss_weight: float = 0.1
    grip_phase_strength: float = 1.0
    grip_clock_conditioning: bool = False
    grip_joint_shared_noise: bool = False
    grip_joint_noise_strength: float = 0.35


class CausalInnovationTransport(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.wrist_index = int(metadata["wrist_index"])
        if cfg.num_nodes < 2 or not 0 <= self.wrist_index < cfg.num_nodes:
            raise ValueError("CIT requires at least two nodes and a valid wrist_index")
        self.register_buffer("free_indices", torch.tensor([i for i in range(cfg.num_nodes) if i != self.wrist_index]))
        n = cfg.num_nodes - 1
        self.register_buffer("persistence", torch.zeros(n, 1))
        self.register_buffer("radius", torch.ones(n, 1))
        self.register_buffer("innovation_scale", torch.ones(cfg.pred_length, n, 1))
        self.register_buffer("calibrated", torch.tensor(False))
        self.register_buffer("calibration_windows", torch.tensor(0, dtype=torch.long))
        if cfg.terminal_kinematic_projection:
            available = {tuple(map(int, edge)) for edge in metadata.get("edges", ())}
            undirected = available | {(child, parent) for parent, child in available}
            if cfg.num_nodes != 21:
                raise ValueError(
                    "Terminal kinematic projection requires a 21-joint hand layout"
                )
            if (self.wrist_index == ASSEMBLY_WRIST
                    and set(ASSEMBLY_PROJECTION_EDGES) <= undirected):
                projection_edges = ASSEMBLY_PROJECTION_EDGES
            else:
                projection_edges = _rooted_projection_edges(
                    cfg.num_nodes, self.wrist_index, available,
                )
            self.projection_edges = projection_edges
            # Non-persistent indices keep this inference map checkpoint-compatible
            # with an otherwise identical CIT-linear state dict.
            self.register_buffer(
                "projection_parents",
                torch.tensor([parent for parent, _ in projection_edges]),
                persistent=False,
            )
            self.register_buffer(
                "projection_children",
                torch.tensor([child for _, child in projection_edges]),
                persistent=False,
            )
        self.denoiser = MRTimeTransformerDenoiser(
            in_feat=3*n, t_in=cfg.input_length, t_out=cfg.pred_length,
            d_model=cfg.denoiser_dim, depth=cfg.denoiser_depth,
            n_heads=cfg.denoiser_heads, dropout=cfg.dropout,
            cond_use_history=cfg.cond_use_history, cond_use_coarse=cfg.cond_use_coarse,
        )
        # Begin at the calibrated reference process.
        nn.init.zeros_(self.denoiser.out.weight)
        nn.init.zeros_(self.denoiser.out.bias)
        # Construct proposal-specific heads after the common velocity field so
        # all variants retain identical denoiser initialization for a seed.
        if cfg.grip_phase_transport:
            self.grip_clock = GripPhaseClock(
                cfg.input_length, cfg.pred_length, n, cfg.grip_clock_hidden_dim,
            )
            if cfg.grip_clock_conditioning:
                # Attach conditioning after constructing the common denoiser so
                # its shared parameters retain exactly the control initialization.
                self.denoiser.latent_dim = cfg.pred_length
                self.denoiser.latent_proj = nn.Sequential(
                    nn.Linear(cfg.pred_length, cfg.denoiser_dim), nn.SiLU(),
                    nn.Linear(cfg.denoiser_dim, cfg.denoiser_dim),
                )
                self.denoiser.latent_gate = nn.Parameter(torch.zeros(()))
            if cfg.grip_joint_shared_noise:
                self.grip_noise_coupler = GripJointNoiseCoupler(
                    cfg.input_length, cfg.pred_length, n, cfg.grip_clock_hidden_dim,
                )
    def select(self, value):
        return value.index_select(-2, self.free_indices)

    def propagate(self, residual):
        return self.persistence * residual

    def propagate_jvp(self, residual, direction):
        return self.persistence * direction

    @staticmethod
    def previous(path):
        return torch.cat((torch.zeros_like(path[:, :1]), path[:, :-1]), dim=1)

    def encode(self, residual):
        return (residual - self.propagate(self.previous(residual))) / self.innovation_scale

    def decode(self, innovations):
        state = torch.zeros_like(innovations[:, 0])
        states = []
        for t in range(self.cfg.pred_length):
            state = self.propagate(state) + self.innovation_scale[t] * innovations[:, t]
            states.append(state)
        return torch.stack(states, dim=1)

    def decode_with_jvp(self, innovations, direction):
        state = torch.zeros_like(innovations[:, 0])
        velocity = torch.zeros_like(state)
        states, velocities = [], []
        for t in range(self.cfg.pred_length):
            velocity = self.propagate_jvp(state, velocity) + self.innovation_scale[t] * direction[:, t]
            state = self.propagate(state) + self.innovation_scale[t] * innovations[:, t]
            states.append(state)
            velocities.append(velocity)
        return torch.stack(states, dim=1), torch.stack(velocities, dim=1)

    def inverse_jvp(self, residual, velocity):
        return (velocity - self.propagate_jvp(self.previous(residual), self.previous(velocity))) / self.innovation_scale

    def require_calibrated(self):
        if not bool(self.calibrated):
            raise RuntimeError("CIT reference is uncalibrated; call calibrate_reference on training data after coarse pretraining")

    def field(self, residual, time, history, coarse, phase_context=None):
        scale = self.radius / math.sqrt(3)
        latent_context = None
        if self.cfg.grip_clock_conditioning:
            if phase_context is None or phase_context.shape != (history.size(0), self.cfg.pred_length):
                raise ValueError("Clock-conditioned grip flow requires one clock logit per future frame")
            latent_context = phase_context
        velocity = self.denoiser(
            (residual / scale).flatten(2), 1000 * time,
            self.select(history).flatten(2), self.select(coarse).flatten(2),
            latent_z=latent_context,
        ).reshape_as(residual)
        return velocity * scale

    def flow_path(self, source, residual, time, coarse=None, history=None):
        """Return the physical residual path and its exact generative velocity."""
        s = time[:, None, None, None]
        clean = self.encode(residual)
        return self.decode_with_jvp((1-s)*source+s*clean, clean-source)

    def training_loss(self, history, future, coarse, return_metrics=False):
        self.require_calibrated()
        residual = self.training_residual(history, future, coarse)
        condition_coarse = coarse
        clock_loss = None
        target_phase = None
        if self.cfg.grip_phase_transport:
            target_phase, clock_target = grip_phase(
                history, future, strength=self.cfg.grip_phase_strength,
            )
            condition_coarse = to_canonical(coarse, target_phase, history[:, -1])
            canonical_future = to_canonical(future, target_phase, history[:, -1])
            residual = self.select(canonical_future-condition_coarse)
        time = torch.rand(history.size(0), device=history.device, dtype=history.dtype)
        source = torch.randn_like(residual)
        if self.cfg.grip_phase_transport:
            clock_source = torch.randn_like(clock_target)
            clock_source -= clock_source.mean(1, keepdim=True)
            if self.cfg.grip_joint_shared_noise:
                clock_source = clock_source/clock_source.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
                source = self.grip_noise_coupler.couple(
                    clock_source, source, self.select(history),
                    self.cfg.grip_joint_noise_strength,
                )
            clock_loss = self.grip_clock.loss(
                self.select(history), clock_target, clock_source, time,
            )
        point, target_velocity = self.flow_path(
            source, residual, time, coarse=condition_coarse, history=history,
        )
        predicted_velocity = self.field(
            point, time, history, condition_coarse,
            phase_context=clock_target if self.cfg.grip_clock_conditioning else None,
        )
        error = predicted_velocity - target_velocity
        geometric_error = error.square().mean()
        normalizer = self.radius.square().mean() / 3
        flow_loss = geometric_error / normalizer
        loss = flow_loss
        metrics = {"flow_matching_loss": flow_loss.detach()}
        if clock_loss is not None:
            loss = loss+self.cfg.grip_clock_loss_weight*clock_loss
            metrics["grip_clock_loss"] = clock_loss.detach()
        return (loss, metrics) if return_metrics else loss

    def training_residual(self, history, future, coarse):
        """Return the endpoint used by flow matching in residual coordinates."""
        return self.select(future - coarse)

    def observed_bone_lengths(self, history):
        """Return deterministic per-subject median lengths from observed frames."""
        if not self.cfg.terminal_kinematic_projection:
            raise RuntimeError("Terminal kinematic projection is disabled")
        vectors = (history.index_select(-2, self.projection_children)
                   - history.index_select(-2, self.projection_parents))
        ordered = vectors.norm(dim=-1).sort(dim=1).values
        middle = ordered.size(1) // 2
        if ordered.size(1) % 2:
            lengths = ordered[:, middle]
        else:
            lengths = 0.5 * (ordered[:, middle - 1] + ordered[:, middle])
        return lengths.clamp_min(1e-8)

    def terminal_projection(self, pose, history):
        """Retract a sampled trajectory onto the history-calibrated hand tree.

        Directions come from the unconstrained terminal sample, while lengths
        come only from the observed history.  Reconstruction follows the rooted
        tree, so every output edge has exactly its observed median length.
        """
        if not self.cfg.terminal_kinematic_projection:
            raise RuntimeError("Terminal kinematic projection is disabled")
        lengths = self.observed_bone_lengths(history)
        projected = pose.clone()
        projected[:, :, self.wrist_index] = 0
        for edge, (parent, child) in enumerate(self.projection_edges):
            direction = pose[:, :, child] - pose[:, :, parent]
            norm = direction.norm(dim=-1, keepdim=True)
            fallback = history[:, -1:, child] - history[:, -1:, parent]
            fallback_norm = fallback.norm(dim=-1, keepdim=True)
            axis = torch.zeros_like(fallback)
            axis[..., 0] = 1
            fallback = torch.where(fallback_norm > 1e-8,
                                   fallback / fallback_norm.clamp_min(1e-8), axis)
            unit = torch.where(norm > 1e-8,
                               direction / norm.clamp_min(1e-8), fallback)
            projected[:, :, child] = (
                projected[:, :, parent] + lengths[:, None, edge, None] * unit
            )
        projected[:, :, self.wrist_index] = 0
        return projected

    @torch.no_grad()
    def sample(self, history, coarse, seed):
        self.require_calibrated()
        generator = torch.Generator(device=history.device).manual_seed(int(seed))
        condition_coarse = coarse
        sampled_phase = None
        sampled_clock = None
        sampled_clock_source = None
        if self.cfg.grip_phase_transport:
            sampled_phase, sampled_clock, sampled_clock_source = self.grip_clock.sample_with_source(
                self.select(history), generator, self.cfg.ddim_steps,
                phase_strength=self.cfg.grip_phase_strength,
                normalize_source=self.cfg.grip_joint_shared_noise,
            )
            condition_coarse = to_canonical(coarse, sampled_phase, history[:, -1])
        shape = (history.size(0), self.cfg.pred_length, self.cfg.num_nodes-1, 3)
        innovations = torch.randn(shape, device=history.device, dtype=history.dtype, generator=generator)
        if self.cfg.grip_joint_shared_noise:
            innovations = self.grip_noise_coupler.couple(
                sampled_clock_source, innovations, self.select(history),
                self.cfg.grip_joint_noise_strength,
            )
        steps = self.cfg.ddim_steps
        for step in range(steps):
            coordinate = self.decode(innovations)
            time = history.new_full((history.size(0),), step / steps)
            velocity = self.field(
                coordinate, time, history, condition_coarse,
                phase_context=sampled_clock if self.cfg.grip_clock_conditioning else None,
            )
            innovations = innovations + self.inverse_jvp(coordinate, velocity) / steps
        free_residual = self.decode(innovations)
        result = condition_coarse.clone()
        result[:, :, self.free_indices] += free_residual
        result[:, :, self.wrist_index] = 0
        if sampled_phase is not None:
            result = from_canonical(result, sampled_phase, history[:, -1])
        if self.cfg.terminal_kinematic_projection:
            result = self.terminal_projection(result, history)
        return result


class CausalInnovationForecaster(nn.Module):
    def __init__(self, cfg: CausalInnovationConfig, metadata=None):
        super().__init__()
        if cfg.propagation != "linear":
            raise ValueError("Only linear CIT propagation is supported")
        if not math.isfinite(cfg.scale_floor) or cfg.scale_floor <= 0:
            raise ValueError("CIT scale_floor must be finite and positive")
        if not 0 <= cfg.max_persistence < 1 or cfg.ddim_steps < 1:
            raise ValueError("CIT requires max_persistence in [0,1) and positive solver steps")
        if cfg.terminal_kinematic_projection != cfg.grip_phase_transport:
            raise ValueError("Terminal kinematic projection is reserved for grip-phase CIT")
        if (not isinstance(cfg.grip_clock_hidden_dim, int) or cfg.grip_clock_hidden_dim < 8
                or not math.isfinite(cfg.grip_clock_loss_weight) or cfg.grip_clock_loss_weight <= 0
                or not math.isfinite(cfg.grip_phase_strength)
                or not 0 < cfg.grip_phase_strength <= 1):
            raise ValueError("Invalid grip-phase clock configuration")
        if cfg.grip_clock_conditioning and not cfg.grip_phase_transport:
            raise ValueError("Grip clock conditioning requires grip-phase transport")
        if cfg.grip_joint_shared_noise and not cfg.grip_phase_transport:
            raise ValueError("Shared grip noise requires grip-phase transport")
        if cfg.grip_joint_shared_noise and not cfg.grip_clock_conditioning:
            raise ValueError("Shared grip noise requires clock-conditioned articulation")
        if (not math.isfinite(cfg.grip_joint_noise_strength)
                or not 0 < cfg.grip_joint_noise_strength < 1):
            raise ValueError("Invalid shared grip-noise strength")
        if (cfg.use_latent_variability or cfg.diffusion_only or not cfg.freeze_coarse
                or not cfg.cond_use_history or not cfg.cond_use_coarse):
            raise ValueError("CIT requires frozen coarse/history context and disabled CARD latent/diffusion_only modes")
        self.cfg = cfg
        self.use_latent_variability = False
        self.explainable_latent_decomposition = False
        # Keep construction order identical to CARD for common coarse initialization.
        self.coarse = CoarseDCTForecaster(cfg)
        self.diffusion = CausalInnovationTransport(cfg, metadata or {})

    def _check_history(self, history):
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")

    @torch.no_grad()
    def calibrate_reference(self, train_loader):
        """Two deterministic TRAIN-only passes; no optimizer or training RNG use.

        The caller must supply the training loader. Do not call this on test/val.
        Stored calibration is used as-is for inference and checkpoint reload.
        """
        if bool(self.diffusion.calibrated):
            raise RuntimeError("CIT is already calibrated; refusing to replace stored training statistics")
        previous_mode = self.training
        device = next(self.coarse.parameters()).device
        calibration_loader = DataLoader(
            train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False,
            drop_last=False, num_workers=0, generator=torch.Generator().manual_seed(0),
        )
        self.eval()
        transport = self.diffusion
        n = self.cfg.num_nodes-1
        rms_sum = torch.zeros(n, 1, device=device, dtype=torch.float64)
        lag_sum, previous_sum = torch.zeros_like(rms_sum), torch.zeros_like(rms_sum)
        count = 0
        def residuals():
            for batch in calibration_loader:
                history = batch[0][..., 4:].to(device=device, dtype=torch.float32)
                future = batch[1][..., 4:].to(device=device, dtype=torch.float32)
                self._check_history(history)
                _assert_shape(future, (history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3), "future")
                coarse = self.coarse(history)
                if self.cfg.grip_phase_transport:
                    phase, _ = grip_phase(
                        history, future, strength=self.cfg.grip_phase_strength,
                    )
                    canonical_coarse = to_canonical(coarse, phase, history[:, -1])
                    canonical_future = to_canonical(future, phase, history[:, -1])
                    value = transport.select(canonical_future-canonical_coarse)
                else:
                    value = transport.training_residual(history, future, coarse)
                value = value.double()
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite training residual during CIT calibration")
                yield value

        try:
            for residual in residuals():
                prev = transport.previous(residual)
                rms_sum += residual.square().sum(dim=(0, 1, 3)).unsqueeze(-1)
                lag_sum += (residual * prev).sum(dim=(0, 1, 3)).unsqueeze(-1)
                previous_sum += prev.square().sum(dim=(0, 1, 3)).unsqueeze(-1)
                count += residual.size(0)
            if count == 0:
                raise ValueError("Empty training dataset for CIT calibration")
            radius = (rms_sum / (count*self.cfg.pred_length)).sqrt().clamp_min(self.cfg.scale_floor)
            persistence = (lag_sum / previous_sum.clamp_min(1e-20)).clamp(0, self.cfg.max_persistence)
            transport.radius.copy_(radius)
            transport.persistence.copy_(persistence)
            innovation_sum = torch.zeros_like(transport.innovation_scale, dtype=torch.float64)
            second_count = 0
            for residual in residuals():
                innovation = residual - transport.propagate(transport.previous(residual))
                innovation_sum += innovation.square().sum(dim=(0, 3)).unsqueeze(-1)
                second_count += residual.size(0)
            if second_count != count:
                raise RuntimeError("CIT calibration passes saw different window counts")
            scale = (innovation_sum / (3*count)).sqrt().clamp_min(self.cfg.scale_floor)
            transport.innovation_scale.copy_(scale)
            transport.calibration_windows.fill_(count)
            transport.calibrated.fill_(True)
        finally:
            self.train(previous_mode)
        return {"source_split": "train", "passes": 2, "windows_per_pass": count,
                "propagation": self.cfg.propagation,
                "persistence_min": float(transport.persistence.min()),
                "persistence_max": float(transport.persistence.max()),
                "radius_min": float(transport.radius.min()), "radius_max": float(transport.radius.max()),
                "innovation_scale_min": float(transport.innovation_scale.min()),
                "innovation_scale_max": float(transport.innovation_scale.max())}

    def diffusion_loss(self, history, future_gt, mamp_feat=None, coarse_future=None,
                       allow_coarse_grad=False, return_metrics=False, **unused):
        self._check_history(history)
        _assert_shape(future_gt, (history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3), "future")
        if mamp_feat is not None or allow_coarse_grad:
            raise ValueError("CIT uses fixed coarse calibration and does not support MAMP or joint coarse updates")
        coarse = self.coarse(history) if coarse_future is None else coarse_future
        loss, diffusion_metrics = self.diffusion.training_loss(
            history, future_gt, coarse.detach(), return_metrics=True)
        metrics = {"diffusion_loss": loss.detach(), "total_loss": loss.detach(), **diffusion_metrics}
        return (loss, coarse, metrics) if return_metrics else (loss, coarse)

    @torch.no_grad()
    def predict(self, history, mamp_feat=None, coarse_future=None, deterministic=True,
                seed=0, return_score=False, action_id=None):
        self._check_history(history)
        if mamp_feat is not None:
            raise ValueError("CIT does not support MAMP conditioning")
        if self.training:
            raise RuntimeError("Call eval() before CIT prediction")
        coarse = self.coarse(history) if coarse_future is None else coarse_future
        result = self.diffusion.sample(history, coarse, seed)
        return (result, history.new_zeros(history.size(0))) if return_score else result

    def forward(self, history):
        return self.coarse(history) if self.training else self.predict(history)
