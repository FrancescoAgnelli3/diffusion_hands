"""Grip-phase clock and temporal resampling for CIT ablations."""
from __future__ import annotations

import math

import torch
from torch import nn
TIPS = (0, 1, 2, 3, 4)


def grip_observables(pose):
    """Rotation/translation-invariant distances describing hand closure."""
    thumb = pose[..., TIPS[0], :]
    fingers = pose[..., list(TIPS[1:]), :]
    thumb_distances = (fingers-thumb.unsqueeze(-2)).norm(dim=-1)
    wrist = pose[..., 5, :].unsqueeze(-2)
    wrist_distances = (pose[..., list(TIPS), :]-wrist).norm(dim=-1)
    adjacent = (pose[..., list(TIPS[:-1]), :]
                - pose[..., list(TIPS[1:]), :]).norm(dim=-1)
    return torch.cat((thumb_distances, wrist_distances, adjacent), dim=-1)


def grip_phase(history, future, floor=1e-4, strength=1.0):
    previous = grip_observables(history[:, -1:])
    current = grip_observables(future)
    sequence = torch.cat((previous, current), dim=1)
    speed = (sequence[:, 1:]-sequence[:, :-1]).square().sum(-1).sqrt()
    adaptive_floor = floor * speed.mean(1, keepdim=True).clamp_min(1.0)
    increments = speed + adaptive_floor
    increments = increments / increments.sum(1, keepdim=True).clamp_min(1e-8)
    uniform = torch.full_like(increments, 1.0/increments.size(1))
    phase = ((1-strength)*uniform+strength*increments).cumsum(1)
    logits = increments.clamp_min(1e-8).log()
    logits = logits-logits.mean(1, keepdim=True)
    return phase, logits


def _interpolate(values, knots, queries, initial):
    """Piecewise-linear batched interpolation along the temporal axis."""
    batch, length = knots.shape
    zero = knots.new_zeros(batch, 1)
    knots = torch.cat((zero, knots), dim=1)
    values = torch.cat((initial[:, None], values), dim=1)
    upper = torch.searchsorted(knots.contiguous(), queries.contiguous(), right=False)
    upper = upper.clamp(1, length)
    lower = upper-1
    batch_index = torch.arange(batch, device=values.device)[:, None]
    lo_k, hi_k = knots[batch_index, lower], knots[batch_index, upper]
    weight = ((queries-lo_k)/(hi_k-lo_k).clamp_min(1e-8))
    view = (batch, queries.size(1)) + (1,)*(values.ndim-2)
    lo_v, hi_v = values[batch_index, lower], values[batch_index, upper]
    return lo_v + weight.view(view)*(hi_v-lo_v)


def to_canonical(values, phase, initial):
    length = values.size(1)
    query = torch.arange(1, length+1, device=values.device, dtype=values.dtype)/length
    return _interpolate(values, phase, query[None].expand(values.size(0), -1), initial)


def from_canonical(values, phase, initial):
    length = values.size(1)
    knots = torch.arange(1, length+1, device=values.device, dtype=values.dtype)/length
    return _interpolate(values, knots[None].expand(values.size(0), -1), phase, initial)


class GripPhaseClock(nn.Module):
    def __init__(self, input_length, pred_length, num_free, hidden=128):
        super().__init__()
        self.pred_length = pred_length
        feature_dim = 9*num_free
        self.history_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.velocity = nn.Sequential(
            nn.LayerNorm(hidden+pred_length+4), nn.Linear(hidden+pred_length+4, hidden),
            nn.SiLU(), nn.Linear(hidden, pred_length),
        )
        nn.init.zeros_(self.velocity[-1].weight)
        nn.init.zeros_(self.velocity[-1].bias)

    @staticmethod
    def history_summary(history):
        velocity = history[:, 1:]-history[:, :-1]
        mean_velocity = velocity.mean(1)
        last_velocity = velocity[:, -1]
        acceleration = velocity[:, 1:]-velocity[:, :-1]
        mean_acceleration = acceleration.mean(1) if acceleration.size(1) else torch.zeros_like(mean_velocity)
        return torch.cat((history[:, -1], mean_velocity, last_velocity), dim=-1).flatten(1)

    def field(self, point, time, history):
        context = self.history_encoder(self.history_summary(history))
        embedding = torch.stack((time, time.square(), torch.sin(math.pi*time),
                                 torch.cos(math.pi*time)), dim=-1)
        velocity = self.velocity(torch.cat((context, point, embedding), dim=-1))
        return velocity-velocity.mean(1, keepdim=True)

    def loss(self, history, target, source, time):
        s = time[:, None]
        point = (1-s)*source+s*target
        desired = target-source
        prediction = self.field(point, time, history)
        return (prediction-desired).square().mean()

    @torch.no_grad()
    def sample_with_source(self, history, generator, steps, phase_strength=1.0,
                           normalize_source=False):
        point = torch.randn(history.size(0), self.pred_length, device=history.device,
                            dtype=history.dtype, generator=generator)
        point -= point.mean(1, keepdim=True)
        source = point.clone()
        if normalize_source:
            source = source/source.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            point = source.clone()
        for step in range(steps):
            time = history.new_full((history.size(0),), step/steps)
            point = point+self.field(point, time, history)/steps
            point -= point.mean(1, keepdim=True)
        increments = torch.softmax(point, dim=1)
        uniform = torch.full_like(increments, 1.0/increments.size(1))
        increments = (1-phase_strength)*uniform+phase_strength*increments
        return increments.cumsum(1), point, source

    @torch.no_grad()
    def sample(self, history, generator, steps, phase_strength=1.0):
        phase, point, _ = self.sample_with_source(
            history, generator, steps, phase_strength=phase_strength,
        )
        return phase, point


class GripJointNoiseCoupler(nn.Module):
    """History-conditioned low-rank copula between clock and trajectory noise."""
    def __init__(self, input_length, pred_length, num_free, hidden=128):
        super().__init__()
        self.pred_length = pred_length
        self.num_free = num_free
        feature_dim = 9*num_free
        self.history_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, pred_length*num_free*3),
        )

    def basis(self, history):
        context = GripPhaseClock.history_summary(history)
        raw = self.history_encoder(context).reshape(
            history.size(0), self.pred_length, self.num_free, 3,
        )
        return torch.tanh(raw)

    def couple(self, clock_source, trajectory_noise, history, strength):
        if not 0 < strength < 1:
            raise ValueError("Shared grip-noise strength must lie strictly between zero and one")
        z = clock_source-clock_source.mean(1, keepdim=True)
        z = z/z.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        basis = self.basis(history)
        shared = z[:, :, None, None]*basis
        private_scale = (1-strength**2*basis.square()).clamp_min(1e-6).sqrt()
        return private_scale*trajectory_noise+strength*shared
