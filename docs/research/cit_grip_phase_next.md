# Grip-phase CIT: second-stage experiment

The first grip-phase experiment improved three-seed ADE by 1.57% over CIT
linear and 1.07% over the terminal kinematic pushforward. Its FDE, APD and FID
were worse. This experiment tests two specific explanations instead of changing
the backbone or training budget.

## Soft phase transport

Let `q_t` be the normalized increments derived from hand-internal grip
distances and let `u_t=1/T` be the physical-time clock. The original model used

```
phase_t = sum_{k<=t} q_k.
```

The soft variant uses

```
q_t(gamma) = (1-gamma) u_t + gamma q_t
phase_t(gamma) = sum_{k<=t} q_k(gamma),   gamma=0.5.
```

Every increment remains positive and the last phase remains exactly one. This
retains half of the observed grip alignment while limiting temporal distortion.
It directly tests whether the full warp caused the endpoint and distributional
degradation.

## Clock-conditioned articulation

The first model factorized its conditional law as

```
p(clock | H) p(canonical articulation | H).
```

The coupled variants instead implement

```
p(clock | H) p(canonical articulation | clock, H).
```

At training time the trajectory velocity field receives the centered log
increments of the target grip clock. At inference it receives the clock drawn
by the clock flow. A zero-gated projection injects this vector into the field's
time embedding. The common trajectory denoiser is constructed first, so all
shared parameters have the same initialization as the independent control.

This dependence can represent correlations such as fast closure accompanying
a particular articulation path. It does not pair a future with another
history, use a test target, or change the causal innovation source.

## Models

- `cit_grip_phase_soft`: gamma 0.5, independent clock and articulation.
- `cit_grip_phase_coupled`: gamma 1.0, clock-conditioned articulation.
- `cit_grip_phase_coupled_soft`: gamma 0.5 and clock conditioning.
Together with the completed gamma 1.0 independent model, these form
a 2x2 ablation of warp strength and clock dependence.

## Launch

```bash
cd /home/fagnelli/diffusion_hands
bash scripts/run_generative_research.sh \
  --config configs/ablations/cit_grip_phase_next_research.yaml \
  --models cit_grip_phase_soft cit_grip_phase_coupled \
           cit_grip_phase_coupled_soft \
  --output results/cit_grip_phase_next_research
```

The protocol remains `pick_up_screwd`, official splits, no source-file cap,
seeds 0/1/2, GPU 0, 100 coarse epochs, 200 generative epochs, 10 candidates,
and 25 Euler steps. The three phase variants add 25 small clock-field
evaluations per candidate.
