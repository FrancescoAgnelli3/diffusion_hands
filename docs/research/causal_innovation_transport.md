# Linear Causal Innovation Transport

CIT-linear models the residual between a frozen coarse hand forecast and the
future trajectory. The wrist is removed, and each remaining joint follows a
calibrated linear causal innovation process

```text
r_t = rho r_{t-1} + sigma_t u_t.
```

`rho` and `sigma_t` are estimated in two deterministic passes over the training
split. A conditional velocity field is then trained by flow matching in this
invertible innovation chart. At inference, Gaussian innovations are integrated
with the configured Euler budget and decoded back to Cartesian residuals.

The implementation requires a frozen coarse predictor, history and coarse
conditioning, and linear propagation.

## Grip-phase ablations

The retained grip-phase variants resample the coarse and target trajectories on
a monotone hand-closure clock, train the same CIT-linear field in canonical
phase, and map generated trajectories back to physical time. The coupled
variants condition the articulation field on the sampled clock. See
[cit_next_directions.md](cit_next_directions.md) and
[cit_grip_phase_next.md](cit_grip_phase_next.md).

## Launch

```bash
cd /home/fagnelli/diffusion_hands
bash scripts/run_generative_research.sh \
  --config configs/ablations/cit_grip_phase_next_research.yaml \
  --models cit_linear \
  --output results/causal_innovation_research
```

The shared protocol uses the `pick_up_screwd` Assembly action, official splits,
seeds 0/1/2, 100 coarse epochs, 200 generative epochs, 10 candidates, and 25
Euler steps.
