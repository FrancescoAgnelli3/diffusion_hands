# Joint grip-clock transport

This experiment compares the existing CIT-linear baseline, the basic
hand-specific grip-phase transport, and a new joint grip-clock model under the
same fixed protocol (`pick_up_screwd`, official splits, seeds 0/1/2, 100 coarse
epochs and 200 flow epochs).

The basic grip model learns a monotone latent phase from the observed and
future hand geometry. It uses that phase to canonicalize the target trajectory
and adds a clock prediction loss. Its clock and trajectory innovations are
otherwise sampled independently.

The proposed model makes the clock part of the stochastic hand state. A small
history encoder predicts a bounded direction for every future frame, joint and
Cartesian coordinate. A centered grip-clock source is normalized per sample
and used as a shared scalar source. The trajectory innovation at coordinate
`(t,j,k)` is coupled as

```text
eps' = sqrt(1 - a^2 b^2) eps + a z_clock b,
```

where `b=tanh(g_psi(H))`, `a` is the configurable coupling strength, and
`eps` is the independent innovation. This preserves unit marginal scale while
introducing a history-dependent low-rank copula between progress and residual
articulation. The velocity field is also conditioned on the sampled grip clock.
At test time the clock and coupled trajectory noise are sampled jointly from
the observed history; no future geometry is used.

The coupling is a change to the source stochastic process, rather than an
extra attention block or a loss-only regularizer. The coarse predictor remains
frozen and every target residual remains paired with its own history.

## Launch

```bash
cd /home/fagnelli/diffusion_hands
bash scripts/run_generative_research.sh \
  --config configs/ablations/cit_joint_clock_research.yaml \
  --models cit_linear cit_grip_phase cit_grip_joint \
  --output results/cit_joint_clock_research
```

The plan uses physical GPU 0 and produces nine runs (three models by three
seeds), together with paired deltas against `cit_linear`. Use `--resume` only
with the same output directory and unchanged source/configuration.
