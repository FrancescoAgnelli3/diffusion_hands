# Grip-phase CIT

This experiment extends the CIT-linear endpoint model with a grip-phase clock.
It keeps the frozen coarse predictor, causal temporal innovation calibration,
linear conditional flow, and terminal bone-length pushforward.

## `cit_grip_phase`

Physical frame index is a poor coordinate when the same grasp is performed at
different speeds. This model derives a monotone phase from 13 translation- and
rotation-invariant hand-closure distances. It resamples the future and coarse
forecast onto a uniform grip-phase grid and trains CIT on the resulting
canonical residual trajectory.

A small history-conditioned flow simultaneously models the centered log
increments of the physical clock. At inference it samples positive increments
with a softmax, accumulates them to a monotone clock ending exactly at one,
generates a canonical future, and maps that future back to physical time. The
main trajectory and clock retain their own independent Gaussian sources.

Central hypothesis: separating *what articulation evolves* from *when grasp
events occur* reduces phase averaging without requiring target-dependent
pairings.

## Shared protocol

- Assembly action: `pick_up_screwd`
- No source-file cap
- Official split and data seed 0
- Training seeds 0, 1, and 2
- 70 observed and 30 future frames, stride 10
- 100 coarse epochs and 200 generative epochs
- Batch size 256, no early stopping or loss scheduler
- 10 candidates and 25 Euler steps per candidate
- GPU index 1

The grip-phase model adds 25 evaluations of its small clock MLP per candidate.
This appears as `auxiliary_evals_per_candidate` in `metrics_long.csv`; the
common main-field budget remains 25 in `nfe_per_candidate`.

## Launch

```bash
cd /home/fagnelli/diffusion_hands
bash scripts/run_generative_research.sh \
  --config configs/ablations/cit_next_directions_research.yaml \
  --models cit_grip_phase \
  --output results/cit_next_directions_research
```

Resume an interrupted cohort with the identical command plus `--resume`.
Results are aggregated in `results/cit_next_directions_research/summary.csv`
and per-seed values in `metrics_long.csv`.
