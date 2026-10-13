# CIT-linear coverage and solver audit

`tools/audit_cit_coverage_solver.py` evaluates saved CIT-linear candidates with
best-of-K ADE/FDE, candidate-mean metrics, APD, and the multivariate energy
score. Its optional solver audit compares matched Gaussian initial draws under
Euler and Heun integration without modifying checkpoints or benchmark
artifacts.

```bash
cd /home/fagnelli/diffusion_hands
.venv/bin/python tools/audit_cit_coverage_solver.py \
  --cohort results/cit_grip_phase_next_research \
  --models cit_linear \
  --output results/cit_linear_solver_audit \
  --run-solvers
```

Coverage uses the saved official-test examples. The solver audit uses the same
histories and matched initial residuals for every integration method.
