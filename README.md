# CARD: Conditional Autoregressive Diffusion for Hand Motion Forecasting

Training and evaluation workspace for CARD, a diffusion-based hand-motion forecasting model, with reproducible experiment configs, ablation runners, and dataset-specific orchestration.

## Project Layout

- `configs/experiment.yaml`: main experiment configuration
- `configs/models/*.yaml`: per-model defaults
- `scripts/run_all_models.sh`: entrypoint script
- `tools/run_all_models.py`: orchestrator
- `results/all_models_metrics_long.csv`: aggregate metrics
- `vendor/`: model repositories used by the orchestrator

## Environment Setup

Use `uv` to create and manage the local virtual environment.

Create `.venv` and install dependencies:

```bash
cd /path/to/diffusion_hands
uv venv .venv
uv pip install -r requirements.txt
```

Activate it:

```bash
source .venv/bin/activate
```

## Running

### Configure dataset locations

The repository does not include any datasets. Before running an experiment,
copy `configs/experiment.yaml` to a local file and add a `data_roots` mapping
that points to your dataset installations. Keep the local file outside the
repository or do not commit it, because paths are machine-specific.

```yaml
# configs/experiment.local.yaml
data_roots:
  assembly: /path/to/assembly/data
  bighands: /path/to/bighands/data
  h2o: /path/to/h2o/data
  fpha: /path/to/fpha/data
```

The selected datasets and enabled models in the same file determine what is
run. `action_filter` applies only to the Assembly dataset. Start with one
dataset, one action, and one enabled model when validating a new installation.

### Standard experiment

Run the three-seed configuration on the GPU selected by `gpu_index`:

```bash
cd /path/to/diffusion_hands
bash scripts/run_all_models.sh configs/experiment.local.yaml
```

`configs/experiment.yaml` runs seeds `0`, `1`, and `2` sequentially on its
configured GPU. Per-seed results are written to
`results/all_models_three_seeds_long.csv`; complete three-seed cohorts are
aggregated into `results/all_models_three_seeds_mean_std.csv` using the sample
standard deviation. An incomplete cohort is retained in both files but has
blank mean/std fields, and the launcher exits unsuccessfully after attempting
all configured runs.

### Run the forecasting-horizon suite

The E1–E5 suite evaluates five observation/horizon pairs under one fixed
protocol. It inherits dataset selection, data roots, action filters, enabled
models, stride, and metrics from its base experiment configuration; it changes
only `input_n`, `output_n`, and the number of generated trajectories (`K=20`).
By default, `configs/forecasting_suite.yaml` uses
`configs/experiment.yaml` as its base configuration and runs on GPU 1.

| Experiment | Past | Horizon | Objective |
|---|---:|---:|---|
| E1 | 30 | 10 | Short-term forecasting |
| E2 | 30 | 30 | Main benchmark |
| E3 | 30 | 60 | Long-term forecasting |
| E4 | 15 | 30 | Short context |
| E5 | 60 | 30 | Long context |

First, validate the resolved configurations without loading data or training:

```bash
bash scripts/run_forecasting_suite_gpu1.sh \
  --base-config configs/experiment.local.yaml \
  --dry-run
```

Then launch the full suite. The suite's physical GPU is set by `gpu_index` in
`configs/forecasting_suite.yaml`; copy that file and pass it with
`--suite-config` if your machine uses a different GPU index.

```bash
bash scripts/run_forecasting_suite_gpu1.sh \
  --base-config configs/experiment.local.yaml
```

The launcher creates a timestamped suite directory under
`results/forecasting_suite/`, unless `--output-root` is provided. Each
experiment directory contains:

- `resolved_config.yaml`: exact configuration passed to the main runner;
- `metrics_long.csv`: one row per completed model/seed run;
- `metrics_mean_std.csv`: aggregate metrics for complete seed cohorts;
- `metrics_by_action.csv`: per-action aggregate metrics; and
- `runs/`: backend-specific training and evaluation artifacts.

For an iterative run, choose explicit experiments and an explicit output
directory:

```bash
bash scripts/run_forecasting_suite_gpu1.sh \
  --base-config configs/experiment.local.yaml \
  --experiments E2 E3 \
  --output-root results/forecasting_suite/main_benchmark
```

The suite runs later experiments even if an earlier one fails, then exits
nonzero after all selected experiments finish. To reuse the most recent suite
directory after correcting an issue, run:

```bash
bash scripts/run_forecasting_suite_gpu1.sh \
  --base-config configs/experiment.local.yaml \
  --resume-latest
```

`--resume-latest` (also available as `--keep-going`) selects the latest suite
directory only when `--output-root` is omitted. The main runner determines
which model/seed rows are already complete from the existing metric files.

The retained grip-phase CIT ablations use the fixed multi-seed research runner:

```bash
bash scripts/run_generative_research.sh \
  --config configs/ablations/cit_grip_phase_next_research.yaml \
  --output results/cit_grip_phase_next_research
```

Run a custom config:

```bash
bash scripts/run_all_models.sh configs/experiment.yaml
```

### CARD-family variants

Three comparable entries use the shared SplineEqNet training and evaluation
pipeline:

- `card`: time-domain residual diffusion with spatial anisotropy and identity
  temporal factor (`V = I`)
- `star_hand`: time-domain residual diffusion with spatial and temporal
  anisotropy
- `fihard`: future-DCT residual diffusion with spatial and frequency
  anisotropy

Enable any combination in `configs/experiment.yaml`:

```yaml
models:
  card:
    enabled: true
  star_hand:
    enabled: true
  fihard:
    enabled: true
```

Their defaults are in `configs/models/card.yaml`,
`configs/models/star_hand.yaml`, and `configs/models/fihard.yaml`.

### Bayesian paired comparison with CoMusion

After training all four models on the same split, run the existing paired
Bayesian signed-rank workflow with their final checkpoints:

```bash
python tools/bayesian_signed_rank_test.py \
  --card-checkpoint out/.../card/checkpoints/final.pt \
  --star-hand-checkpoint out/.../star_hand/checkpoints/final.pt \
  --fihard-checkpoint out/.../fihard/checkpoints/final.pt \
  --comusion-checkpoint out/.../comusion/checkpoints/final.pt \
  --dataset assembly \
  --action-filter pick_up_screwd
```

The workflow evaluates every model on identical test samples and tests all six
unique pairs: the three CARD-family models against CoMusion and against one
another. It writes:

- `all_sample_metrics.csv`, containing aligned per-sample metrics
- `paired_metric_tables/<left>_vs_<right>/`, containing each paired input
- `reports/`, containing the autorank Bayesian signed-rank reports
- `bayesian_pairwise_summary.csv`, containing posterior probabilities that
  the left model is better, practically equivalent, or the right model is
  better

As in the original workflow, APD is retained in the paired metric tables but
excluded from signed-rank inference; ADE, FDE, MMADE, MMFDE, CMD, and FID are
tested with ascending (lower-is-better) ordering.

To force the evaluation onto physical GPU 0, use the launcher:

```bash
bash scripts/run_bayesian_signed_rank_gpu0.sh \
  out/.../card/checkpoints/final.pt \
  out/.../star_hand/checkpoints/final.pt \
  out/.../fihard/checkpoints/final.pt \
  out/.../comusion/checkpoints/final.pt \
  --dataset assembly \
  --action-filter pick_up_screwd
```

If checkpoints do not exist yet, train all four models sequentially on GPU 1
with 70 observed and 30 predicted frames:

```bash
bash scripts/train_card_variants_gpu0.sh
```

To train and immediately run all paired Bayesian comparisons:

```bash
RUN_BAYESIAN_AFTER_TRAIN=1 bash scripts/train_card_variants_gpu0.sh
```

The training launcher uses
`configs/train_card_variants_vs_comusion.yaml`, verifies that every model
produced a new `final.pt`, prints all four paths, and prints the complete
Bayesian comparison command. Edit `data_roots.assembly` in that YAML if the
dataset is stored elsewhere.

The main launcher scripts are:

- `scripts/run_all_models.sh`
- `scripts/profile_all_models.sh`

## Profiling model efficiency

Profile every model from random initialization on one synthetic batch, without
loading data or checkpoints, on physical GPU 0:

```bash
bash scripts/profile_all_models_random_gpu0.sh \
  --batch-size 1 \
  --num-candidates 1 \
  --warmup 5 \
  --repeats 30
```

This freshly runs the complete inference path on the same randomly generated
input shape for every model and writes `results/all_model_efficiency.*`. The
warm-up and timed repetitions reuse that tensor; they do not access datasets,
checkpoints, earlier profile rows, or update model weights. Increase
`--num-candidates` to measure multi-hypothesis sampling (up to 10 for DLow).

Profile the models already enabled in `configs/experiment.yaml`:

```bash
bash scripts/profile_all_models.sh \
  --config configs/experiment.yaml \
  --warmup 5 \
  --repeats 30
```

To enable every configured model (and add model configs that are not listed in
the experiment file), pass `--enable-all`:

```bash
bash scripts/profile_all_models.sh \
  --config configs/experiment.yaml \
  --enable-all
```

The normal runner trains models before its final evaluation, so `--enable-all`
can be expensive when checkpoints/runs are missing. Profiling uses one real test
example, the experiment's `num_candidates`, and each model's complete sampling
path. It writes:

- `results/model_efficiency.jsonl`: raw records and environment metadata
- `results/model_efficiency.csv`: flat machine-readable table
- `results/model_efficiency.md`: publication-style Markdown table

Rebuild CSV and Markdown without launching models again:

```bash
bash scripts/profile_all_models.sh --aggregate-only
```

Retry only selected models while preserving existing JSONL records:

```bash
bash scripts/profile_all_models.sh \
  --models card star_hand fihard dlow_cvae \
  --append
```

Latency is the median CUDA-event time after warm-up. Peak memory is model
parameter/buffer storage plus allocations made by the profiled sampling call,
which prevents optimizer state retained by a training script from contaminating
inference memory. FLOPs use PyTorch's operation-level flop counter and cover the
whole iterative sampler; unsupported vendor operations are recorded in the
`flop_error` column rather than silently reported as zero.

## Experiment Config

`configs/experiment.yaml` controls:

- global options: `seed`, `gpu_index`, `num_candidates`, `humanmac_multimodal_threshold`, `save_model`
- shared preprocessing: `preprocessing.input_n/output_n/stride/...`
- dataset selection:
  - one or more datasets with `datasets: [assembly, h2o, bighands, fpha]`
- model enable switches under `models.<model_name>.enabled`

### Defaults in Code

If omitted from `experiment.yaml`, these are provided by `tools/run_all_models.py`:

- `data_roots`
  - `assembly`: `change_with_your_dataset_path`
  - `h2o`: `change_with_your_dataset_path`
  - `bighands`: `change_with_your_dataset_path`
  - `fpha`: `change_with_your_dataset_path`
- `runtime`
  - `output_root`: `results/`
  - `aggregate_csv`: `results/all_models_metrics_long.csv`
- model config path fallback:
  - `configs/models/{model_name}.yaml`

### Action Filter Behavior

`action_filter` is applied only for `assembly`.
It can be a single string or a list of action strings. When a list is provided, the runner processes each assembly action as a separate run, similar to how `datasets` are processed.
For all other datasets (`h2o`, `bighands`, `fpha`), the runner forces an empty filter.

## Outputs

- Aggregate metrics are appended to:
  - `results/all_models_metrics_long.csv`
- Model-specific training artifacts are written in vendor folders:
  - `vendor/splineeqnet/out/diffusion_hands_runs/card/<run_id>/`
  - `vendor/skeletondiffusion/out/diffusion_hands_runs/skeletondiffusion/<run_id>/`
  - other models store artifacts in their respective `vendor/<model>/results/...` folders (some are cleaned by the runner after metric extraction)
