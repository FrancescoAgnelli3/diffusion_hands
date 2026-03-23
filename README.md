# diffusion_hands

Unified runner for training and evaluating multiple hand-motion models (twostage_dct_diffusion, BeLFusion, CoMusion, DLow-CVAE, HumanMAC, SkeletonDiffusion) with one experiment config.

## Project Layout

- `configs/experiment.yaml`: main experiment configuration
- `configs/models/*.yaml`: per-model defaults
- `scripts/run_all_models.sh`: entrypoint script
- `tools/run_all_models.py`: orchestrator
- `results/all_models_metrics_long.csv`: aggregate metrics
- `vendor/`: model repositories used by the orchestrator

## Environment Setup

This repo uses a local virtual environment at `.venv`.

Activate it:

```bash
cd /home/agnelli/projects/diffusion_hands
source .venv/bin/activate
```

Install dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Note: your current `.venv` does not include `pip` by default. If needed, bootstrap it first:

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

## Running

Default run:

```bash
cd /home/agnelli/projects/diffusion_hands
bash scripts/run_all_models.sh
```

Custom config:

```bash
bash scripts/run_all_models.sh configs/experiment.yaml
```

## Experiment Config

`configs/experiment.yaml` controls:

- global options: `seed`, `gpu_index`, `num_candidates`, `humanmac_multimodal_threshold`
- shared preprocessing: `preprocessing.input_n/output_n/stride/...`
- dataset selection:
  - single dataset with `dataset: assembly`
  - multiple datasets with `datasets: [assembly, h2o, bighands, fpha]`
- model enable switches under `models.<model_name>.enabled`

### Defaults in Code

If omitted from `experiment.yaml`, these are provided by `tools/run_all_models.py`:

- `data_roots`
  - `assembly`: `/mnt/turing-datasets/AssemblyHands/assembly101-download-scripts/data_our/`
  - `h2o`: `/mnt/turing-datasets/h2o/`
  - `bighands`: `/mnt/turing-datasets/BigHands/BigHand2.2M/data/`
  - `fpha`: `/mnt/turing-datasets/FPHA/data/`
- `runtime`
  - `output_root`: `/home/agnelli/projects/diffusion_hands/results`
  - `aggregate_csv`: `/home/agnelli/projects/diffusion_hands/results/all_models_metrics_long.csv`
- model config path fallback:
  - `configs/models/{model_name}.yaml`

### Action Filter Behavior

`action_filter` is applied only for `assembly`.
For all other datasets (`h2o`, `bighands`, `fpha`), the runner forces an empty filter.

## Outputs

- Aggregate metrics are appended to:
  - `results/all_models_metrics_long.csv`
- Model-specific training artifacts are written in vendor folders:
  - `vendor/splineeqnet/out/diffusion_hands_runs/twostage_dct_diffusion/<run_id>/`
  - `vendor/skeletondiffusion/out/diffusion_hands_runs/skeletondiffusion_<run_id>/`
  - other models store artifacts in their respective `vendor/<model>/results/...` folders (some are cleaned by the runner after metric extraction)
