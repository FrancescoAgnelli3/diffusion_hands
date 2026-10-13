#!/usr/bin/env python3
"""Coverage and numerical-solver audit for trained CIT checkpoints.

By default, coverage reuses the saved ten-candidate examples from the official
test split. Solver comparisons use matched Gaussian draws. No model parameters
or benchmark artifacts are modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/splineeqnet"))
from models.causal_innovation_transport import CausalInnovationConfig, CausalInnovationForecaster


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def trajectory_errors(pred, target):
    """Return candidate-wise ADE/FDE with the repository's flatten-all-joints convention."""
    diff = pred.flatten(3) - target[:, None].flatten(3)
    distance = torch.linalg.vector_norm(diff, dim=-1)
    return distance.mean(-1), distance[:, :, -1]


def energy_score(pred, target):
    """Multivariate energy score, normalized by sqrt(trajectory dimension)."""
    flat = pred.flatten(2)
    truth = target.flatten(1)
    scale = math.sqrt(flat.size(-1))
    observation = torch.linalg.vector_norm(flat-truth[:, None], dim=-1).mean(-1)
    pairwise = torch.cdist(flat, flat).mean((1, 2))
    return (observation-.5*pairwise)/scale


def summarize_candidates(pred, target, ks):
    ade, fde = trajectory_errors(pred, target)
    result = {
        "examples": pred.size(0), "candidates": pred.size(1),
        "candidate_mean_ADE": float(ade.mean()),
        "candidate_mean_FDE": float(fde.mean()),
        "sample_mean_ADE": float(trajectory_errors(pred.mean(1, keepdim=True), target)[0].mean()),
        "sample_mean_FDE": float(trajectory_errors(pred.mean(1, keepdim=True), target)[1].mean()),
    }
    for k in ks:
        if k > pred.size(1):
            raise ValueError(f"Requested K={k}, but artifacts contain only {pred.size(1)} candidates")
        result[f"best_ADE@{k}"] = float(ade[:, :k].min(1).values.mean())
        result[f"best_FDE@{k}"] = float(fde[:, :k].min(1).values.mean())
        result[f"energy_score@{k}"] = float(energy_score(pred[:, :k], target).mean())
        if k == 1:
            result[f"APD@{k}"] = 0.
        else:
            flat = pred[:, :k].flatten(2)
            upper = torch.triu_indices(k, k, offset=1)
            result[f"APD@{k}"] = float(torch.cdist(flat, flat)[:, upper[0], upper[1]].mean())
    return result


def result_rows(result_dir, selected_models):
    rows = list(csv.DictReader((result_dir / "metrics_long.csv").open()))
    return [r for r in rows if r["status"] == "ok" and r["model"] in selected_models]


def sample_path(row):
    return Path(row["checkpoint"]).parent.parent / "eval_samples.npz"


def coverage_audit(cohorts, models, ks):
    records, reference = [], None
    for cohort in cohorts:
        for row in result_rows(cohort, models):
            path = sample_path(row)
            with np.load(path) as data:
                if "pred_all" not in data:
                    raise ValueError(f"No all-candidate predictions in {path}")
                obs, target, pred = data["obs"], data["target"], data["pred_all"]
            # Stored layout is (examples,K,T,J,3); fail closed on older layouts.
            if pred.ndim != 5 or pred.shape[0] != len(obs) or pred.shape[2:] != target.shape[1:]:
                raise ValueError(f"Unexpected saved candidate shape in {path}: {pred.shape}")
            identity = (digest(obs), digest(target))
            if reference is None:
                reference = identity
            elif identity != reference:
                raise RuntimeError("Saved validation examples differ between compared runs")
            metrics = summarize_candidates(torch.from_numpy(pred), torch.from_numpy(target), ks)
            records.append({"cohort": cohort.name, "model": row["model"],
                            "seed": int(row["seed"]), "artifact": str(path), **metrics})
    missing = set(models)-{r["model"] for r in records}
    if missing:
        raise ValueError(f"Models absent from all supplied cohorts: {sorted(missing)}")
    return records


def load_model(row, device):
    payload = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    cfg = CausalInnovationConfig(**metadata["forecaster_config"])
    if cfg.propagation != "linear":
        raise ValueError("Solver audit currently requires linear CIT")
    model = CausalInnovationForecaster(cfg, metadata["graph_metadata"]).to(device).eval()
    model.load_state_dict(payload["final_model_state"]["card"])
    return model


@torch.inference_mode()
def initial_residual(model, history, coarse, seed):
    d = model.diffusion
    generator = torch.Generator(device=history.device).manual_seed(int(seed))
    shape = (history.size(0), d.cfg.pred_length, d.cfg.num_nodes-1, 3)
    noise = torch.randn(shape, device=history.device, dtype=history.dtype, generator=generator)
    return d.decode(noise)


@torch.inference_mode()
def solve(model, history, coarse, initial, method, steps):
    d, value = model.diffusion, initial.clone()
    h = 1./steps
    for index in range(steps):
        time0 = history.new_full((history.size(0),), index/steps)
        first = d.field(value, time0, history, coarse)
        if method == "euler":
            value = value+h*first
        elif method == "heun":
            trial = value+h*first
            time1 = history.new_full((history.size(0),), (index+1)/steps)
            second = d.field(trial, time1, history, coarse)
            value = value+.5*h*(first+second)
        else:
            raise ValueError(f"Unknown solver: {method}")
    output = coarse.clone()
    output[:, :, d.free_indices] += value
    output[:, :, d.wrist_index] = 0
    return output


def parse_solver(value):
    try:
        method, steps = value.split(":", 1)
        steps = int(steps)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Solver must be METHOD:STEPS") from exc
    if method not in {"euler", "heun"} or steps < 1:
        raise argparse.ArgumentTypeError("Solver must use euler/heun and positive steps")
    return method, steps


def solver_audit(cohorts, models, device, candidates, max_examples, solvers):
    records = []
    for cohort in cohorts:
        for row in result_rows(cohort, models):
            model = load_model(row, device)
            with np.load(sample_path(row)) as data:
                histories = torch.from_numpy(data["obs"][:max_examples]).to(device)
                targets = torch.from_numpy(data["target"][:max_examples]).to(device)
            initials = []
            coarse_examples = model.coarse(histories)
            for example_index in range(len(histories)):
                history = histories[example_index:example_index+1]
                coarse = coarse_examples[example_index:example_index+1]
                for candidate in range(candidates):
                    seed = example_index*1000003+candidate
                    initials.append(initial_residual(model, history, coarse, seed))
            initial_batch = torch.cat(initials)
            history_batch = histories.repeat_interleave(candidates, dim=0)
            coarse_batch = coarse_examples.repeat_interleave(candidates, dim=0)
            generated, durations = {}, {}
            started = time.perf_counter()
            for solver in solvers:
                solver_started = time.perf_counter()
                generated[solver] = solve(model,history_batch,coarse_batch,initial_batch,*solver).cpu()
                durations[solver] = time.perf_counter()-solver_started
            tensors = {key:value.reshape(len(histories),candidates,*targets.shape[1:])
                       for key,value in generated.items()}
            reference_key = solvers[-1]
            reference = tensors[reference_key]
            for solver, predictions in tensors.items():
                metrics = summarize_candidates(predictions, targets.cpu(), [1,candidates])
                drift = torch.linalg.vector_norm(
                    (predictions-reference).flatten(3),dim=-1).mean((-1,-2))
                records.append({"cohort":cohort.name,"model":row["model"],"seed":int(row["seed"]),
                    "solver":solver[0],"steps":solver[1],"nfe":solver[1]*(2 if solver[0]=="heun" else 1),
                    "reference_solver":f"{reference_key[0]}:{reference_key[1]}",
                    "mean_prediction_drift":float(drift.mean()),"solver_seconds":durations[solver],
                    "seconds_total_model":time.perf_counter()-started,
                    **metrics})
    return records


def means(records, keys):
    groups = {}
    for record in records:
        key = tuple(record[k] for k in keys)
        groups.setdefault(key, []).append(record)
    output = []
    for key, rows in groups.items():
        item = dict(zip(keys,key)); item["seeds"] = len(rows)
        numeric = set.intersection(*[{k for k,v in row.items() if isinstance(v,(int,float))} for row in rows])
        for field in sorted(numeric-set(keys)-{"seed"}):
            item[field] = sum(float(row[field]) for row in rows)/len(rows)
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, action="append", required=True)
    parser.add_argument("--models", nargs="+", default=["cit_linear"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1,5,10])
    parser.add_argument("--run-solvers", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--solver-candidates", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=13)
    parser.add_argument("--solvers", type=parse_solver, nargs="+",
                        default=[("euler",10),("euler",25),("euler",50),("euler",100),
                                 ("heun",5),("heun",12),("heun",25),("heun",50)])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new or empty directory")
    if any(k < 1 for k in args.ks) or args.solver_candidates < 1 or args.max_examples < 1:
        parser.error("K, solver candidates and examples must be positive")
    cohorts = [path.resolve() for path in args.cohort]
    args.output.mkdir(parents=True,exist_ok=True)
    coverage = coverage_audit(cohorts,args.models,args.ks)
    report = {"scope":"saved official test examples: first item from each evaluation batch",
              "coverage_records":coverage,"coverage_means":means(coverage,["model"]),
              "solver_records":[],"solver_means":[]}
    if args.run_solvers:
        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        solver = solver_audit(cohorts,args.models,args.device,args.solver_candidates,args.max_examples,args.solvers)
        report["solver_records"] = solver
        report["solver_means"] = means(solver,["model","solver","steps","nfe"])
    (args.output/"coverage_report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    for name, rows in (("coverage",coverage),("solver",report["solver_records"])):
        if rows:
            with (args.output/f"{name}.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({"coverage_means":report["coverage_means"],"solver_means":report["solver_means"]},indent=2))


if __name__ == "__main__":
    main()
