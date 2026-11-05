"""Leave-one-training-trajectory-out evaluation for Q3 generalization.

Configuration is fixed before evaluation. For each fold, normalization,
POD, and AR are fitted on four training trajectories only. The held-out
trajectory is never used for statistics or parameters.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PYLIBS = os.path.join(ROOT, ".pylibs")
if os.path.isdir(PYLIBS):
    sys.path.insert(0, PYLIBS)

from data.h5_loader import BridgeFlowData  # noqa: E402
from eval.metrics import compute_rmse_per_frame  # noqa: E402
from models.pod_ar import PODFlowPredictor  # noqa: E402


def fold_statistics(fields):
    stacked = np.concatenate(fields, axis=0)
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True) + 1e-8
    return mean, std


def evaluate_held_out(model, field, global_mean, global_std, mask):
    # Starts are separated by the full prediction length so that windows do
    # not overlap and are not treated as independent repeated samples.
    starts = list(range(9, 380, 20))
    window_rows = []
    all_rmse = []
    all_persistence = []
    for start in starts:
        normalized = (field - global_mean) / global_std
        history = normalized[start - 9:start + 1]
        target = field[start + 1:start + 21]
        prediction = model.predict(history, 20) * global_std + global_mean
        persistence = np.repeat(field[start:start + 1], 20, axis=0)
        rmse = compute_rmse_per_frame(prediction, target, mask)
        persistence_rmse = compute_rmse_per_frame(
            persistence, target, mask
        )
        all_rmse.append(rmse)
        all_persistence.append(persistence_rmse)
        window_rows.append({
            "start": start,
            "history_start": start - 9,
            "history_end": start,
            "rmse_mean": float(rmse.mean()),
            "rmse_final": float(rmse[-1]),
            "persistence_rmse_mean": float(persistence_rmse.mean()),
            "persistence_rmse_final": float(persistence_rmse[-1]),
        })
    rmse_array = np.asarray(all_rmse)
    persistence_array = np.asarray(all_persistence)
    return {
        "windows": len(starts),
        "model_rmse_mean": float(rmse_array.mean()),
        "model_rmse_std_across_windows": float(
            rmse_array.mean(axis=1).std()
        ),
        "persistence_rmse_mean": float(persistence_array.mean()),
        "rmse_final_mean": float(rmse_array[:, -1].mean()),
        "per_step_rmse_mean": rmse_array.mean(axis=0).tolist(),
        "window_rows": window_rows,
    }


def run(h5_path, no_write=False):
    bd = BridgeFlowData(h5_path).load()
    mask = bd.get_fluid_mask()
    train_ids = sorted(bd.trajectories["train"])
    folds = {}
    eval_rows = []

    for held_out_id in train_ids:
        fit_ids = [trajectory_id for trajectory_id in train_ids
                   if trajectory_id != held_out_id]
        fit_fields = [
            bd.trajectories["train"][trajectory_id]["field"]
            for trajectory_id in fit_ids
        ]
        global_mean, global_std = fold_statistics(fit_fields)
        normalized_fit = [
            (field - global_mean) / global_std for field in fit_fields
        ]
        model = PODFlowPredictor(50, 3, 3.0).fit(normalized_fit, mask)
        held_out = bd.trajectories["train"][held_out_id]["field"]
        result = evaluate_held_out(
            model, held_out, global_mean, global_std, mask
        )
        result["fit_trajectories"] = fit_ids
        result["pod_energy_percent"] = float(model.pod_energy)
        result["spectral_radius"] = float(model.spectral_radius())
        folds[held_out_id] = result
        for row in result.pop("window_rows"):
            eval_rows.append({"held_out": held_out_id, **row})

    finite_ids = [
        trajectory_id
        for trajectory_id, fold in folds.items()
        if np.isfinite(fold["model_rmse_mean"])
        and fold["model_rmse_mean"] < 1.0
    ]
    diverged_ids = [
        trajectory_id for trajectory_id in folds if trajectory_id not in finite_ids
    ]
    per_step = np.asarray([
        folds[trajectory_id]["per_step_rmse_mean"]
        for trajectory_id in finite_ids
    ])
    fold_rmse = {
        trajectory_id: fold["model_rmse_mean"]
        for trajectory_id, fold in folds.items()
    }
    finite_mean = float(np.mean([
        folds[trajectory_id]["model_rmse_mean"]
        for trajectory_id in finite_ids
    ]))
    finite_persistence = float(np.mean([
        folds[trajectory_id]["persistence_rmse_mean"]
        for trajectory_id in finite_ids
    ]))
    summary = {
        "role": "train leave-one-trajectory-out cross-validation",
        "protocol": {
            "configuration": {
                "n_modes": 50,
                "ar_order": 3,
                "ridge_alpha": 3.0,
            },
            "normalization": "per-fold, fit trajectories only",
            "windows": "starts 9..369, step 20, non-overlapping",
            "transition_role": "not used",
            "test_role": "not used",
        },
        "finite_folds": finite_ids,
        "diverged_folds": diverged_ids,
        "finite_fold_model_rmse_mean": finite_mean,
        "finite_fold_model_rmse_std": float(np.std([
            folds[trajectory_id]["model_rmse_mean"]
            for trajectory_id in finite_ids
        ])),
        "finite_fold_persistence_rmse_mean": finite_persistence,
        "finite_fold_relative_improvement_percent": float(
            (1.0 - finite_mean / finite_persistence) * 100
        ),
        "best_fold": min(fold_rmse, key=fold_rmse.get),
        "worst_finite_fold": max(
            finite_ids, key=lambda trajectory_id: folds[trajectory_id]["model_rmse_mean"]
        ),
        "overall_persistence_rmse_mean": float(
            np.mean([
                fold["persistence_rmse_mean"] for fold in folds.values()
            ])
        ),
        "finite_fold_per_step_rmse_mean": per_step.mean(axis=0).tolist(),
        "folds": folds,
        "conclusion": (
            "Per-fold normalization and fitting avoid held-out trajectory "
            "leakage. Four folds remain finite, but the fold that removes "
            "train_a6 diverges, showing that the fixed ridge configuration is "
            "not uniformly stable across all training trajectories. This does "
            "not establish transfer to a new geometry or Reynolds number."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if no_write:
        return summary

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(
        os.path.join(results_dir, "leave_one_out_eval.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eval_rows[0]))
        writer.writeheader()
        writer.writerows(eval_rows)
    with open(
        os.path.join(results_dir, "leave_one_out_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5", default=os.path.join(ROOT, "bridge_a_flow_light.h5")
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    run(args.h5, no_write=args.no_write)
