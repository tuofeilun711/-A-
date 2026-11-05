"""Q3: held-out trajectory analysis with and without reverse_a1 in training.

The original all-train model is the submitted model. Because the train split
already contains reverse_a1, it is not a zero-shot reverse-condition model.
The forward-only ablation excludes every train trajectory whose ID starts with
``reverse_`` and quantifies how much accuracy is retained without that sample.
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
from eval.metrics import appendix_b_eval, compute_rmse_per_frame  # noqa: E402
from models.pod_ar import PODFlowPredictor  # noqa: E402


def evaluate_on_transition(model, bd, input_len=10, pred_len=20):
    """Return per-window and per-step errors on the two transition trajectories."""
    fluid_mask = bd.get_fluid_mask()
    results = {}
    windows = []
    for trajectory_id in ("reverse_a0", "reverse_a3"):
        data = bd.trajectories["transition"][trajectory_id]
        field = bd.normalize(data["field"])
        trajectory_rmses = []
        for start in range(input_len - 1, data["n_frames"] - pred_len, 10):
            history = field[start - input_len + 1:start + 1]
            target = bd.denormalize(field[start + 1:start + 1 + pred_len])
            prediction = bd.denormalize(model.predict(history, pred_len))
            persistence = bd.denormalize(
                np.repeat(field[start:start + 1], pred_len, axis=0)
            )
            rmse_per = compute_rmse_per_frame(
                prediction, target, fluid_mask
            )
            persistence_per = compute_rmse_per_frame(
                persistence, target, fluid_mask
            )
            trajectory_rmses.append(rmse_per)
            windows.append({
                "trajectory": trajectory_id,
                "start": start,
                "history_start": start - input_len + 1,
                "history_end": start,
                "rmse_mean": float(np.mean(rmse_per)),
                "rmse_final": float(rmse_per[-1]),
                "persistence_rmse_mean": float(np.mean(persistence_per)),
            })

        trajectory_array = np.asarray(trajectory_rmses)
        results[trajectory_id] = {
            "windows": int(len(trajectory_array)),
            "model_rmse_mean": float(trajectory_array.mean()),
            "model_rmse_std": float(trajectory_array.std()),
            "persistence_rmse_mean": float(
                np.mean([row["persistence_rmse_mean"] for row in windows
                         if row["trajectory"] == trajectory_id])
            ),
            "per_step_rmse_mean": trajectory_array.mean(axis=0).tolist(),
        }
    results["overall_model_rmse_mean"] = float(
        np.mean([results["reverse_a0"]["model_rmse_mean"],
                 results["reverse_a3"]["model_rmse_mean"]])
    )
    return results, windows


def fit_model(bd, fluid_mask, trajectory_ids, n_modes=50, ar_order=3):
    fields = [
        bd.normalize(bd.trajectories["train"][trajectory_id]["field"])
        for trajectory_id in trajectory_ids
    ]
    return PODFlowPredictor(n_modes, ar_order, 3.0).fit(fields, fluid_mask)


def run_q3(h5_path: str, n_modes=50, ar_order=3, no_write=False):
    bd = BridgeFlowData(h5_path).load()
    fluid_mask = bd.get_fluid_mask()
    all_train_ids = sorted(bd.trajectories["train"])
    forward_only_ids = [
        trajectory_id for trajectory_id in all_train_ids
        if not trajectory_id.startswith("reverse_")
    ]
    models = {
        "all_train": fit_model(
            bd, fluid_mask, all_train_ids, n_modes=n_modes, ar_order=ar_order
        ),
        "forward_only": fit_model(
            bd, fluid_mask, forward_only_ids, n_modes=n_modes, ar_order=ar_order
        ),
    }

    summary = {
        "purpose": (
            "Separate held-out transition trajectories from true reverse-condition "
            "ablation; transition trajectories are never used for fitting."
        ),
        "training_boundary": {
            "all_train": all_train_ids,
            "forward_only": forward_only_ids,
            "transition_used_for_training": False,
        },
        "models": {},
        "conclusion": (
            "The all-train result is a held-out-trajectory test, not a zero-shot "
            "reverse-condition test. The forward-only ablation measures transfer "
            "without reverse_a1; its small degradation supports limited short-horizon "
            "transfer within the same geometry and operating family, not physical "
            "universality."
        ),
    }
    window_rows = []
    for model_name, model in models.items():
        _, test_steps = appendix_b_eval(
            bd,
            lambda history, steps, fitted=model: fitted.predict(history, steps),
            input_len=10,
            pred_len=20,
        )
        transition, windows = evaluate_on_transition(model, bd)
        summary["models"][model_name] = {
            "test_rmse": float(np.mean(test_steps)),
            "transition": transition,
        }
        for row in windows:
            window_rows.append({"model": model_name, **row})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if no_write:
        return summary

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(
        os.path.join(results_dir, "transition_transfer_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(results_dir, "transition_transfer_eval.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default=os.path.join(ROOT, "bridge_a_flow_light.h5"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    run_q3(args.h5, no_write=args.no_write)
