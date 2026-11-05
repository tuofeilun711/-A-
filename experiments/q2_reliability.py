"""Q2: Validation-only error-growth and component-reliability analysis."""
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


STARTS = (99, 199, 299, 379)


def analyse_validation(model, bd, input_len=10, pred_len=20):
    mask = bd.get_fluid_mask()
    field = bd.normalize(
        bd.trajectories["validation"]["validation_a05"]["field"]
    )
    rows = []
    model_steps, persistence_steps = [], []
    w_steps, u_steps = [], []

    for start in STARTS:
        history = field[start - input_len + 1 : start + 1]
        target = bd.denormalize(field[start + 1 : start + 1 + pred_len])
        prediction = bd.denormalize(model.predict(history, pred_len))
        persistence = bd.denormalize(
            np.repeat(field[start : start + 1], pred_len, axis=0)
        )
        errors = compute_rmse_per_frame(prediction, target, mask)
        persistence_errors = compute_rmse_per_frame(persistence, target, mask)
        w_errors = np.sqrt(
            ((prediction[:, 0] - target[:, 0])[:, mask] ** 2).mean(axis=1)
        )
        u_errors = np.sqrt(
            ((prediction[:, 1] - target[:, 1])[:, mask] ** 2).mean(axis=1)
        )
        rows.append(
            {
                "start": start,
                "rmse_mean": float(errors.mean()),
                "rmse_final": float(errors[-1]),
            }
        )
        model_steps.append(errors)
        persistence_steps.append(persistence_errors)
        w_steps.append(w_errors)
        u_steps.append(u_errors)

    return (
        rows,
        np.asarray(model_steps),
        np.asarray(persistence_steps),
        np.asarray(w_steps),
        np.asarray(u_steps),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5", default=os.path.join(ROOT, "bridge_a_flow_light.h5")
    )
    parser.add_argument("--n-modes", type=int, default=50)
    parser.add_argument("--ar-order", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=3.0)
    args = parser.parse_args()

    bd = BridgeFlowData(args.h5).load()
    mask = bd.get_fluid_mask()
    train = [
        bd.normalize(data["field"])
        for data in bd.trajectories["train"].values()
    ]
    model = PODFlowPredictor(
        args.n_modes, args.ar_order, args.ridge_alpha
    ).fit(train, mask)
    rows, model_steps, persistence_steps, w_steps, u_steps = analyse_validation(
        model, bd
    )

    results = os.path.join(ROOT, "results")
    os.makedirs(results, exist_ok=True)
    with open(
        os.path.join(results, "reliability_error_growth.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "horizon",
                "rmse_mean",
                "rmse_std_across_windows",
                "persistence_rmse",
                "w_rmse",
                "u_rmse",
            ]
        )
        for step in range(model_steps.shape[1]):
            writer.writerow(
                [
                    step + 1,
                    float(model_steps[:, step].mean()),
                    float(model_steps[:, step].std()),
                    float(persistence_steps[:, step].mean()),
                    float(w_steps[:, step].mean()),
                    float(u_steps[:, step].mean()),
                ]
            )

    validation_rmse = float(model_steps.mean())
    persistence_rmse = float(persistence_steps.mean())
    summary = {
        "role": "validation",
        "n_modes": int(model.n_modes),
        "ar_order": int(model.ar_order),
        "ridge_alpha": args.ridge_alpha,
        "pod_energy_percent": float(model.pod_energy),
        "one_step_in_sample_r2": float(model.r2),
        "spectral_radius": float(model.spectral_radius()),
        "validation_rmse": validation_rmse,
        "persistence_rmse": persistence_rmse,
        "relative_improvement_percent": float(
            (1 - validation_rmse / persistence_rmse) * 100
        ),
        "rmse_horizon_1": float(model_steps[:, 0].mean()),
        "rmse_horizon_20": float(model_steps[:, -1].mean()),
        "ensemble_members": 20,
        "windows": rows,
    }
    with open(
        os.path.join(results, "reliability_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(
        "Q2 validation RMSE %.6f; persistence %.6f; step 1 %.6f; step 20 %.6f"
        % (
            validation_rmse,
            persistence_rmse,
            summary["rmse_horizon_1"],
            summary["rmse_horizon_20"],
        )
    )


if __name__ == "__main__":
    main()
