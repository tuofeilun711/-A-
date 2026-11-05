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
from eval.metrics import (  # noqa: E402
    compute_ensemble_per_frame,
    compute_rmse_per_frame,
)
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
    ensemble_rows = []

    for window_index, start in enumerate(STARTS):
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
        ensemble = bd.denormalize(
            model.predict_ensemble(
                history, pred_len, n_members=20, seed=20261026 + window_index
            )
        )
        ensemble_diagnostics = compute_ensemble_per_frame(
            ensemble, target, mask
        )
        for horizon in range(pred_len):
            ensemble_rows.append(
                {
                    "window": window_index + 1,
                    "start": start,
                    "horizon": horizon + 1,
                    "actual_rmse": float(errors[horizon]),
                    "ensemble_mean_rmse": float(
                        ensemble_diagnostics["ensemble_mean_rmse"][horizon]
                    ),
                    "ensemble_spread": float(
                        ensemble_diagnostics["spread"][horizon]
                    ),
                    "coverage_90": float(
                        ensemble_diagnostics["coverage_90"][horizon]
                    ),
                    "coverage_95": float(
                        ensemble_diagnostics["coverage_95"][horizon]
                    ),
                }
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
        ensemble_rows,
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
    (
        rows,
        model_steps,
        persistence_steps,
        w_steps,
        u_steps,
        ensemble_rows,
    ) = analyse_validation(model, bd)

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
                "ensemble_mean_rmse",
                "ensemble_spread",
                "coverage_90",
                "coverage_95",
            ]
        )
        for step in range(model_steps.shape[1]):
            step_ensemble = [
                row for row in ensemble_rows if row["horizon"] == step + 1
            ]
            writer.writerow(
                [
                    step + 1,
                    float(model_steps[:, step].mean()),
                    float(model_steps[:, step].std()),
                    float(persistence_steps[:, step].mean()),
                    float(w_steps[:, step].mean()),
                    float(u_steps[:, step].mean()),
                    float(
                        np.mean(
                            [row["ensemble_mean_rmse"] for row in step_ensemble]
                        )
                    ),
                    float(
                        np.mean([row["ensemble_spread"] for row in step_ensemble])
                    ),
                    float(np.mean([row["coverage_90"] for row in step_ensemble])),
                    float(np.mean([row["coverage_95"] for row in step_ensemble])),
                ]
            )
    with open(
        os.path.join(results, "ensemble_validation.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window",
                "start",
                "horizon",
                "actual_rmse",
                "ensemble_mean_rmse",
                "ensemble_spread",
                "coverage_90",
                "coverage_95",
            ],
        )
        writer.writeheader()
        writer.writerows(ensemble_rows)

    validation_rmse = float(model_steps.mean())
    persistence_rmse = float(persistence_steps.mean())
    ensemble_rmse = float(
        np.mean([row["ensemble_mean_rmse"] for row in ensemble_rows])
    )
    ensemble_spread = float(
        np.mean([row["ensemble_spread"] for row in ensemble_rows])
    )
    coverage_90 = float(np.mean([row["coverage_90"] for row in ensemble_rows]))
    coverage_95 = float(np.mean([row["coverage_95"] for row in ensemble_rows]))
    spread_correlation = float(
        np.corrcoef(
            [row["ensemble_spread"] for row in ensemble_rows],
            [row["actual_rmse"] for row in ensemble_rows],
        )[0, 1]
    )
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
        "ensemble_seed": 20261026,
        "ensemble_mean_rmse": ensemble_rmse,
        "ensemble_spread_mean": ensemble_spread,
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        "spread_error_correlation": spread_correlation,
        "windows": rows,
    }
    with open(
        os.path.join(results, "reliability_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(
        "Q2 validation RMSE %.6f; persistence %.6f; step 1 %.6f; step 20 %.6f; "
        "ensemble spread %.6f; coverage 90/95 %.3f/%.3f"
        % (
            validation_rmse,
            persistence_rmse,
            summary["rmse_horizon_1"],
            summary["rmse_horizon_20"],
            ensemble_spread,
            coverage_90,
            coverage_95,
        )
    )


if __name__ == "__main__":
    main()
