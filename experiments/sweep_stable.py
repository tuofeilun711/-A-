"""Validation-only stability sweep: how sensitive is the model to k / AR order / ridge alpha?

Writes results/stability_sweep.csv (and results/validation_sweep.csv for the alpha slice).
Selection is done on validation only; test is never touched here.
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.h5_loader import BridgeFlowData  # noqa: E402
from eval.metrics import validation_eval  # noqa: E402
from models.pod_ar import PODFlowPredictor  # noqa: E402


def one(bd, train, mask, k, order, alpha):
    model = PODFlowPredictor(n_modes=k, ar_order=order, ridge_alpha=alpha)
    model.fit(train, mask)
    _, overall = validation_eval(
        bd,
        lambda f, n: model.predict(f, n),
        input_len=10,
        pred_len=20,
        starts=(99, 199, 299),
    )
    return model, float(np.mean(overall))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=os.path.join(ROOT, "bridge_a_flow_light.h5"))
    a = ap.parse_args()

    bd = BridgeFlowData(a.h5).load()
    mask = bd.get_fluid_mask()
    train = [
        bd.normalize(bd.trajectories["train"][t]["field"])
        for t in sorted(bd.trajectories["train"])
    ]

    rows = []
    for k in [30, 50, 80]:
        for order in [2, 3]:
            for alpha in [0.01, 0.1, 1.0, 3.0, 10.0]:
                try:
                    model, rmse = one(bd, train, mask, k, order, alpha)
                    rows.append({"n_modes": model.n_modes, "ar_order": model.ar_order,
                                 "ridge_alpha": alpha,
                                 "pod_energy_percent": round(model.pod_energy, 6),
                                 "one_step_in_sample_r2": round(model.one_step_r2, 8),
                                 "spectral_radius": round(model.spectral_radius(), 8),
                                 "validation_rmse": round(rmse, 8),
                                 "stable": bool(np.isfinite(rmse) and rmse < 1.0)})
                    print("k=%3d ar=%d alpha=%5.2f  val=%.6f  energy=%.1f%%"
                          % (k, order, alpha, rmse, model.pod_energy))
                except Exception as exc:  # numerical blow-up
                    rows.append({"n_modes": k, "ar_order": order, "ridge_alpha": alpha,
                                 "validation_rmse": float("inf"), "stable": False})
                    print("k=%3d ar=%d alpha=%5.2f  FAILED: %s" % (k, order, alpha, exc))

    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "stability_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    frozen = [r for r in rows if r["n_modes"] == 50 and r["ar_order"] == 3]
    with open(os.path.join(out, "validation_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frozen[0]))
        w.writeheader()
        w.writerows(frozen)
    print("written to", out)


if __name__ == "__main__":
    main()
