"""Q1: Validation-only hyperparameter comparison for the POD-AR model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_pylibs = os.path.join(os.path.dirname(__file__), '..', '.pylibs')
if os.path.isdir(_pylibs):
    sys.path.insert(0, _pylibs)

import numpy as np
from data.h5_loader import BridgeFlowData
from models.pod_ar import PODFlowPredictor
from eval.metrics import validation_eval


def run_q1(h5_path: str):
    bd = BridgeFlowData(h5_path).load()
    fluid_mask = bd.get_fluid_mask()

    # Prepare training data
    train_trajectories = [
        bd.normalize(d['field']) for d in bd.trajectories['train'].values()
    ]

    # Compare a small validation-only grid. The frozen submission uses (50, 3),
    # selected as a parameter/accuracy trade-off rather than the grid minimum.
    configs = [
        (30, 3), (50, 2), (50, 3), (80, 3),
    ]
    best_rmse = float('inf')
    best_cfg = None
    results = []

    for n_modes, ar_order in configs:
        print(f"\n{'='*50}")
        print(f"Config: n_modes={n_modes}, ar_order={ar_order}")
        model = PODFlowPredictor(n_modes=n_modes, ar_order=ar_order)
        model.fit(train_trajectories, fluid_mask)

        _, overall = validation_eval(
            bd, lambda inp, n: model.predict(inp, n), input_len=10, pred_len=20
        )
        mean_rmse = np.mean(overall)
        results.append((n_modes, ar_order, mean_rmse, overall))
        print(f"  Validation 4-window RMSE: {mean_rmse:.6f}")
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_cfg = (n_modes, ar_order)

    print(f"\n{'='*50}")
    print("Q1 SUMMARY")
    for nm, ao, mrmse, _ in results:
        star = '*' if (nm, ao) == best_cfg else ' '
        print(f"  {star} modes={nm:3d} ar={ao:2d}  RMSE={mrmse:.6f}")
    print(
        f"\nLowest grid RMSE: n_modes={best_cfg[0]}, ar_order={best_cfg[1]}, "
        f"RMSE={best_rmse:.6f}"
    )
    print("Frozen submission config: n_modes=50, ar_order=3, ridge_alpha=3.0")
    return best_cfg, best_rmse


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--h5', default='bridge_a_flow_light.h5')
    args = p.parse_args()
    run_q1(args.h5)
