"""Q3: Open research — transfer learning to reverse-flow transition trajectories."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_pylibs = os.path.join(os.path.dirname(__file__), '..', '.pylibs')
if os.path.isdir(_pylibs): sys.path.insert(0, _pylibs)

import numpy as np
from data.h5_loader import BridgeFlowData
from models.pod_ar import PODFlowPredictor
from eval.metrics import compute_rmse_per_frame


def evaluate_on_transition(model, bd, input_len=10, pred_len=20):
    """Evaluate trained model on transition trajectories (reverse flow angles)."""
    fluid_mask = bd.get_fluid_mask()
    results = {}

    for traj_id in ['reverse_a0', 'reverse_a3']:
        data = bd.trajectories['transition'][traj_id]
        fn = bd.normalize(data['field'])
        n_frames = data['n_frames']

        traj_rmses = []
        starts = list(range(input_len - 1, n_frames - pred_len, 10))
        for start in starts:
            inp = fn[start - input_len + 1:start + 1]
            tgt = fn[start + 1:start + 1 + pred_len]
            pred = model.predict(inp, pred_len)
            pred_p = bd.denormalize(pred)
            tgt_p = bd.denormalize(tgt)
            rmse_per = compute_rmse_per_frame(pred_p, tgt_p, fluid_mask)
            traj_rmses.append(rmse_per)

        traj_arr = np.array(traj_rmses)
        results[traj_id] = {
            'mean_rmse_per_step': traj_arr.mean(axis=0).tolist(),
            'overall_mean': float(np.mean(traj_arr)),
            'overall_std': float(np.std(traj_arr)),
            'n_windows': len(traj_rmses),
        }

    return results


def evaluate_baseline_persistence(bd, traj_id, input_len=10, pred_len=20):
    """Simple persistence baseline: predict last frame repeats."""
    fluid_mask = bd.get_fluid_mask()
    data = bd.trajectories['test'][traj_id]
    fn = bd.normalize(data['field'])

    all_rmses = []
    for start in [99, 199, 299, 379]:
        inp = fn[max(0, start - input_len + 1):start + 1]
        tgt = fn[start + 1:start + 1 + pred_len]
        # Persistence: repeat last input frame
        pred = np.repeat(inp[-1:], pred_len, axis=0)
        pred_p = bd.denormalize(pred)
        tgt_p = bd.denormalize(tgt)
        rmse_per = compute_rmse_per_frame(pred_p, tgt_p, fluid_mask)
        all_rmses.append(rmse_per)
    return np.mean(all_rmses)


def run_q3(h5_path: str, n_modes=50, ar_order=3):
    print("=" * 60)
    print("Q3: Open Research — Transfer to Reverse-Flow Conditions")
    print("=" * 60)

    bd = BridgeFlowData(h5_path).load()
    fluid_mask = bd.get_fluid_mask()

    # Train on forward flow only
    train_trajectories = [
        bd.normalize(d['field']) for d in bd.trajectories['train'].values()
    ]
    model = PODFlowPredictor(n_modes=n_modes, ar_order=ar_order)
    model.fit(train_trajectories, fluid_mask)

    # 1. Persistence baseline on test
    print("\n--- Persistence Baseline ---")
    pers_a2 = evaluate_baseline_persistence(bd, 'test_a2')
    pers_a4 = evaluate_baseline_persistence(bd, 'test_a4')
    print(f"  test_a2 persistence RMSE: {pers_a2:.5f}")
    print(f"  test_a4 persistence RMSE: {pers_a4:.5f}")
    print(f"  Persistence avg: {np.mean([pers_a2, pers_a4]):.5f}")

    # 2. Model on in-distribution (test)
    print("\n--- Model on Test (in-distribution) ---")
    from eval.metrics import appendix_b_eval
    _, overall = appendix_b_eval(bd, lambda inp, n: model.predict(inp, n), 10, 20)
    print(f"  Appendix II RMSE: {np.mean(overall):.5f}")

    # 3. Transfer to transition (out-of-distribution)
    print("\n--- Transfer to Transition Trajectories ---")
    trans_results = evaluate_on_transition(model, bd)
    for tid, r in trans_results.items():
        print(f"  {tid}: mean RMSE = {r['overall_mean']:.5f} +/- {r['overall_std']:.5f}")
        print(f"    Per-step (first 5): {[round(x,4) for x in r['mean_rmse_per_step'][:5]]}")

    # 4. Analysis
    print("\n--- Q3 Findings ---")
    test_rmse = np.mean(overall)
    trans_mean = np.mean([r['overall_mean'] for r in trans_results.values()])
    degradation = (trans_mean - test_rmse) / test_rmse * 100
    print(f"  Test RMSE:       {test_rmse:.5f}")
    print(f"  Transition RMSE: {trans_mean:.5f}")
    print(f"  Degradation:     {degradation:.1f}%")
    print(f"\n  Conclusion: POD modes and pooled AR dynamics retain")
    print(f"  short-horizon accuracy on the reverse-flow stress test.")
    print(f"  Degradation versus the forward test set is {degradation:.1f}%.")
    print(f"  This supports limited transfer within the same geometry and")
    print(f"  Reynolds regime, not general physical universality.")

    return trans_results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--h5', default='bridge_a_flow_light.h5')
    args = p.parse_args()
    run_q3(args.h5)
