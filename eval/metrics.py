"""Evaluation metrics for flow prediction."""
import numpy as np


def compute_rmse(pred, target, mask):
    """RMSE over fluid points only. pred,target: (*, 2, H, W), mask: (H,W) bool."""
    mask_2ch = np.stack([mask, mask], axis=-3)
    while mask_2ch.ndim < pred.ndim:
        mask_2ch = mask_2ch[np.newaxis, ...]
    mask_exp = np.broadcast_to(mask_2ch, pred.shape)
    pred_f = pred[mask_exp]
    tgt_f = target[mask_exp]
    N = pred_f.size
    if N == 0:
        return 0.0
    return np.sqrt(np.mean((pred_f - tgt_f) ** 2))


def compute_rmse_per_frame(pred_seq, tgt_seq, mask):
    """Compute RMSE for each prediction step. Returns (pred_len,) array."""
    pred_len = pred_seq.shape[0]
    rmses = np.zeros(pred_len)
    for m in range(pred_len):
        rmses[m] = compute_rmse(pred_seq[m], tgt_seq[m], mask)
    return rmses


def compute_correlation(pred, target, mask):
    """Pearson correlation over fluid points."""
    mask_2ch = np.stack([mask, mask], axis=-3)
    while mask_2ch.ndim < pred.ndim:
        mask_2ch = mask_2ch[np.newaxis, ...]
    mask_exp = np.broadcast_to(mask_2ch, pred.shape)
    p = pred[mask_exp]
    t = target[mask_exp]
    p = p - p.mean()
    t = t - t.mean()
    num = np.sum(p * t)
    den = np.sqrt(np.sum(p * p) * np.sum(t * t)) + 1e-12
    return num / den


def compute_ensemble_per_frame(ensemble_pred, target, mask):
    """Compute prediction RMSE, ensemble spread, and interval coverage.

    ensemble_pred: (members, pred_len, 2, H, W)
    target: (pred_len, 2, H, W)
    mask: (H, W) bool
    """
    ensemble_pred = np.asarray(ensemble_pred)
    target = np.asarray(target)
    if ensemble_pred.ndim != 5 or target.ndim != 4:
        raise ValueError("Expected ensemble_pred=(M,T,2,H,W) and target=(T,2,H,W)")
    if ensemble_pred.shape[1:] != target.shape:
        raise ValueError("Ensemble and target shapes do not match")

    ensemble_mean = ensemble_pred.mean(axis=0)
    ensemble_rmse = compute_rmse_per_frame(ensemble_mean, target, mask)

    flat_pred = ensemble_pred.reshape(
        ensemble_pred.shape[0], ensemble_pred.shape[1], 2, -1
    )[:, :, :, mask.reshape(-1)]
    flat_target = target.reshape(target.shape[0], 2, -1)[
        :, :, mask.reshape(-1)
    ]
    spread = np.std(flat_pred, axis=0, ddof=0)
    spread = np.sqrt(np.mean(spread ** 2, axis=(1, 2)))

    lower_90, upper_90 = np.quantile(flat_pred, [0.05, 0.95], axis=0)
    lower_95, upper_95 = np.quantile(flat_pred, [0.025, 0.975], axis=0)
    coverage_90 = np.mean(
        (flat_target >= lower_90) & (flat_target <= upper_90), axis=(1, 2)
    )
    coverage_95 = np.mean(
        (flat_target >= lower_95) & (flat_target <= upper_95), axis=(1, 2)
    )
    return {
        "ensemble_mean_rmse": ensemble_rmse,
        "spread": spread,
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
    }


def appendix_b_eval(bridge_data, predict_fn, input_len=10, pred_len=20):
    """
    Evaluate per Appendix II: 8 fixed windows on test_a2, test_a4.
    
    predict_fn: fn(input_frames_norm, n_steps) -> pred_frames_norm
    Returns: (window_results, overall_mean_rmse)
    """
    test_trajs = ['test_a2', 'test_a4']
    start_frames = [99, 199, 299, 379]
    fluid_mask = bridge_data.get_fluid_mask()

    window_results = []
    all_rmses = []

    for traj_id in test_trajs:
        data = bridge_data.trajectories['test'][traj_id]
        field_norm = bridge_data.normalize(data['field'])

        for start in start_frames:
            # Input: frames up to start (inclusive), up to input_len
            avail = min(input_len, start + 1)
            input_frames = field_norm[start - avail + 1:start + 1].copy()
            if len(input_frames) < input_len:
                pad = np.repeat(input_frames[0:1], input_len - len(input_frames), axis=0)
                input_frames = np.concatenate([pad, input_frames], axis=0)

            # Target
            tgt_start = start + 1
            tgt_end = tgt_start + pred_len
            targets = field_norm[tgt_start:tgt_end]

            # Predict
            preds_norm = predict_fn(input_frames, pred_len)

            # Denormalize to physical (dimensionless) space
            preds_phys = bridge_data.denormalize(preds_norm)
            tgts_phys = bridge_data.denormalize(targets)

            rmse_per = compute_rmse_per_frame(preds_phys, tgts_phys, fluid_mask)

            window_results.append({
                'traj': traj_id,
                'start': start,
                'rmse': rmse_per,
                'input_actual': len(input_frames),
            })
            all_rmses.append(rmse_per)

    overall = np.mean(all_rmses, axis=0) if all_rmses else np.zeros(pred_len)
    return window_results, overall


def validation_eval(bridge_data, predict_fn, input_len=10, pred_len=20,
                    starts=(99, 199, 299, 379), trajectory='validation_a05'):
    """Same protocol as Appendix II, but on the validation split (for model selection).

    Returns (rows, overall_per_step) where rows mirror annotation-table fields.
    """
    mask = bridge_data.get_fluid_mask()
    field = bridge_data.normalize(bridge_data.trajectories['validation'][trajectory]['field'])
    rows, all_steps = [], []
    for s in starts:
        hist = field[max(0, s - input_len + 1):s + 1]
        if len(hist) < input_len:
            hist = np.concatenate([np.repeat(hist[:1], input_len - len(hist), axis=0), hist], axis=0)
        tgt = bridge_data.denormalize(field[s + 1:s + 1 + pred_len])
        err = compute_rmse_per_frame(bridge_data.denormalize(predict_fn(hist, pred_len)), tgt, mask)
        rows.append({'role': 'validation', 'trajectory': trajectory, 'start': s,
                     'history_frames': int(len(hist)), 'rmse_mean': float(np.mean(err)),
                     'rmse_final': float(err[-1]), 'per_step': err})
        all_steps.append(err)
    return rows, np.mean(all_steps, axis=0)


def print_results(window_results, overall):
    """Print Appendix II format results."""
    print("=" * 70)
    print("APPENDIX II — Common Reference Evaluation")
    print("=" * 70)
    for wr in window_results:
        r5 = ", ".join([f"{r:.4f}" for r in wr['rmse'][:5]])
        print(f"  {wr['traj']:10s} start={wr['start']:3d}  " +
              f"RMSE[m=1..5]: [{r5} ...]")
    print("-" * 70)
    print("Overall average RMSE per step (8 windows):")
    print("  " + "  ".join([f"{v:.4f}" for v in overall]))
    print(f"  Mean: {np.mean(overall):.6f}  Std: {np.std(overall):.6f}")
