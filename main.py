"""Entry point: fit the frozen config, evaluate validation + official test windows, write results/.

Only numpy + h5py are needed here. Figures and the PPT/PDF build live in
experiments/ and need extra packages (see README).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYLIB = os.path.join(_HERE, '.pylibs')
if os.path.isdir(_PYLIB):
    sys.path.insert(0, _PYLIB)

from data.h5_loader import BridgeFlowData, build_snapshot_matrix   # noqa: E402
from models.pod_ar import PODFlowPredictor                          # noqa: E402
from eval.metrics import compute_ensemble_per_frame, compute_rmse_per_frame  # noqa: E402

RESULTS = os.path.join(_HERE, 'results')
CONFIG = os.path.join(_HERE, 'configs', 'model_config.json')


def load_config():
    cfg = {'n_modes': 50, 'ar_order': 3, 'ridge_alpha': 3.0, 'input_len': 10, 'pred_len': 20}
    if os.path.exists(CONFIG):
        with open(CONFIG) as f:
            raw = json.load(f)
        cfg.update({k: v for k, v in raw.items() if not k.startswith('_')})
    return cfg


def build_history(field, start, input_len):
    hist = field[max(0, start - input_len + 1):start + 1]
    if len(hist) < input_len:
        hist = np.concatenate([np.repeat(hist[:1], input_len - len(hist), axis=0), hist], axis=0)
    return hist


def eval_split(bd, model, split, ids, starts, mask, input_len, pred_len):
    rows = []
    for tid in ids:
        field = bd.normalize(bd.trajectories[split][tid]['field'])
        for s in starts:
            hist = build_history(field, s, input_len)
            tgt = bd.denormalize(field[s + 1:s + 1 + pred_len])
            err = compute_rmse_per_frame(bd.denormalize(model.predict(hist, pred_len)), tgt, mask)
            pers = compute_rmse_per_frame(
                bd.denormalize(np.repeat(field[s][None], pred_len, axis=0)), tgt, mask)
            pr = bd.denormalize(model.predict(hist, pred_len))
            w_step = np.sqrt(((pr[:, 0] - tgt[:, 0])[:, mask] ** 2).mean(axis=1))
            u_step = np.sqrt(((pr[:, 1] - tgt[:, 1])[:, mask] ** 2).mean(axis=1))
            rows.append({'role': split, 'trajectory': tid, 'name': '%s@%d' % (tid, s), 'start': s,
                         'history_frames': int(len(hist)),
                         'history_start': int(max(0, s - input_len + 1)),
                         'history_end': int(s),
                         'rmse_mean': float(np.mean(err)), 'rmse_final': float(err[-1]),
                         'persistence_rmse_mean': float(np.mean(pers)),
                         'persistence_rmse_final': float(pers[-1]),
                         'per_step': err, 'persistence_per_step': pers,
                         'w_step': w_step, 'u_step': u_step})
    return rows


def eval_validation_ensemble(bd, model, field, mask, starts, input_len, pred_len,
                             n_members=20, seed=20261026):
    """Evaluate a reproducible residual-bootstrap ensemble on validation windows."""
    rows = []
    for window_index, start in enumerate(starts):
        hist = build_history(field, start, input_len)
        target = bd.denormalize(field[start + 1:start + 1 + pred_len])
        ensemble = bd.denormalize(
            model.predict_ensemble(
                hist, pred_len, n_members=n_members, seed=seed + window_index
            )
        )
        deterministic = bd.denormalize(model.predict(hist, pred_len))
        deterministic_rmse = compute_rmse_per_frame(
            deterministic, target, mask
        )
        diagnostics = compute_ensemble_per_frame(ensemble, target, mask)
        for horizon in range(pred_len):
            rows.append({
                'window': window_index + 1,
                'start': start,
                'horizon': horizon + 1,
                'actual_rmse': float(deterministic_rmse[horizon]),
                'ensemble_mean_rmse': float(
                    diagnostics['ensemble_mean_rmse'][horizon]
                ),
                'ensemble_spread': float(diagnostics['spread'][horizon]),
                'coverage_90': float(diagnostics['coverage_90'][horizon]),
                'coverage_95': float(diagnostics['coverage_95'][horizon]),
            })
    return rows


def write_csv(path, rows, cols):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='bridge_a_flow_light.h5')
    ap.add_argument('--n-modes', type=int, default=cfg['n_modes'])
    ap.add_argument('--ar-order', type=int, default=cfg['ar_order'])
    ap.add_argument('--ridge-alpha', type=float, default=cfg['ridge_alpha'])
    ap.add_argument('--input-len', type=int, default=cfg['input_len'])
    ap.add_argument('--pred-len', type=int, default=cfg['pred_len'])
    ap.add_argument('--no-write', action='store_true')
    a = ap.parse_args()

    bd = BridgeFlowData(a.h5).load()
    mask = bd.get_fluid_mask()
    train = [bd.normalize(bd.trajectories['train'][k]['field'])
             for k in sorted(bd.trajectories['train'])]
    model = PODFlowPredictor(a.n_modes, a.ar_order, a.ridge_alpha).fit(train, mask)

    energy = model.pod_energy
    rho = model.spectral_radius()

    val = eval_split(bd, model, 'validation', ['validation_a05'], [99, 199, 299, 379],
                     mask, a.input_len, a.pred_len)
    test = eval_split(bd, model, 'test', ['test_a2', 'test_a4'], [99, 199, 299, 379],
                      mask, a.input_len, a.pred_len)
    trans = []
    for tid in ['reverse_a0', 'reverse_a3']:
        n = len(bd.trajectories['transition'][tid]['field'])
        trans += eval_split(bd, model, 'transition', [tid], list(range(9, n - 20, 10)),
                            mask, a.input_len, a.pred_len)
    validation_field = bd.normalize(
        bd.trajectories['validation']['validation_a05']['field']
    )
    ensemble_rows = eval_validation_ensemble(
        bd, model, validation_field, mask, [99, 199, 299, 379],
        a.input_len, a.pred_len, n_members=20, seed=20261026
    )

    val_rmse = float(np.mean([r['rmse_mean'] for r in val]))
    val_pers = float(np.mean([r['persistence_rmse_mean'] for r in val]))
    test_rmse = float(np.mean([r['rmse_mean'] for r in test]))
    test_pers = float(np.mean([r['persistence_rmse_mean'] for r in test]))
    ensemble_rmse = float(np.mean([r['ensemble_mean_rmse'] for r in ensemble_rows]))
    ensemble_spread = float(np.mean([r['ensemble_spread'] for r in ensemble_rows]))
    coverage_90 = float(np.mean([r['coverage_90'] for r in ensemble_rows]))
    coverage_95 = float(np.mean([r['coverage_95'] for r in ensemble_rows]))
    spread_correlation = float(np.corrcoef(
        [r['ensemble_spread'] for r in ensemble_rows],
        [r['actual_rmse'] for r in ensemble_rows],
    )[0, 1])

    print('=' * 64)
    print('POD-%d + AR(%d), ridge alpha=%.3g' % (model.n_modes, model.ar_order, a.ridge_alpha))
    print('POD energy kept %.1f%%   one-step in-sample R2 %.4f   spectral radius %.6f'
          % (energy, model.r2, rho))
    print('validation 4 windows RMSE %.6f  (persistence %.6f)' % (val_rmse, val_pers))
    print('validation 20-member ensemble mean RMSE %.6f, spread %.6f, '
          'coverage 90/95 %.3f/%.3f'
          % (ensemble_rmse, ensemble_spread, coverage_90, coverage_95))
    print('test       8 windows RMSE %.6f  (persistence %.6f, improvement %.2f%%)'
          % (test_rmse, test_pers, (1 - test_rmse / test_pers) * 100))
    for r in test:
        print('   %-12s mean %.4f   step1 %.4f   step20 %.4f'
              % (r['name'], r['rmse_mean'], r['per_step'][0], r['per_step'][-1]))
    print('=' * 64)

    if a.no_write:
        return
    os.makedirs(RESULTS, exist_ok=True)
    cols = [
        'role', 'trajectory', 'start', 'history_start', 'history_end',
        'history_frames', 'rmse_mean', 'rmse_final'
    ]
    write_csv(os.path.join(RESULTS, 'validation_reference.csv'), val, cols)
    write_csv(os.path.join(RESULTS, 'common_reference.csv'), test, cols)
    write_csv(os.path.join(RESULTS, 'transition_eval.csv'), trans,
              cols + ['persistence_rmse_mean', 'persistence_rmse_final'])
    with open(os.path.join(RESULTS, 'common_reference_per_step.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'role', 'trajectory', 'start', 'history_start', 'history_end',
            'history_frames', 'horizon', 'rmse', 'persistence_rmse'
        ])
        for row in test:
            for horizon, (rmse, persistence_rmse) in enumerate(
                zip(row['per_step'], row['persistence_per_step']), start=1
            ):
                writer.writerow([
                    row['role'], row['trajectory'], row['start'],
                    row['history_start'], row['history_end'],
                    row['history_frames'], horizon, float(rmse),
                    float(persistence_rmse)
                ])
    T = np.array([r['per_step'] for r in test])
    TP = np.array([r['persistence_per_step'] for r in test])
    with open(os.path.join(RESULTS, 'test_error_growth.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['horizon', 'rmse_mean', 'persistence_rmse'])
        for h in range(a.pred_len):
            writer.writerow([h + 1, float(T[:, h].mean()), float(TP[:, h].mean())])
    E = np.array([r['per_step'] for r in val])
    P = np.array([r['persistence_per_step'] for r in val])
    W = np.array([r['w_step'] for r in val])
    U = np.array([r['u_step'] for r in val])
    write_csv(
        os.path.join(RESULTS, 'ensemble_validation.csv'),
        ensemble_rows,
        ['window', 'start', 'horizon', 'actual_rmse', 'ensemble_mean_rmse',
         'ensemble_spread', 'coverage_90', 'coverage_95']
    )
    ensemble_by_horizon = {
        h: [r for r in ensemble_rows if r['horizon'] == h + 1]
        for h in range(a.pred_len)
    }
    with open(os.path.join(RESULTS, 'reliability_error_growth.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['horizon', 'rmse_mean', 'rmse_std_across_windows', 'persistence_rmse',
                    'w_rmse', 'u_rmse', 'ensemble_mean_rmse', 'ensemble_spread',
                    'coverage_90', 'coverage_95'])
        for h in range(a.pred_len):
            ens_rows = ensemble_by_horizon[h]
            w.writerow([h + 1, float(E[:, h].mean()), float(E[:, h].std()), float(P[:, h].mean()),
                        float(W[:, h].mean()), float(U[:, h].mean()),
                        float(np.mean([r['ensemble_mean_rmse'] for r in ens_rows])),
                        float(np.mean([r['ensemble_spread'] for r in ens_rows])),
                        float(np.mean([r['coverage_90'] for r in ens_rows])),
                        float(np.mean([r['coverage_95'] for r in ens_rows]))])
    diag = {'n_modes': int(model.n_modes), 'ar_order': int(model.ar_order),
            'ridge_alpha': a.ridge_alpha, 'pod_energy_percent': energy,
            'one_step_in_sample_r2': model.r2, 'spectral_radius': rho,
            'input_len': a.input_len, 'pred_len': a.pred_len,
            'fluid_cells': int(mask.sum()), 'state_dimension': int(2 * mask.sum())}
    with open(os.path.join(RESULTS, 'model_diagnostics.json'), 'w') as f:
        json.dump(diag, f, indent=2)

    reliability_summary = {
        'role': 'validation',
        'n_modes': int(model.n_modes),
        'ar_order': int(model.ar_order),
        'ridge_alpha': a.ridge_alpha,
        'pod_energy_percent': energy,
        'one_step_in_sample_r2': model.r2,
        'spectral_radius': rho,
        'validation_rmse': val_rmse,
        'persistence_rmse': val_pers,
        'relative_improvement_percent': float((1 - val_rmse / val_pers) * 100),
        'rmse_horizon_1': float(E[:, 0].mean()),
        'rmse_horizon_20': float(E[:, -1].mean()),
        'ensemble_members': 20,
        'ensemble_seed': 20261026,
        'ensemble_mean_rmse': ensemble_rmse,
        'ensemble_spread_mean': ensemble_spread,
        'coverage_90': coverage_90,
        'coverage_95': coverage_95,
        'spread_error_correlation': spread_correlation,
        'windows': [{'start': r['start'], 'rmse_mean': r['rmse_mean'],
                     'rmse_final': r['rmse_final']} for r in val],
    }
    with open(os.path.join(RESULTS, 'reliability_summary.json'), 'w') as f:
        json.dump(reliability_summary, f, indent=2)

    transition_summary = {
        'role_note': (
            'validation is the in-distribution reference. The two transition '
            'trajectories are held out from fitting and tuning, but reverse_a1 '
            'is present in train, so this result is not a zero-shot reverse test.'
        ),
        'validation_reference_rmse': val_rmse,
        'transition': {},
        'conclusion': (
            'The all-train model retains short-horizon accuracy on two held-out '
            'transition trajectories. See transition_transfer_summary.json for '
            'the forward-only ablation that excludes reverse_a1. Neither result '
            'establishes general physical universality.'
        ),
        'strict_ablation_file': 'transition_transfer_summary.json',
    }
    for tid in ['reverse_a0', 'reverse_a3']:
        selected = [r for r in trans if r['trajectory'] == tid]
        transition_summary['transition'][tid] = {
            'windows': len(selected),
            'model_rmse_mean': float(np.mean([r['rmse_mean'] for r in selected])),
            'model_rmse_std': float(np.std([r['rmse_mean'] for r in selected])),
            'persistence_rmse_mean': float(
                np.mean([r['persistence_rmse_mean'] for r in selected])
            ),
        }
    with open(os.path.join(RESULTS, 'transition_summary.json'), 'w') as f:
        json.dump(transition_summary, f, indent=2)

    summary = {
        'validation_rmse': val_rmse,
        'test_rmse': test_rmse,
        'validation_persistence_rmse': val_pers,
        'test_persistence_rmse': test_pers,
        'test_improvement_percent': float((1 - test_rmse / test_pers) * 100),
        'test_window_mean': {r['name']: r['rmse_mean'] for r in test},
        'validation_window_mean': {r['name']: r['rmse_mean'] for r in val},
        'validation_ensemble': {
            'members': 20,
            'mean_rmse': ensemble_rmse,
            'mean_spread': ensemble_spread,
            'coverage_90': coverage_90,
            'coverage_95': coverage_95,
            'spread_error_correlation': spread_correlation,
        },
        'diagnostics': diag,
    }
    with open(os.path.join(RESULTS, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('results written to %s' % RESULTS)


if __name__ == '__main__':
    main()
