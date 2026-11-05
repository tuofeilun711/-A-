import unittest

import numpy as np

from data.h5_loader import create_sequences
from eval.metrics import compute_rmse
from models.pod_ar import PODFlowPredictor


class MetricTests(unittest.TestCase):
    def test_rmse_ignores_solid_cells(self):
        mask = np.ones((3, 4), dtype=bool)
        mask[1, 1:3] = False
        truth = np.zeros((2, 3, 4), dtype=np.float32)
        prediction = truth.copy()
        prediction[:, mask] = 2.0
        prediction[:, ~mask] = 999.0
        self.assertAlmostEqual(compute_rmse(prediction, truth, mask), 2.0)

    def test_create_sequences_shapes(self):
        data = np.arange(12, dtype=np.float32).reshape(12, 1, 1)
        inputs, targets = create_sequences(data, input_len=3, pred_len=2)
        self.assertEqual(inputs.shape, (8, 3, 1, 1))
        self.assertEqual(targets.shape, (8, 2, 1, 1))


class ModelTests(unittest.TestCase):
    def test_ar_fit_does_not_cross_trajectory_boundaries(self):
        rng = np.random.default_rng(3)
        mask = np.ones((3, 4), dtype=bool)
        first = rng.normal(size=(10, 2, 3, 4))
        second = rng.normal(size=(10, 2, 3, 4))
        model = PODFlowPredictor(n_modes=2, ar_order=2, ridge_alpha=0.1)
        model.fit([first, second], mask)
        self.assertEqual(model.ar_residuals.shape[0], 16)

    def test_forecast_ensemble_is_reproducible(self):
        rng = np.random.default_rng(1)
        mask = np.ones((4, 5), dtype=bool)
        train = rng.normal(size=(2, 20, 2, 4, 5))
        model = PODFlowPredictor(n_modes=3, ar_order=2, ridge_alpha=3.0)
        model.fit(list(train), mask)
        inputs = train[0, 10:15]
        first = model.predict_ensemble(inputs, 4, n_members=3, seed=7)
        second = model.predict_ensemble(inputs, 4, n_members=3, seed=7)
        np.testing.assert_allclose(first, second)


if __name__ == '__main__':
    unittest.main()
