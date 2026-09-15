"""Regression checks for the forecasting example (run with unittest)."""
import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

PATH = Path(__file__).with_name('Prophet_ETS_LSTM.py')


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fail safely instead of running the old training loop during import.
        tree = ast.parse(PATH.read_text())
        cls.import_safe = any(isinstance(n, ast.FunctionDef) and n.name == 'run_experiment' for n in tree.body)
        if cls.import_safe:
            spec = importlib.util.spec_from_file_location('forecast_example', PATH)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertTrue(self.import_safe, 'Expose run_experiment without training on import')

    def test_windows_and_short_series(self):
        dataset = self.module.ResidDataset(np.arange(6), input_len=3)
        x, y = dataset[0]
        np.testing.assert_array_equal(x.numpy().ravel(), [0, 1, 2])
        np.testing.assert_array_equal(y.numpy(), [3])
        self.assertEqual(len(dataset), 3)
        with self.assertRaises(ValueError):
            self.module.ResidDataset(np.arange(3), input_len=3)

    def test_rejects_unsupported_horizon(self):
        with self.assertRaises(ValueError):
            self.module.ResidDataset(np.arange(10), horizon=2)
        with self.assertRaises(ValueError):
            self.module.ResidLSTM(horizon=2)
        with self.assertRaises(ValueError):
            self.module.rolling_resid_predict(self.module.ResidLSTM(), np.zeros(28), 3, horizon=2)

    def test_recursive_forecast_uses_predictions(self):
        class Increment(torch.nn.Module):
            def forward(self, x):
                return x[:, -1, :] + 1
        values = self.module.rolling_resid_predict(Increment(), np.arange(4), 3, input_len=4)
        np.testing.assert_array_equal(values, [4, 5, 6])

    def test_pipeline_alignment_and_repeatability(self):
        torch.set_num_threads(1)
        a = self.module.run_experiment(epochs=1, make_plots=False)
        b = self.module.run_experiment(epochs=1, make_plots=False)
        for name in ['prophet_yhat_test', 'ets_yhat_test', 'hybrid_yhat_test']:
            self.assertEqual(a[name].shape, (183,))
            self.assertTrue(np.isfinite(a[name]).all())
            np.testing.assert_allclose(a[name], b[name], rtol=0, atol=0)
        expected = a['ets_model'].forecast(365)[182:]
        np.testing.assert_allclose(a['ets_yhat_test'], expected)
        np.testing.assert_allclose(a['resid_train'], a['df_train']['y'].values - a['train_prophet_forecast'] - a['residual_ets_model'].fittedvalues)
        np.testing.assert_allclose(a['hybrid_yhat_test'], a['prophet_yhat_test'] + a['residual_ets_pred_all'][182:] + a['resid_pred_test'])
        self.assertLess(abs(a['resid_train'].mean()), 1)


if __name__ == '__main__':
    unittest.main()
