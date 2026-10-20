"""Regression checks for the forecasting example (run with unittest)."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

PATH = Path(__file__).with_name('Prophet_ETS_LSTM.py')


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('forecast_example', PATH)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

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

    @unittest.skipUnless(
        importlib.util.find_spec("prophet") and importlib.util.find_spec("statsmodels"),
        "Full pipeline requires requirements-forecast.txt",
    )
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


    def test_metrics_accept_lists_and_reject_broadcasting(self):
        self.assertEqual(self.module.mae([1, 3], [2, 1]), 1.5)
        self.assertAlmostEqual(self.module.rmse([1, 3], [2, 1]), np.sqrt(2.5))
        self.assertEqual(self.module.mape([1, 4], [2, 2]), 75)
        for metric in (self.module.mae, self.module.rmse, self.module.mape):
            for a, b in [([1, 2], [[1], [2]]), ([], []), ([np.nan], [1]), ([1, 2], [1])]:
                with self.subTest(metric=metric.__name__, a=a, b=b):
                    with self.assertRaises(ValueError):
                        metric(a, b)

    def test_integer_arguments_are_validated(self):
        for value in [True, 2.5, 0, -1]:
            with self.subTest(input_len=value):
                with self.assertRaises(ValueError):
                    self.module.ResidDataset(np.arange(10), input_len=value)
        for value in [True, 1.5, -1]:
            with self.subTest(total_steps=value):
                with self.assertRaises(ValueError):
                    self.module.rolling_resid_predict(self.module.ResidLSTM(), np.zeros(28), value)
        with self.assertRaises(ValueError):
            self.module.run_experiment(epochs=1.5, make_plots=False)

    def test_dataset_does_not_materialize_overlapping_windows(self):
        dataset = self.module.ResidDataset(np.arange(10000), input_len=100)
        self.assertTrue(np.shares_memory(dataset.X, dataset.series))
        self.assertTrue(np.shares_memory(dataset.y, dataset.series))
        x, y = dataset[len(dataset) - 1]
        np.testing.assert_array_equal(x.numpy().ravel(), np.arange(9899, 9999))
        np.testing.assert_array_equal(y.numpy(), [9999])

    def test_batched_indexing_matches_per_sample_collation(self):
        series = np.random.default_rng(0).normal(size=200)
        dataset = self.module.ResidDataset(series, input_len=28)
        idx = [5, 0, 171, 42]
        xb, yb = dataset[idx]
        self.assertEqual((xb.shape, yb.shape), ((4, 28, 1), (4, 1)))
        expected = torch.utils.data.default_collate([dataset[i] for i in idx])
        torch.testing.assert_close(xb, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(yb, expected[1], rtol=0, atol=0)
        xb[:] = -1
        self.assertFalse((dataset.series == -1).any())

    def test_dataset_retains_snapshot_of_input(self):
        source = np.arange(6, dtype=np.float32)
        dataset = self.module.ResidDataset(source, input_len=3)
        source[:] = -1
        x, y = dataset[0]
        np.testing.assert_array_equal(x.numpy().ravel(), [0, 1, 2])
        np.testing.assert_array_equal(y.numpy(), [3])

    def test_recursive_inference_preserves_model_mode_and_context(self):
        class Increment(torch.nn.Module):
            def forward(self, x):
                return x[:, -1, :] + 1
        model = Increment()
        context = np.arange(4, dtype=np.float32)
        actual = self.module.rolling_resid_predict(model, context, 100, input_len=4)
        np.testing.assert_array_equal(actual, np.arange(4, 104))
        np.testing.assert_array_equal(context, np.arange(4))
        self.assertTrue(model.training)
        empty = self.module.rolling_resid_predict(model, context, 0, input_len=4)
        self.assertEqual(empty.shape, (0,))

    def test_recursive_inference_rejects_nonfinite_predictions(self):
        class Invalid(torch.nn.Module):
            def forward(self, x):
                return torch.full((1, 1), float('nan'))
        with self.assertRaises(ValueError):
            self.module.rolling_resid_predict(Invalid(), np.arange(4), 3, input_len=4)


if __name__ == '__main__':
    unittest.main()
