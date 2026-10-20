"""Offline checks for the Timestral example; no model download required."""
import importlib.util
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PATH = Path(__file__).with_name('timestral_example.py')
SCRIPT = PATH.with_name('timestral.py')
try:
    distribution('timestral')
    HAS_BACKEND = True
except PackageNotFoundError:
    HAS_BACKEND = False


class TimestralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('timestral_example', PATH)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_invalid_context_is_rejected_before_model_loading(self):
        for values in [[], np.ones((1, 0)), np.ones((1, 1, 1, 2)), [np.inf], [np.nan], ['bad']]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.module.forecast_series(values)
        for horizon in [0, -1, True, 1.5]:
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                self.module.forecast_series([1, 2], prediction_length=horizon)

    def test_loading_options_reach_backend(self):
        class Loader:
            @staticmethod
            def from_pretrained(model_id, **options):
                return model_id, options
        class Backend:
            TimestralForecaster = Loader
        with patch.object(self.module, '_backend', return_value=Backend):
            model_id, options = self.module.load_forecaster(
                Path('/tmp/local-model'), revision='pinned-commit',
                cache_dir=Path('/tmp/model-cache'), token=False,
                local_files_only=True,
            )
        self.assertEqual(model_id, '/tmp/local-model')
        self.assertEqual(options, dict(device='cpu', revision='pinned-commit',
                                      cache_dir=Path('/tmp/model-cache'), token=False,
                                      local_files_only=True))

    def test_forecast_loading_options_are_forwarded(self):
        class Loader:
            @staticmethod
            def from_pretrained(model_id, **options):
                class Forecaster:
                    def forecast(self, context, prediction_length):
                        return model_id, options, prediction_length
                return Forecaster()
        class Backend:
            TimestralForecaster = Loader
        with patch.object(self.module, '_backend', return_value=Backend):
            model_id, options, horizon = self.module.forecast_series(
                [1, 2, 3], 8, model_id='org/model', revision='v1',
                cache_dir='/tmp/cache', token='test-token', local_files_only=True,
            )
        self.assertEqual((model_id, horizon), ('org/model', 8))
        self.assertEqual(options['revision'], 'v1')
        self.assertEqual(options['cache_dir'], '/tmp/cache')
        self.assertEqual(options['token'], 'test-token')
        self.assertTrue(options['local_files_only'])

    def test_demo_is_reproducible_without_changing_global_rng(self):
        np.random.seed(17)
        expected = np.random.random()
        np.random.seed(17)
        first = self.module.make_demo_context(seed=42)
        second = self.module.make_demo_context(seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 1, 2000))
        self.assertEqual(np.random.random(), expected)

    def test_forecast_passes_missing_values_to_injected_backend(self):
        # A lightweight backend is used to exercise forwarding without weights.
        class Backend:
            def forecast(self, context, prediction_length):
                np.testing.assert_array_equal(context, [1, np.nan, 3])
                return np.full(prediction_length, context[-1])
        result = self.module.forecast_series([1, np.nan, 3], prediction_length=4, forecaster=Backend())
        np.testing.assert_array_equal(result, [3, 3, 3, 3])

    def test_covariates_preserve_missing_values_and_indicators(self):
        past = np.array([[[2, np.nan, 4]], [[5, 6, 7]]])
        known = np.array([[[0, 1, np.nan, 0, 1]], [[1, 0, 1, 0, 1]]])
        class Backend:
            def forecast(self, context, prediction_length, past_covariates, known_covariates):
                np.testing.assert_array_equal(past_covariates, past)
                np.testing.assert_array_equal(known_covariates, known)
                # Return targets only; conditioning rows are not output variates.
                return np.repeat(context[..., -1:], prediction_length, axis=-1)
        result = self.module.forecast_series(np.ones((2, 3, 3)), 2, forecaster=Backend(),
                                             past_covariates=past, known_covariates=known)
        self.assertEqual(result.shape, (2, 3, 2))

    def test_invalid_covariates_are_rejected_before_loading(self):
        for name, invalid in [
            ('past_covariates', np.ones((2, 1, 3))),
            ('past_covariates', np.ones((1, 1, 4))),
            ('past_covariates', np.ones((1, 3))),
            ('past_covariates', np.full((1, 1, 3), np.inf)),
            ('known_covariates', np.ones((1, 1, 3))),
            ('known_covariates', np.ones((2, 1, 5))),
            ('known_covariates', np.full((1, 1, 5), -np.inf)),
        ]:
            with self.subTest(name=name, shape=invalid.shape), self.assertRaises(ValueError):
                self.module.forecast_series([1, 2, 3], prediction_length=2, **{name: invalid})

    def test_categorical_covariates_preserve_codes_and_missing_values(self):
        # Adjacent large codes would collapse if converted to float32.
        past = np.array([[[2**24, 2**24 + 1, np.nan]]], dtype=np.float64)
        known = np.array([[[7, np.nan, 9, 10, 11]]], dtype=np.float64)
        class Backend:
            def forecast(self, context, prediction_length, **covariates):
                self_test.assertEqual(set(covariates), set(arguments))
                for name, codes in covariates.items():
                    expected = past if name == 'categorical_past_covariates' else known
                    np.testing.assert_array_equal(codes, expected)
                return np.repeat(context[-1:], prediction_length)
        self_test = self
        for arguments in [dict(categorical_past_covariates=past),
                          dict(categorical_known_covariates=known),
                          dict(categorical_past_covariates=past, categorical_known_covariates=known)]:
            with self.subTest(arguments=list(arguments)):
                result = self.module.forecast_series([1, 2, 3], 2, forecaster=Backend(), **arguments)
                np.testing.assert_array_equal(result, [3, 3])

    def test_invalid_categorical_covariates_are_rejected_before_loading(self):
        for name, length in [('categorical_past_covariates', 3), ('categorical_known_covariates', 5)]:
            for invalid in [np.ones((2, 1, length)), np.ones((1, 1, length + 1)),
                            np.ones((1, length)), np.full((1, 1, length), np.inf),
                            np.full((1, 1, length), 'category')]:
                with self.subTest(name=name, shape=invalid.shape), self.assertRaises(ValueError):
                    self.module.forecast_series([1, 2, 3], 2, **{name: invalid})

    def test_export_contains_quantiles_and_metadata(self):
        import torch
        class Forecast:
            quantiles = torch.arange(12).reshape(1, 1, 4, 3)
            quantile_levels = (.1, .5, .9)
            median = quantiles[..., 1]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'forecast.npz'
            self.module.save_forecast(Forecast(), target)
            with np.load(target, allow_pickle=False) as result:
                np.testing.assert_array_equal(result['median'], [[[1, 4, 7, 10]]])
                self.assertEqual(result['quantiles'].shape, (1, 1, 4, 3))
                np.testing.assert_array_equal(result['quantile_levels'], [.1, .5, .9])

    @unittest.skipUnless(HAS_BACKEND, 'Install requirements-timestral.txt for backend integration')
    def test_real_backend_with_tiny_random_model(self):
        backend = self.module._backend()
        config = backend.TimestralConfig(
            patch_size=4, d_model=16, num_heads=2, mlp_hidden_dim=32,
            block_schedule=('time',), quantile_knots=(.1, .5, .9),
            block_size=8, context_length=32,
        )
        forecaster = backend.TimestralForecaster(backend.TimestralModel(config).eval())
        forecast = self.module.forecast_series([1, np.nan, 3, 4], prediction_length=5, forecaster=forecaster)
        self.assertEqual(tuple(forecast.quantiles.shape), (1, 1, 5, 3))
        self.assertTrue(np.isfinite(forecast.median.cpu().numpy()).all())

    @unittest.skipUnless(HAS_BACKEND, 'Install requirements-timestral.txt for backend integration')
    def test_real_backend_covariates_are_conditioning_rows(self):
        backend = self.module._backend()
        config = backend.TimestralConfig(
            patch_size=4, d_model=16, num_heads=2, mlp_hidden_dim=32,
            block_schedule=('time_variate', 'time_variate'), quantile_knots=(.1, .5, .9),
            block_size=8, context_length=32,
        )
        forecaster = backend.TimestralForecaster(backend.TimestralModel(config).eval())
        forecast = self.module.forecast_series(
            np.ones((1, 2, 4)), prediction_length=5, forecaster=forecaster,
            past_covariates=np.array([[[1, np.nan, 2, 3]]]),
            known_covariates=np.array([[[0, 1, 0, 1, 0, 1, 0, 1, 0]]]),
        )
        self.assertEqual(tuple(forecast.quantiles.shape), (1, 2, 5, 3))
        self.assertTrue(np.isfinite(forecast.median.cpu().numpy()).all())

    @unittest.skipUnless(HAS_BACKEND, 'Install requirements-timestral.txt for backend integration')
    def test_real_backend_categorical_covariates(self):
        backend = self.module._backend()
        config = backend.TimestralConfig(
            patch_size=4, d_model=16, num_heads=2, mlp_hidden_dim=32,
            block_schedule=('time_variate', 'time_variate'), quantile_knots=(.1, .5, .9),
            block_size=8, context_length=32,
        )
        forecaster = backend.TimestralForecaster(backend.TimestralModel(config).eval())
        for variates in (1, 2):
            with self.subTest(variates=variates):
                forecast = self.module.forecast_series(
                    np.arange(1, 5)[None, None, :].repeat(variates, axis=1), 5,
                    forecaster=forecaster,
                    categorical_past_covariates=np.array([[[2**24, 2**24 + 1, np.nan, 2**24]]]),
                    categorical_known_covariates=np.array([[[1, 2, np.nan, 1, 3, 1, 2, 3, np.nan]]]),
                )
                self.assertEqual(tuple(forecast.quantiles.shape), (1, variates, 5, 3))
                self.assertTrue(np.isfinite(forecast.median.cpu().numpy()).all())

    @unittest.skipUnless(HAS_BACKEND, 'Install requirements-timestral.txt for backend integration')
    def test_reusable_module_survives_backend_import(self):
        code = (
            "import timestral_example as example; "
            "example._backend(); "
            "import timestral_example as again; "
            "assert again is example; "
            "assert callable(again.forecast_series)"
        )
        result = subprocess.run([sys.executable, '-c', code], cwd=PATH.parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_script_help_does_not_load_model(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--prediction-length', result.stdout)


if __name__ == '__main__':
    unittest.main()
