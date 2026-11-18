"""Four analytical guards; real notebook execution covers model integration."""
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd


def api():
    return importlib.import_module('platform_revenue')


def test_sample_is_reproducible_and_accounts_for_revenue():
    m = api()
    a = m.make_sample()
    pd.testing.assert_frame_equal(a, m.make_sample())
    assert len(a) == 1825 and a.dt.is_unique
    np.testing.assert_allclose(a.revenue, a.dau * a.payer_rate * a.revenue_per_payer)
    assert (a.select_dtypes('number') >= 0).all().all()
    assert not any('event' in c for cols in m.GROUPS.values() for c in cols)


def test_forecast_uses_only_available_history_and_delays_features():
    m = api()
    calls = []
    class Recorder:
        def forecast(self, context, prediction_length, **kwargs):
            calls.append((context.copy(), kwargs))
            return SimpleNamespace(median=np.repeat(context[..., -1:], prediction_length, axis=-1))
    df = m.make_sample()
    cols = m.GROUPS['discovery']
    first = m.forecast_at(Recorder(), df, 500, cols, delay=7)
    modified = df.copy()
    modified.loc[500:, ['revenue'] + cols] *= 100
    second = m.forecast_at(Recorder(), modified, 500, cols, delay=7)
    np.testing.assert_equal(first, second)
    np.testing.assert_equal(calls[0][0], calls[1][0])
    for key in calls[0][1]:
        np.testing.assert_equal(calls[0][1][key], calls[1][1][key])
    np.testing.assert_equal(calls[0][1]['past_covariates'][0, :, -1], df.loc[492, cols].to_numpy(np.float32))
    assert calls[0][0].shape == (1, 1, 365)
    assert calls[0][1]['known_covariates'].shape[-1] == 455


def test_decomposition_allocates_interactions_exactly():
    m = api()
    previous = pd.Series(dict(dau=100, payer_rate=.1, revenue_per_payer=20))
    current = pd.Series(dict(dau=120, payer_rate=.15, revenue_per_payer=30))
    parts = m.decomposition(previous, current)
    assert np.isclose(parts.contribution.sum(), 340)
    assert (parts.contribution > 0).all()
    np.testing.assert_allclose(m.decomposition(current, previous).contribution, -parts.contribution)


def test_calibration_uses_finite_sample_rank_without_hiding_insufficient_data():
    m = api()
    assert m.calibration_radius(np.arange(1, 10), coverage=.9) == 9
    assert np.isinf(m.calibration_radius([1, 2], coverage=.9))
