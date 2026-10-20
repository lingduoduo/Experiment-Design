"""Regression checks for the hypothesis-test helpers (run with pytest)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).parent))

import categorical as cat
import confidence_interval as ci
import numeric as nm


# ---------------- confidence_interval ----------------

@pytest.mark.parametrize("k, n", [(245, 1000), (0, 10), (10, 10), (3, 7)])
def test_wilson_matches_scipy(k, n):
    ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    assert ci.ci_proportion_wilson(k, n)["ci"] == pytest.approx((ref.low, ref.high))


def test_wald_is_clipped_to_unit_interval():
    assert ci.ci_proportion_normal(0, 10)["ci"] == (0.0, 0.0)
    lo, hi = ci.ci_proportion_normal(1, 10, conf=0.999)["ci"]
    assert lo == 0.0 and 0 < hi < 1


@pytest.mark.parametrize("k, n", [(1.5, 10), (11, 10), (1, 0), (-1, 10), (np.nan, 10)])
def test_invalid_counts_raise(k, n):
    with pytest.raises(ValueError):
        ci.ci_proportion_wilson(k, n)


def test_daily_counts_reject_fractional_and_unpaired():
    with pytest.raises(ValueError):
        ci.ci_rate_from_daily_counts([1.5, 2], [10, 10])
    with pytest.raises(ValueError):
        ci.ci_rate_from_daily_counts([1, np.nan], [10, 10])


def test_rolling_mean_matches_scalar_loop_with_nans():
    rng = np.random.default_rng(1)
    x = rng.normal(5, 2, 200)
    x[[5, 40]] = np.nan
    window = 10
    mid, lo, hi = ci.rolling_ci_mean(x, window=window)
    assert np.isnan(mid[: window - 1]).all()
    for i in range(window - 1, len(x)):
        ref = ci.ci_mean_from_daily(x[i - window + 1 : i + 1])
        assert mid[i] == pytest.approx(ref["mean"], rel=1e-10)
        assert (lo[i], hi[i]) == pytest.approx(ref["ci"], rel=1e-10)


def test_rolling_mean_constant_window_has_zero_width():
    x = np.r_[np.random.default_rng(0).normal(1e6, 1e3, 50), np.full(10, 3.3)]
    mid, lo, hi = ci.rolling_ci_mean(x, window=7)
    assert hi[-1] - lo[-1] == 0.0


@pytest.mark.parametrize("method", ["wilson", "wald"])
def test_rolling_rate_matches_scalar_loop(method):
    rng = np.random.default_rng(2)
    s = rng.integers(0, 50, 200)
    n = s + rng.integers(0, 1000, 200)
    window = 7
    phat, lo, hi = ci.rolling_ci_rate(s, n, window=window, method=method)
    for i in range(window - 1, len(s)):
        ref = ci.ci_rate_from_daily_counts(s[i - window + 1 : i + 1], n[i - window + 1 : i + 1], method=method)
        assert phat[i] == pytest.approx(ref["p_hat"])
        assert (lo[i], hi[i]) == pytest.approx(ref["ci"])


def test_rolling_rate_zero_trial_window_is_nan():
    phat, lo, hi = ci.rolling_ci_rate([0, 0, 1], [0, 0, 5], window=2)
    assert np.isnan(phat[1]) and np.isnan(lo[1]) and np.isnan(hi[1])
    assert phat[2] == pytest.approx(0.2)


# ---------------- categorical ----------------

def test_categorical_imports_without_side_effects():
    # Previously the module ran plotting code against an undefined `student`.
    assert hasattr(cat, "two_proportion_ci_test")


def test_fisher_rejects_fractional_counts():
    with pytest.raises(ValueError):
        cat.fisher_exact_test([[1.5, 2], [3, 4]])
    odds, p = cat.fisher_exact_test([[8, 92], [18, 82]])
    ref = stats.fisher_exact([[8, 92], [18, 82]])
    assert (odds, p) == pytest.approx((ref[0], ref[1]))


def test_two_proportion_handles_degenerate_and_extreme_tails():
    res = cat.two_proportion_ci_test(0, 10, 0, 10)
    assert res["z"] == 0.0 and res["p_value"] == 1.0
    # 1 - cdf would underflow to exactly 0 here; sf keeps precision.
    assert cat.two_proportion_ci_test(900, 1000, 100, 1000)["p_value"] > 0


# ---------------- numeric ----------------

def test_jsd_uses_probability_kl():
    p, q = [0.1, 0.2, 0.7], [0.2, 0.2, 0.6]
    assert nm.jensen_shannon_distance(p, q) == pytest.approx(jensenshannon(p, q, base=2))


def test_sample_kl_is_finite_for_disjoint_samples():
    x = np.linspace(0, 1, 100)
    y = np.linspace(10, 11, 100)
    assert np.isfinite(nm.kl_divergence(x, y))


def test_welch_matches_scipy_and_keeps_tail_precision():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 30), rng.normal(0.5, 2, 40)
    res = nm.two_mean_ci_test(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert res["t"] == pytest.approx(ref.statistic)
    assert res["p_value"] == pytest.approx(ref.pvalue)
    assert nm.two_mean_ci_test(np.arange(50) + 100.0, np.arange(50))["p_value"] > 0


@pytest.mark.parametrize("a, b", [([1, 1, 1], [1, 1, 1]), ([1], [2, 3])])
def test_welch_rejects_undefined_inputs(a, b):
    with pytest.raises(ValueError):
        nm.two_mean_ci_test(a, b)


# ---------------- anova ----------------

def test_two_way_anova_effects_exclude_residual():
    pytest.importorskip("statsmodels")
    import anova
    values = [78, 85, 88, 92, 75, 88, 90, 94, 91, 89, 70, 72]
    method = ["A", "A", "A", "B", "B", "A", "A", "B", "B", "B", "A", "B"]
    gender = ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "F", "M"]
    effects = anova.anova_two_way(values, method, gender)["effects"]
    assert "Residual" not in effects
    assert set(effects) == {"C(A)", "C(B)", "C(A):C(B)"}
