import numpy as np
import pandas as pd
from scipy.stats import norm, t

def _validate_conf(conf):
    if not np.isfinite(conf) or not 0 < conf < 1:
        raise ValueError("conf must be between 0 and 1")


def _validate_counts(k, n):
    if (not np.isfinite([k, n]).all() or n <= 0 or not 0 <= k <= n
            or k != int(k) or n != int(n)):
        raise ValueError("Require integer counts with n>0 and 0<=k<=n")


def _daily_counts(successes, trials):
    s = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if s.ndim != 1 or n.ndim != 1 or s.shape != n.shape:
        raise ValueError("daily counts must be matching 1D arrays")
    missing = np.isnan(s) & np.isnan(n)
    s, n = np.where(missing, 0, s), np.where(missing, 0, n)
    if (not np.isfinite(s).all() or not np.isfinite(n).all()
            or np.any(s < 0) or np.any(n < s)
            or np.any(s != np.floor(s)) or np.any(n != np.floor(n))):
        raise ValueError("daily counts must be paired non-negative integers with successes<=trials")
    return s, n


def _validate_window(window, minimum=1):
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < minimum:
        raise ValueError(f"window must be an integer >={minimum}")


def _proportion_bounds(phat, n, conf, method):
    """Vectorized Wald/Wilson bounds clipped to [0, 1]."""
    if method not in ("wald", "wilson"):
        raise ValueError("method must be 'wald' or 'wilson'")
    z = norm.ppf(0.5 + conf / 2)
    if method == "wald":
        center = phat
        half = z * np.sqrt(phat * (1 - phat) / n)
    else:
        z2 = z * z
        denom = 1 + z2 / n
        center = (phat + z2 / (2 * n)) / denom
        half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return np.maximum(0.0, center - half), np.minimum(1.0, center + half)


# -----------------------------
# 1) Population proportion CI
# -----------------------------

def ci_proportion_normal(k, n, conf=0.95):
    """
    Wald (normal approx) CI for a population proportion p.
    Works best when n is large and p not too close to 0/1.
    """
    _validate_conf(conf)
    _validate_counts(k, n)
    phat = k / n
    lo, hi = _proportion_bounds(phat, n, conf, "wald")
    return {"p_hat": phat, "ci": (float(lo), float(hi)), "method": "wald"}


def ci_proportion_wilson(k, n, conf=0.95):
    """
    Wilson score CI for a population proportion p.
    Generally recommended over Wald.
    """
    _validate_conf(conf)
    _validate_counts(k, n)
    phat = k / n
    lo, hi = _proportion_bounds(phat, n, conf, "wilson")
    return {"p_hat": phat, "ci": (float(lo), float(hi)), "method": "wilson"}


# ------------------------------------
# 2) Population mean CI (daily metrics)
# ------------------------------------

def ci_mean_from_daily(daily_values, conf=0.95, use_t=True):
    """
    CI for the population mean using daily metric values (each day = one observation).

    - If use_t=True: uses t-interval (recommended, sigma unknown).
    - If use_t=False: uses z-interval (assumes known sigma; rarely true in practice).
    """
    _validate_conf(conf)
    x = np.asarray(daily_values, dtype=float)
    if x.ndim != 1 or np.isinf(x).any():
        raise ValueError("daily_values must be 1D without infinities")
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 daily values for a mean CI.")

    mean = x.mean()
    s = x.std(ddof=1)
    se = s / np.sqrt(n)

    if use_t:
        crit = t.ppf(0.5 + conf / 2, df=n - 1)
        method = "t"
    else:
        crit = norm.ppf(0.5 + conf / 2)
        method = "z"

    lo, hi = mean - crit * se, mean + crit * se
    return {
        "mean": mean,
        "n_days": n,
        "std": s,
        "se": se,
        "ci": (lo, hi),
        "method": method,
    }


# -----------------------------------------------------------
# 3) Daily metrics → confidence bounds for a RATE/proportion
# -----------------------------------------------------------
# Example: daily conversion rate = conversions / visits per day.
# If you want a CI for the overall proportion over the full window,
# aggregate k=sum(conversions), n=sum(visits) and use Wilson CI.

def ci_rate_from_daily_counts(daily_successes, daily_trials, conf=0.95, method="wilson"):
    """
    Computes CI for an overall proportion using daily counts:
      k = sum(successes_d), n = sum(trials_d)
    """
    successes, trials = _daily_counts(daily_successes, daily_trials)
    k = successes.sum()
    n = trials.sum()
    if method == "wald":
        return ci_proportion_normal(k, n, conf=conf)
    elif method == "wilson":
        return ci_proportion_wilson(k, n, conf=conf)
    else:
        raise ValueError("method must be 'wald' or 'wilson'")


# -----------------------------------------------------------
# 4) Rolling confidence bounds (monitoring daily metrics)
# -----------------------------------------------------------

def rolling_ci_mean(daily_values, window=7, conf=0.95):
    """
    Rolling CI bounds for the mean over a moving window of days.
    Returns arrays aligned to the input with NaNs for the first window-1 days.
    NaNs are omitted within windows; bounds need at least two valid values.
    """
    _validate_conf(conf)
    _validate_window(window, minimum=2)
    x = np.asarray(daily_values, dtype=float)
    if x.ndim != 1 or np.isinf(x).any():
        raise ValueError("daily_values must be 1D without infinities")
    rolling = pd.Series(x).rolling(window, min_periods=1)
    mid = rolling.mean().to_numpy()
    count = rolling.count().to_numpy()
    se = rolling.std(ddof=1).to_numpy() / np.sqrt(np.maximum(count, 1))
    half = np.full(len(x), np.nan)
    valid = count >= 2
    half[valid] = t.ppf(0.5 + conf / 2, count[valid] - 1) * se[valid]
    lo, hi = mid - half, mid + half
    for result in (mid, lo, hi):
        result[:window - 1] = np.nan
    return mid, lo, hi


def rolling_ci_rate(daily_successes, daily_trials, window=7, conf=0.95, method="wilson"):
    """
    Rolling CI bounds for a proportion/rate over a moving window using aggregated counts.
    Paired missing days contribute zero counts; zero-trial windows return NaNs.
    """
    _validate_conf(conf)
    _validate_window(window)
    if method not in ("wilson", "wald"):
        raise ValueError("method must be 'wald' or 'wilson'")
    s, n = _daily_counts(daily_successes, daily_trials)
    # Rolling sums avoid re-summing overlapping slices.
    k = pd.Series(s).rolling(window, min_periods=window).sum().to_numpy()
    total = pd.Series(n).rolling(window, min_periods=window).sum().to_numpy()
    valid = total > 0
    phat = np.full(len(s), np.nan)
    lo, hi = phat.copy(), phat.copy()
    phat[valid] = k[valid] / total[valid]
    lo[valid], hi[valid] = _proportion_bounds(phat[valid], total[valid], conf, method)
    return phat, lo, hi


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # A) Proportion CI (single snapshot)
    k, n = 245, 1000
    print(ci_proportion_wilson(k, n, conf=0.95))

    # B) Mean CI using daily metric values (e.g., daily revenue per user)
    daily_revenue_per_user = [1.2, 1.0, 1.4, 0.9, 1.1, 1.3, 1.25, 1.05, 1.6, 1.15]
    print(ci_mean_from_daily(daily_revenue_per_user, conf=0.95))

    # C) Rate CI using daily counts (e.g., conversions / visits)
    daily_conv = [30, 28, 35, 27, 32, 31, 29]
    daily_visits = [1200, 1150, 1300, 1100, 1250, 1230, 1180]
    print(ci_rate_from_daily_counts(daily_conv, daily_visits, conf=0.95, method="wilson"))

    # D) Rolling bounds (7-day window)
    mean_mid, mean_lo, mean_hi = rolling_ci_mean(daily_revenue_per_user, window=5, conf=0.95)
    print("Rolling mean CI last:", mean_mid[-1], (mean_lo[-1], mean_hi[-1]))

    rate_mid, rate_lo, rate_hi = rolling_ci_rate(daily_conv, daily_visits, window=3, conf=0.95)
    print("Rolling rate CI last:", rate_mid[-1], (rate_lo[-1], rate_hi[-1]))
