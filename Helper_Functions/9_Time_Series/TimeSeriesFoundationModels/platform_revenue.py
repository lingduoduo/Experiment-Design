"""Synthetic platform-revenue experiments; predictive diagnostics, not causal effects."""
from itertools import permutations
from math import ceil

import numpy as np
import pandas as pd

GROUPS = {
    'audience': ['dau', 'returning_viewer_rate'],
    'supply': ['active_rooms', 'content_diversity'],
    'discovery': ['views_per_dau', 'ctr', 'room_join_rate'],
    'engagement': ['watch_minutes_per_join', 'repeat_participation_rate', 'follow_rate'],
    'monetization': ['payer_rate', 'revenue_per_payer'],
}
HORIZONS = (7, 30, 90)
ANOMALY_TARGETS = ['revenue', 'dau', 'room_join_rate', 'watch_minutes_per_join']


def make_sample(n=1825, seed=2026):
    """Daily accounting-consistent data with AR(1) signals and delayed dependencies.

    Generating equations describe this fictional system only. Event labels and
    hidden recommendation quality are excluded from all forecasting inputs.
    """
    if n < 365:
        raise ValueError('Use at least 365 daily observations')
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    dt = pd.date_range('2021-01-01', periods=n, freq='D')
    weekly = np.sin(2 * np.pi * dt.dayofweek.to_numpy() / 7)
    annual = np.sin(2 * np.pi * dt.dayofyear.to_numpy() / 365.25)
    campaign = ((dt.day >= 15) & (dt.day <= 18)).astype(float)
    def ar1(scale):
        out = np.zeros(n)
        for i in range(1, n):
            out[i] = .94 * out[i - 1] + rng.normal(0, scale)
        return out
    def lag(x, days):
        return np.r_[np.repeat(x[0], days), x[:-days]]
    quality = np.clip(.62 + ar1(.025) + .03 * annual
                      - .20 * ((t >= 1400) & (t < 1440)), .15, .95)
    supply = np.clip(1 + ar1(.015) - .24 * ((t >= 1510) & (t < 1535)), .4, 1.5)
    dau = np.maximum(100, 22000 + 8 * t + 1400 * weekly + 900 * annual
                     + 1800 * lag(quality, 14) + rng.normal(0, 300, n))
    rooms = np.maximum(50, 1300 * supply + 70 * weekly + rng.normal(0, 20, n))
    join_rate = np.clip(.16 + .28 * quality + .04 * supply + rng.normal(0, .006, n), .01, .95)
    watch = np.maximum(1, 8 + 15 * lag(quality, 7) + 3 * supply + rng.normal(0, .35, n))
    repeat = np.clip(.18 + .40 * lag(join_rate, 14) + .006 * lag(watch, 7)
                     + rng.normal(0, .004, n), .01, .95)
    payer_rate = np.clip(.012 + .04 * lag(quality, 7) + .055 * lag(repeat, 14)
                         + .003 * campaign + rng.normal(0, .0008, n), .001, .5)
    payer_rate *= np.where((t >= 1640) & (t < 1647), .55, 1.)
    spend = np.maximum(1, 26 + .35 * lag(watch, 7) + 1.2 * annual + rng.normal(0, .35, n))
    df = pd.DataFrame({
        'dt': dt, 'dau': dau,
        'returning_viewer_rate': np.clip(.35 + .5 * repeat + rng.normal(0, .008, n), .01, .99),
        'active_rooms': rooms, 'content_diversity': np.clip(.65 * supply + rng.normal(0, .015, n), .01, .99),
        'views_per_dau': np.maximum(1, 5 + 3 * quality + .2 * weekly + rng.normal(0, .1, n)),
        'ctr': np.clip(.025 + .06 * quality + rng.normal(0, .002, n), .001, .99),
        'room_join_rate': join_rate, 'watch_minutes_per_join': watch,
        'repeat_participation_rate': repeat,
        'follow_rate': np.clip(.015 + .025 * quality + rng.normal(0, .004, n), .001, .99),
        'payer_rate': payer_rate, 'revenue_per_payer': spend,
    })
    df['paying_users'] = dau * payer_rate  # Continuous expected count in this simulation.
    df['revenue'] = df.paying_users * spend
    df['cumulative_revenue'] = df.revenue.cumsum()
    df['event'] = 'none'
    df.loc[(t >= 1400) & (t < 1470), 'event'] = 'discovery decline + delayed effects'
    df.loc[(t >= 1510) & (t < 1565), 'event'] = 'supply decline + delayed effects'
    df.loc[(t >= 1640) & (t < 1647), 'event'] = 'payment outage'
    return df


def calendar(dates):
    """Predetermined calendar only; no realized future KPIs."""
    dates = pd.DatetimeIndex(dates)
    return np.array([
        (dates.dayofweek >= 5).astype(float),
        np.sin(2 * np.pi * dates.dayofyear / 365.25),
        np.cos(2 * np.pi * dates.dayofyear / 365.25),
        ((dates.day >= 15) & (dates.day <= 18)).astype(float),
    ], dtype=np.float32)


def forecast_at(model, df, cutoff, columns=(), delay=0, horizon=90, context=365,
                delay_columns=None):
    """Forecast from df[:cutoff]; delay 7 means latest KPI is df[cutoff-8]."""
    if not isinstance(cutoff, (int, np.integer)) or not context <= cutoff <= len(df):
        raise ValueError('cutoff must have a full historical context within the data')
    if delay < 0 or int(delay) != delay or horizon < 1 or int(horizon) != horizon:
        raise ValueError('delay must be a nonnegative integer; horizon a positive integer')
    history = df.iloc[:cutoff]
    y = history.revenue.to_numpy(np.float32)[-context:][None, None, :]
    dates = pd.date_range(history.dt.iloc[-context], periods=context + horizon, freq='D')
    kwargs = {'known_covariates': calendar(dates)[None]}
    if columns:
        past = history[list(columns)].copy()
        delayed = list(columns) if delay_columns is None else list(delay_columns)
        past[delayed] = past[delayed].shift(delay)
        kwargs['past_covariates'] = past.iloc[-context:].to_numpy(np.float32).T[None]
    out = model.forecast(y, prediction_length=horizon, **kwargs).median
    if hasattr(out, 'detach'):
        out = out.detach().cpu().numpy()
    return np.asarray(out, dtype=float).reshape(-1)


def decomposition(previous, current):
    """Exact symmetric Shapley allocation of change in a three-factor product.

    Average each factor's marginal accounting contribution over all six orders.
    This allocates interactions and does not imply causal effects.
    """
    factors = ['dau', 'payer_rate', 'revenue_per_payer']
    before = previous[factors].to_numpy(float)
    after = current[factors].to_numpy(float)
    if not np.isfinite(np.r_[before, after]).all():
        raise ValueError('Accounting factors must be finite')
    contributions = np.zeros(3)
    for order in permutations(range(3)):
        state = before.copy()
        for i in order:
            old = np.prod(state)
            state[i] = after[i]
            contributions[i] += (np.prod(state) - old) / 6
    return pd.DataFrame({'factor': factors, 'contribution': contributions})


def calibration_radius(scores, coverage=.9):
    """Split-conformal order statistic; infinity when sample cannot support rank.

    Time-series dependence/distribution shifts invalidate general coverage
    guarantees; this function computes the finite-sample rank, not a guarantee.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('scores must be a nonempty finite nonnegative vector')
    if not 0 < coverage < 1:
        raise ValueError('coverage must be between zero and one')
    rank = ceil((len(scores) + 1) * coverage)
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float('inf')


def experiment_specs():
    """Fixed in advance, without selecting configurations on evaluation results."""
    audience = GROUPS['audience']
    full = sum(GROUPS.values(), [])
    specs = {'revenue_only': ([], 0, []), 'audience': (audience, 0, [])}
    for group in ['supply', 'discovery', 'engagement', 'monetization']:
        specs['audience_plus_' + group] = (audience + GROUPS[group], 0, [])
    specs['full'] = (full, 0, [])
    for group, cols in GROUPS.items():
        specs['full_minus_' + group] = ([c for c in full if c not in cols], 0, [])
    selected = GROUPS['discovery'] + GROUPS['engagement']
    for delay in [0, 1, 7, 14, 30]:
        specs[f'signals_delayed_{delay:02d}'] = (audience + selected, delay, selected)
    return specs


def run_study(df, model, progress=print):
    """Run fixed 365-day contexts and disjoint 90-day forecast windows.

    Returns raw origin-level evidence, paired test scores, cumulative intervals,
    and independently calibrated multivariate daily KPI anomaly diagnostics.
    """
    if len(df) < 1805 or not df.dt.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError('Study requires at least 1805 consecutive daily rows')
    specs = experiment_specs()
    origins = list(range(365, 1716, 90))
    records, daily = [], []
    for j, cutoff in enumerate(origins):
        split = 'development' if j < 3 else 'calibration' if j < 12 else 'evaluation'
        progress(f'{j + 1}/{len(origins)}: {df.dt.iloc[cutoff].date()} ({split})')
        history = df.iloc[:cutoff]
        truth = df.revenue.iloc[cutoff:cutoff + 90].to_numpy()
        scale = float(history.revenue.iloc[-28:].median())
        forecasts = {'seasonal_naive': np.resize(history.revenue.iloc[-7:].to_numpy(), 90)}
        for name, (cols, delay, delayed) in specs.items():
            forecasts[name] = forecast_at(model, df, cutoff, cols, delay=delay, delay_columns=delayed)
        for name, pred in forecasts.items():
            if len(pred) != 90 or not np.isfinite(pred).all():
                raise ValueError(f'Invalid model output for {name} at {cutoff}')
            for h in HORIZONS:
                actual, predicted = float(truth[:h].sum()), float(pred[:h].sum())
                records.append(dict(cutoff=cutoff, origin=df.dt.iloc[cutoff], split=split,
                                    experiment=name, horizon=h, actual=actual, predicted=predicted,
                                    abs_error=abs(actual - predicted), error=predicted - actual,
                                    scale=scale * h))
        # Joint target forecasts for detecting anomalous observed KPI movements.
        context = history[ANOMALY_TARGETS].iloc[-365:].to_numpy(np.float32).T[None]
        dates = pd.date_range(history.dt.iloc[-365], periods=455, freq='D')
        predicted = model.forecast(context, prediction_length=90,
                                   known_covariates=calendar(dates)[None]).median
        if hasattr(predicted, 'detach'):
            predicted = predicted.detach().cpu().numpy()
        predicted = np.asarray(predicted)[0]
        for k, metric in enumerate(ANOMALY_TARGETS):
            metric_scale = float(history[metric].iloc[-28:].median())
            for step in range(90):
                row = df.iloc[cutoff + step]
                daily.append(dict(dt=row['dt'], cutoff=cutoff, split=split, metric=metric,
                                  actual=row[metric], predicted=float(predicted[k, step]),
                                  scale=metric_scale, event=row.event, lead_day=step + 1))
    forecasts = pd.DataFrame(records)
    evaluation = forecasts[forecasts.split == 'evaluation'].copy()
    summaries = []
    for (name, h), group in evaluation.groupby(['experiment', 'horizon']):
        summaries.append(dict(experiment=name, horizon=h, origins=len(group),
                              cumulative_wape=group.abs_error.sum() / group.actual.sum(),
                              cumulative_mae=group.abs_error.mean(), bias=group.error.mean()))
    scores = pd.DataFrame(summaries)
    paired = []
    for name in specs:
        reference = ('full' if name.startswith('full_minus_') else
                     'audience' if name.startswith(('audience_plus_', 'signals_delayed_')) else 'revenue_only')
        if name == reference:
            continue
        for h in HORIZONS:
            a = evaluation[(evaluation.experiment == name) & (evaluation.horizon == h)].set_index('cutoff')
            b = evaluation[(evaluation.experiment == reference) & (evaluation.horizon == h)].set_index('cutoff')
            delta = a.abs_error - b.abs_error
            paired.append(dict(experiment=name, reference=reference, horizon=h,
                               delta_mae=float(delta.mean()), wins=int((delta < 0).sum()), origins=len(delta),
                               # Positive means the experiment improves cumulative absolute error.
                               improvement_pct=100 * (1 - a.abs_error.sum() / b.abs_error.sum())))
    intervals = []
    for h in HORIZONS:
        selected = forecasts[(forecasts.experiment == 'full') & (forecasts.horizon == h)]
        cal = selected[selected.split == 'calibration']
        radius = calibration_radius((cal.abs_error / cal.scale).to_numpy(), .9)
        out = selected[selected.split == 'evaluation'].copy()
        out['lower'] = np.maximum(0, out.predicted - radius * out.scale)
        out['upper'] = out.predicted + radius * out.scale
        out['covered'] = out.actual.between(out.lower, out.upper)
        out['calibration_origins'] = len(cal)
        intervals.append(out)
    daily = pd.DataFrame(daily)
    anomaly_rows = []
    for metric, rows in daily.groupby('metric', sort=False):
        cal = rows[rows.split == 'calibration']
        radius = calibration_radius((abs(cal.actual - cal.predicted) / cal.scale).to_numpy(), .95)
        out = rows[rows.split == 'evaluation'].copy()
        out['lower'] = np.maximum(0, out.predicted - radius * out.scale)
        out['upper'] = out.predicted + radius * out.scale
        out['flagged'] = ~out.actual.between(out.lower, out.upper)
        out['calibration_days'] = len(cal)
        anomaly_rows.append(out)
    return {'forecasts': forecasts, 'scores': scores, 'paired': pd.DataFrame(paired),
            'intervals': pd.concat(intervals, ignore_index=True), 'daily_forecasts': daily,
            'anomalies': pd.concat(anomaly_rows, ignore_index=True)}
