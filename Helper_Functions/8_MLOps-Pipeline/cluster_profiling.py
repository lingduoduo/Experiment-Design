"""Reproducible local version of the sample cluster profiling workflow."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

FEATURES = ['original_posts', 'reblogs', 'likes', 'replies', 'searches',
            'messaging', 'push_launched']
REFERENCE_DATE = '2018-05-12'
GEOS = ['US', 'GB', 'KR', 'CA', 'BR', 'DE', 'MX', 'TR', 'MY', 'AU']


def generate_sample_data(rows=1200, seed=2016, reference_date=REFERENCE_DATE):
    """Generate synthetic weekly app activity, including rows excluded by filters."""
    if rows < 1:
        raise ValueError('rows must be positive')
    rng = np.random.default_rng(seed)
    segment = rng.choice(3, rows, p=[.5, .3, .2])
    days = np.minimum(7, 1 + rng.poisson(1 + segment))
    sessions = days * (1 + rng.poisson(1 + segment))
    rates = np.array([[.1, .4, 2, .1, 1, .2, .5],
                      [1, 3, 5, 1, 2, 2, 1],
                      [3, 1, 8, 3, 4, 5, 2]])
    counts = rng.poisson(sessions[:, None] * rates[segment])
    data = pd.DataFrame(counts, columns=FEATURES)
    data.index = pd.Index([f'user_{i:06d}' for i in range(rows)], name='user_id')
    age_days = rng.integers(1, 2000, rows)
    data['regi_dt'] = (pd.Timestamp(reference_date) - pd.to_timedelta(age_days, unit='D')).strftime('%Y-%m-%d')
    data['device'] = rng.choice(['ios', 'android'], rows, p=[.8, .2])
    data['geo'] = rng.choice(GEOS + ['JP', 'FR'], rows)
    data['active_dt'] = days
    data['csi_cnt'] = sessions
    data['user_age'] = rng.integers(4, 85, rows)
    data['timespent'] = np.round(sessions * rng.gamma(3, 4, rows), 2)
    for column in ['users_with_custom', 'linked']:
        data[column] = rng.binomial(1, .4, rows)
    for column, rate in [('postimpressions', 25), ('activity', 8), ('notes', 2),
                         ('follows', .5), ('unfollows', .1), ('received_follows', .4),
                         ('blog_subscribes', .1), ('blogs', .2),
                         ('received_subscribers', .1), ('subscribers', 2)]:
        data[column] = rng.poisson(sessions * rate)
    data['followers'] = rng.integers(0, 3000, rows)
    data['feature_name'] = rng.choice(['0', 'feed_ranking', 'notifications'], rows)
    data['feature_subgroup'] = rng.choice(['control', 'treatment'], rows)
    # Missing activity exercises the original workflow's fill-with-zero rule.
    data.loc[data.index[rng.random(rows) < .02], 'likes'] = np.nan
    return data


def preprocess(data, reference_date=REFERENCE_DATE):
    """Filter established iOS users and normalize event counts per session."""
    required = FEATURES + ['regi_dt', 'device', 'followers', 'active_dt', 'csi_cnt',
        'user_age', 'timespent', 'users_with_custom', 'linked', 'postimpressions',
        'activity', 'notes', 'follows', 'unfollows', 'received_follows',
        'blog_subscribes', 'geo', 'feature_name', 'feature_subgroup']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if not data.index.is_unique or data.index.hasnans:
        raise ValueError('User IDs must be unique and non-null')
    days_reg = (pd.Timestamp(reference_date) - pd.to_datetime(data.regi_dt, errors='raise')).dt.days
    raw = data.loc[(data.followers <= 2686) & (days_reg > 7) & (data.device == 'ios')].copy()
    numeric = FEATURES + ['active_dt', 'csi_cnt', 'timespent', 'postimpressions',
                         'activity', 'notes', 'follows', 'unfollows', 'received_follows',
                         'blog_subscribes', 'users_with_custom', 'linked']
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors='raise').fillna(0)
    if not np.isfinite(raw[numeric].to_numpy()).all() or (raw[numeric] < 0).any().any():
        raise ValueError('Activity must be finite and nonnegative')
    if raw.empty or (raw.csi_cnt <= 0).any() or (~raw.active_dt.between(1, 7)).any():
        raise ValueError('Need retained users with positive sessions and 1–7 active days')
    features = raw[FEATURES].div(raw.csi_cnt, axis=0).rename(columns={'push_launched': 'notification opened'})
    profile = raw[['active_dt', 'csi_cnt', 'user_age', 'timespent', 'users_with_custom',
                   'linked', 'postimpressions', 'activity', 'notes', 'follows',
                   'unfollows', 'received_follows', 'blog_subscribes']].rename(
        columns={'active_dt': 'days_visited', 'csi_cnt': 'session', 'user_age': 'age', 'users_with_custom': 'custom'})
    profile['age'] = pd.to_numeric(profile.age, errors='raise').where(profile.age.between(5, 75))
    profile['days_reg'] = days_reg.loc[raw.index]
    profile['session'] = raw.csi_cnt / raw.active_dt
    metadata = raw[['geo', 'feature_name', 'feature_subgroup']].copy()
    metadata['geo'] = metadata.geo.where(metadata.geo.isin(GEOS), 'other')
    return features, profile, metadata


def _heatmap(table, path, title, show=False):
    fig, ax = plt.subplots(figsize=(max(8, table.shape[1]), max(4, table.shape[0] * .4)))
    image = ax.imshow(table.to_numpy(dtype=float), aspect='auto', cmap='Greens')
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel('Cluster ID')
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def run_profiling(data, output_dir, ks=(2, 3, 4, 5, 6, 7, 8, 9), seed=2016,
                  reference_date=REFERENCE_DATE, show=False):
    """Fit each requested k and export profiles with consistent cluster IDs."""
    features, profile, metadata = preprocess(data, reference_date)
    ks = sorted(set(ks))
    if not ks or any(not isinstance(k, (int, np.integer)) or k < 2 or k >= len(features) for k in ks):
        raise ValueError('Each cluster count must be >=2 and smaller than retained sample size')
    standardized = StandardScaler().fit_transform(np.log1p(features))
    pca = PCA(n_components=min(len(features), len(FEATURES)), random_state=seed)
    transformed = pca.fit_transform(standardized)
    if len(np.unique(transformed, axis=0)) < max(ks):
        raise ValueError('Not enough distinct activity profiles for requested cluster count')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'component': np.arange(1, pca.n_components_ + 1),
                  'explained_variance_ratio': pca.explained_variance_ratio_}).to_csv(output / 'pca_variance.csv', index=False)
    diagnostics = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=32, random_state=seed).fit(transformed)
        labels = pd.Series(model.labels_, index=features.index, name='cluster')
        labels.index.name = 'user_id'
        labels.to_csv(output / f'user_labels_k{k}.csv')
        means = profile.groupby(labels).mean().sort_values('days_visited')
        order = means.index
        means.insert(0, 'size(%)', labels.value_counts().reindex(order) / len(labels) * 100)
        means.insert(1, 'DAU/WAU', means.days_visited / 7)
        medians = profile.groupby(labels).median().reindex(order)
        centroids = features.groupby(labels).mean().reindex(order)
        for name, table in [('centroids_rawdata', centroids), ('profile_mean', means), ('profile_median', medians)]:
            table.to_csv(output / f'{name}_k{k}.csv')
            _heatmap(table.T, output / f'{name}_k{k}.png', f'{name.replace("_", " ")} (k={k})', show=show)
        geo = pd.crosstab(labels, metadata.geo).reindex(order, fill_value=0)
        geo.to_csv(output / f'geo_counts_k{k}.csv')
        _heatmap(geo.div(geo.sum(axis=1), axis=0).T, output / f'geo_proportions_k{k}.png', 'Geographic proportions within each cluster', show=show)
        # Standard Davies–Bouldin score; spherical Gaussian mixture BIC approximation.
        n, d = transformed.shape
        variance = max(model.inertia_ / ((n - k) * d), np.finfo(float).eps)
        counts = labels.value_counts().to_numpy()
        log_likelihood = (np.sum(counts * np.log(counts / n))
                          - n * d / 2 * np.log(2 * np.pi * variance)
                          - model.inertia_ / (2 * variance))
        bic = -2 * log_likelihood + (k * d + k) * np.log(n)
        diagnostics.append({'k': k, 'inertia': model.inertia_, 'bic': bic,
                            'davies_bouldin': davies_bouldin_score(transformed, labels)})
        experiment = pd.concat([features, profile[['days_visited', 'session', 'postimpressions', 'activity', 'notes']], metadata, labels], axis=1)
        experiment = experiment.loc[experiment.feature_name.astype(str) != '0']
        values = list(features.columns) + ['days_visited', 'session', 'postimpressions', 'activity', 'notes']
        for suffix, keys in [('', ['feature_name', 'feature_subgroup']),
                             ('_by_segments', ['cluster', 'feature_name', 'feature_subgroup'])]:
            grouped = experiment.groupby(keys)[values]
            for statistic in ['mean', 'std', 'sem', 'count']:
                getattr(grouped, statistic)().to_csv(output / f'abtest_{statistic}{suffix}_k{k}.csv')
        # Last requested k provides the Tableau export; IDs match labels and charts.
        if k == ks[-1]:
            centroids.reset_index().melt(id_vars='cluster', var_name='metric', value_name='value').rename(columns={'cluster': 'segment'}).to_csv(output / 'seg_def_matrix_tableau.csv', index=False)
    metrics = pd.DataFrame(diagnostics)
    metrics.to_csv(output / 'cluster_diagnostics.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ['inertia', 'bic', 'davies_bouldin']):
        ax.plot(metrics.k, metrics[metric], 'o-')
        ax.set(xlabel='Number of clusters', ylabel=metric, xticks=ks)
    fig.tight_layout()
    fig.savefig(output / 'cluster_diagnostics.png', dpi=140)
    if show:
        plt.show()
    plt.close(fig)
    return metrics


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show', action='store_true', help='Display charts interactively as well as saving PNGs')
    parser.add_argument('--input', type=Path, help='CSV with user_id as first column; omit to generate synthetic data')
    parser.add_argument('--output-dir', type=Path, default=base / 'cluster_example_output')
    parser.add_argument('--rows', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=2016)
    parser.add_argument('--reference-date', default=REFERENCE_DATE)
    parser.add_argument('--clusters', type=int, nargs='+', default=list(range(2, 10)))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input:
        data = pd.read_csv(args.input, index_col=0, na_values=['\\N'])
    else:
        data = generate_sample_data(args.rows, args.seed, args.reference_date)
        data.to_csv(args.output_dir / 'sampledata_5.csv')
    metrics = run_profiling(data, args.output_dir, args.clusters, args.seed, args.reference_date, show=args.show)
    print(f'Input users: {len(data)}; retained users: {len(preprocess(data, args.reference_date)[0])}')
    print(metrics.to_string(index=False))
    print(f'Outputs: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
