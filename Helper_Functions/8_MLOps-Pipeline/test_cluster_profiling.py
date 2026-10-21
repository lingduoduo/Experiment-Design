import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('profiling', Path(__file__).with_name('cluster_profiling.py'))
profiling = importlib.util.module_from_spec(spec)
if spec.loader and Path(spec.origin).exists():
    spec.loader.exec_module(profiling)


def test_reproducible_sample_and_filters():
    assert hasattr(profiling, 'generate_sample_data'), 'sample generator is missing'
    data = profiling.generate_sample_data(180, seed=7)
    pd.testing.assert_frame_equal(data, profiling.generate_sample_data(180, seed=7))
    features, profile, metadata = profiling.preprocess(data)
    expected = data.loc[(data.device == 'ios') & (data.followers <= 2686)
                        & ((pd.Timestamp('2018-05-12') - pd.to_datetime(data.regi_dt)).dt.days > 7)]
    assert features.index.equals(expected.index)
    assert np.isfinite(features.to_numpy()).all()
    assert np.allclose(features['likes'], expected.likes.fillna(0) / expected.csi_cnt)
    assert np.allclose(profile.session, expected.csi_cnt / expected.active_dt)
    assert metadata.index.equals(features.index)


def test_pipeline_exports_match_membership(tmp_path):
    assert hasattr(profiling, 'run_profiling'), 'local pipeline is missing'
    data = profiling.generate_sample_data(180, seed=7)
    metrics = profiling.run_profiling(data, tmp_path, ks=[2, 3], seed=7)
    labels = pd.read_csv(tmp_path / 'user_labels_k3.csv')
    means = pd.read_csv(tmp_path / 'profile_mean_k3.csv')
    assert labels.user_id.is_unique
    assert labels.cluster.nunique() == 3
    assert means['size(%)'].sum() == pytest.approx(100)
    assert np.isfinite(metrics[['inertia', 'bic', 'davies_bouldin']]).all().all()
    assert len(pd.read_csv(tmp_path / 'seg_def_matrix_tableau.csv')) == 21
    assert (tmp_path / 'cluster_diagnostics.png').stat().st_size > 1000
    for k in [2, 3]:
        exported = pd.read_csv(tmp_path / f'centroids_rawdata_k{k}.csv', index_col='cluster')
        features, _, _ = profiling.preprocess(data)
        assignments = pd.read_csv(tmp_path / f'user_labels_k{k}.csv', index_col='user_id')
        actual = features.groupby(assignments.cluster).mean().loc[exported.index]
        np.testing.assert_allclose(exported, actual)


def test_invalid_sample_and_cluster_counts(tmp_path):
    assert hasattr(profiling, 'generate_sample_data'), 'sample generator is missing'
    with pytest.raises(ValueError):
        profiling.generate_sample_data(0)
    with pytest.raises(ValueError):
        profiling.run_profiling(profiling.generate_sample_data(30), tmp_path, ks=[100])


def test_show_heatmaps_displays_figures_before_closing(tmp_path, monkeypatch):
    displayed = []

    def capture_show():
        figure = profiling.plt.gcf()
        assert figure.axes
        displayed.append(figure.number)

    monkeypatch.setattr(profiling.plt, 'show', capture_show)
    profiling.run_profiling(profiling.generate_sample_data(60), tmp_path, ks=[3], show=True)
    assert len(displayed) == 5  # Four heatmaps and cluster diagnostics.
    assert not profiling.plt.get_fignums()


def test_entry_script_has_valid_python_syntax():
    entry = Path(__file__).with_name('5-Cluster_Profiling.py')
    compile(entry.read_text(), str(entry), 'exec')
