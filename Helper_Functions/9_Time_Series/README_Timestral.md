# Timestral example

Install the optional backend separately (Python 3.10+, PyTorch 2.4+):

```sh
python -m pip install -r Helper_Functions/9_Time_Series/requirements-timestral.txt
```

Run the seeded demo and export forecasts:

```sh
python Helper_Functions/9_Time_Series/timestral.py --output forecast.npz
# Smaller checkpoint for a quick CPU run:
python Helper_Functions/9_Time_Series/timestral.py --model mistralai/Timestral-5M --context-length 64 --prediction-length 8 --output small_forecast.npz
```

The default remains `mistralai/Timestral-209M`, with two univariate random walks,
2,000 observations each, and a 96-step forecast. Weights are downloaded on the
first forecast and cached by Hugging Face. Imports and `--help` do not load weights.
Use `--local-files-only` for cached/local checkpoints and `--device cuda` for CUDA.

For your own data, save a numeric `.npy` array and pass `--input context.npy`.
Accepted shapes are `(time,)`, `(series, time)` and `(series, variates, time)`.
The final axis is chronological history. NaNs denote missing observations;
infinities, empty arrays and entirely missing variates are rejected. Data is
converted to float32. Supply training history only when evaluating held-out data.

The `.npz` output contains `median`, `quantiles`, and `quantile_levels`, readable
with `np.load('forecast.npz', allow_pickle=False)`. Median shape is
`(series, variates, horizon)`; quantiles add a final quantile-level axis. Output
files are overwritten if they already exist.

For programmatic reuse without colliding with the installed package's name:

```python
from importlib import import_module

example = import_module('Helper_Functions.9_Time_Series.timestral_example')
forecaster = example.load_forecaster('mistralai/Timestral-5M')
forecast = example.forecast_series([1, 2, 3, 4], prediction_length=8,
                                   forecaster=forecaster)
median = forecast.median
```

`forecast_series` returns the upstream Forecast object. Passing a loaded
forecaster avoids repeated model loading. `timestral.py` is the CLI launcher; the reusable helpers live in
`timestral_example.py` to keep their module name distinct from the backend. The
loader resolves the installed package explicitly so direct execution works.

## Loading options

`from_pretrained` accepts a Hub repository ID or a local directory containing
`config.json` and `model.safetensors`. `load_forecaster` and `forecast_series`
expose the same loading options:

- `revision` pins a branch, tag or commit.
- `cache_dir` and `token` follow `huggingface_hub` behavior. Leave `token=None`
  to use the Hub's authentication settings, or pass `False` to disable credentials.
- `local_files_only=True` or `HF_HUB_OFFLINE=1` uses local/cached files without
  network access. Offline mode requires the requested checkpoint to exist locally.

```python
forecaster = example.load_forecaster(
    'mistralai/Timestral-209M', revision='main',
    cache_dir='./model-cache', token=None, local_files_only=True,
)
# A checkpoint directory can be passed instead of a Hub ID:
forecaster = example.load_forecaster('./checkpoints/timestral', local_files_only=True)
```

CLI equivalents are `--revision`, `--cache-dir`, `--token` and
`--local-files-only`. For environment-based authentication, set `HF_TOKEN` instead
of including a token in command arguments. Existing forecasters passed to
`forecast_series` are reused; loading options apply only when a model is loaded.

```sh
python Helper_Functions/9_Time_Series/timestral.py --model mistralai/Timestral-5M --revision main --cache-dir ./model-cache --local-files-only
HF_HUB_OFFLINE=1 python Helper_Functions/9_Time_Series/timestral.py --model ./checkpoints/timestral
```

## Covariates

Pass additional conditioning data through `forecast_series` or directly to a
loaded forecaster:

```python
forecast = example.forecast_series(
    context,                         # (B, V, T)
    prediction_length=96,
    forecaster=forecaster,
    past_covariates=past,             # (B, C_past, T)
    known_covariates=known,           # (B, C_known, T + 96)
)
# The upstream call supports the same covariate arguments:
forecast = forecaster.forecast(context, prediction_length=96,
                               past_covariates=past, known_covariates=known)
```

`B` is the number of series, `V` the target variates and `T` the history length.
Covariates are conditioning rows: they are not forecast, and their rows never
read the targets. Known-future rows whose observed values are all 0 or 1 are
handled as indicators by the backend. NaN denotes a missing covariate value.
Past-only data must stop at the context cutoff; known-future data includes the
whole requested horizon and must be available at forecast time.

The helper requires explicit 3D covariate arrays, validates matching batch/time
sizes and rejects infinities before loading weights. For 1D target context,
use `B=1`. Covariates remain optional and output shapes contain only targets.

For the CLI, save each covariate array as a numeric `.npy` file:

```sh
python Helper_Functions/9_Time_Series/timestral.py --input context.npy --prediction-length 96 --past-covariates past.npy --known-covariates known.npy --output forecast.npz
```

## Categorical covariates

Numeric category codes can condition forecasts alongside numeric covariates:

```python
forecast = example.forecast_series(
    context, prediction_length=96, forecaster=forecaster,
    categorical_past_covariates=codes_past,     # (B, C, T)
    categorical_known_covariates=codes_known,   # (B, C, T + 96)
)
# Or call the loaded backend directly:
forecast = forecaster.forecast(
    context, prediction_length=96,
    categorical_past_covariates=codes_past,
    categorical_known_covariates=codes_known,
)
```

Use a consistent numeric code for each category across history and future;
NaN marks missing values. Raw string labels must be encoded before calling the
helper. Both arguments are optional and accept explicit 3D arrays. Batch/time
sizes and infinities are validated before model loading. Categorical codes are
passed as float64 so adjacent integer codes above 2**24 remain distinct.

The backend handles category encoding: for univariate targets it derives category
statistics from the available context; multivariate targets use raw codes.
Categorical conditioning rows do not add output variates. Known-future categories
must be available at forecast time.

The CLI accepts numeric `.npy` files:

```sh
python Helper_Functions/9_Time_Series/timestral.py --input context.npy --prediction-length 96 --categorical-past-covariates codes_past.npy --categorical-known-covariates codes_known.npy --output forecast.npz
```

These flags can be combined with `--past-covariates` and `--known-covariates`.

## Verification

```sh
python -m pytest Helper_Functions/9_Time_Series/test_timestral.py Helper_Functions/9_Time_Series/test_timestral_backend.py -q
```

Offline tests exercise validation, demo RNG isolation, output archives and CLI
help without downloading weights. When the backend is installed, a tiny randomly
initialized model additionally checks real inference without network access.
Additional backend tests use locally generated checkpoints to verify context
capping, multi-block forecasts, finite ordered quantiles, sign symmetry with
covariates, categorical target-encoding equivalence, and exact local/Hub loading
round trips. Hub downloads are replaced with a local snapshot in those tests;
no model download or credentials are required. Backend tests skip when the optional
Timestral package is absent.

A manual pretrained 5M run also verified direct execution, eight-step forecasts,
finite ordered quantiles, and NPZ export. The 209M checkpoint and CUDA were not
run during verification.

The backend is pinned to the inspected upstream commit. See the
[official implementation and model documentation](https://github.com/mistralai/timestral)
for model details and [package metadata](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/pyproject.toml)
for dependencies.
