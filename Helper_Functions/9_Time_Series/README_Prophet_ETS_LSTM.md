# Prophet, ETS, and LSTM forecast comparison

This English-language example generates 1,825 daily synthetic observations and
compares standalone Prophet, standalone ETS, and a sequential hybrid.

## Run

Use Python 3.12 in a virtual environment:

```sh
python -m pip install -r Helper_Functions/9_Time_Series/requirements-forecast.txt
python Helper_Functions/9_Time_Series/Prophet_ETS_LSTM.py
# Training and printed metrics without opening plots:
python Helper_Functions/9_Time_Series/Prophet_ETS_LSTM.py --no-plots --epochs 30 --seed 42
python -m unittest discover -s Helper_Functions/9_Time_Series -p 'test_prophet_ets_lstm.py'
```

Importing the module does not train models. `run_experiment` accepts `epochs`,
`seed`, and `make_plots` and returns a dictionary of experiment results.
The current `prophet` package is required; errors are allowed to propagate.
Prophet may report that optional Plotly is missing; this example uses Matplotlib.

## Evaluation and hybrid construction

All models fit only the first 1,460 observations and forecast the following 365
days from that same cutoff. The first 182 forecast days form a diagnostic
validation set, and the remaining 183 form the test set. Validation observations
are not used for refitting, early stopping, or hyperparameter selection.

The standalone ETS model fits the original training signal. In the hybrid:

1. Prophet fits the training signal.
2. A separate ETS model fits the training signal minus Prophet's fitted values.
3. The LSTM fits the remaining residuals, standardized using training statistics.
4. Forecasts add Prophet, residual ETS, and inverse-transformed LSTM predictions.

The LSTM uses 28-day windows and recursively predicts one day at a time.
`horizon` values other than 1 are rejected. MAE, RMSE, and MAPE are printed;
six diagnostic figures are displayed unless `--no-plots` is used.

## Limits

This is a synthetic-data demonstration, not evidence that hybrid models always
outperform individual models. Noise and random holiday spikes are not inherently
predictable. The trend includes an intentional downward step at 60% of the series.
Residual targets use in-sample fitted values; a more rigorous model-selection
study should use rolling-origin out-of-sample residuals and multiple backtests.
The full-series periodogram is descriptive and does not select model parameters.
NumPy and PyTorch are seeded, but exact results can differ across library versions
and CPU/GPU backends. Warnings remain visible to aid diagnosis.

## Performance and validation improvements

`ResidDataset` retains one float32 snapshot of the input and uses views for its
windows, rather than materializing every overlapping window. Returned tensors
are independent copies, so callers cannot overwrite the dataset through a batch.
Storage scales with series length rather than series length times window size.

Recursive inference preallocates history on the model device, uses inference mode,
and transfers the complete forecast back to the CPU once. It preserves the model's
training/evaluation mode and rejects nonfinite predictions. Input lengths, step
counts and epochs must be integers; zero forecast steps are supported. Forecasts
still restart the LSTM for every sliding window, preserving the training setup.

MAE, RMSE and MAPE accept lists and arrays, and reject empty, nonfinite or mismatched
inputs instead of silently broadcasting them. MAPE retains its 1e-6 denominator
floor, so values near zero can still produce very large percentage errors.

Forecast dependencies are loaded by `run_experiment`; plotting dependencies are
loaded only when plots are requested. Dataset, model and metric helpers require
NumPy, pandas and PyTorch. Tests skip the full-pipeline check when Prophet or
statsmodels is absent; install `requirements-forecast.txt` to run that check.

A local CPU benchmark against the preceding implementation measured:

| Operation | Before | After |
| --- | ---: | ---: |
| Dataset construction, 100,000 values and 100-day windows | 0.839 s | 0.0007 s |
| Peak additional dataset allocation (tracemalloc) | 64.03 MiB | 0.48 MiB |
| Actual LSTM, 365 recursive steps, one CPU thread | 0.161 s | 0.136 s |
| Arithmetic-model inference, 10,000 steps | 0.091 s | 0.052 s |
| Batch loading, 30 epochs of 1,432 windows (batch 32) | 0.190 s | 0.014 s |

Inference times are medians of three runs; dataset construction is one run.
Training fetches whole batches by indexing the window view once per batch
(`BatchSampler` with `batch_size=None`) instead of collating 32 per-sample
tensors. Batches and forecasts are bit-identical to `shuffle=True`. End to end,
a 30-epoch CPU run takes about 8 s, almost all of it LSTM forward/backward, so
the loading change saves roughly 2%.
LSTM forecasts matched within rtol=1e-5, atol=1e-7; arithmetic forecasts matched
exactly. These measurements do not establish GPU performance or forecast accuracy
improvements. Existing tests also verify repeatability and training/test alignment
with two actual one-epoch experiment runs.

## Investigation of the historical notebooks

The eight NeuralProphet notebooks require a private CSV at
`../src/util/aws/vertica/vertica_data_20230530.csv`, which is absent from this
repository. NeuralProphet is separate from Prophet and is not included in
`requirements-forecast.txt`. Those notebooks were inspected but were not executed
or modified. Before relying on their results:

- Preserve account identifiers in a multi-series dataset or select one account.
  They group by account/date, then omit the account identifier from `ts_data`,
  potentially leaving duplicate dates in what appears to be a single series.
- Assign the result of `sort_values` and validate chronology per account.
- Validate positive denominators and values before dividing and taking logarithms.
- In `08-Uncertainty-20230605.ipynb`, split training and calibration data before
  fitting the quantile model. It currently fits all data before constructing the
  calibration split, so the calibration observations have already been seen.
- Treat plots of predictions on fitted data as diagnostics; use held-out future
  observations for forecast accuracy and interval coverage evaluation.

## Verification environment

`test_prophet_ets_lstm.py` passes 11 tests (plus 19 subtests) with Python 3.12,
pandas 2.2.2, torch 2.11, Prophet 1.5 and statsmodels 0.15. Without Prophet and
statsmodels, the full-pipeline test is skipped and the other 10 pass.
The plotting path also completed a one-epoch run using the noninteractive Agg
backend; its display warnings and Prophet's optional Plotly warning are expected.
Prophet emitted dependency deprecation warnings with the newer NumPy version.

An additional pandas 3 run passed the time-series tests. It also exposed a
read-only `to_numpy` failure in `5_Hypothesis_Tests/confidence_interval.py`,
which is fixed separately in that module's pull request.
