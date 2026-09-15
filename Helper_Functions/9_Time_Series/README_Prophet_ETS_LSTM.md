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
