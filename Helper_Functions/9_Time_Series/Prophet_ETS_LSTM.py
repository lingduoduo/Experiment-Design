"""Compare Prophet, ETS, and a sequential Prophet/ETS/LSTM residual model."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from prophet import Prophet


class ResidDataset(Dataset):
    def __init__(self, series, input_len=28, horizon=1):
        if horizon != 1:
            raise ValueError("Only horizon=1 is supported; forecasts recurse one step at a time.")
        if input_len < 1 or len(series) < input_len + horizon:
            raise ValueError("The series must contain at least input_len + horizon values.")
        self.series = np.asarray(series, dtype=np.float32)
        if self.series.ndim != 1 or not np.isfinite(self.series).all():
            raise ValueError("The series must be one-dimensional and finite.")
        self.input_len = input_len
        self.horizon = horizon
        self.X, self.y = [], []
        for i in range(0, len(series) - input_len - horizon + 1):
            self.X.append(self.series[i:i+input_len])
            self.y.append(self.series[i+input_len:i+input_len+horizon])
        self.X = np.array(self.X)
        self.y = np.array(self.y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).unsqueeze(-1), torch.from_numpy(self.y[idx])


class ResidLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, horizon=1):
        super().__init__()
        if horizon != 1:
            raise ValueError("Only horizon=1 is supported.")
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h)


def rolling_resid_predict(
    model: nn.Module, resid_init: np.ndarray, total_steps: int,
    input_len: int = 28, horizon: int = 1,
) -> np.ndarray:
    """
    resid_init: Initial context from the last window of training residuals.
    total_steps: Number of future time steps to predict.
    """
    if horizon != 1:
        raise ValueError("Only horizon=1 is supported.")
    context = np.asarray(resid_init, dtype=np.float32).copy()
    if input_len < 1 or context.ndim != 1 or len(context) < input_len:
        raise ValueError("Provide a one-dimensional context of at least input_len values.")
    if total_steps < 0 or not np.isfinite(context).all():
        raise ValueError("Steps must be nonnegative and context values finite.")
    device = next(model.parameters(), torch.empty(0)).device
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(total_steps):
            x = torch.from_numpy(context[-input_len:].astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
            yhat = model(x).cpu().numpy().flatten()[0]
            preds.append(yhat)
            context = np.append(context, yhat)
    return np.array(preds)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(1e-6, np.abs(y_true)))) * 100


def run_experiment(epochs: int = 30, seed: int = 42, make_plots: bool = True) -> dict:
    """Run a fixed-origin forecast comparison; return results for inspection.

    Validation and test forecasts both start at the training cutoff. The
    validation set is diagnostic only; it is not used for fitting or tuning.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    palette = {
        "raw": "#FF00FF",
        "trend": "#00FFFF",
        "season_w": "#00FF00",
        "season_y": "#FFA500",
        "noise": "#FF0000",
        "prophet": "#1E90FF",
        "ets": "#32CD32",
        "lstm": "#DA70D6",
        "hybrid": "#FF1493",
        "resid": "#8A2BE2",
        "error": "#FFD700",
    }
    sns.set(style="whitegrid")

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_days = 5 * 365
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    # Trend: linear growth, scaled down to 70% after 60% of the time span.
    trend = 0.01 * t
    trend[int(n_days*0.6):] = trend[int(n_days*0.6):] * 0.7  # Reduce the later trend values.

    # Weekly seasonality (7-day period).
    season_w = 2.5 * np.sin(2 * np.pi * t / 7.0) + 1.5 * np.cos(2 * np.pi * t / 7.0)

    # Annual seasonality (365-day period).
    season_y = 8.0 * np.sin(2 * np.pi * t / 365.0)

    # Random holiday spikes.
    holiday_idx = np.random.choice(n_days, size=25, replace=False)
    holiday_spikes = np.zeros(n_days)
    holiday_spikes[holiday_idx] = np.random.uniform(10, 30, size=len(holiday_idx))

    # Noise (standard deviation gradually increases over time).
    noise = np.random.normal(loc=0.0, scale=1.5 + 0.005*t, size=n_days)

    y = 50 + trend + season_w + season_y + holiday_spikes + noise

    df = pd.DataFrame({"ds": dates, "y": y,
                       "trend": trend, "season_w": season_w, "season_y": season_y,
                       "holiday_spikes": holiday_spikes, "noise": noise})

    # Split the dataset
    train_end = int(4 * 365)
    val_end = train_end + int(0.5 * 365)
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    # Prophet: model low-frequency patterns and multiple seasonalities.
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                seasonality_mode="additive")
    m.fit(df_train[["ds", "y"]])

    # Forecast the validation and test periods.
    future = pd.DataFrame({"ds": pd.concat([df_val["ds"], df_test["ds"]], ignore_index=True)})
    prophet_forecast = m.predict(future)
    prophet_yhat_val = prophet_forecast["yhat"].values[:len(df_val)]
    prophet_yhat_test = prophet_forecast["yhat"].values[len(df_val):]

    # ETS: exponential smoothing with a damped trend and weekly seasonality.
    ets_model = ExponentialSmoothing(
        df_train["y"].values, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7
    ).fit()

    # Every model forecasts from the end of training, without validation observations.
    ets_pred_all = ets_model.forecast(steps=len(df_val) + len(df_test))
    ets_yhat_val = ets_pred_all[:len(df_val)]
    ets_yhat_test = ets_pred_all[len(df_val):]

    # Compute residuals for the LSTM to learn nonlinear, high-frequency patterns.
    train_future = pd.DataFrame({"ds": df_train["ds"]})
    train_prophet_forecast = m.predict(train_future)["yhat"].values

    # Fit the hybrid ETS component to Prophet residuals, not the original signal.
    prophet_residuals = df_train["y"].values - train_prophet_forecast
    residual_ets_model = ExponentialSmoothing(
        prophet_residuals, trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=7,
    ).fit()
    residual_ets_pred_all = residual_ets_model.forecast(len(df_val) + len(df_test))
    resid_train = prophet_residuals - residual_ets_model.fittedvalues

    # Learn on standardized training residuals; invert scaling after forecasting.
    resid_mean = float(resid_train.mean())
    resid_scale = max(float(resid_train.std()), 1e-8)
    resid_scaled = (resid_train - resid_mean) / resid_scale

    # Prepare the LSTM dataset.

    input_len = 28
    horizon = 1
    dataset = ResidDataset(resid_scaled, input_len=input_len, horizon=horizon)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # LSTM model.

    device = torch.device("cuda"if torch.cuda.is_available() else"cpu")
    model = ResidLSTM(horizon=horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the LSTM.
    for ep in range(epochs):
        model.train()
        losses = []
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            yhat = model(Xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (ep+1) % 5 == 0:
            print(f"Epoch {ep+1}/{epochs}, train MSE: {np.mean(losses):.4f}")


    # Use the final training residual window as the recursive forecast context.
    resid_context = resid_scaled[-input_len:]

    # Forecast the combined validation and test periods.
    total_future_steps = len(df_val) + len(df_test)
    resid_pred_all = rolling_resid_predict(model, resid_context, total_future_steps, input_len=input_len, horizon=horizon)
    resid_pred_all = resid_pred_all * resid_scale + resid_mean
    resid_pred_val = resid_pred_all[:len(df_val)]
    resid_pred_test = resid_pred_all[len(df_val):]

    # Combine the forecasts.
    hybrid_yhat_val = prophet_yhat_val + residual_ets_pred_all[:len(df_val)] + resid_pred_val
    hybrid_yhat_test = prophet_yhat_test + residual_ets_pred_all[len(df_val):] + resid_pred_test

    if make_plots:
        fs = 1.0  # Sampling frequency: one observation per day.
        freqs, pxx = periodogram(df["y"].values, fs=fs)
        plt.figure(figsize=(12,4))
        plt.plot(freqs, pxx, color=palette["season_y"], linewidth=1.5)
        plt.axvline(x=1/7, color=palette["season_w"], linestyle="--", label="Weekly freq ~ 1/7")
        plt.axvline(x=1/365, color=palette["trend"], linestyle="--", label="Yearly freq ~ 1/365")
        plt.title("Periodogram: energy peaks at weekly and yearly frequencies")
        plt.xlim(0, 0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ACF/PACF
        plt.figure(figsize=(12,4))
        plot_acf(resid_train, lags=40, ax=plt.gca(), color=palette["resid"])
        plt.title("ACF of training residuals (post Prophet+ETS)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12,4))
        plot_pacf(resid_train, lags=40, ax=plt.gca(), method="ywm", color=palette["lstm"])
        plt.title("PACF of training residuals (post Prophet+ETS)")
        plt.tight_layout()
        plt.show()

        # Compare Prophet, ETS, and hybrid forecasts on the validation set.
        plt.figure(figsize=(14,5))
        plt.plot(df_val["ds"], df_val["y"].values, color=palette["raw"], label="Actual (val)", linewidth=1.5)
        plt.plot(df_val["ds"], prophet_yhat_val, color=palette["prophet"], label="Prophet (val)", linewidth=1.2)
        plt.plot(df_val["ds"], ets_yhat_val, color=palette["ets"], label="ETS (val)", linewidth=1.2)
        plt.plot(df_val["ds"], hybrid_yhat_val, color=palette["hybrid"], label="Hybrid (val)", linewidth=1.5)
        plt.title("Validation set: Prophet vs ETS vs Hybrid")
        plt.legend(loc="upper left", ncol=2)
        plt.tight_layout()
        plt.show()

    # Evaluation metrics

    y_true_test = df_test["y"].values
    metrics = {
        "Prophet": {"MAE": mae(y_true_test, prophet_yhat_test),
                    "RMSE": rmse(y_true_test, prophet_yhat_test),
                    "MAPE(%)": mape(y_true_test, prophet_yhat_test)},
        "ETS": {"MAE": mae(y_true_test, ets_yhat_test),
                "RMSE": rmse(y_true_test, ets_yhat_test),
                "MAPE(%)": mape(y_true_test, ets_yhat_test)},
        "Hybrid": {"MAE": mae(y_true_test, hybrid_yhat_test),
                   "RMSE": rmse(y_true_test, hybrid_yhat_test),
                   "MAPE(%)": mape(y_true_test, hybrid_yhat_test)}
    }


    print(pd.DataFrame(metrics).T.round(4).to_string())

    if make_plots:
        plt.figure(figsize=(14,5))
        plt.plot(df_test["ds"], y_true_test, color=palette["raw"], label="Actual (test)", linewidth=1.5)
        plt.plot(df_test["ds"], prophet_yhat_test, color=palette["prophet"], label="Prophet (test)", linewidth=1.2)
        plt.plot(df_test["ds"], ets_yhat_test, color=palette["ets"], label="ETS (test)", linewidth=1.2)
        plt.plot(df_test["ds"], hybrid_yhat_test, color=palette["hybrid"], label="Hybrid (test)", linewidth=1.6)
        plt.title("Test set: Prophet vs ETS vs Hybrid")
        plt.legend(loc="upper left", ncol=2)
        plt.tight_layout()
        plt.show()

        err_prophet = y_true_test - prophet_yhat_test
        err_ets = y_true_test - ets_yhat_test
        err_hybrid = y_true_test - hybrid_yhat_test

        plt.figure(figsize=(12,5))
        sns.kdeplot(err_prophet, fill=True, color=palette["prophet"], label="Prophet error", alpha=0.6)
        sns.kdeplot(err_ets, fill=True, color=palette["ets"], label="ETS error", alpha=0.6)
        sns.kdeplot(err_hybrid, fill=True, color=palette["hybrid"], label="Hybrid error", alpha=0.6)
        plt.title("Error distribution (Test set)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return locals()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run_experiment(epochs=args.epochs, seed=args.seed, make_plots=not args.no_plots)
