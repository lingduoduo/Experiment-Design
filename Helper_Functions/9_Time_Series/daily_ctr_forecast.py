"""Generate synthetic clicks/impressions and forecast held-out daily CTR with Timestral."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from timestral_example import DEFAULT_MODEL, _numpy, forecast_series, load_forecaster


def generate_daily_ctr(days=758, seed=42, start_date="2024-01-01"):
    """Sample clicks from binomial trials with trend, seasonality and daily noise."""
    if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or days < 1:
        raise ValueError("days must be a positive integer")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=days, freq="D")
    day = np.arange(days)
    weekly = np.sin(2 * np.pi * day / 7)
    impressions = rng.poisson(20000 + 3000 * weekly)
    impressions = np.maximum(impressions, 1)
    log_odds = (-3.5 + 0.0003 * day + 0.22 * weekly
                + 0.10 * np.sin(2 * np.pi * day / 365.25)
                + rng.normal(0, 0.05, days))
    true_probability = 1 / (1 + np.exp(-log_odds))
    clicks = rng.binomial(impressions, true_probability)
    return pd.DataFrame({
        "date": dates, "impressions": impressions, "clicks": clicks,
        "ctr": clicks / impressions, "synthetic_probability": true_probability,
    })


def run_demo(output_dir, history_days=730, prediction_length=28, seed=42,
             model_id=DEFAULT_MODEL, device="cpu", local_files_only=False):
    """Forecast raw observed CTR; never expose holdout values to the model."""
    if history_days < 7 or prediction_length < 1:
        raise ValueError("Require history_days>=7 and prediction_length>=1")
    data = generate_daily_ctr(history_days + prediction_length, seed=seed)
    data["split"] = np.where(np.arange(len(data)) < history_days, "train", "holdout")
    train = data.iloc[:history_days]
    holdout = data.iloc[history_days:]
    model = load_forecaster(model_id, device=device, local_files_only=local_files_only)
    forecast = forecast_series(train["ctr"].to_numpy(), prediction_length, forecaster=model)
    # The general-purpose model is unconstrained; bound displayed CTRs to [0, 1].
    median = np.clip(_numpy(forecast.median)[0, 0], 0, 1)
    lower = np.clip(_numpy(forecast.quantile(.1))[0, 0], 0, 1)
    upper = np.clip(_numpy(forecast.quantile(.9))[0, 0], 0, 1)
    baseline = np.resize(train["ctr"].to_numpy()[-7:], prediction_length)
    actual = holdout["ctr"].to_numpy()
    predictions = pd.DataFrame({
        "date": holdout["date"].to_numpy(), "actual_ctr": actual,
        "predicted_ctr": median, "q10_ctr": lower, "q90_ctr": upper,
        "seasonal_naive_ctr": baseline,
    })
    if (not np.isfinite(predictions.iloc[:, 1:].to_numpy()).all()
            or not (lower <= median).all() or not (median <= upper).all()):
        raise ValueError("Forecast contains invalid or unordered quantiles")
    metrics = {
        "synthetic_data": True, "model": str(model_id), "seed": seed,
        "history_days": history_days, "holdout_days": prediction_length,
        "train_end": str(train["date"].iloc[-1].date()),
        "forecast_start": str(holdout["date"].iloc[0].date()),
        "timestral_mae_percentage_points": float(np.mean(np.abs(actual - median)) * 100),
        "timestral_rmse_percentage_points": float(np.sqrt(np.mean((actual - median)**2)) * 100),
        "seasonal_naive_mae_percentage_points": float(np.mean(np.abs(actual - baseline)) * 100),
        "q10_q90_holdout_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "note": "Single synthetic holdout; model quantiles are not a calibrated confidence guarantee. CTR predictions clipped to [0, 1].",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "sampled_daily_ctr.csv", index=False)
    predictions.to_csv(output_dir / "ctr_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    recent = train.iloc[-90:]
    ax.plot(recent["date"], recent["ctr"], color="#475569", label="Observed training CTR", linewidth=1.3)
    ax.plot(holdout["date"], actual, color="#0f172a", label="Actual held-out CTR", linewidth=1.4)
    ax.plot(holdout["date"], median, color="#2563eb", label="Timestral median", linewidth=1.8)
    ax.fill_between(holdout["date"], lower, upper, color="#60a5fa", alpha=.25, label="Timestral q10–q90")
    ax.plot(holdout["date"], baseline, color="#d97706", linestyle="--", label="Weekly seasonal baseline", alpha=.75)
    ax.axvline(holdout["date"].iloc[0], color="#64748b", linestyle=":")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set(title=f"Synthetic daily CTR · {str(model_id).split('/')[-1]} · {prediction_length}-day holdout",
           xlabel="Date", ylabel="CTR")
    ax.grid(alpha=.2)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.savefig(output_dir / "ctr_forecast.png", dpi=160)
    plt.close(fig)
    return predictions, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("ctr_example_output"))
    parser.add_argument("--history-days", type=int, default=730)
    parser.add_argument("--prediction-length", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    predictions, metrics = run_demo(args.output_dir, args.history_days, args.prediction_length,
                                    args.seed, args.model, args.device, args.local_files_only)
    print(predictions.head(7).to_string(index=False))
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
