"""Import-safe Timestral forecasting example; run with --help for CLI usage.

The optional upstream package is loaded only when a model is requested.
"""
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path
import sys

import numpy as np

DEFAULT_MODEL = "mistralai/Timestral-209M"


def _backend():
    """Find the installed package without importing this same-named script."""
    loaded = sys.modules.get("timestral")
    if loaded is not None and hasattr(loaded, "TimestralForecaster"):
        return loaded
    directory = Path(__file__).resolve().parent
    search_paths = [entry for entry in sys.path if Path(entry or '.').resolve() != directory]
    spec = PathFinder.find_spec("timestral", search_paths)
    if spec is None or spec.loader is None or spec.submodule_search_locations is None:
        raise ImportError(
            "Install Timestral first: python -m pip install "
            "-r Helper_Functions/9_Time_Series/requirements-timestral.txt"
        )
    package = module_from_spec(spec)
    # Upstream uses absolute timestral.* imports, so it needs its canonical name.
    sys.modules["timestral"] = package
    try:
        spec.loader.exec_module(package)
    except BaseException:
        if loaded is None:
            sys.modules.pop("timestral", None)
        else:
            sys.modules["timestral"] = loaded
        raise
    return package


def load_forecaster(model_id=DEFAULT_MODEL, device="cpu", *, revision=None,
                    cache_dir=None, token=None, local_files_only=False):
    """Load a Hub ID or directory containing config.json and model.safetensors.

    revision pins a branch, tag or commit. cache_dir and token follow the
    huggingface_hub API (token may be a string, True, False, or None).
    local_files_only=True or HF_HUB_OFFLINE=1 prevents Hub network access.
    """
    return _backend().TimestralForecaster.from_pretrained(
        str(model_id), device=device, revision=revision, cache_dir=cache_dir,
        token=token, local_files_only=local_files_only,
    )


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def forecast_series(context, prediction_length=96, *, forecaster=None,
                    model_id=DEFAULT_MODEL, device="cpu", revision=None, cache_dir=None,
                    token=None, local_files_only=False, past_covariates=None,
                    known_covariates=None, categorical_past_covariates=None,
                    categorical_known_covariates=None):
    """Forecast (time,), (series,time), or (series,variates,time) numeric data.

    NaNs mark missing values. Each variate must have at least one observation.
    Returns the upstream Forecast, with tensor quantiles, median and quantile().
    Supply an already loaded forecaster to avoid loading weights on every call.
    Loading options apply only when forecaster is not supplied.
    past_covariates must be (B, C_past, T); known_covariates must be
    (B, C_known, T + prediction_length). NaNs mark missing covariate values.
    Categorical covariates use the same batch/time shapes, with numeric category
    codes and NaNs for missing values. Codes are kept in float64 for the backend.
    Covariates condition target forecasts and are not themselves forecast.
    """
    _positive_integer(prediction_length, "prediction_length")
    values = np.asarray(context, dtype=np.float32)
    if (values.ndim not in (1, 2, 3) or values.size == 0
            or np.isinf(values).any() or not np.isfinite(values).any(axis=-1).all()):
        raise ValueError("context must be a nonempty 1D, 2D or 3D array with observations in each variate and no infinities")
    batch = 1 if values.ndim == 1 else values.shape[0]
    length = values.shape[-1]
    covariates = {}
    for name, data, expected_length, dtype in (
        ("past_covariates", past_covariates, length, np.float32),
        ("known_covariates", known_covariates, length + prediction_length, np.float32),
        ("categorical_past_covariates", categorical_past_covariates, length, np.float64),
        ("categorical_known_covariates", categorical_known_covariates, length + prediction_length, np.float64),
    ):
        if data is None:
            continue
        array = np.asarray(data, dtype=dtype)
        if (array.ndim != 3 or array.shape[0] != batch
                or array.shape[-1] != expected_length or np.isinf(array).any()):
            raise ValueError(f"{name} must have shape ({batch}, C, {expected_length}) with no infinities")
        covariates[name] = array
    if forecaster is None:
        forecaster = load_forecaster(model_id, device, revision=revision, cache_dir=cache_dir,
                                     token=token, local_files_only=local_files_only)
    return forecaster.forecast(values, prediction_length=int(prediction_length), **covariates)


def make_demo_context(seed=42, n_series=2, n_variates=1, context_length=2000):
    """Generate seeded random walks without changing NumPy's global RNG state."""
    for name, value in (("n_series", n_series), ("n_variates", n_variates), ("context_length", context_length)):
        _positive_integer(value, name)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_series, n_variates, context_length)).astype(np.float32).cumsum(axis=-1)


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_forecast(forecast, output):
    """Save median, quantiles and quantile levels as a pickle-free .npz archive."""
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("output must have a .npz extension")
    np.savez_compressed(output, median=_numpy(forecast.median),
                        quantiles=_numpy(forecast.quantiles),
                        quantile_levels=_numpy(forecast.quantile_levels))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub ID or local checkpoint directory")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--input", type=Path, help="Numeric .npy context; otherwise use seeded demo data")
    parser.add_argument("--past-covariates", type=Path, help=".npy conditioning rows with shape (B, C_past, T)")
    parser.add_argument("--known-covariates", type=Path, help=".npy conditioning rows with shape (B, C_known, T + horizon)")
    parser.add_argument("--categorical-past-covariates", type=Path, help=".npy category codes with shape (B, C, T); NaN = missing")
    parser.add_argument("--categorical-known-covariates", type=Path, help=".npy category codes with shape (B, C, T + horizon); NaN = missing")
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--context-length", type=int, default=2000, help="Demo history length")
    parser.add_argument("--seed", type=int, default=42, help="Demo data seed")
    parser.add_argument("--revision", help="Hub branch, tag or commit")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face model cache directory")
    parser.add_argument("--token", help="Hugging Face access token; defaults to Hub authentication settings")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached/local model files only")
    parser.add_argument("--output", type=Path, help="Save forecast tensors to .npz")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.suffix != ".npz":
        parser.error("--output must have a .npz extension")
    try:
        context = (np.load(args.input, allow_pickle=False) if args.input is not None
                   else make_demo_context(seed=args.seed, context_length=args.context_length))
        covariates = {
            name: np.load(path, allow_pickle=False)
            for name in ("past_covariates", "known_covariates", "categorical_past_covariates", "categorical_known_covariates")
            if (path := getattr(args, name)) is not None
        }
        forecast = forecast_series(context, prediction_length=args.prediction_length,
                                   model_id=args.model, device=args.device, revision=args.revision,
                                   cache_dir=args.cache_dir, token=args.token,
                                   local_files_only=args.local_files_only,
                                   **covariates)
        if args.output is not None:
            save_forecast(forecast, args.output)
    except (ValueError, ImportError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Quantiles: {tuple(forecast.quantiles.shape)}")
    print(f"Quantile levels: {forecast.quantile_levels}")
    print(f"Median: {tuple(forecast.median.shape)}")
    if args.output is not None:
        print(f"Saved forecast to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
