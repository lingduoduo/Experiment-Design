"""Bayesian marketing-mix example with differentiable adstock and Pyro NUTS.

Requires torch and pyro-ppl. Run this file to fit the synthetic data; importing
it only defines helpers. For a quick smoke run, use --samples 10 --warmup 10.
The default single chain is for demonstration: use multiple chains and inspect
ESS, R-hat, and divergences before interpreting estimates. Media coefficients,
decay, and saturation can be weakly identified even with synthetic data.
"""

import argparse

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS


def geometric_adstock(x, alpha):
    """Apply s[t] = x[t] + alpha*s[t-1], with zero initial carryover.

    Accepts [time] or [time, channels]; alpha broadcasts across channels.
    Vectorize channels while keeping the exact time recurrence. Avoid in-place
    writes so gradients through every carryover step remain valid for NUTS.
    """
    state = torch.zeros_like(x[0])
    states = []
    for spend in x.unbind(0):
        state = spend + alpha * state
        states.append(state)
    return torch.stack(states)


def saturation(x, lam):
    """Exponential saturation, evaluated accurately near zero."""
    return -torch.expm1(-lam * x)


def media_contribution(X, beta, alpha, lam):
    """Total media effect; beta is the maximum sales lift per channel."""
    return (saturation(geometric_adstock(X, alpha), lam) * beta).sum(-1)


def generate_synthetic_data(seed=42):
    """Create 120 observations for three channels without changing global RNG."""
    rng = torch.Generator().manual_seed(seed)
    X = torch.rand(120, 3, generator=rng) * 100
    promo = torch.randn(120, generator=rng)
    truth = {
        "intercept": torch.tensor(20.0),
        # Sales units, equivalent to the original [0.08, 0.15, 0.05] * 100.
        "beta": torch.tensor([8.0, 15.0, 5.0]),
        "alpha": torch.tensor([0.3, 0.6, 0.2]),
        "lam": torch.tensor([0.02, 0.01, 0.03]),
        "promo_beta": torch.tensor(2.0),
        "sigma": torch.tensor(2.0),
    }
    mu = (truth["intercept"]
          + media_contribution(X, truth["beta"], truth["alpha"], truth["lam"])
          + truth["promo_beta"] * promo)
    y = mu + truth["sigma"] * torch.randn(120, generator=rng)
    return X, promo, y, truth


def marketing_mix_model(X, promo, y=None):
    """Positive media lifts with channel-specific carryover and saturation."""
    T, n_channels = X.shape
    ones = X.new_ones(n_channels)
    intercept = pyro.sample("intercept", dist.Normal(X.new_tensor(0.), 50.))
    # Use the same sales units as the synthetic truth.
    beta = pyro.sample("beta", dist.HalfNormal(20. * ones).to_event(1))
    alpha = pyro.sample("alpha", dist.Beta(2. * ones, 2. * ones).to_event(1))
    lam = pyro.sample("lam", dist.LogNormal(-3. * ones, ones).to_event(1))
    promo_beta = pyro.sample("promo_beta", dist.Normal(X.new_tensor(0.), 5.))
    sigma = pyro.sample("sigma", dist.HalfNormal(X.new_tensor(5.)))

    mu = intercept + media_contribution(X, beta, alpha, lam) + promo_beta * promo
    with pyro.plate("time", T):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def run_example(num_samples=1000, warmup_steps=500, num_chains=1, seed=42):
    """Fit reproducible data and print diagnostics plus parameter comparisons."""
    if num_samples < 4 or warmup_steps < 1 or num_chains < 1:
        raise ValueError("need samples >= 4, warmup >= 1, and chains >= 1")
    pyro.set_rng_seed(seed)
    X, promo, y, truth = generate_synthetic_data(seed)
    kernel = NUTS(marketing_mix_model, target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps,
                num_chains=num_chains)
    mcmc.run(X, promo, y)
    mcmc.summary()
    print("\nSynthetic truth and posterior means (not a convergence check):")
    for name, draws in mcmc.get_samples().items():
        print(f"{name:12s} truth={truth[name].tolist()} "
              f"mean={draws.mean(0).tolist()}")
    return mcmc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_example(args.samples, args.warmup, args.chains, args.seed)
