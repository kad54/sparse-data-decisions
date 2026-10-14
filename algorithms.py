"""Three decisions from sparse observations; all delays are synthetic minutes.

The truth is fixed within an episode. Decisions never receive it. Seeds give
paired, nested observations as the sample count changes, not fresh easier cases.
"""

import numpy as np
import pandas as pd

CORRIDORS = ("Corridor A", "Corridor B", "Corridor C")
METHODS = ("Naive", "Bayesian", "Scenario minimax regret")
PRIOR_MEAN = 60.0
PRIOR_SD = 25.0
POSTERIOR_DRAWS = 2000
COUNTS = (1, 2, 3, 5, 8, 12, 20)


def _observations(observations):
    if len(observations) != 3:
        raise ValueError("Provide observations for exactly three corridors")
    rows = [np.asarray(row, dtype=float) for row in observations]
    if any(row.ndim != 1 or not np.isfinite(row).all() for row in rows):
        raise ValueError("Observations must be finite one-dimensional arrays")
    return rows


def naive_decision(observations, fallback=PRIOR_MEAN):
    """Sample means; an unobserved corridor gets the shared prior mean."""
    if not np.isfinite(fallback):
        raise ValueError("Fallback must be finite")
    means = np.array([row.mean() if len(row) else fallback for row in _observations(observations)])
    return int(np.argmin(means)), means


def normal_posterior(observations, noise_sd, prior_mean=PRIOR_MEAN, prior_sd=PRIOR_SD):
    """Independent Normal means with known Normal observation noise.

    posterior precision = prior precision + n / noise variance
    posterior mean = posterior variance * (prior mean/prior variance + sum(y)/noise variance)
    """
    if not np.isfinite([noise_sd, prior_mean, prior_sd]).all() or min(noise_sd, prior_sd) <= 0:
        raise ValueError("Noise and prior SD must be positive and finite")
    rows = _observations(observations)
    variance = np.array([1 / (1 / prior_sd**2 + len(row) / noise_sd**2) for row in rows])
    means = variance * np.array(
        [prior_mean / prior_sd**2 + row.sum() / noise_sd**2 for row in rows]
    )
    return means, np.sqrt(variance)


def bayesian_decision(observations, noise_sd):
    means, sd = normal_posterior(observations, noise_sd)
    return int(np.argmin(means)), means, sd


def scenario_minimax_regret(scenarios):
    """Minimise maximum regret over these finite posterior mean configurations.

    This protects only against the sampled scenarios, not every possible state.
    Ties in every rule use corridor order A, B, C.
    """
    values = np.asarray(scenarios, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != 3
        or not np.isfinite(values).all()
    ):
        raise ValueError("Scenarios must have shape (positive sample count, 3) and be finite")
    regret = values - values.min(axis=1, keepdims=True)
    scores = regret.max(axis=0)
    return int(np.argmin(scores)), scores


def simulate(seed=42, n=3, noise_sd=25.0, draws=POSTERIOR_DRAWS):
    """One synthetic episode; latent means Uniform(20,100), inference prior N(60,25²).

    The broad Normal prior approximates, rather than exactly matches, the truth
    generator. Normal observations and posterior samples can be negative.
    """
    if not isinstance(n, (int, np.integer)) or not 1 <= n <= 20:
        raise ValueError("n must be an integer from 1 to 20")
    if not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        raise ValueError("Noise SD must be positive and finite")
    truth_seed, observation_seed, posterior_seed = np.random.SeedSequence(seed).spawn(3)
    truth = np.random.default_rng(truth_seed).uniform(20, 100, 3)
    standard_noise = np.random.default_rng(observation_seed).normal(size=(3, 20))
    observations = truth[:, None] + noise_sd * standard_noise[:, :n]
    naive, sample_means = naive_decision(observations)
    bayes, posterior_mean, posterior_sd = bayesian_decision(observations, noise_sd)
    samples = np.random.default_rng(posterior_seed).normal(posterior_mean, posterior_sd, (draws, 3))
    robust, robust_scores = scenario_minimax_regret(samples)
    decisions = dict(zip(METHODS, (naive, bayes, robust)))
    estimates = {
        "Naive": sample_means,
        "Bayesian": posterior_mean,
        "Scenario minimax regret": posterior_mean,
    }
    return {
        "seed": seed,
        "n": n,
        "noise_sd": noise_sd,
        "draws": draws,
        "truth": truth,
        "observations": observations,
        "sample_means": sample_means,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "robust_scores": robust_scores,
        "decisions": decisions,
        "regrets": {
            method: float(truth[index] - truth.min()) for method, index in decisions.items()
        },
        "rmse": {
            method: float(np.sqrt(np.mean((estimate - truth) ** 2)))
            for method, estimate in estimates.items()
        },
    }


def run_experiments(seed=2026, repeats=100, noise_sd=25.0):
    """100 independent episodes at each count, paired across methods and counts."""
    if not isinstance(repeats, int) or repeats < 2:
        raise ValueError("At least two independent episodes are required")
    episode_seeds = np.random.SeedSequence(seed).generate_state(repeats)
    records = []
    for episode, episode_seed in enumerate(episode_seeds):
        for n in COUNTS:
            result = simulate(int(episode_seed), n, noise_sd)
            records.extend(
                {
                    "episode": episode,
                    "observations": n,
                    "method": method,
                    "regret": result["regrets"][method],
                    "rmse": result["rmse"][method],
                }
                for method in METHODS
            )
    return pd.DataFrame(records)


def summarise_experiments(records):
    """Means and approximate 95% Monte Carlo intervals across independent episodes.

    Intervals describe simulation mean uncertainty, not predictive guarantees or
    simultaneous coverage across counts. Comparison is paired; intervals are marginal.
    """
    rows = []
    for (method, n), group in records.groupby(["method", "observations"], sort=False):
        row = {"method": method, "observations": n, "episodes": len(group)}
        for metric in ("regret", "rmse"):
            mean = float(group[metric].mean())
            half = float(1.96 * group[metric].std(ddof=1) / np.sqrt(len(group)))
            row.update(
                {
                    metric: mean,
                    f"{metric}_low": max(0.0, mean - half),
                    f"{metric}_high": mean + half,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)
