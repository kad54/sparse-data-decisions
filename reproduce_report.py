"""Regenerate the report's synthetic evidence: python reproduce_report.py."""

from pathlib import Path

import numpy as np
import pandas as pd

from algorithms import run_experiments, summarise_experiments


def main():
    output = Path(__file__).resolve().parent / "simulation-results"
    output.mkdir(exist_ok=True)
    records = pd.concat(
        [
            run_experiments(seed=2026, repeats=100, noise_sd=noise).assign(noise_sd=noise)
            for noise in (10.0, 25.0, 50.0)
        ],
        ignore_index=True,
    )
    summary = pd.concat(
        [
            summarise_experiments(group).assign(noise_sd=noise)
            for noise, group in records.groupby("noise_sd")
        ],
        ignore_index=True,
    )
    comparisons = []
    for (noise, n), group in records.groupby(["noise_sd", "observations"]):
        paired = group.pivot(index="episode", columns="method", values="regret")
        delta = paired["Scenario minimax regret"] - paired["Bayesian"]
        half = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
        comparisons.append(
            {
                "noise_sd": noise,
                "observations": n,
                "robust_minus_bayes": delta.mean(),
                "difference_low": delta.mean() - half,
                "difference_high": delta.mean() + half,
            }
        )
    assert len(records) == 6300 and len(summary) == 63
    assert not records.duplicated(["noise_sd", "observations", "episode", "method"]).any()
    assert (records[["regret", "rmse"]] >= 0).all().all()
    records.to_csv(output / "episodes.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "paired-comparisons.csv", index=False)
    print(f"Saved {len(records)} method-level results to {output}")


if __name__ == "__main__":
    main()
