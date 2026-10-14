# Good choices can precede accurate delay estimates

The synthetic experiment illustrates **decision sufficiency**: choosing a good corridor can require less information than accurately estimating all three corridors. At medium observation noise, three observations per corridor produced **2.09 minutes of average regret**, despite **10.61 minutes of average Bayesian estimation error**. This supports the idea within this small model; it does not establish a general guarantee.

## Experiment

The experiment uses the repository's [three algorithms](algorithms.py), seed **2026**, and **100 independent episodes**. Each episode has three hidden mean delays drawn from Uniform(20, 100), held fixed while observations are collected. Observations add independent Normal noise. Counts of 1, 2, 3, 5, 8, 12 and 20 use nested observations from the same episode; every method receives identical data.

The main comparison uses noise SD **25 minutes**. Sensitivity checks use SD 10 and 50, reusing the same episodes. These are 2,100 episode/count/noise combinations, not 2,100 independent hidden states. All data is synthetic; corridors A, B and C are hypothetical.

**Regret** is the chosen corridor's hidden mean minus the smallest hidden mean. **Estimation error** is the root mean squared error (RMSE) across the three hidden means, calculated per episode and then averaged. Both are in minutes, but measure different things: the cost of a choice versus accuracy about the whole state.

## Main results: medium noise

| Observations per corridor | Naive / Bayesian mean regret | Scenario minimax mean regret | Bayesian mean RMSE |
| --- | ---: | ---: | ---: |
| 1 | 5.49 | 5.34 | 14.44 |
| 2 | 3.43 | 3.86 | 11.96 |
| 3 | 2.09 | 2.31 | 10.61 |
| 5 | 2.61 | 2.48 | 9.09 |
| 8 | 1.63 | 1.38 | 7.77 |
| 12 | 1.10 | 1.02 | 6.62 |
| 20 | 0.55 | 0.55 | 5.27 |

**A good choice does not require exact knowledge of every delay.** At three observations, substantial uncertainty remains about the means, yet the average cost of choosing incorrectly is comparatively small. A corridor can be close enough to the best option that choosing it incurs little regret, even when the estimates are inaccurate. There is no predefined threshold here for when a decision is “good enough.”

**More data helps overall, without a smooth improvement at every step.** Naive/Bayesian mean regret falls from 5.49 to 0.55 minutes between one and 20 observations. It rises between three and five observations in this finite batch, while estimation error falls. Additional noisy samples can change rankings unfavourably in individual episodes; the reported averages are not a monotonicity guarantee.

**Bayesian estimation improves without changing the decision.** With one observation, naive mean RMSE is 22.66 minutes versus 14.44 for Bayesian estimation, yet their regrets are identical. The common prior, equal counts and common noise apply the same positive shrinkage factor to every sample mean, preserving the corridor ranking. This equality is structural, rather than evidence from a competition between unrelated methods.

**The sampled robust rule has no consistent advantage in these results.** It is slightly better at some counts and worse at others. At one observation, its paired mean regret difference from Bayesian is −0.15 minutes, with an approximate 95% Monte Carlo interval of **[−1.38, 1.08]**. That comparison does not establish an advantage. Its worst-case objective concerns 2,000 posterior samples, not every possible hidden state; sampled extremes and the seed affect its choice. Robust uses the same Bayesian state estimate, so their RMSE values coincide.

## Sensitivity to observation noise

| Noise SD | Naive / Bayesian mean regret: 1 observation | 3 observations | 20 observations |
| --- | ---: | ---: | ---: |
| 10 min | 1.60 | 0.69 | 0.10 |
| 25 min | 5.49 | 2.09 | 0.55 |
| 50 min | 10.32 | 4.76 | 2.14 |

In these paired simulations, noisier observations require more information to achieve similar average regret. Twenty high-noise observations still leave more regret than three low-noise observations. Information quality matters alongside quantity.

## Limits and interpretation

These are descriptive results from 100 episodes per setting. For example, the medium-noise Bayesian mean regret interval is **[1.15, 3.04]** minutes at three observations and **[0.15, 0.95]** at 20. Intervals use mean ± 1.96 × sample SD / √100; nonnegative metric intervals are clipped at zero. They quantify Monte Carlo uncertainty about simulated means, not real-world coverage. Paired difference intervals use episode-level differences and are not clipped. No correction for multiple exploratory comparisons is applied.

The model has only three independent actions, known noise and equal observation counts. Its Normal prior N(60, 25²) approximates the uniform truth generator; Normal observations and posterior samples can be negative. Regret uses latent mean loss, not a separately randomised delivery outcome. There are no real route constraints or public observations. Low average regret can hide occasional costly mistakes, and neither posterior intervals nor finite-scenario minimax regret provide a universal guarantee.

The useful conclusion is a separation between **estimating well** and **deciding well**, not proof that one algorithm always wins. A focused next experiment would give corridors unequal observation counts, testing when Bayesian updating changes the ranking and whether that improves regret.

## Reproduce and inspect

Run from the repository root after installing `requirements.txt`:

```sh
python reproduce_report.py
```

The script generates `simulation-results/episodes.csv`, `summary.csv`, and `paired-comparisons.csv`. These reproducible outputs are excluded from Git; run the script to inspect the evidence behind the rounded values above. The app's default **Run 100 experiments** reproduces the medium-noise setting when its noise control is set to 25 minutes.
