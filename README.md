# How little information is enough?

A minimal offline Python research POC about **decision sufficiency**: can we identify a good decision before accurately understanding an entire system?

## What it demonstrates

Choose among three hypothetical civilian supply-delivery corridors inspired by the border logistics. A few noisy observations inform a one-off decision. Hidden simulated means let us measure the cost of choosing incorrectly. These are synthetic alternatives, not verified routes or operational recommendations.

Four tabs: Simulation, Research Experiments, Possible Extensions, and Data & Methodology. The main view has one chart, three decisions, simulated regret and a plain-language explanation. Experiments run 100 independent episodes at each of seven observation counts (700 combinations), with shared observations across methods and nested samples across counts.

## Mathematical methods

See [Simulation results and analysis](SIMULATION_REPORT.md) for a short reproducible report comparing observation counts, noise levels and the three methods.

All three algorithms are in `algorithms.py`:

- **Naive:** pick the lowest sample mean. Missing observations fall back to the shared prior mean, 60 minutes.
- **Bayesian:** independent Normal–Normal updates with known observation noise and prior N(60, 25²). Pick the lowest posterior mean; show model-based 95% credible intervals.
- **Scenario-based minimax regret:** sample 2,000 configurations of mean delays from the same posterior. Pick the action with the smallest maximum regret over those configurations. This is a finite-scenario rule, not a universal guarantee.

Latent means are independent Uniform(20, 100) draws, fixed within an episode. Observations have independent Normal noise with known SD 10, 25 or 50 minutes. The inference prior approximates rather than matches the truth generator. Actual regret = chosen latent mean − smallest latent mean. RMSE measures errors across all three means. Evaluation uses latent mean loss, not a separately randomised realised travel time.

With equal counts, shared prior and common noise, naive and Bayesian rankings are identical even when estimation errors differ. The demo explains this instead of implying an advantage. The robust rule can differ because of its finite posterior scenarios; it is not assumed to win.

## Install and run

Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
python -m pytest -q
```

No API key, editable package install, external data or internet connection is needed after dependencies are installed. Run simulation keeps the current hidden state. Generate new scenario increments the seed, initially 42. Experiments use seed 2026. Separate random streams preserve truth and nested observations when sample count changes. Experiment results can be downloaded as CSV.

## Interpretation and limitations

The research charts show mean regret vs. count and mean estimation RMSE vs. mean regret. Error bars are approximate marginal 95% Monte Carlo intervals: mean ± 1.96 × sample SD / √100, clipped at zero. They describe simulation means, not predictive coverage, paired significance tests or real-world guarantees.

This static three-action model assumes independent delays, known noise, equal data counts and a broad approximate prior. Normal observations and posterior draws can be negative: mathematical approximations, not physical negative delays. There is no routing, sequential learning, distributional guarantee or public-data connector. Results depend on the simulator, seed and posterior sample count. Low regret alongside large estimation error illustrates a possible separation; it does not establish a general theorem.

## Future research

The educational extensions tab discusses POMDPs, PAC-Bayes, value of information, anytime optimisation, tropical algebra and distributionally robust optimisation. None is implemented. Each asks what additional assumptions, data or computation would extend the decision-sufficiency question.
