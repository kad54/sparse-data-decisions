"""Minimal offline research demo. Launch with: streamlit run app.py."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from algorithms import (
    CORRIDORS,
    METHODS,
    PRIOR_MEAN,
    PRIOR_SD,
    run_experiments,
    simulate,
    summarise_experiments,
)

COLORS = ("#7d8a99", "#247989", "#ba7b37")
NOISE = {"Low · 10 min": 10.0, "Medium · 25 min": 25.0, "High · 50 min": 50.0}

st.set_page_config(page_title="How little information is enough?", layout="centered")
st.markdown(
    """<style>
.block-container {max-width:1000px;padding-top:2.4rem;padding-bottom:3rem}
h1 {letter-spacing:-1.2px;font-size:2.5rem!important}
h2 {font-size:1.35rem!important}
[data-baseweb="tab-list"] {gap:1.1rem}
</style>""",
    unsafe_allow_html=True,
)
st.title("How little information is enough?")
st.write("Can we make a good decision before we fully understand the system?")
simulation, experiments, extensions, methodology = st.tabs(
    ["Simulation", "Research Experiments", "Possible Extensions", "Data & Methodology"]
)


def chart_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#d9dfe4")
    ax.grid(axis="y", color="#edf0f2")
    ax.set_axisbelow(True)


with simulation:
    st.write(
        "Imagine you must choose one of three ways to complete a task, but you have only a few observations of their delays. Which option should you choose?"
    )
    st.caption(
        "Three abstract alternatives, labelled Corridor A, B and C. All delays are simulated; no real locations or routes are represented."
    )
    a, b = st.columns(2)
    n = a.slider("Number of observations per corridor", 1, 20, 3)
    noise_label = b.selectbox("Observation noise", list(NOISE), index=1)
    noise = NOISE[noise_label]
    a, b = st.columns(2)
    run = a.button("Run simulation", type="primary", width="stretch")
    new = b.button("Generate new scenario", width="stretch")
    if "poc_seed" not in st.session_state:
        st.session_state.poc_seed = 42
    if new:
        st.session_state.poc_seed += 1
    if run or new or "poc_result" not in st.session_state:
        st.session_state.poc_result = simulate(st.session_state.poc_seed, n, noise)
    result = st.session_state.poc_result
    st.caption(
        f"Displayed run: seed {result['seed']} · {result['n']} observations per corridor · noise SD {result['noise_sd']:.0f} min. Run applies the controls; a new scenario changes the hidden delays."
    )
    if n != result["n"] or noise != result["noise_sd"]:
        st.info("Controls changed. Press Run simulation to update the displayed result.")
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    x = np.arange(3)
    ax.bar(
        x,
        result["posterior_mean"],
        color="#247989",
        width=0.5,
        alpha=0.85,
        yerr=1.96 * result["posterior_sd"],
        capsize=6,
        error_kw={"ecolor": "#274952", "linewidth": 1.4},
        label="Bayesian mean + 95% credible interval",
    )
    ax.scatter(
        x,
        result["sample_means"],
        marker="D",
        s=35,
        color="#323b43",
        zorder=4,
        label="Observed sample mean (naive estimate)",
    )
    ax.set_xticks(x, CORRIDORS)
    ax.set_ylabel("Estimated mean delay (minutes)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=9)
    chart_style(ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        "Whiskers describe uncertainty about each hidden mean under the Bayesian model, not future-trip variation or a real-world guarantee."
    )
    columns = st.columns(3)
    for column, method in zip(columns, METHODS):
        with column:
            st.markdown(f"**{method}**")
            st.write(CORRIDORS[result["decisions"][method]])
            st.caption(f"Actual simulated regret: {result['regrets'][method]:.1f} min")
    choices = set(result["decisions"].values())
    if len(choices) == 1:
        index = next(iter(choices))
        st.success(
            f"The methods agree: all choose {CORRIDORS[index]}, losing {result['regrets']['Naive']:.1f} minutes against the best option in the hidden simulated state."
        )
    else:
        st.info(
            "The methods disagree: the scenario-based rule trades average delay against the largest regret in its sampled possible states."
        )
    st.caption(
        "Regret is checked only after choosing. None of the algorithms sees the hidden truth."
    )

with experiments:
    st.subheader("Does a good decision need a good system estimate?")
    st.write(
        "Repeat 100 independent scenarios at each observation count: 1, 2, 3, 5, 8, 12 and 20. All methods receive the same observations; counts use nested samples from the same episode."
    )
    st.caption(
        f"Experiment noise uses the current control: SD {noise:.0f} min. Fixed experiment seed: 2026. Total: 700 episode/count combinations."
    )
    if st.button("Run 100 experiments", type="primary"):
        with st.spinner("Comparing the three rules on shared synthetic observations…"):
            records = run_experiments(noise_sd=noise)
            st.session_state.poc_experiments = (records, noise)
    if "poc_experiments" in st.session_state:
        records, experiment_noise = st.session_state.poc_experiments
        summary = summarise_experiments(records)
        st.caption(
            f"Displayed experiment noise: SD {experiment_noise:.0f} min. Bands and error bars are approximate 95% Monte Carlo intervals for the means across 100 episodes."
        )
        st.markdown("**1. Average decision regret vs. observations**")
        fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
        for method, color, style in zip(METHODS, COLORS, ("-", "--", ":")):
            rows = summary[summary.method == method]
            ax.plot(
                rows.observations,
                rows.regret,
                style,
                color=color,
                marker="o",
                markersize=4,
                label=method,
            )
            ax.fill_between(
                rows.observations, rows.regret_low, rows.regret_high, color=color, alpha=0.12
            )
        ax.set(xlabel="Observations per corridor", ylabel="Average regret (minutes)")
        ax.legend(frameon=False, fontsize=9)
        chart_style(ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.write(
            "Lower means better decisions. More observations can help on average, but an individual run can get worse. The naive and Bayesian curves overlap here because they always rank corridors identically under these symmetric assumptions."
        )
        st.markdown("**2. State-estimation error vs. decision regret**")
        fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
        for method, color, marker in zip(METHODS, COLORS, ("o", "s", "^")):
            rows = summary[summary.method == method]
            ax.errorbar(
                rows.rmse,
                rows.regret,
                xerr=np.array([rows.rmse - rows.rmse_low, rows.rmse_high - rows.rmse]),
                yerr=np.array([rows.regret - rows.regret_low, rows.regret_high - rows.regret]),
                color=color,
                marker=marker,
                linestyle="--",
                linewidth=1,
                capsize=2,
                markersize=5,
                label=method,
            )
        ax.set(
            xlabel="Mean error in estimating all three delays (RMSE, minutes)",
            ylabel="Average regret (minutes)",
        )
        ax.legend(frameon=False, fontsize=9)
        chart_style(ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.write(
            "Each point is one observation count; the path connects increasing counts. Left means more accurate estimates of all delays; down means better choices. A point near the bottom can still sit far from the left: choosing well and estimating everything well are different tasks. This chart describes the simulation, not a causal relationship or proof of a general theorem."
        )
        st.caption(
            "Robust uses the Bayesian estimate for RMSE, since it changes the decision rule rather than the state estimator. Intervals are marginal, not paired significance tests or simultaneous bands."
        )
        st.download_button(
            "Download experiment results",
            records.to_csv(index=False),
            "synthetic-experiments.csv",
            "text/csv",
        )
    else:
        st.info("Run the experiment to see both charts. No result is assumed in advance.")

with extensions:
    st.subheader("Future research · not implemented")
    ideas = {
        "POMDPs": "Choose a sequence of actions while updating beliefs as observations arrive. This could extend a one-off choice to repeated decisions: how much history is needed for a useful policy?",
        "PAC-Bayes": "Study statistical bounds for decision rules learned from limited data. Could a bound on decision loss require less information than accurately estimating the entire state, under explicit assumptions?",
        "Value of information": "Compare the expected benefit of another observation with its cost. When should we stop collecting data because the next sample is unlikely to change the choice?",
        "Anytime optimisation": "Return a usable answer under a strict computation budget, improving it as time allows. How much decision quality is lost when posterior sampling must stop early?",
        "Tropical algebra": "Study max-plus or min-plus calculations for propagation through structured networks. In an extended network model, which structure would permit faster updates without changing the relevant decision?",
        "Distributionally robust optimisation": "Choose decisions that tolerate errors in an estimated probability distribution. How should an ambiguity set be calibrated from sparse observations, and what performance cost buys that protection?",
    }
    for title, explanation in ideas.items():
        with st.expander(title):
            st.write(explanation)

with methodology:
    st.subheader("What is known, estimated and hidden?")
    st.write(
        "**All data is synthetic.** No public API is connected in this version. The corridor names are placeholders; this demo studies general decision rules, not a specific operational system."
    )
    st.write(
        "**Hidden state:** three independent mean delays, drawn uniformly from 20 to 100 minutes, fixed within an episode. **Observations:** independent Normal noise around each mean, with a known SD of 10, 25 or 50 minutes. **Estimates:** sample means or Bayesian posterior means computed only from those observations."
    )
    st.write(
        f"**Shared Bayesian assumptions:** independent Normal priors with mean {PRIOR_MEAN:.0f} minutes and SD {PRIOR_SD:.0f} minutes. This broad prior approximates, rather than exactly matches, the uniform truth generator. A Normal approximation permits negative observations and posterior means; these are model artefacts, not physical negative delays."
    )
    st.markdown("**The three algorithms**")
    st.write(
        "**Naive:** pick the lowest sample mean. If a corridor has no observations, use the shared prior mean of 60 minutes; the UI always supplies at least one sample per corridor."
    )
    st.write(
        "**Bayesian:** combine prior and observations with Normal–Normal conjugate updating, then pick the lowest posterior mean. With equal counts, common noise and a common prior, this shrinks every sample mean by the same factor, preserving its ranking. Better state estimates therefore need not change the decision."
    )
    st.code(
        "posterior_variance = 1 / (1 / prior_sd² + n / noise_sd²)\nposterior_mean = posterior_variance × (prior_mean / prior_sd² + sum(observations) / noise_sd²)\n95% credible interval = posterior_mean ± 1.96 × sqrt(posterior_variance)",
        language=None,
    )
    st.write(
        "**Scenario-based minimax regret:** draw 2,000 configurations of the three unknown means from that posterior. In each configuration, subtract its best corridor’s delay from every corridor’s delay, then choose the smallest maximum gap over those finite samples. This is not a universal robustness guarantee; results depend on the sample count and seed, and finite-sample extremes can change the decision."
    )
    st.code(
        "regret[scenario, corridor] = sampled_mean[scenario, corridor] − min(sampled_mean[scenario, :])\nrobust_choice = argmin(max(regret, over scenarios))",
        language=None,
    )
    st.write(
        "**Evaluation:** actual regret = hidden mean of chosen corridor − smallest hidden mean. Loss is the latent mean, not a newly randomised travel time. State-estimation error is RMSE across the three latent means. The same hidden state evaluates every rule."
    )
    st.write(
        "**Limits:** only three independent actions; static delays; known noise; equal data counts; an approximate prior; no network constraints, costs of observation or sequential decisions. No method is assumed to win. Low regret with large RMSE is evidence about this toy setup, not proof that sparse information always suffices."
    )
    st.write(
        "**Reproducibility:** run seed starts at 42; Generate new scenario increments it. Separate seeded streams generate truth, observations and posterior draws. Counts use nested observations and the same latent state. Experiments use seed 2026 and 100 independent episodes at each count; approximate mean intervals are mean ± 1.96 × sample SD / √100, clipped at zero for nonnegative metrics."
    )
    with st.expander("Inspect this simulated episode after the decisions"):
        st.caption(
            "Hidden truth below is used only for evaluation. It is never passed to a decision rule."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Corridor": CORRIDORS,
                    "Hidden mean": result["truth"],
                    "Sample mean": result["sample_means"],
                    "Posterior mean": result["posterior_mean"],
                    "Posterior SD": result["posterior_sd"],
                }
            ),
            hide_index=True,
        )
        st.dataframe(pd.DataFrame(result["observations"].T, columns=CORRIDORS), hide_index=True)
