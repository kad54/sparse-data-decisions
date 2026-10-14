import numpy as np
import pytest

from algorithms import (
    bayesian_decision,
    naive_decision,
    normal_posterior,
    run_experiments,
    scenario_minimax_regret,
    simulate,
    summarise_experiments,
)


def test_closed_form_update_and_no_observation_fallback():
    observations = [[40.0, 60.0], [], [80.0]]
    mean, sd = normal_posterior(observations, 10.0, prior_mean=60.0, prior_sd=20.0)
    assert mean == pytest.approx([51.111111111, 60.0, 76.0])
    assert sd == pytest.approx([np.sqrt(1 / 0.0225), 20.0, np.sqrt(80.0)])
    action, means = naive_decision(observations)
    assert action == 0
    assert means == pytest.approx([50.0, 60.0, 80.0])


def test_scenario_regret_differs_from_lowest_average():
    scenarios = np.array([[0.0, 10.0, 100.0]] * 9 + [[40.0, 10.0, 100.0]])
    action, worst = scenario_minimax_regret(scenarios)
    assert worst == pytest.approx([30.0, 10.0, 100.0])
    assert action == 1
    assert scenarios.mean(axis=0).argmin() == 0
    assert scenario_minimax_regret(np.ones((5, 3)))[0] == 0


def test_reproducible_nested_observations_and_fixed_truth():
    small = simulate(42, n=2)
    large = simulate(42, n=20)
    same = simulate(42, n=2)
    np.testing.assert_array_equal(small["truth"], large["truth"])
    np.testing.assert_array_equal(small["observations"], large["observations"][:, :2])
    np.testing.assert_array_equal(small["posterior_mean"], same["posterior_mean"])
    assert small["decisions"] == same["decisions"]
    assert not np.array_equal(small["truth"], simulate(43)["truth"])
    assert all(0 <= choice < 3 for choice in small["decisions"].values())
    for method, choice in small["decisions"].items():
        assert small["regrets"][method] == small["truth"][choice] - small["truth"].min()


def test_symmetric_bayes_ranking_and_uncertainty_reduction():
    for seed in range(10):
        result = simulate(seed, n=3)
        assert result["decisions"]["Naive"] == result["decisions"]["Bayesian"]
    assert np.all(simulate(n=20)["posterior_sd"] < simulate(n=1)["posterior_sd"])
    assert np.all(simulate(noise_sd=50)["posterior_sd"] > simulate(noise_sd=10)["posterior_sd"])


def test_repeated_experiments_are_paired_complete_and_summarised():
    data = run_experiments(repeats=4)
    assert len(data) == 4 * 7 * 3
    assert (data.regret >= 0).all()
    summary = summarise_experiments(data)
    assert len(summary) == 21
    assert (summary.regret_low <= summary.regret).all()
    assert (summary.regret_high >= summary.regret).all()
    assert (summary.episodes == 4).all()


@pytest.mark.parametrize("noise", [0.0, -1.0, float("nan")])
def test_invalid_noise_rejected(noise):
    with pytest.raises(ValueError):
        bayesian_decision([[1.0], [2.0], [3.0]], noise)


def test_invalid_scenarios_and_observations_rejected():
    with pytest.raises(ValueError):
        scenario_minimax_regret([[1.0, 2.0]])
    with pytest.raises(ValueError):
        naive_decision([[np.nan], [], []])
    with pytest.raises(ValueError):
        simulate(n=0)
