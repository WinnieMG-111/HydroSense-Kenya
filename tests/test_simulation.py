"""
test_simulation.py
Automated tests for Euler method, RK4 method, and Monte Carlo simulation.
Run with: pytest tests/test_simulation.py -v
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation import (euler_simulation, rk4_simulation,
                              monte_carlo_rainfall, run_monte_carlo_soil_moisture)
from src.numerical_methods import compute_et, water_balance_step

# Shared fixtures

N = 10
RAINFALL = np.array([3.0, 0.0, 5.0, 0.0, 0.0, 10.0, 2.0, 0.0, 1.0, 0.0])
IRRIGATION = np.zeros(N)
ET = np.full(N, 2.5)
FC = 40.0
DC = 0.15
S0 = 32.0
MIN_MOISTURE = 22.0

# Euler method tests

class TestEulerSimulation:

    def test_output_length(self):
        result = euler_simulation(S0, RAINFALL, IRRIGATION, ET, FC, DC)
        assert len(result) == N + 1

    def test_initial_value_preserved(self):
        result = euler_simulation(S0, RAINFALL, IRRIGATION, ET, FC, DC)
        assert result[0] == S0

    def test_non_negative_moisture(self):
        result = euler_simulation(S0, RAINFALL, IRRIGATION, ET, FC, DC)
        assert np.all(result >= 0.0)

    def test_rainfall_increases_moisture(self):
        """A large rain event should raise moisture."""
        rain_event = np.zeros(N)
        rain_event[0] = 20.0
        no_rain = np.zeros(N)
        S_rain = euler_simulation(S0, rain_event, IRRIGATION, ET, FC, DC)
        S_no_rain = euler_simulation(S0, no_rain, IRRIGATION, ET, FC, DC)
        assert S_rain[1] > S_no_rain[1]


# RK4 method tests

class TestRK4Simulation:

    def test_output_length(self):
        result = rk4_simulation(S0, RAINFALL, IRRIGATION, ET, FC, DC)
        assert len(result) == N + 1

    def test_initial_value_preserved(self):
        result = rk4_simulation(S0, RAINFALL, IRRIGATION, ET, FC, DC)
        assert result[0] == S0

    def test_non_negative_moisture(self):
        result = rk4_simulation(S0, RAINFALL, IRRIGATION, ET, FC, DC)
        assert np.all(result >= 0.0)

    def test_rk4_vs_euler_close(self):
        """For small time steps, RK4 and Euler should produce similar results."""
        euler = euler_simulation(S0, RAINFALL, IRRIGATION, ET, FC, DC)
        rk4 = rk4_simulation(S0, RAINFALL, IRRIGATION, ET, FC, DC)
        # They should be within 5% of each other for this simple case
        assert np.max(np.abs(euler - rk4)) < 5.0


# Monte Carlo tests

class TestMonteCarlo:

    def test_scenario_shape(self):
        scenarios = monte_carlo_rainfall(RAINFALL, n_scenarios=500, seed=0)
        assert scenarios.shape == (500, N)

    def test_scenarios_non_negative(self):
        scenarios = monte_carlo_rainfall(RAINFALL, n_scenarios=100, seed=1)
        assert np.all(scenarios >= 0.0)

    def test_reproducibility(self):
        s1 = monte_carlo_rainfall(RAINFALL, n_scenarios=100, seed=42)
        s2 = monte_carlo_rainfall(RAINFALL, n_scenarios=100, seed=42)
        assert np.allclose(s1, s2)

    def test_run_monte_carlo_keys(self):
        scenarios = monte_carlo_rainfall(RAINFALL, n_scenarios=200, seed=7)
        result = run_monte_carlo_soil_moisture(
            S0, scenarios, IRRIGATION, ET, FC, DC, MIN_MOISTURE
        )
        expected_keys = {"all_moisture", "prob_shortage", "prob_over_irrigation",
                         "expected_demand", "worst_demand", "percentiles"}
        assert expected_keys.issubset(result.keys())

    def test_prob_shortage_in_range(self):
        scenarios = monte_carlo_rainfall(RAINFALL, n_scenarios=200, seed=7)
        result = run_monte_carlo_soil_moisture(
            S0, scenarios, IRRIGATION, ET, FC, DC, MIN_MOISTURE
        )
        assert 0.0 <= result["prob_shortage"] <= 1.0

# Water balance helper tests

class TestWaterBalanceStep:

    def test_drainage_above_fc(self):
        S_next, D = water_balance_step(45.0, 0, 0, 0, FC, DC)
        assert D > 0

    def test_no_drainage_below_fc(self):
        S_next, D = water_balance_step(30.0, 0, 0, 0, FC, DC)
        assert D == 0.0

    def test_et_reduces_moisture(self):
        S_next, _ = water_balance_step(30.0, 0, 0, 3.0, FC, DC)
        assert S_next < 30.0


# ET function tests

class TestComputeET:

    def test_non_negative_output(self):
        assert compute_et(10.0, 1.0, 0.3, 90.0) >= 0.0

    def test_zero_when_all_low(self):
        """Very cold, calm, dark, humid day → ET should be 0."""
        assert compute_et(0.0, 0.0, 0.0, 100.0) == 0.0

    def test_increases_with_temperature(self):
        et_low = compute_et(20.0, 2.0, 0.6, 65.0)
        et_high = compute_et(30.0, 2.0, 0.6, 65.0)
        assert et_high > et_low
