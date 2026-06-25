"""
HydroSense-Kenya — Simulation Module

Implements:
  - Euler method for soil moisture ODE
  - Runge-Kutta (RK4) method for soil moisture ODE
  - Monte Carlo rainfall uncertainty simulation
  - Probability estimates: water shortage, over-irrigation, demand statistics
"""

import numpy as np
from typing import Tuple, Optional

# 1. ODE helpers — soil moisture dynamics

def soil_moisture_ode(S: float, R: float, I: float, ET: float,
                       field_capacity: float, drainage_coeff: float) -> float:
    """
    dS/dt for continuous soil-moisture model.

    dS/dt = R + I - ET - D(S)
    where D(S) = drainage_coeff * max(0, S - field_capacity)
    """
    D = drainage_coeff * max(0.0, S - field_capacity)
    return R + I - ET - D



# 2. Euler Method

def euler_simulation(S0: float,
                     rainfall: np.ndarray,
                     irrigation: np.ndarray,
                     et: np.ndarray,
                     field_capacity: float,
                     drainage_coeff: float,
                     dt: float = 1.0) -> np.ndarray:
    """
    Simulate soil moisture over N days using the Euler method.

    Parameters
    S0             : initial soil moisture (%)
    rainfall       : daily rainfall array (mm), length N
    irrigation     : daily irrigation array (mm), length N
    et             : daily ET array (mm), length N
    field_capacity : field capacity threshold (%)
    drainage_coeff : drainage coefficient (0–1)
    dt             : time step in days (default 1)

    Returns
    moisture : array of length N+1 (includes S0)
    """
    N = len(rainfall)
    moisture = np.zeros(N + 1)
    moisture[0] = S0

    for t in range(N):
        dS = soil_moisture_ode(moisture[t], rainfall[t], irrigation[t],
                                et[t], field_capacity, drainage_coeff)
        moisture[t + 1] = max(0.0, moisture[t] + dt * dS)

    return moisture


# 3. Runge-Kutta (RK4) Method
def rk4_simulation(S0: float,
                   rainfall: np.ndarray,
                   irrigation: np.ndarray,
                   et: np.ndarray,
                   field_capacity: float,
                   drainage_coeff: float,
                   dt: float = 1.0) -> np.ndarray:
    """
    Simulate soil moisture over N days using the RK4 method.

    Uses the same interface as euler_simulation for direct comparison.
    Since inputs are daily discrete values, we interpolate R, I, ET
    as piecewise constant within each day.
    """
    N = len(rainfall)
    moisture = np.zeros(N + 1)
    moisture[0] = S0

    for t in range(N):
        R, I_irr, ET = rainfall[t], irrigation[t], et[t]
        S = moisture[t]

        def f(s):
            return soil_moisture_ode(s, R, I_irr, ET, field_capacity, drainage_coeff)

        k1 = f(S)
        k2 = f(S + 0.5 * dt * k1)
        k3 = f(S + 0.5 * dt * k2)
        k4 = f(S + dt * k3)

        moisture[t + 1] = max(0.0, S + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    return moisture


# 4. Monte Carlo Simulation
def monte_carlo_rainfall(
        observed_rainfall: np.ndarray,
        n_scenarios: int = 1000,
        noise_std_pct: float = 0.30,
        seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate n_scenarios rainfall uncertainty scenarios.

    Each scenario perturbs the observed rainfall with multiplicative noise:
      R_scenario = R_observed * (1 + N(0, noise_std_pct))
    Negative values are clipped to 0.

    Parameters
    observed_rainfall : array of N daily rainfall values
    n_scenarios       : number of Monte Carlo scenarios
    noise_std_pct     : standard deviation of fractional noise (default 30%)
    seed              : random seed for reproducibility

    Returns
    scenarios : (n_scenarios, N) array of simulated rainfall
    """
    rng = np.random.default_rng(seed)
    N = len(observed_rainfall)
    noise = rng.normal(1.0, noise_std_pct, size=(n_scenarios, N))
    scenarios = np.clip(observed_rainfall[np.newaxis, :] * noise, 0, None)
    return scenarios


def run_monte_carlo_soil_moisture(
        S0: float,
        rainfall_scenarios: np.ndarray,
        irrigation: np.ndarray,
        et: np.ndarray,
        field_capacity: float,
        drainage_coeff: float,
        min_moisture: float
) -> dict:
    """
    Run soil-moisture simulation for each rainfall scenario.

    Returns
    dict with keys:
      all_moisture    : (n_scenarios, N+1) moisture trajectories
      prob_shortage   : probability of any day falling below min_moisture
      prob_over_irrig : probability of any day exceeding field_capacity
      expected_demand : mean total irrigation demand across scenarios
      worst_demand    : 95th percentile total irrigation demand
      percentiles     : dict of 5th, 50th, 95th percentile trajectories
    """
    n_scenarios, N = rainfall_scenarios.shape
    all_moisture = np.zeros((n_scenarios, N + 1))

    for i in range(n_scenarios):
        all_moisture[i] = rk4_simulation(
            S0, rainfall_scenarios[i], irrigation, et,
            field_capacity, drainage_coeff
        )

    # Shortage: at least one day below minimum moisture
    shortage_flags = np.any(all_moisture < min_moisture, axis=1)
    prob_shortage = shortage_flags.mean()

    # Over-irrigation: at least one day above field capacity
    over_flags = np.any(all_moisture > field_capacity, axis=1)
    prob_over = over_flags.mean()

    # Irrigation demand proxy: total irrigation supplied
    total_irrig = irrigation.sum()
    irrig_totals = np.full(n_scenarios, total_irrig)  # same schedule each scenario
    expected_demand = irrig_totals.mean()
    worst_demand = np.percentile(irrig_totals, 95)

    percentiles = {
        "p05": np.percentile(all_moisture, 5, axis=0),
        "p50": np.percentile(all_moisture, 50, axis=0),
        "p95": np.percentile(all_moisture, 95, axis=0),
    }

    return {
        "all_moisture": all_moisture,
        "prob_shortage": prob_shortage,
        "prob_over_irrigation": prob_over,
        "expected_demand": expected_demand,
        "worst_demand": worst_demand,
        "percentiles": percentiles,
    }
