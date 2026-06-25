"""

HydroSense-Kenya — Irrigation Optimization Module

Implements a greedy threshold-based irrigation optimizer and a simple
gradient-descent-based minimizer that minimises total irrigation water use
while keeping soil moisture above the crop minimum threshold at all times.
"""

import numpy as np
from typing import Tuple
from src.simulation import rk4_simulation
from src.numerical_methods import compute_et_vectorised

# 1. Greedy threshold scheduler

def greedy_irrigation_schedule(
        S0: float,
        rainfall: np.ndarray,
        et: np.ndarray,
        field_capacity: float,
        drainage_coeff: float,
        min_moisture: float,
        target_moisture: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy irrigation scheduler.

    On each day, if the predicted soil moisture (without irrigation) would
    fall below min_moisture, apply just enough irrigation to reach target_moisture.
    Never irrigate above field capacity.

    Parameters
    S0             : initial soil moisture (%)
    rainfall       : daily rainfall array (mm)
    et             : daily ET array (mm)
    field_capacity : field capacity (%)
    drainage_coeff : drainage coefficient
    min_moisture   : stress threshold — irrigate if S would fall below this
    target_moisture: refill target

    Returns
    irrigation  : daily irrigation schedule (mm)
    moisture    : resulting soil moisture trajectory (N+1 points)
    """
    N = len(rainfall)
    irrigation = np.zeros(N)
    moisture = np.zeros(N + 1)
    moisture[0] = S0

    for t in range(N):
        # Predict next-step moisture without irrigation
        zero_irr = np.array([0.0])
        S_pred = rk4_simulation(moisture[t], rainfall[t:t+1], zero_irr,
                                 et[t:t+1], field_capacity, drainage_coeff)[1]
        if S_pred < min_moisture:
            # Determine irrigation needed to reach target_moisture
            needed = target_moisture - S_pred
            irrigation[t] = max(0.0, min(needed, field_capacity - S_pred))

        # Simulate actual step with chosen irrigation
        irr = np.array([irrigation[t]])
        moisture[t + 1] = rk4_simulation(
            moisture[t], rainfall[t:t+1], irr,
            et[t:t+1], field_capacity, drainage_coeff
        )[1]

    return irrigation, moisture


# 2. Gradient-descent optimizer

def _penalty_objective(irrigation: np.ndarray,
                        S0: float,
                        rainfall: np.ndarray,
                        et: np.ndarray,
                        field_capacity: float,
                        drainage_coeff: float,
                        min_moisture: float,
                        penalty_weight: float = 100.0) -> float:
    """
    Objective function for optimization:
      J = sum(I_t) + penalty_weight * sum(max(0, min_moisture - S(t+1))^2)
    
    Minimising J minimises total water use while penalising moisture stress.
    """
    moisture = rk4_simulation(S0, rainfall, irrigation, et,
                               field_capacity, drainage_coeff)
    total_water = irrigation.sum()
    stress = np.sum(np.maximum(0.0, min_moisture - moisture[1:]) ** 2)
    return total_water + penalty_weight * stress


def optimise_irrigation(
        S0: float,
        rainfall: np.ndarray,
        et: np.ndarray,
        field_capacity: float,
        drainage_coeff: float,
        min_moisture: float,
        target_moisture: float,
        learning_rate: float = 0.05,
        max_iter: int = 500,
        tol: float = 1e-4,
        penalty_weight: float = 100.0
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Gradient-descent irrigation optimizer.

    Minimises total irrigation while penalising moisture stress.
    Uses numerical gradient (finite differences on the objective).

    Returns
    irrigation  : optimised daily irrigation schedule (mm)
    moisture    : resulting soil moisture trajectory
    loss_history: list of objective values per iteration
    """
    N = len(rainfall)
    # Initialise with greedy schedule as warm start
    irrigation, _ = greedy_irrigation_schedule(
        S0, rainfall, et, field_capacity, drainage_coeff, min_moisture, target_moisture
    )
    irrigation = irrigation.astype(float).copy()

    loss_history = []
    h = 1e-3  # finite difference step

    for iteration in range(max_iter):
        J0 = _penalty_objective(irrigation, S0, rainfall, et,
                                 field_capacity, drainage_coeff,
                                 min_moisture, penalty_weight)
        loss_history.append(J0)

        # Numerical gradient
        grad = np.zeros(N)
        for i in range(N):
            irr_plus = irrigation.copy()
            irr_plus[i] += h
            J_plus = _penalty_objective(irr_plus, S0, rainfall, et,
                                         field_capacity, drainage_coeff,
                                         min_moisture, penalty_weight)
            grad[i] = (J_plus - J0) / h

        # Gradient step
        irrigation = irrigation - learning_rate * grad
        irrigation = np.clip(irrigation, 0.0, field_capacity)  # non-negative, bounded

        # Convergence check
        if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            print(f"[optimise_irrigation] Converged at iteration {iteration+1}")
            break

    moisture = rk4_simulation(S0, rainfall, irrigation, et,
                               field_capacity, drainage_coeff)
    return irrigation, moisture, loss_history


# 3. Multi-zone water allocation summary

def zone_water_summary(zone_ids: list,
                        irrigation_schedules: dict,
                        areas_m2: dict) -> dict:
    """
    Summarise total water allocation per zone.

    """
    summary = {}
    for zone in zone_ids:
        total_mm = float(irrigation_schedules[zone].sum())
        total_litres = total_mm * areas_m2[zone]
        summary[zone] = {"total_mm": round(total_mm, 2),
                        "total_litres": round(total_litres, 1)
                        }
    return summary
