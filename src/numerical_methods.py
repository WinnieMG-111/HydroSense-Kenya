"""
HydroSense-Kenya — Numerical Methods Engine

Implements from scratch (no SciPy calls for the core logic):
  - Evapotranspiration and water balance functions (scalar + vectorised)
  - Root finding: Bisection, Newton-Raphson, Secant
  - Finite differences: Forward, Backward, Central
  - Numerical integration: Trapezoidal Rule, Simpson's Rule
  - Linear systems: Gaussian Elimination (with partial pivoting), LU Decomposition
"""

import numpy as np
from typing import Callable, Tuple, Optional


# 1. Evapotranspiration and water balance

def compute_et(T: float, W: float, Solar: float, H: float) -> float:
    """
    Simplified empirical daily evapotranspiration (mm).

    ET = max(0, 0.12*T + 0.35*W + 2.4*Solar - 0.025*H)

    Parameters
    T     : temperature (°C)
    W     : wind speed (m/s)
    Solar : solar index (dimensionless, 0–1)
    H     : humidity (%)
    """
    return float(np.maximum(0.0, 0.12 * T + 0.35 * W + 2.4 * Solar - 0.025 * H))


def compute_et_vectorised(T: np.ndarray, W: np.ndarray,
                           Solar: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Vectorised version of compute_et for NumPy arrays."""
    return np.maximum(0.0, 0.12 * T + 0.35 * W + 2.4 * Solar - 0.025 * H)


def water_balance_step(S_t: float, R_t: float, I_t: float,
                       ET_t: float, field_capacity: float,
                       drainage_coeff: float) -> Tuple[float, float]:
    """
    One discrete water-balance step.

    Computes a single daily soil moisture step tracking physical boundaries.

    S(t+1) = S(t) + R(t) + I(t) - ET(t) - D(t)

    Drainage D_t is proportional to water above field capacity.

    Returns
    (S_next, D_t) : updated moisture and drainage
    """
    S_raw = S_t + R_t + I_t - ET_t
    excess = max(0.0, S_raw - field_capacity)
    D_t = drainage_coeff * excess
    S_next = max(0.0, S_raw - D_t)
    return S_next, D_t


# 2. Root Finding

def bisection(f: Callable, a: float, b: float,
              tol: float = 1e-5, max_iter: int = 100) -> dict:
    """
    Bisection root-finding method.

    Finds x in [a, b] such that f(x) ≈ 0.

    Returns
    dict with keys: root, iterations, final_error, converged, history
    """
    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    history = []
    for i in range(1, max_iter + 1):
        mid = (a + b) / 2.0
        f_mid = f(mid)
        error = abs(b - a) / 2.0
        history.append({"iter": i, "root_est": mid, "error": error})

        if error < tol or f_mid == 0.0:
            return {"root": mid, "iterations": i,
                    "final_error": error, "converged": True, "history": history}
        if f(a) * f_mid < 0:
            b = mid
        else:
            a = mid

    return {"root": (a + b) / 2.0, "iterations": max_iter,
            "final_error": abs(b - a) / 2.0, "converged": False, "history": history}


def newton_raphson(f: Callable, df: Callable, x0: float,
                   tol: float = 1e-5, max_iter: int = 100) -> dict:
    """
    Newton-Raphson root-finding method.

    Parameters
    f   : function
    df  : derivative of f
    x0  : initial guess
    """
    x = x0
    history = []
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:
            raise ValueError("Derivative near zero; Newton-Raphson failed.")
        x_new = x - fx / dfx
        error = abs(x_new - x)
        history.append({"iter": i, "root_est": x_new, "error": error})
        x = x_new
        if error < tol:
            return {"root": x, "iterations": i,
                    "final_error": error, "converged": True, "history": history}

    return {"root": x, "iterations": max_iter,
            "final_error": error, "converged": False, "history": history}


def secant(f: Callable, x0: float, x1: float,
           tol: float = 1e-5, max_iter: int = 100) -> dict:
    """
    Secant root-finding method (derivative-free).
    """
    history = []
    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        if abs(f1 - f0) < 1e-12:
            raise ValueError("Secant denominator near zero; method failed.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x2 - x1)
        history.append({"iter": i, "root_est": x2, "error": error})
        x0, x1 = x1, x2
        if error < tol:
            return {"root": x2, "iterations": i,
                    "final_error": error, "converged": True, "history": history}

    return {"root": x1, "iterations": max_iter,
            "final_error": error, "converged": False, "history": history}


def irrigation_root_function(I: float, S_t: float, R_t: float,
                              ET_t: float, field_capacity: float,
                              drainage_coeff: float, target: float) -> float:
    """
    Helper: returns S(t+1) - target_moisture given irrigation amount I.
    Used by root finders to determine how much irrigation is needed.
    """
    S_next, _ = water_balance_step(S_t, R_t, I, ET_t, field_capacity, drainage_coeff)
    return S_next - target


# 3. Finite Differences

def forward_diff(f: Callable, x: float, h: float = 1e-4) -> float:
    """First-order forward finite difference approximation of f'(x)."""
    return (f(x + h) - f(x)) / h


def backward_diff(f: Callable, x: float, h: float = 1e-4) -> float:
    """First-order backward finite difference approximation of f'(x)."""
    return (f(x) - f(x - h)) / h


def central_diff(f: Callable, x: float, h: float = 1e-4) -> float:
    """Second-order central finite difference approximation of f'(x)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def soil_moisture_rate(moisture_series: np.ndarray) -> np.ndarray:
    """
    Estimate daily rate of change in soil moisture using finite differences.
    Uses forward diff for first point, backward diff for last, central elsewhere.
    """
    n = len(moisture_series)
    rates = np.zeros(n)
    rates[0] = moisture_series[1] - moisture_series[0]          # forward
    rates[-1] = moisture_series[-1] - moisture_series[-2]       # backward
    rates[1:-1] = (moisture_series[2:] - moisture_series[:-2]) / 2.0  # central
    return rates


# 4. Numerical Integration

def trapezoidal(y: np.ndarray, x: Optional[np.ndarray] = None,
                dx: float = 1.0) -> float:
    """
    Trapezoidal rule for numerical integration.

    Parameters
    y  : function values
    x  : x-coordinates (if None, uniform spacing dx is assumed)
    dx : spacing (used only when x is None)
    """
    # np.trapz was renamed to np.trapezoid in NumPy 2.0
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    if x is not None:
        return float(_trapz(y, x))
    return float(_trapz(y, dx=dx))


def simpsons(y: np.ndarray, dx: float = 1.0) -> float:
    """
    Simpson's 1/3 rule for numerical integration.
    Requires an odd number of equally-spaced points (even number of intervals).
    If n is even (odd intervals), fall back to trapezoidal for the last interval.
    """
    n = len(y)
    if n < 3:
        raise ValueError("Simpson's rule requires at least 3 points.")
    if n % 2 == 0:
        # odd number of intervals: apply Simpson on [0..n-2], trap on last interval
        result = simpsons(y[:-1], dx) + 0.5 * (y[-2] + y[-1]) * dx
        return result

    coeffs = np.ones(n)
    coeffs[1:-1:2] = 4.0
    coeffs[2:-2:2] = 2.0
    return float((dx / 3.0) * np.dot(coeffs, y))


def cumulative_water_deficit(et_series: np.ndarray,
                              rainfall_series: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute cumulative daily water deficit = ET - Rainfall (when positive).
    Integrates using the trapezoidal rule.

    Returns
    -------
    daily_deficit : array of daily deficits
    cumulative    : scalar total deficit over the period
    """
    daily_deficit = np.maximum(0.0, et_series - rainfall_series)
    cumulative = trapezoidal(daily_deficit, dx=1.0)
    return daily_deficit, cumulative


# 5. Linear Systems — Gaussian Elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using Gaussian Elimination with partial pivoting.

    Parameters
    A : (n, n) coefficient matrix
    b : (n,) right-hand side vector

    Returns
    x : solution vector
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col + np.argmax(np.abs(A[col:, col]))
        if abs(A[max_row, col]) < 1e-12:
            raise ValueError(f"Matrix is singular at column {col}.")
        # Swap rows
        A[[col, max_row]] = A[[max_row, col]]
        b[[col, max_row]] = b[[max_row, col]]
        # Eliminate below
        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]
            A[row, col:] -= factor * A[col, col:]
            b[row] -= factor * b[col]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) < 1e-12:
            raise ValueError(f"Zero division detected on diagonal matrix row {i} during back-substitution.")
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    LU decomposition without pivoting (Doolittle's method).
    Returns L (lower triangular, unit diagonal) and U (upper triangular).
    """
    A = A.astype(float).copy()
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        # Upper triangular
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        # Lower triangular
        for j in range(i + 1, n):
            if abs(U[i, i]) < 1e-12:
                raise ValueError(f"Zero pivot at row {i}; LU without pivoting fails.")
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

    return L, U


def lu_solve(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b given LU decomposition: forward substitution then back substitution."""
    b = b.astype(float).copy()
    n = len(b)

    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Back substitution: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def solve_water_allocation(A: np.ndarray, b: np.ndarray,
                            method: str = "gaussian") -> np.ndarray:
    """
    Solve a multi-zone water allocation linear system Ax = b.

    Parameters
    A      : allocation coefficient matrix
    b      : demand / constraint vector
    method : 'gaussian' or 'lu'
    """
    if method == "gaussian":
        return gaussian_elimination(A, b)
    elif method == "lu":
        L, U = lu_decomposition(A)
        return lu_solve(L, U, b)
    else:
        raise ValueError("method must be 'gaussian' or 'lu'.")
