"""
test_root_finding.py
Automated tests for bisection, Newton-Raphson, and secant root-finding methods.
Run with: pytest tests/test_root_finding.py -v
"""

import pytest
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.numerical_methods import bisection, newton_raphson, secant


# Shared test functions

def f_quadratic(x):
    """f(x) = x^2 - 4;  roots at x = ±2"""
    return x**2 - 4.0

def df_quadratic(x):
    return 2.0 * x

def f_cubic(x):
    """f(x) = x^3 - x - 2;  root near x ≈ 1.5214"""
    return x**3 - x - 2.0

def df_cubic(x):
    return 3.0 * x**2 - 1.0

def f_trig(x):
    """f(x) = cos(x) - x;  root near x ≈ 0.7391"""
    return math.cos(x) - x

def df_trig(x):
    return -math.sin(x) - 1.0

TOL = 1e-5

# Bisection tests

class TestBisection:

    def test_quadratic_positive_root(self):
        res = bisection(f_quadratic, 0, 5, tol=TOL)
        assert res["converged"], "Bisection did not converge"
        assert abs(res["root"] - 2.0) < TOL * 10

    def test_quadratic_negative_root(self):
        res = bisection(f_quadratic, -5, 0, tol=TOL)
        assert abs(res["root"] - (-2.0)) < TOL * 10

    def test_cubic_root(self):
        res = bisection(f_cubic, 1.0, 2.0, tol=TOL)
        assert abs(f_cubic(res["root"])) < 1e-4

    def test_history_length(self):
        res = bisection(f_quadratic, 0, 5, tol=TOL)
        assert len(res["history"]) == res["iterations"]

    def test_invalid_bracket_raises(self):
        with pytest.raises(ValueError):
            bisection(f_quadratic, 3, 5)  # both positive

    def test_iterations_positive(self):
        res = bisection(f_quadratic, 0, 5, tol=TOL)
        assert res["iterations"] > 0


# Newton-Raphson tests
class TestNewtonRaphson:

    def test_quadratic_root(self):
        res = newton_raphson(f_quadratic, df_quadratic, x0=3.0, tol=TOL)
        assert res["converged"]
        assert abs(res["root"] - 2.0) < TOL * 10

    def test_cubic_root(self):
        res = newton_raphson(f_cubic, df_cubic, x0=1.5, tol=TOL)
        assert abs(f_cubic(res["root"])) < 1e-4

    def test_trig_root(self):
        res = newton_raphson(f_trig, df_trig, x0=0.5, tol=TOL)
        assert abs(res["root"] - 0.7390851332) < 1e-4

    def test_faster_convergence_than_bisection(self):
        res_nr = newton_raphson(f_quadratic, df_quadratic, x0=3.0, tol=TOL)
        res_bi = bisection(f_quadratic, 0, 5, tol=TOL)
        assert res_nr["iterations"] <= res_bi["iterations"]


# Secant tests

class TestSecant:

    def test_quadratic_root(self):
        res = secant(f_quadratic, 1.0, 3.0, tol=TOL)
        assert res["converged"]
        assert abs(res["root"] - 2.0) < TOL * 10

    def test_cubic_root(self):
        res = secant(f_cubic, 1.0, 2.0, tol=TOL)
        assert abs(f_cubic(res["root"])) < 1e-4

    def test_trig_root(self):
        res = secant(f_trig, 0.5, 1.0, tol=TOL)
        assert abs(res["root"] - 0.7390851332) < 1e-4

    def test_history_non_empty(self):
        res = secant(f_quadratic, 1.0, 3.0, tol=TOL)
        assert len(res["history"]) > 0
