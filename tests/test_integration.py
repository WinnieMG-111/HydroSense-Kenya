"""
test_integration.py
Automated tests for trapezoidal rule, Simpson's rule, and finite differences.
Run with: pytest tests/test_integration.py -v
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.numerical_methods import (trapezoidal, simpsons,
                                    forward_diff, backward_diff, central_diff)

TOL = 1e-3  # tolerance for integration tests

# Trapezoidal rule

class TestTrapezoidal:

    def test_constant_function(self):
        """Integral of 5 over [0, 10] = 50"""
        y = np.full(11, 5.0)
        result = trapezoidal(y, dx=1.0)
        assert abs(result - 50.0) < TOL

    def test_linear_function_exact(self):
        """Integral of x over [0, 1] = 0.5; trapezoidal is exact for linear."""
        x = np.linspace(0, 1, 101)
        y = x
        result = trapezoidal(y, x=x)
        assert abs(result - 0.5) < 1e-6

    def test_sine_approx(self):
        """Integral of sin(x) over [0, pi] = 2.0"""
        x = np.linspace(0, np.pi, 1000)
        y = np.sin(x)
        result = trapezoidal(y, x=x)
        assert abs(result - 2.0) < TOL

    def test_non_negative_output(self):
        """Trapezoidal result should be non-negative for non-negative input."""
        y = np.array([0.5, 1.0, 1.5, 2.0])
        assert trapezoidal(y, dx=1.0) > 0


# Simpson's rule


class TestSimpsons:

    def test_quadratic_exact(self):
        """Simpson's rule is exact for polynomials up to degree 3.
        Integral of x^2 over [0, 1] = 1/3."""
        x = np.linspace(0, 1, 101)
        y = x**2
        result = simpsons(y, dx=x[1] - x[0])
        assert abs(result - 1.0 / 3.0) < TOL

    def test_constant_function(self):
        y = np.ones(11)
        result = simpsons(y, dx=1.0)
        assert abs(result - 10.0) < TOL

    def test_more_accurate_than_trapezoidal_cubic(self):
        """For a cubic function, Simpson's should be more accurate."""
        x = np.linspace(0, 1, 11)
        y = x**3
        exact = 0.25
        trap = trapezoidal(y, x=x)
        simp = simpsons(y, dx=x[1] - x[0])
        assert abs(simp - exact) < abs(trap - exact)

    def test_requires_at_least_3_points(self):
        with pytest.raises(ValueError):
            simpsons(np.array([1.0, 2.0]), dx=1.0)


# Finite differences

class TestFiniteDifferences:

    def f(self, x):
        return x**2

    def df_exact(self, x):
        return 2.0 * x

    def test_forward_diff_accuracy(self):
        for x in [0.5, 1.0, 2.0, 3.0]:
            approx = forward_diff(self.f, x, h=1e-5)
            exact = self.df_exact(x)
            assert abs(approx - exact) < 1e-3

    def test_backward_diff_accuracy(self):
        for x in [0.5, 1.0, 2.0]:
            approx = backward_diff(self.f, x, h=1e-5)
            exact = self.df_exact(x)
            assert abs(approx - exact) < 1e-3

    def test_central_diff_more_accurate(self):
        """Central diff should have smaller error than forward diff."""
        x = 1.5
        h = 1e-3
        exact = self.df_exact(x)
        err_fwd = abs(forward_diff(self.f, x, h) - exact)
        err_cen = abs(central_diff(self.f, x, h) - exact)
        assert err_cen < err_fwd
