"""
test_linear_systems.py
Automated tests for Gaussian elimination and LU decomposition.
Run with: pytest tests/test_linear_systems.py -v
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.numerical_methods import gaussian_elimination, lu_decomposition, lu_solve

TOL = 1e-8


# Gaussian Elimination

class TestGaussianElimination:

    def test_2x2_simple(self):
        A = np.array([[2.0, 1.0], [5.0, 7.0]])
        b = np.array([11.0, 13.0])
        x = gaussian_elimination(A, b)
        assert np.allclose(A @ x, b, atol=TOL)

    def test_3x3_water_allocation(self):
        """
        Three-zone water allocation problem:
        Zone A + Zone B = 50 mm
        Zone B + Zone C = 60 mm
        Zone A + Zone C = 45 mm
        """
        A = np.array([[1.0, 1.0, 0.0],
                      [0.0, 1.0, 1.0],
                      [1.0, 0.0, 1.0]])
        b = np.array([50.0, 60.0, 45.0])
        x = gaussian_elimination(A, b)
        assert np.allclose(A @ x, b, atol=TOL)

    def test_identity_matrix(self):
        A = np.eye(4)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = gaussian_elimination(A, b)
        assert np.allclose(x, b, atol=TOL)

    def test_singular_raises(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            gaussian_elimination(A, b)

    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        A = rng.random((5, 5)) + np.eye(5) * 5  # diagonally dominant
        b = rng.random(5)
        x_ours = gaussian_elimination(A, b)
        x_np = np.linalg.solve(A, b)
        assert np.allclose(x_ours, x_np, atol=1e-6)

# LU Decomposition

class TestLUDecomposition:

    def test_lu_factors_correct(self):
        A = np.array([[2.0, 1.0, -1.0],
                      [-3.0, -1.0, 2.0],
                      [-2.0, 1.0, 2.0]])
        L, U = lu_decomposition(A)
        assert np.allclose(L @ U, A, atol=TOL)

    def test_lu_solve_2x2(self):
        A = np.array([[4.0, 3.0], [6.0, 3.0]])
        b = np.array([10.0, 12.0])
        L, U = lu_decomposition(A)
        x = lu_solve(L, U, b)
        assert np.allclose(A @ x, b, atol=TOL)

    def test_lu_solve_3x3(self):
        A = np.array([[1.0, 2.0, 3.0],
                      [0.0, 1.0, 4.0],
                      [5.0, 6.0, 0.0]])
        b = np.array([14.0, 14.0, 14.0])
        L, U = lu_decomposition(A)
        x = lu_solve(L, U, b)
        assert np.allclose(A @ x, b, atol=TOL)

    def test_l_is_lower_triangular(self):
        A = np.array([[2.0, 1.0], [4.0, 3.0]])
        L, _ = lu_decomposition(A)
        assert L[0, 1] == 0.0  # upper-off-diagonal of L is 0

    def test_u_is_upper_triangular(self):
        A = np.array([[2.0, 1.0], [4.0, 3.0]])
        _, U = lu_decomposition(A)
        assert U[1, 0] == 0.0  # lower-off-diagonal of U is 0
