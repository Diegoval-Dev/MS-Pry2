"""Unit tests for analytical M/M/1 metrics."""

import math

import pytest

from sim.metrics import mm1_theory, relative_error, rho


def test_rho_basic_value():
    assert math.isclose(rho(0.5, 1.0), 0.5)


def test_mm1_theory_matches_known_case():
    theory = mm1_theory(0.5, 1.0)
    assert math.isclose(theory.rho, 0.5)
    assert math.isclose(theory.L, 1.0)
    assert math.isclose(theory.Lq, 0.5)
    assert math.isclose(theory.W, 2.0)
    assert math.isclose(theory.Wq, 1.0)
    assert math.isclose(theory.L, 0.5 * theory.W)  # Little's law


def test_mm1_theory_invalid_rho():
    with pytest.raises(ValueError):
        mm1_theory(1.0, 1.0)


def test_relative_error_guard_zero_reference():
    assert relative_error(0.0, 0.0) == 0.0
    assert math.isinf(relative_error(1.0, 0.0))


def test_rho_requires_positive_mu():
    with pytest.raises(ValueError):
        rho(0.5, 0.0)
