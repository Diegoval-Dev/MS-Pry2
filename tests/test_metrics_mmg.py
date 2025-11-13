"""Unit tests for analytical M/M/g metrics."""

import math

import pytest

from sim.metrics import mm1_theory
from sim.metrics_mmg import mmg_theory


def test_mmg_reduces_to_mm1_when_g_is_one():
    lam = 0.7
    mu = 1.0
    mm1 = mm1_theory(lam, mu)
    mmg = mmg_theory(lam, mu, g=1)

    assert math.isclose(mmg.rho, mm1.rho, rel_tol=1e-9)
    assert math.isclose(mmg.L, mm1.L, rel_tol=1e-9)
    assert math.isclose(mmg.Lq, mm1.Lq, rel_tol=1e-9)
    assert math.isclose(mmg.W, mm1.W, rel_tol=1e-9)
    assert math.isclose(mmg.Wq, mm1.Wq, rel_tol=1e-9)
    assert math.isclose(mmg.Pwait, mm1.rho, rel_tol=1e-9)


def test_mmg_theory_properties_for_two_servers():
    lam = 1.7
    mu = 1.0
    g = 2
    theory = mmg_theory(lam, mu, g)

    assert 0 < theory.Pwait < 1
    assert theory.Lq > 0
    assert theory.Wq > 0
    assert math.isclose(theory.L, lam * theory.W, rel_tol=1e-9)
    assert math.isclose(theory.Lq, lam * theory.Wq, rel_tol=1e-9)


def test_mmg_invalid_parameters_raise():
    with pytest.raises(ValueError):
        mmg_theory(1.0, 1.0, g=0)
    with pytest.raises(ValueError):
        mmg_theory(2.0, 1.0, g=2)  # rho >= 1
