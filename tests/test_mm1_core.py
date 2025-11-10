"""Integration-style checks for the simulation core."""

from sim.mm1_core import MM1Params, run_mm1


def test_little_law_errors_within_ten_percent():
    """Short run should keep Little's law deviation within 10% tolerances."""
    params = MM1Params(lam=0.5, mu=1.0, seed=42, warmup=500.0, horizon=5000.0)
    result = run_mm1(params)
    assert result.little_L_error < 0.10
    assert result.little_Lq_error < 0.10
