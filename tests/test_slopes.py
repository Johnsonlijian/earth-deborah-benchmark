import numpy as np

from edb.signal.slopes import local_loglog_slope


def test_local_loglog_slope_recovers_exact_power_curve() -> None:
    freq = np.geomspace(1e-3, 1.0, 200)
    beta = 1.5
    psd = 4.0 * freq ** (-beta)

    out = local_loglog_slope(freq, psd, window_decades=0.75, min_points=10)

    assert not out.empty
    np.testing.assert_allclose(out["beta"].median(), beta, atol=1e-10)
    assert (out["n_points"] >= 10).all()
