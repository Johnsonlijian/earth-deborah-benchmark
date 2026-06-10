import numpy as np

from edb.signal.timescales import autocorrelation, integral_autocorrelation_time


def test_autocorrelation_basic_properties() -> None:
    x = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

    acf = autocorrelation(x, max_lag=3)

    assert acf.shape == (4,)
    assert abs(acf[0] - 1.0) < 1e-12


def test_integral_autocorrelation_time_positive_for_ar1() -> None:
    rng = np.random.default_rng(123)
    x = np.zeros(500)
    for idx in range(1, x.size):
        x[idx] = 0.8 * x[idx - 1] + rng.normal()

    tau = integral_autocorrelation_time(x, dt_seconds=86_400.0, max_lag=120)

    assert tau > 86_400.0


def test_integral_autocorrelation_time_rejects_bad_dt() -> None:
    try:
        integral_autocorrelation_time([1, 2, 3], dt_seconds=0.0)
    except ValueError as exc:
        assert "dt_seconds" in str(exc)
    else:
        raise AssertionError("expected ValueError")

