import numpy as np
import pandas as pd

from edb.signal.nulls import ar1_surrogate, phase_randomized_surrogate, seasonal_noise_surrogate, white_noise_surrogate


def test_white_noise_surrogate_is_reproducible() -> None:
    x = np.arange(20, dtype=float)

    a = white_noise_surrogate(x, seed=7)
    b = white_noise_surrogate(x, seed=7)

    assert a.shape == x.shape
    np.testing.assert_allclose(a, b)


def test_ar1_surrogate_shape() -> None:
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=50))

    out = ar1_surrogate(x, seed=3)

    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_phase_randomized_surrogate_shape() -> None:
    x = np.sin(np.linspace(0, 6 * np.pi, 128))

    out = phase_randomized_surrogate(x, seed=4)

    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_seasonal_noise_surrogate_shape() -> None:
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    x = np.sin(np.arange(120) / 365 * 2 * np.pi)

    out = seasonal_noise_surrogate(dates, x, seed=5)

    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_seasonal_noise_surrogate_month_key_shape() -> None:
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    x = np.tile(np.arange(12, dtype=float), 3)

    out = seasonal_noise_surrogate(dates, x, seed=6, seasonal_key="month")

    assert out.shape == x.shape
    assert np.isfinite(out).all()
