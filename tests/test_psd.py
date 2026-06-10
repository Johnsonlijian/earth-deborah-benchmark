import numpy as np

from edb.signal.psd import lomb_scargle_psd, multitaper_psd, welch_psd
from edb.signal.slopes import local_loglog_slope


def _powerlaw_noise(beta: float, n: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(n, d=1.0)
    amplitude = np.zeros_like(freq)
    amplitude[1:] = freq[1:] ** (-beta / 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=freq.size)
    spectrum = amplitude * np.exp(1j * phase)
    spectrum[0] = 0.0
    if n % 2 == 0:
        spectrum[-1] = amplitude[-1] * rng.choice([-1.0, 1.0])
    x = np.fft.irfft(spectrum, n=n)
    return (x - x.mean()) / x.std()


def test_welch_psd_returns_positive_frequencies() -> None:
    t = np.arange(512)
    x = np.sin(2 * np.pi * t / 32)

    out = welch_psd(t, x, fs=1.0)

    assert {"frequency", "psd"}.issubset(out.columns)
    assert (out["frequency"] > 0).all()
    assert (out["psd"] > 0).all()


def test_lomb_scargle_psd_returns_positive_power() -> None:
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0.0, 100.0, size=200))
    x = np.sin(2 * np.pi * t / 12.0)

    out = lomb_scargle_psd(t, x, min_freq=0.01, max_freq=0.4, n_freqs=128)

    assert {"frequency", "psd"}.issubset(out.columns)
    assert (out["frequency"] > 0).all()
    assert (out["psd"] > 0).all()


def test_synthetic_powerlaw_slope_recovers_five_thirds() -> None:
    beta_true = 5.0 / 3.0
    x = _powerlaw_noise(beta_true, n=16384, seed=42)
    t = np.arange(x.size)

    psd = welch_psd(t, x, fs=1.0, nperseg=2048)
    slopes = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=12)
    interior = slopes[(slopes["f_center"] >= 0.01) & (slopes["f_center"] <= 0.2)]

    recovered = float(interior["beta"].median())
    assert abs(recovered - beta_true) < 0.35


def test_multitaper_placeholder() -> None:
    try:
        multitaper_psd()
    except NotImplementedError:
        return
    raise AssertionError("multitaper_psd should raise NotImplementedError")
