import numpy as np

from edb.signal.de import beta_vs_de, deborah_number


def test_deborah_number_returns_frequency_times_tau() -> None:
    freq = np.array([0.1, 1.0, 10.0])
    tau = 4.0

    np.testing.assert_allclose(deborah_number(freq, tau), np.array([0.4, 4.0, 40.0]))


def test_beta_vs_de_shapes_values() -> None:
    freq = np.array([0.5, 1.0])
    beta = np.array([1.5, 1.7])

    out = beta_vs_de(freq, beta, tau=2.0)

    assert list(out.columns) == ["frequency", "beta", "tau", "de", "log10_de"]
    np.testing.assert_allclose(out["de"], np.array([1.0, 2.0]))
