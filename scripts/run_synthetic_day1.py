"""Run the Day 1 synthetic PSD slope recovery diagnostic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from edb.plots.figures import save_figure
from edb.signal.de import beta_vs_de
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope


def powerlaw_noise(beta: float, n: int, seed: int = 42) -> np.ndarray:
    """Generate a zero-mean unit-variance synthetic series with approximate PSD slope beta."""

    rng = np.random.default_rng(seed)
    frequency = np.fft.rfftfreq(n, d=1.0)
    amplitude = np.zeros_like(frequency)
    amplitude[1:] = frequency[1:] ** (-beta / 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=frequency.size)
    spectrum = amplitude * np.exp(1j * phase)
    spectrum[0] = 0.0
    if n % 2 == 0:
        spectrum[-1] = amplitude[-1] * rng.choice([-1.0, 1.0])
    x = np.fft.irfft(spectrum, n=n)
    return (x - x.mean()) / x.std()


def main() -> None:
    beta_true = 5.0 / 3.0
    tau = 64.0
    n = 16_384
    t = np.arange(n)
    x = powerlaw_noise(beta_true, n=n, seed=42)

    psd = welch_psd(t, x, fs=1.0, nperseg=2048)
    slopes = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=12)
    beta_de = beta_vs_de(slopes["f_center"], slopes["beta"], tau=tau)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    psd.to_csv(tables_dir / "synthetic_psd.csv", index=False)
    slopes.to_csv(tables_dir / "synthetic_local_beta.csv", index=False)
    beta_de.to_csv(tables_dir / "synthetic_beta_vs_de.csv", index=False)

    interior = slopes[(slopes["f_center"] >= 0.01) & (slopes["f_center"] <= 0.2)]
    recovered = float(interior["beta"].median())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].loglog(psd["frequency"], psd["psd"], color="#1f77b4", linewidth=1.2)
    axes[0].set_xlabel("frequency")
    axes[0].set_ylabel("PSD")
    axes[0].set_title("Synthetic PSD")

    axes[1].semilogx(slopes["f_center"], slopes["beta"], color="#2ca02c", linewidth=1.2)
    axes[1].axhline(beta_true, color="#222222", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("frequency")
    axes[1].set_ylabel("local beta")
    axes[1].set_title(f"median beta={recovered:.2f}")

    axes[2].semilogx(beta_de["de"], beta_de["beta"], color="#d62728", linewidth=1.2)
    axes[2].axhline(beta_true, color="#222222", linestyle="--", linewidth=1.0)
    axes[2].axvline(1.0, color="#666666", linestyle=":", linewidth=1.0)
    axes[2].set_xlabel("De")
    axes[2].set_ylabel("local beta")
    axes[2].set_title("Beta vs De")

    fig.tight_layout()
    output_path = save_figure(fig, "synthetic_day1_diagnostic.png")
    plt.close(fig)

    print(f"Recovered median beta over 0.01-0.2 cycles/unit: {recovered:.3f}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
