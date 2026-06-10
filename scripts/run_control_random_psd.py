"""Control experiment: random PSD shapes should NOT exhibit storage-memory collapse.

If the tau_acf -> beta(De) relationship is circular (both derived from the same signal),
random non-physical spectra should also show apparent "collapse". If they don't,
the observed collapse in river data is not a methodological artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edb.signal.de import beta_vs_de
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window
from edb.signal.timescales import integral_autocorrelation_time

SECONDS_PER_DAY = 86_400.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-series", type=int, default=100)
    p.add_argument("--series-length", type=int, default=15000)
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--output-prefix", default="control_random_psd")
    return p.parse_args()


def random_psd_series(length, rng):
    nf = length // 2 + 1
    amps = 10 ** rng.uniform(-3, 0, nf)
    amps[0] = 0
    phases = rng.uniform(0, 2 * np.pi, nf)
    if length % 2 == 0:
        phases[-1] = 0
    return np.fft.irfft(amps * np.exp(1j * phases), n=length)


def colored_noise_series(length, rng):
    alpha = rng.uniform(0.0, 3.0)
    nf = length // 2 + 1
    freq = np.fft.rfftfreq(length, d=1.0)
    safe = np.where(freq > 0, freq, freq[1])
    amps = safe ** (-alpha / 2)
    amps[0] = 0
    phases = rng.uniform(0, 2 * np.pi, nf)
    if length % 2 == 0:
        phases[-1] = 0
    return np.fft.irfft(amps * np.exp(1j * phases), n=length)


def beta_quick(dates, values, tau_sec):
    psd = welch_psd(dates, values, fs=1.0 / SECONDS_PER_DAY)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    bd = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_sec)
    return bd, summarize_beta_window(bd)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    base = pd.Timestamp("1980-01-01")
    dates = pd.DatetimeIndex([base + pd.Timedelta(days=i) for i in range(args.series_length)])

    rows = []
    for i in range(args.n_series):
        # Random PSD
        s = random_psd_series(args.series_length, rng)
        t = integral_autocorrelation_time(s, dt_seconds=SECONDS_PER_DAY, max_lag=730, stop_at_zero=True)
        tau_d = t / SECONDS_PER_DAY
        if tau_d > 0.1:
            try:
                _, ob = beta_quick(dates, s, t)
                rows.append({"id": "r%d" % i, "kind": "random_psd", "tau_d": tau_d, **ob})
            except Exception:
                pass

        # Colored noise
        sc = colored_noise_series(args.series_length, rng)
        tc = integral_autocorrelation_time(sc, dt_seconds=SECONDS_PER_DAY, max_lag=730, stop_at_zero=True)
        tau_dc = tc / SECONDS_PER_DAY
        if tau_dc > 0.1:
            try:
                _, oc = beta_quick(dates, sc, tc)
                rows.append({"id": "c%d" % i, "kind": "colored_noise", "tau_d": tau_dc, **oc})
            except Exception:
                pass

    df = pd.DataFrame(rows)
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{args.output_prefix}_summary.csv", index=False)

    # Compare with real CAMELS
    rp = Path("reports/tables/camels_674_memory_collapse_summary.csv")
    cr = pd.read_csv(rp) if rp.exists() else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(7, 5))
    if not cr.empty:
        v = cr[cr["mean_beta_de_window"].notna()]
        ax.scatter(v["tau_acf_days"], v["mean_beta_de_window"], c="#2166ac", s=5, alpha=0.35, label="CAMELS real (n=%d)" % len(v), zorder=3)
    rd = df[df["kind"] == "random_psd"]
    cn = df[df["kind"] == "colored_noise"]
    ax.scatter(rd["tau_d"], rd["mean_beta_de_window"], c="#b2182b", s=8, alpha=0.4, marker="x", label="Random PSD (n=%d)" % len(rd), zorder=2)
    ax.scatter(cn["tau_d"], cn["mean_beta_de_window"], c="#d6604d", s=8, alpha=0.4, marker="+", label="Colored noise (n=%d)" % len(cn), zorder=2)
    ax.axhline(5.0/3.0, color="#777", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("tau [days]")
    ax.set_ylabel("mean beta(0.5 <= De <= 2)")
    ax.set_title("Control experiment: random spectra do NOT exhibit collapse")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fdir = Path("reports/figures")
    fdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fdir / "control_random_psd.png", dpi=300)
    plt.close(fig)

    rn = (rd["n_beta_de_window"].fillna(0) > 0).sum()
    cn_n = (cn["n_beta_de_window"].fillna(0) > 0).sum()
    print("Random PSD De windows: %d/%d" % (rn, len(rd)))
    print("Colored noise De windows: %d/%d" % (cn_n, len(cn)))
    if not cr.empty:
        rv = cr["mean_beta_de_window"].dropna().var()
        rav = rd["mean_beta_de_window"].dropna().var()
        cv = cn["mean_beta_de_window"].dropna().var()
        print("Variance: real=%.2f random=%.2f colored=%.2f" % (rv, rav, cv))
    print("Saved %s" % (out_dir / ("%s_summary.csv" % args.output_prefix)))


if __name__ == "__main__":
    main()
