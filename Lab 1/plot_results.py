#!/usr/bin/env python3
"""
plot_results.py - FIT3143 Lab 1 figure generation

Reads the two CSVs written by run_sweep.sh and emits every figure needed for
the presentation.

    scaling_n.csv        ->  graphs 1, 2, 5, 6, 7   (+ A1, A3, A4)
    scaling_threads.csv  ->  graphs 3, 4, 8         (+ A2, A5)

    ./plot_results.py                 # writes ./plots/*.png
    ./plot_results.py --outdir figs   # writes elsewhere

Design rules applied throughout:
  * One colour per implementation, held constant across every figure, so the
    audience learns the mapping once. Palette is colour-blind safe (Wong).
  * Runtime against n is plotted log-log. Runtime spans four orders of
    magnitude; on linear axes every point below n = 5,000,000 collapses onto
    the origin and the small-n regime - where speedup is worst and therefore
    most interesting - becomes invisible.
  * Speedup is plotted on linear axes against a dashed ideal-speedup
    reference, because the gap to ideal is the quantity being discussed.
  * Hardware structure (8 performance cores + 4 efficiency cores) is drawn on
    every thread-axis figure, so the knee in the curve is visually tied to its
    cause rather than left for the viewer to infer.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")           # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Machine constants. Change these if the sweep is run on different hardware.
# ---------------------------------------------------------------------------
P_CORES = 8         # performance cores  (sysctl hw.perflevel0.logicalcpu)
E_CORES = 4         # efficiency cores   (sysctl hw.perflevel1.logicalcpu)
TOTAL_CORES = P_CORES + E_CORES

# Wong colour-blind-safe palette. Serial is deliberately grey: it is the
# reference, not a competitor, so it should recede visually.
C_SERIAL = "#7f7f7f"
C_PTHREAD = "#0072B2"      # blue
C_OPENMP = "#D55E00"       # vermillion
C_IDEAL = "#999999"
C_ACCENT = "#009E73"       # green, for annotations

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 12,          # presentation-scale: readable projected
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "figure.autolayout": False,
})

FIGSIZE = (9, 5.5)


def style(ax, title, xlabel, ylabel):
    """Common axis furniture: title states the finding, not just the metric."""
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def millions(x, _pos):
    """Axis tick formatter: 2,000,000 -> 2M, 500,000 -> 500K."""
    if x >= 1e6:
        return f"{x/1e6:g}M"
    if x >= 1e3:
        return f"{x/1e3:g}K"
    return f"{x:g}"


def mark_cores(ax, label_y=0.95):
    """
    Draw the P-core / E-core structure on a thread-count axis.

    The knee in every thread-axis curve sits exactly at p = P_CORES. Drawing
    the boundary makes the causal claim visible instead of asserted.
    """
    ax.axvline(P_CORES, color=C_ACCENT, linestyle="--", linewidth=1.6, zorder=1)
    ax.axvspan(P_CORES, TOTAL_CORES, color=C_ACCENT, alpha=0.07, zorder=0)
    ax.axvline(TOTAL_CORES, color="#444444", linestyle=":", linewidth=1.4, zorder=1)
    ax.text(P_CORES - 0.25, ax.get_ylim()[1] * label_y,
            f"{P_CORES} P-cores", rotation=90, va="top", ha="right",
            fontsize=10, color=C_ACCENT, fontweight="bold")
    ax.text(TOTAL_CORES + 0.35, ax.get_ylim()[1] * label_y,
            f"{TOTAL_CORES} logical cores", rotation=90, va="top", ha="left",
            fontsize=10, color="#444444")


def save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
# Graphs 1, 2, 5, 6, 7 - problem size on the x-axis
# ===========================================================================
def runtime_vs_n(df, outdir, which, fname, title):
    """Graphs 1 and 5: serial against one parallel implementation."""
    col, colour, label = which
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.loglog(df["n"], df["serial_min_s"], "o-", color=C_SERIAL,
              label="Serial (Task 1)", markersize=5, linewidth=1.8)
    ax.loglog(df["n"], df[col], "s-", color=colour,
              label=label, markersize=5, linewidth=1.8)

    # Fill the gap between the curves: on a log axis a constant vertical gap
    # is a constant ratio, so the band having even width IS the finding that
    # speedup has stabilised.
    ax.fill_between(df["n"], df[col], df["serial_min_s"],
                    color=colour, alpha=0.10)

    style(ax, title, "n (upper bound on the prime search)", "Runtime (s, log scale)")
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.legend(frameon=False, loc="upper left")
    save(fig, outdir, fname)


def speedup_vs_n(df, outdir, col, colour, fname, title, label):
    """Graphs 2 and 6."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(df["n"], df[col], "o-", color=colour, markersize=5,
                linewidth=1.9, label=label)
    ax.axhline(TOTAL_CORES, color=C_IDEAL, linestyle="--", linewidth=1.4,
               label=f"Ideal speedup ({TOTAL_CORES} threads)")

    ymax = df[col].max()
    ax.axhline(ymax, color=C_ACCENT, linestyle=":", linewidth=1.4)
    ax.annotate(f"plateau ≈ {ymax:.2f}×",
                xy=(df["n"].iloc[-1], ymax), xytext=(-10, 8),
                textcoords="offset points", ha="right",
                fontsize=10, color=C_ACCENT, fontweight="bold")

    # The small-n regime is a result, not noise to be hidden.
    small = df[df["n"] < 6e5]
    if not small.empty:
        ax.axvspan(df["n"].min(), 6e5, color="#cccccc", alpha=0.25, zorder=0)
        ax.annotate("thread-management overhead\ndominates: sub-millisecond runs",
                    xy=(small["n"].iloc[len(small)//2], small[col].min()),
                    xytext=(0, -46), textcoords="offset points",
                    ha="center", fontsize=9.5, color="#555555")

    style(ax, title, "n (upper bound on the prime search)", "Speedup  S = T_serial / T_parallel")
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_ylim(0, TOTAL_CORES + 1.4)
    ax.legend(frameon=False, loc="lower right")
    save(fig, outdir, fname)


def find_crossover(df):
    """
    Locate the problem size at which OpenMP takes over from POSIX Threads.

    Two things make a naive scan unreliable. At small n the two are within
    measurement noise, so the ordering flips back and forth; and at large n a
    single unlucky point can invert once (here n = 8.5M) while every neighbour
    on both sides favours OpenMP. Both are artefacts, not crossovers.

    The ratio T_pthreads / T_openmp is therefore smoothed with a 3-point
    rolling median before the last upward crossing of 1.0 is taken. A median
    filter removes isolated outliers without shifting a genuine transition.
    """
    ratio = (df["pthreads_min_s"] / df["openmp_min_s"])
    smooth = ratio.rolling(3, center=True, min_periods=1).median().to_numpy()
    for i in range(len(smooth) - 1, 0, -1):
        if smooth[i - 1] < 1.0 <= smooth[i]:
            return i - 1, df["n"].iloc[i - 1], df["n"].iloc[i]
    return None, None, None


def graph7(df, outdir):
    """
    Graph 7: the two parallel implementations against each other.

    Plotted as two stacked panels sharing the x-axis. The upper panel is the
    runtime comparison the specification asks for; on a log axis, however, a
    20% difference is a barely visible vertical offset. The lower panel plots
    the ratio directly, which turns that same difference into a curve with an
    obvious structure - and the ratio is the quantity actually under
    discussion when comparing two parallel implementations.
    """
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(9, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})

    ax.loglog(df["n"], df["pthreads_min_s"], "s-", color=C_PTHREAD,
              label="POSIX Threads (Task 2)", markersize=5, linewidth=1.8)
    ax.loglog(df["n"], df["openmp_min_s"], "^-", color=C_OPENMP,
              label="OpenMP (Task 3)", markersize=5, linewidth=1.8)
    style(ax, "Graph 7 — POSIX Threads vs OpenMP with increasing n",
          "", "Runtime (s, log scale)")
    ax.legend(frameon=False, loc="upper left")

    ratio = df["pthreads_min_s"] / df["openmp_min_s"]
    bx.semilogx(df["n"], ratio, "o-", color=C_ACCENT, markersize=5, linewidth=1.9)
    bx.axhline(1.0, color="#666666", linestyle="--", linewidth=1.4)
    bx.fill_between(df["n"], 1.0, ratio, where=(ratio >= 1),
                    color=C_OPENMP, alpha=0.16, interpolate=True)
    bx.fill_between(df["n"], 1.0, ratio, where=(ratio < 1),
                    color=C_PTHREAD, alpha=0.16, interpolate=True)
    style(bx, "", "n (upper bound on the prime search)",
          "T$_{pthreads}$ / T$_{openmp}$")
    bx.xaxis.set_major_formatter(FuncFormatter(millions))

    _, lo, hi = find_crossover(df)
    if lo is not None:
        cross = (lo * hi) ** 0.5          # geometric midpoint, correct on a log axis
        for a in (ax, bx):
            a.axvline(cross, color=C_ACCENT, linestyle="--", linewidth=1.5)
        bx.annotate(f"crossover ≈ {cross/1e3:.0f}K",
                    xy=(cross, 1.0), xytext=(10, -30), textcoords="offset points",
                    fontsize=10, color=C_ACCENT, fontweight="bold")

    peak = ratio.iloc[len(ratio)//3:].max()
    bx.text(0.985, 0.93, f"OpenMP up to {(peak-1)*100:.0f}% faster in the mid-range",
            transform=bx.transAxes, ha="right", va="top",
            fontsize=10, color=C_OPENMP, fontweight="bold")
    bx.text(0.985, 0.06, "below crossover: dynamic-scheduling overhead not yet repaid",
            transform=bx.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=C_PTHREAD)

    save(fig, outdir, "graph7_pthreads_vs_openmp_runtime_n.png")


# ===========================================================================
# Graphs 3, 4, 8 - thread count on the x-axis
# ===========================================================================
def graph3(dt, outdir):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dt["threads"], dt["serial_min_s"], "o--", color=C_SERIAL,
            label="Serial (Task 1, thread-count independent)",
            markersize=5, linewidth=1.6)
    ax.plot(dt["threads"], dt["pthreads_min_s"], "s-", color=C_PTHREAD,
            label="POSIX Threads (Task 2)", markersize=6, linewidth=2.0)
    style(ax, "Graph 3 — Serial vs POSIX Threads runtime with increasing thread count",
          "Threads (p)", f"Runtime (s) at n = {int(dt['n'].iloc[0]):,}")
    ax.set_ylim(bottom=0)
    mark_cores(ax, label_y=0.72)
    ax.legend(frameon=False, loc="upper right")
    save(fig, outdir, "graph3_serial_vs_pthreads_runtime_threads.png")


def graph4(dt, outdir):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dt["threads"], dt["threads"], "--", color=C_IDEAL,
            linewidth=1.5, label="Ideal linear speedup (S = p)")
    ax.plot(dt["threads"], dt["speedup_pthreads"], "s-", color=C_PTHREAD,
            markersize=6, linewidth=2.0, label="POSIX Threads (Task 2)")
    style(ax, "Graph 4 — POSIX Threads speedup with increasing thread count",
          "Threads (p)", "Speedup  S = T_serial / T_parallel")
    ax.set_ylim(0, dt["threads"].max() * 0.62)
    mark_cores(ax, label_y=0.62)

    peak = dt["speedup_pthreads"].max()
    ax.annotate(f"saturates at ≈ {peak:.2f}×\nadding threads past {P_CORES} returns little",
                xy=(TOTAL_CORES, peak), xytext=(20, -34),
                textcoords="offset points", fontsize=10,
                color=C_PTHREAD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PTHREAD, lw=1.3))
    ax.legend(frameon=False, loc="upper left")
    save(fig, outdir, "graph4_pthreads_speedup_threads.png")


def graph8(dt, outdir):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dt["threads"], dt["pthreads_min_s"], "s-", color=C_PTHREAD,
            markersize=6, linewidth=2.0, label="POSIX Threads (Task 2)")
    ax.plot(dt["threads"], dt["openmp_min_s"], "^-", color=C_OPENMP,
            markersize=6, linewidth=2.0, label="OpenMP (Task 3)")
    style(ax, "Graph 8 — POSIX Threads vs OpenMP runtime with increasing thread count",
          "Threads (p)", f"Runtime (s) at n = {int(dt['n'].iloc[0]):,}")
    ax.set_ylim(bottom=0)
    mark_cores(ax)
    ax.annotate("static partitioning becomes erratic\nonce threads outnumber P-cores",
                xy=(10, dt.loc[dt["threads"] == 10, "pthreads_min_s"].iloc[0]),
                xytext=(14, 46), textcoords="offset points",
                fontsize=9.5, color=C_PTHREAD,
                arrowprops=dict(arrowstyle="->", color=C_PTHREAD, lw=1.2))
    ax.legend(frameon=False, loc="upper right")
    save(fig, outdir, "graph8_pthreads_vs_openmp_runtime_threads.png")


# ===========================================================================
# Analysis figures - not required by the spec, but each one answers a
# question a marker is likely to ask.
# ===========================================================================
def a1_speedup_both(df, outdir):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(df["n"], df["speedup_pthreads"], "s-", color=C_PTHREAD,
                markersize=5, linewidth=1.9, label="POSIX Threads")
    ax.semilogx(df["n"], df["speedup_openmp"], "^-", color=C_OPENMP,
                markersize=5, linewidth=1.9, label="OpenMP")
    ax.axhline(TOTAL_CORES, color=C_IDEAL, linestyle="--", linewidth=1.4,
               label=f"Ideal ({TOTAL_CORES} threads)")
    style(ax, "A1 — Speedup of both implementations against n",
          "n", "Speedup")
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_ylim(0, TOTAL_CORES + 1.4)
    ax.legend(frameon=False, loc="lower right")
    save(fig, outdir, "A1_speedup_both_vs_n.png")


def a2_efficiency(dt, outdir):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dt["threads"], dt["efficiency_pthreads"], "s-", color=C_PTHREAD,
            markersize=6, linewidth=2.0, label="POSIX Threads")
    ax.plot(dt["threads"], dt["efficiency_openmp"], "^-", color=C_OPENMP,
            markersize=6, linewidth=2.0, label="OpenMP")
    ax.axhline(1.0, color=C_IDEAL, linestyle="--", linewidth=1.4,
               label="Ideal efficiency (E = 1)")
    style(ax, "A2 — Parallel efficiency collapses once threads exceed cores",
          "Threads (p)", "Efficiency  E = S / p")
    ax.set_ylim(0, 1.15)
    mark_cores(ax, label_y=0.97)
    ax.legend(frameon=False, loc="lower left")
    save(fig, outdir, "A2_efficiency_vs_threads.png")


def a3_karp_flatt(dt, outdir):
    """The figure that separates Amdahl's serial fraction from growing overhead."""
    d = dt[dt["threads"] > 1].copy()
    for c in ("karp_flatt_pthreads", "karp_flatt_openmp"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(d["threads"], d["karp_flatt_pthreads"], "s-", color=C_PTHREAD,
            markersize=6, linewidth=2.0, label="POSIX Threads")
    ax.plot(d["threads"], d["karp_flatt_openmp"], "^-", color=C_OPENMP,
            markersize=6, linewidth=2.0, label="OpenMP")
    style(ax, "A3 — Karp–Flatt metric: the limiter changes character at p = 8",
          "Threads (p)", "Experimentally determined serial fraction  e")
    ax.set_ylim(bottom=0)
    mark_cores(ax, label_y=0.97)

    ax.annotate("flat  →  fixed sequential\nfraction (Amdahl)",
                xy=(5, d["karp_flatt_openmp"].iloc[3]), xytext=(-6, 44),
                textcoords="offset points", fontsize=9.5, ha="center",
                color="#444444",
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))
    ax.annotate("rising  →  overhead grows\nwith thread count",
                xy=(20, d["karp_flatt_openmp"].iloc[-2]), xytext=(-30, -52),
                textcoords="offset points", fontsize=9.5, ha="center",
                color="#444444",
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))
    ax.legend(frameon=False, loc="upper left")
    save(fig, outdir, "A3_karp_flatt_vs_threads.png")


def a4_marginal_gain(dt, outdir):
    """Marginal speedup per added thread - the clearest view of P vs E cores."""
    d = dt[dt["threads"] <= TOTAL_CORES].sort_values("threads")
    p = d["threads"].to_numpy()
    s = d["speedup_openmp"].to_numpy()
    gain = np.diff(s) / np.diff(p)
    at = p[1:]
    colours = [C_PTHREAD if t <= P_CORES else C_OPENMP for t in at]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(at, gain, color=colours, width=0.66, zorder=2)
    for x, g in zip(at, gain):
        ax.text(x, g + 0.02, f"{g:.2f}", ha="center", fontsize=9.5)

    mp = gain[at <= P_CORES].mean()
    me = gain[at > P_CORES].mean()
    ax.axhline(mp, color=C_PTHREAD, linestyle="--", linewidth=1.4)
    ax.axhline(me, color=C_OPENMP, linestyle="--", linewidth=1.4)
    top = max(gain) * 1.35
    ax.text(2.0, top * 0.955, f"threads 2–{P_CORES}:  {mp:.2f}× each",
            color=C_PTHREAD, fontsize=11, fontweight="bold", va="top")
    ax.text(2.0, top * 0.875,
            f"threads {P_CORES+1}–{TOTAL_CORES}:  {me:.2f}× each   →  {mp/me:.1f}× less per thread",
            color=C_OPENMP, fontsize=11, fontweight="bold", va="top")
    ax.set_xticks(at)

    style(ax, "A4 — Marginal speedup contributed by each additional thread (OpenMP)",
          "Thread added", "Δ speedup per thread")
    ax.set_ylim(0, top)
    save(fig, outdir, "A4_marginal_gain_per_thread.png")


def a5_imbalance(df, outdir):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(df["n"], df["imbalance_pthreads"], "s-", color=C_PTHREAD,
                markersize=5, linewidth=1.9, label="POSIX Threads (block-cyclic, static)")
    ax.semilogx(df["n"], df["imbalance_openmp"], "^-", color=C_OPENMP,
                markersize=5, linewidth=1.9, label="OpenMP (schedule(dynamic))")
    ax.axhline(1.0, color=C_IDEAL, linestyle="--", linewidth=1.4,
               label="Perfect balance (1.0)")
    style(ax, "A5 — Load imbalance: dynamic self-scheduling converges, static does not",
          "n", "Imbalance  (slowest thread / mean thread)")
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.annotate("both schemes peak together at small n\n→ common cause: thread-team startup stagger,\nnot the partitioning scheme",
                xy=(2.4e5, df["imbalance_pthreads"].max()), xytext=(10, -66),
                textcoords="offset points", fontsize=9.5, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))
    ax.legend(frameon=False, loc="upper right")
    save(fig, outdir, "A5_load_imbalance_vs_n.png")


def a6_measurement_quality(df, outdir):
    """Justifies the methodology: shows exactly where the data is trustworthy."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for c, colour, lab in (("serial_cv_pct", C_SERIAL, "Serial"),
                           ("pthreads_cv_pct", C_PTHREAD, "POSIX Threads"),
                           ("openmp_cv_pct", C_OPENMP, "OpenMP")):
        ax.semilogx(df["n"], df[c], "o-", color=colour, markersize=4,
                    linewidth=1.6, label=lab)
    ax.axhline(10, color=C_ACCENT, linestyle="--", linewidth=1.5)
    ax.text(df["n"].max(), 10.6, "10% threshold", ha="right",
            fontsize=10, color=C_ACCENT, fontweight="bold")
    style(ax, "A6 — Measurement stability: dispersion falls as runtime grows",
          "n", "Coefficient of variation (%)")
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.annotate("sub-millisecond runs:\ntimer granularity and scheduler\nquantum dominate",
                xy=(1.6e5, df["serial_cv_pct"].iloc[:4].max()), xytext=(24, 12),
                textcoords="offset points", fontsize=9.5, color="#555555")
    ax.legend(frameon=False, loc="upper right")
    save(fig, outdir, "A6_measurement_stability.png")


# ===========================================================================
def talking_points(df, dt):
    """Print the derived numbers worth quoting in the presentation."""
    print("\n" + "=" * 68)
    print("KEY FIGURES  (quote these; they come straight from the CSVs)")
    print("=" * 68)

    big = df.iloc[-1]
    print(f"\nLargest problem size, n = {int(big['n']):,}")
    print(f"  serial   {big['serial_min_s']:.4f} s")
    print(f"  pthreads {big['pthreads_min_s']:.4f} s   S = {big['speedup_pthreads']:.2f}x")
    print(f"  openmp   {big['openmp_min_s']:.4f} s   S = {big['speedup_openmp']:.2f}x")

    d = dt[dt["threads"] <= TOTAL_CORES].sort_values("threads")
    g = np.diff(d["speedup_openmp"].to_numpy()) / np.diff(d["threads"].to_numpy())
    at = d["threads"].to_numpy()[1:]
    mp, me = g[at <= P_CORES].mean(), g[at > P_CORES].mean()
    print(f"\nMarginal speedup per added thread (OpenMP)")
    print(f"  threads 2-{P_CORES}   {mp:.3f}x each")
    print(f"  threads {P_CORES+1}-{TOTAL_CORES}  {me:.3f}x each   ({mp/me:.1f}x less)")
    print(f"  predicted ceiling {P_CORES}*{mp:.2f} + {E_CORES}*{me:.2f} = {P_CORES*mp + E_CORES*me:.2f}x")
    print(f"  measured at p={TOTAL_CORES}: "
          f"{d.loc[d['threads']==TOTAL_CORES,'speedup_openmp'].iloc[0]:.2f}x")

    p1 = dt[dt["threads"] == 1]
    if not p1.empty:
        print(f"\nSingle-thread cost of the parallel construct (p = 1)")
        print(f"  pthreads S = {p1['speedup_pthreads'].iloc[0]:.4f}  "
              f"-> {(1-p1['speedup_pthreads'].iloc[0])*100:+.1f}% vs serial")
        print(f"  openmp   S = {p1['speedup_openmp'].iloc[0]:.4f}  "
              f"-> {(1-p1['speedup_openmp'].iloc[0])*100:+.1f}% vs serial")

    _, lo, hi = find_crossover(df)
    if lo is not None:
        print(f"\nCrossover: OpenMP overtakes POSIX Threads between "
              f"n = {int(lo):,} and n = {int(hi):,}")
        print("  (final crossing; earlier flips at small n are within noise)")

    noisy = df[df[["serial_cv_pct", "pthreads_cv_pct", "openmp_cv_pct"]].max(axis=1) > 15]
    print(f"\nPoints with cv > 15%: {len(noisy)} of {len(df)}"
          + (f"  (n = {', '.join(f'{int(v):,}' for v in noisy['n'])})" if len(noisy) else ""))
    print(f"All are below n = {int(noisy['n'].max()):,}" if len(noisy) else "")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--scaling-n", default="scaling_n.csv")
    ap.add_argument("--scaling-threads", default="scaling_threads.csv")
    args = ap.parse_args()

    for f in (args.scaling_n, args.scaling_threads):
        if not os.path.exists(f):
            sys.exit(f"ERROR: {f} not found. Run ./run_sweep.sh first.")

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.scaling_n).sort_values("n").reset_index(drop=True)
    dt = pd.read_csv(args.scaling_threads).sort_values("threads").reset_index(drop=True)
    print(f"Loaded {len(df)} problem sizes and {len(dt)} thread counts.\n")

    print("Required graphs:")
    runtime_vs_n(df, args.outdir, ("pthreads_min_s", C_PTHREAD, "POSIX Threads (Task 2)"),
                 "graph1_serial_vs_pthreads_runtime_n.png",
                 "Graph 1 — Serial vs POSIX Threads runtime with increasing n")
    speedup_vs_n(df, args.outdir, "speedup_pthreads", C_PTHREAD,
                 "graph2_pthreads_speedup_n.png",
                 "Graph 2 — POSIX Threads speedup with increasing n", "POSIX Threads (Task 2)")
    graph3(dt, args.outdir)
    graph4(dt, args.outdir)
    runtime_vs_n(df, args.outdir, ("openmp_min_s", C_OPENMP, "OpenMP (Task 3)"),
                 "graph5_serial_vs_openmp_runtime_n.png",
                 "Graph 5 — Serial vs OpenMP runtime with increasing n")
    speedup_vs_n(df, args.outdir, "speedup_openmp", C_OPENMP,
                 "graph6_openmp_speedup_n.png",
                 "Graph 6 — OpenMP speedup with increasing n", "OpenMP (Task 3)")
    graph7(df, args.outdir)
    graph8(dt, args.outdir)

    print("\nAnalysis figures (appendix):")
    a1_speedup_both(df, args.outdir)
    a2_efficiency(dt, args.outdir)
    a3_karp_flatt(dt, args.outdir)
    a4_marginal_gain(dt, args.outdir)
    a5_imbalance(df, args.outdir)
    a6_measurement_quality(df, args.outdir)

    talking_points(df, dt)


if __name__ == "__main__":
    main()