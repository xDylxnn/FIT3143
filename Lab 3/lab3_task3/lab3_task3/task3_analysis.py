"""
task3_analysis.py - FIT3143 Lab 3, Task 3: communication delay analysis

Input : task3_results.csv  (one row per CAAS run, appended by the base station)
Output: figures/  graphs for the slides
        tables/   CSV + Markdown tables, trend fits, and a findings summary

Usage : pip install pandas numpy matplotlib
        python3 task3_analysis.py [task3_results.csv]
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "task3_results.csv"
MAIN_MODE = 3            # base station version used for the main Task 3 graphs
CORES_PER_MACHINE = 16   # CAAS job scripts use --ntasks-per-node=16
REQUIRED_SIZES = 30      # HD: at least 30 grid sizes and 30 line sizes

# (column prefix, title, column holding the number of messages measured)
MSG_TYPES = [
    ("neighbour", "Neighbour exchange (node <-> node)", "neighbour_msgs"),
    ("report",    "Report (node -> base)",              "alerts"),
    ("reply",     "Reply (base -> node)",               "reply_msgs"),
    ("terminate", "Terminate (base -> node)",           None),
]
COLOURS = {1: "tab:blue", 2: "tab:red"}
MACHINE_LABEL = {1: "1 machine (shared memory)", 2: "2 machines (network)"}

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)


# ---------------------------------------------------------------- 1. load
def load(path):
    df = pd.read_csv(path)
    df["topology"] = np.where(df["m"] == 1, "line", "grid")
    df["size"] = df["m"].astype(str) + "x" + df["n"].astype(str)
    # +1 for the base station process: more than 16 processes -> 2 machines
    df["machines"] = np.where(df["nodes"] + 1 > CORES_PER_MACHINE, 2, 1)
    # A run that sent no messages of a type has NO delay for it (not 0 ms).
    for t, _, count_col in MSG_TYPES:
        if count_col:
            cols = [c for c in (f"{t}_avg_ms", f"{t}_max_ms") if c in df]
            df.loc[df[count_col] == 0, cols] = np.nan
    return df


# --------------------------------------------------- 2. check completeness
def check(df, out):
    out.append("DATA CHECK")
    for topo in ("grid", "line"):
        sizes = df[df.topology == topo]["size"].nunique()
        status = "OK" if sizes >= REQUIRED_SIZES else f"ONLY {sizes} - need {REQUIRED_SIZES}"
        out.append(f"  {topo:4s}: {sizes} distinct sizes, "
                   f"{(df.topology == topo).sum()} runs  [{status}]")
    for t, _, count_col in MSG_TYPES:
        if not count_col:
            continue
        missing = df[df[count_col] == 0]
        if len(missing):
            out.append(f"  {t}: no {t} messages in {len(missing)} run(s) "
                       f"({', '.join(missing['size'])}) -> shown as gaps, not as 0")
    out.append("")


# ---------------------------------------------- 3. average repeated runs
def aggregate(df):
    """One row per topology/size; repeated runs are averaged (std kept)."""
    agg = {"nodes": "first", "machines": "first", "mode": "size"}
    for t, _, count_col in MSG_TYPES:
        agg[f"{t}_avg_ms"] = ["mean", "std"]
        if f"{t}_max_ms" in df:
            agg[f"{t}_max_ms"] = "max"
        if count_col:
            agg[count_col] = "sum"
    g = df.groupby(["topology", "size"]).agg(agg)
    g.columns = ["_".join(c).rstrip("_") if c[1] not in ("first", "max", "sum", "mean")
                 else c[0] for c in g.columns]
    g = g.rename(columns={"mode_size": "runs"}).reset_index()
    return g.sort_values(["topology", "nodes", "size"])


# ------------------------------------------------------ 4. trend fitting
def fit(x, y):
    """Least-squares line y = a*x + b. Returns slope (us/node), intercept, R^2."""
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - pred) ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
    return a * 1000, b, r2, len(x)   # slope ms/node -> us/node


def trends(g):
    rows = []
    for topo in ("grid", "line"):
        d = g[g.topology == topo]
        for t, _, _ in MSG_TYPES:
            for seg, part in [("1 machine", d[d.machines == 1]),
                              ("2 machines", d[d.machines == 2]),
                              ("all", d)]:
                r = fit(part["nodes"].to_numpy(float), part[f"{t}_avg_ms"].to_numpy(float))
                if r:
                    rows.append({"topology": topo, "message": t, "segment": seg,
                                 "slope_us_per_node": round(r[0], 3),
                                 "intercept_ms": round(r[1], 4),
                                 "r2": round(r[2], 3), "points": r[3]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 5. plots
def plot_topology(g, topo, tr):
    d = g[g.topology == topo]
    if d.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (t, title, _) in zip(axes.flat, MSG_TYPES):
        for mach in (1, 2):
            part = d[d.machines == mach]
            if part.empty:
                continue
            yerr = part.get(f"{t}_avg_ms_std")
            ax.errorbar(part["nodes"], part[f"{t}_avg_ms"], yerr=yerr, fmt="o",
                        color=COLOURS[mach], capsize=3, label=f"average - {MACHINE_LABEL[mach]}")
            if f"{t}_max_ms" in part:
                ax.plot(part["nodes"], part[f"{t}_max_ms"], "x", color=COLOURS[mach],
                        alpha=0.45, label=f"max - {MACHINE_LABEL[mach]}")
            row = tr[(tr.topology == topo) & (tr.message == t) &
                     (tr.segment == ("1 machine" if mach == 1 else "2 machines"))]
            if len(row):
                r = row.iloc[0]
                xs = np.array([part["nodes"].min(), part["nodes"].max()], float)
                ax.plot(xs, r.slope_us_per_node / 1000 * xs + r.intercept_ms, "--",
                        color=COLOURS[mach],
                        label=f"trend: {r.slope_us_per_node:+.2f} us/node (R²={r.r2:.2f})")
        ax.axvline(CORES_PER_MACHINE - 0.5, color="grey", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("number of EV charging nodes")
        ax.set_ylabel("delay (ms)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    name = "EV grid size (m x n)" if topo == "grid" else "line topology size (1 x n)"
    fig.suptitle(f"Communication delay vs {name} - CAAS, {d['size'].nunique()} sizes, "
                 f"{int(d['runs'].sum())} runs", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"figures/task3_{topo}.png", dpi=150)
    plt.close(fig)


def plot_grid_vs_line(g):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (t, title, _) in zip(axes, MSG_TYPES):
        for topo, marker in (("grid", "o-"), ("line", "s--")):
            d = g[g.topology == topo]
            ax.plot(d["nodes"], d[f"{t}_avg_ms"], marker, markersize=4, label=topo)
        ax.axvline(CORES_PER_MACHINE - 0.5, color="grey", linestyle=":", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("number of nodes")
        ax.set_ylabel("average delay (ms)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Grid vs line topology at the same number of nodes")
    fig.tight_layout()
    fig.savefig("figures/task3_grid_vs_line.png", dpi=150)
    plt.close(fig)


def plot_versions(df):
    """Task 2 HD: compare base station versions (only if several were run)."""
    modes = sorted(df["mode"].unique())
    if len(modes) < 2:
        return None
    sizes = sorted(df[df["mode"] != MAIN_MODE]["size"].unique(),
                   key=lambda s: (s.startswith("1x"), int(s.split("x")[0]) * int(s.split("x")[1])))
    d = df[df["size"].isin(sizes)]
    table = d.groupby(["size", "mode"])[["report_avg_ms", "reply_avg_ms"]].mean().unstack("mode")
    table = table.reindex(sizes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    width = 0.8 / len(modes)
    names = {1: "v1 blocking", 2: "v2 nonblocking", 3: "v3 Testsome"}
    for ax, t in zip(axes, ("report", "reply")):
        for i, mode in enumerate(modes):
            ax.bar(np.arange(len(sizes)) + i * width, table[(f"{t}_avg_ms", mode)],
                   width, label=names.get(mode, f"v{mode}"))
        ax.set_xticks(np.arange(len(sizes)) + width * (len(modes) - 1) / 2)
        ax.set_xticklabels(sizes, rotation=45)
        ax.set_title(f"{t.capitalize()} delay by base station version")
        ax.set_ylabel("average delay (ms)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("figures/task2_versions.png", dpi=150)
    plt.close(fig)
    table.columns = [f"{t}_v{m}" for t, m in table.columns]
    return table


# ------------------------------------------------------- 6. tables/findings
def write_tables(g, tr, versions):
    cols = ["size", "nodes", "machines", "runs"]
    for t, _, _ in MSG_TYPES:
        cols += [c for c in (f"{t}_avg_ms", f"{t}_max_ms") if c in g]
    with open("tables/task3_tables.md", "w") as md:
        for topo in ("grid", "line"):
            d = g[g.topology == topo][cols]
            d.to_csv(f"tables/task3_{topo}_table.csv", index=False, float_format="%.4f")
            md.write(f"## Communication delay (ms) - {topo}\n\n")
            md.write(d.to_markdown(index=False, floatfmt=".3f") + "\n\n")
        md.write("## Trend fits (average delay vs number of nodes)\n\n")
        md.write(tr.to_markdown(index=False) + "\n\n")
        if versions is not None:
            md.write("## Base station versions (average delay, ms)\n\n")
            md.write(versions.to_markdown(floatfmt=".4f") + "\n")
    tr.to_csv("tables/task3_trends.csv", index=False)


def findings(g, tr, out):
    out.append("FINDINGS (numbers to quote when answering the two questions)")
    for topo, q in (("grid", "a) grid size"), ("line", "b) line topology")):
        d = g[g.topology == topo]
        if d.empty:
            continue
        out.append(f"\n{q}: {d['nodes'].min()}-{d['nodes'].max()} nodes")
        for t, _, _ in MSG_TYPES:
            one = d[d.machines == 1][f"{t}_avg_ms"].mean()
            two = d[d.machines == 2][f"{t}_avg_ms"].mean()
            line = f"  {t:9s}: avg {one:.4f} ms on 1 machine"
            if not np.isnan(two):
                line += f", {two:.4f} ms on 2 machines (x{two / one:.1f})"
            r = tr[(tr.topology == topo) & (tr.message == t) & (tr.segment == "1 machine")]
            if len(r):
                r = r.iloc[0]
                shape = ("no clear trend with size" if r.r2 < 0.3
                         else "grows with size" if r.slope_us_per_node > 0 else "falls with size")
                line += (f"; within 1 machine {r.slope_us_per_node:+.2f} us/node "
                         f"(R²={r.r2:.2f}) -> {shape}")
            out.append(line)


def main():
    df = load(CSV_PATH)
    out = []
    check(df, out)
    versions = plot_versions(df)
    main_df = df[df["mode"] == MAIN_MODE] if (df["mode"] == MAIN_MODE).any() else df
    g = aggregate(main_df)
    tr = trends(g)
    plot_topology(g, "grid", tr)
    plot_topology(g, "line", tr)
    plot_grid_vs_line(g)
    write_tables(g, tr, versions)
    findings(g, tr, out)
    text = "\n".join(out)
    print(text)
    with open("tables/task3_findings.txt", "w") as f:
        f.write(text + "\n")
    print("\nSaved figures/ and tables/")


if __name__ == "__main__":
    main()
