#!/usr/bin/env python3
"""
mpi_bench.py
============

Empirical evaluation + theoretical analysis (Amdahl's law) harness for a
parallel prime-counting assignment.

Problem sizes are capped at 10,000,000 by default (--n-cap). Because the size is
fixed rather than grown with the machine, Amdahl's law is the applicable model;
Gustafson's scaled-size law is deliberately not used.

It builds and benchmarks four programs:

    serial   Week 4 Task 1   (baseline for every speed-up in this report)
    pthread  Week 4 Task 2   (POSIX threads / OpenMP shared-memory version)
    mpi      This week Task 1 (Open MPI)
    hybrid   This week Task 2 (Open MPI + OpenMP)

and produces:

    results/<label>/raw_runs.csv        one row per individual run
    results/<label>/aggregate.csv       one row per (impl, n, procs, threads)
    results/<label>/theory.csv          Amdahl fractions, curves, Karp-Flatt
    results/<label>/sysinfo.json        machine + toolchain description
    results/<label>/figures/fig1..7.png the seven required graphs (+ extras)
    results/<label>/report.md           tables and a data-driven discussion

Quick start
-----------
    python3 mpi_bench.py --quick            # ~5 min sanity run
    python3 mpi_bench.py                    # full run (auto-calibrated)
    python3 mpi_bench.py --plots-only       # redraw figures from existing CSV

Source files are auto-detected in --src-dir by what they #include:
    mpi.h + omp.h -> hybrid, mpi.h -> mpi, pthread.h/omp.h -> pthread,
    neither -> serial.  Override with --serial/--pthread/--mpi/--hybrid.

Requires: gcc, mpicc, mpirun, python3 + numpy + matplotlib.
Tested on Linux and macOS; on Windows use WSL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Constants and small helpers
# --------------------------------------------------------------------------

IMPLS = ("serial", "pthread", "mpi", "hybrid")

LABEL = {
    "serial":  "Serial (Week 4 Task 1)",
    "pthread": "POSIX threads (Week 4 Task 2)",
    "mpi":     "Open MPI (Task 1)",
    "hybrid":  "Hybrid MPI+OpenMP (Task 2)",
}

COLOR = {"serial": "#444444", "pthread": "#1f77b4", "mpi": "#d62728", "hybrid": "#2ca02c"}
MARKER = {"serial": "o", "pthread": "s", "mpi": "^", "hybrid": "D"}

# Output parsers -- these match the printf() formats in the four programs.
RE_PRIMES       = re.compile(r"primes=(\d+)")
RE_SERIAL_TIME  = re.compile(r"Time taken:\s*([0-9.]+)")
RE_PTHREAD_TIME = re.compile(r"\btime=([0-9.]+)\s*s")
RE_MPI_REGION   = re.compile(r"Search \+ gather time:\s*([0-9.]+)")
RE_MPI_COMPUTE  = re.compile(r"Slowest local search:\s*([0-9.]+)")
RE_IMBALANCE    = re.compile(r"Imbalance \(slowest/average\)\s*=\s*([0-9.]+)")
RE_BUSY         = re.compile(r"busy=([0-9.]+)")

# The prime test costs ~sqrt(n) per candidate, so total work grows ~ n^1.5.
# Used to calibrate the problem sizes to this machine's speed.
WORK_EXPONENT = 1.5

RAW_FIELDS = [
    "timestamp", "experiment", "label", "impl", "n", "procs", "threads",
    "workers", "rep", "wall_s", "region_s", "compute_s", "primes",
    "imbalance", "rc",
]


def say(msg: str = "") -> None:
    print(msg, flush=True)


def hr(title: str) -> None:
    say("")
    say("=" * 74)
    say(title)
    say("=" * 74)


def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def stat_of(xs, how):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    if how == "min":
        return min(xs)
    if how == "mean":
        return statistics.fmean(xs)
    return statistics.median(xs)


def geom_ints(lo: int, hi: int, count: int):
    """`count` geometrically spaced integers in [lo, hi], rounded to 3 s.f."""
    lo, hi = int(lo), int(hi)
    if count < 2 or hi <= lo:
        return [lo]
    out = []
    for i in range(count):
        v = lo * (hi / lo) ** (i / (count - 1))
        mag = 10 ** max(0, int(math.floor(math.log10(v))) - 2)
        out.append(int(round(v / mag) * mag))
    out = sorted(set(out))
    # Rounding can collide; nudge duplicates upward so we keep `count` points.
    while len(out) < count and out:
        gaps = [(out[i + 1] - out[i], i) for i in range(len(out) - 1)]
        gaps.sort(reverse=True)
        if not gaps or gaps[0][0] < 2:
            break
        g, i = gaps[0]
        out.insert(i + 1, out[i] + g // 2)
        out = sorted(set(out))
    return out


# --------------------------------------------------------------------------
# Machine / toolchain description
# --------------------------------------------------------------------------

def which(name):
    return shutil.which(name)


def run_text(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except Exception:
        return ""


def physical_cores():
    """Physical (not hyper-threaded) core count, or None if undetectable."""
    try:
        if sys.platform.startswith("linux"):
            seen = set()
            phys = core = None
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        seen.add((phys, core))
                    phys = core = None
            if phys is not None and core is not None:
                seen.add((phys, core))
            if seen:
                return len(seen)
        elif sys.platform == "darwin":
            out = run_text(["sysctl", "-n", "hw.physicalcpu"])
            if out.isdigit():
                return int(out)
    except Exception:
        pass
    return None


def cpu_model():
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif sys.platform == "darwin":
        out = run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    return platform.processor() or "unknown"


def mem_available_bytes():
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) * 1024
        elif sys.platform == "darwin":
            out = run_text(["sysctl", "-n", "hw.memsize"])
            if out.isdigit():
                return int(out) // 2
    except Exception:
        pass
    return 2 * 1024 ** 3


def collect_sysinfo(cfg) -> dict:
    logical = os.cpu_count() or 1
    return {
        "label": cfg.label,
        "hostname": platform.node(),
        "collected": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "logical_cores": logical,
        "physical_cores": physical_cores(),
        "mem_available_gb": round(mem_available_bytes() / 1024 ** 3, 2),
        "python": sys.version.split()[0],
        "cc": run_text([cfg.cc, "--version"]).splitlines()[:1],
        "mpicc": run_text([cfg.mpicc, "--version"]).splitlines()[:1],
        "mpirun": run_text([cfg.mpirun, "--version"]).splitlines()[:1],
        "mpirun_flags": cfg.mpi_flags,
        "hostfile": cfg.hostfile,
    }


# --------------------------------------------------------------------------
# Source discovery and build
# --------------------------------------------------------------------------

def classify_source(path: Path):
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None
    has_mpi = "mpi.h" in text
    has_omp = "omp.h" in text
    has_pth = "pthread.h" in text
    if has_mpi and has_omp:
        return "hybrid"
    if has_mpi:
        return "mpi"
    if has_pth or has_omp:
        return "pthread"
    if "main(" in text:
        return "serial"
    return None


def discover_sources(cfg) -> dict:
    found = {}
    for impl in IMPLS:
        override = getattr(cfg, impl)
        if override:
            p = Path(override)
            if not p.exists():
                sys.exit(f"error: --{impl} {p} does not exist")
            found[impl] = p

    src_dir = Path(cfg.src_dir)
    if not src_dir.exists():
        sys.exit(f"error: --src-dir {src_dir} does not exist")

    candidates = sorted(src_dir.rglob("*.c"))
    for path in candidates:
        kind = classify_source(path)
        if kind and kind not in found:
            found[kind] = path
        elif kind and found.get(kind) != path:
            # Keep the first match but warn so mistakes are visible.
            say(f"  note: ignoring extra {kind} candidate {path}")

    missing = [i for i in IMPLS if i not in found]
    if missing:
        say("")
        say("Sources found:")
        for impl, p in found.items():
            say(f"  {impl:8s} {p}")
        say("")
        say(f"Could not classify: {', '.join(missing)}")
        say("Pass them explicitly, e.g. --serial week4/task1.c --mpi task1.c")
        sys.exit(1)
    return found


def build_all(cfg, sources: dict) -> dict:
    bindir = Path(cfg.workdir) / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    base = ["-O2", "-std=gnu11", "-Wall"]
    recipes = {
        "serial":  [[cfg.cc] + base],
        "pthread": [[cfg.cc] + base + ["-pthread", "-fopenmp"],
                    [cfg.cc] + base + ["-pthread"]],
        "mpi":     [[cfg.mpicc] + base],
        "hybrid":  [[cfg.mpicc] + base + ["-fopenmp"],
                    [cfg.mpicc] + base + ["-Xpreprocessor", "-fopenmp", "-lomp"]],
    }
    binaries = {}
    for impl in IMPLS:
        out = bindir / impl
        src = sources[impl]
        if cfg.skip_build and out.exists():
            binaries[impl] = out
            continue
        last = ""
        for recipe in recipes[impl]:
            cmd = recipe + ["-o", str(out), str(src)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0:
                say(f"  built {impl:8s} <- {src}")
                if p.stderr.strip():
                    for line in p.stderr.strip().splitlines()[:6]:
                        say(f"           warning: {line}")
                binaries[impl] = out
                break
            last = p.stderr or p.stdout
        else:
            say(f"  FAILED to build {impl} from {src}:")
            say(last[:2000])
            sys.exit(1)
    return binaries


# --------------------------------------------------------------------------
# mpirun flag probing
# --------------------------------------------------------------------------

def probe_mpi_flags(cfg):
    """Keep only the mpirun flags this installation actually accepts."""
    flags = []
    candidates = []
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        candidates.append(["--allow-run-as-root"])
    candidates += [["--oversubscribe"], ["--bind-to", "none"]]
    probe_target = "/bin/true" if Path("/bin/true").exists() else "/usr/bin/true"
    for cand in candidates:
        cmd = [cfg.mpirun] + flags + cand + ["-np", "1", probe_target]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if p.returncode == 0:
                flags += cand
        except Exception:
            pass
    # --bind-to none is only wanted for the hybrid runs; keep it separate.
    base = [f for f in flags if f not in ("--bind-to", "none")]
    bind_none = ["--bind-to", "none"] if "--bind-to" in flags else []
    return base, bind_none


# --------------------------------------------------------------------------
# The measurement harness
# --------------------------------------------------------------------------

class Harness:
    def __init__(self, cfg, binaries, outdir: Path):
        self.cfg = cfg
        self.bin = binaries
        self.outdir = outdir
        self.scratch = Path(cfg.scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        for impl in IMPLS:
            (self.scratch / impl).mkdir(exist_ok=True)
        self.raw_path = outdir / "raw_runs.csv"
        new = not self.raw_path.exists()
        self.raw_fh = open(self.raw_path, "a", newline="")
        self.raw_csv = csv.DictWriter(self.raw_fh, fieldnames=RAW_FIELDS)
        if new:
            self.raw_csv.writeheader()
            self.raw_fh.flush()
        self.store = {}          # (impl,n,procs,threads) -> list of run dicts
        self.failures = []
        self.run_count = 0
        self.spent = 0.0
        if cfg.resume:
            self._load_previous()

    # -- persistence -------------------------------------------------------
    def _load_previous(self):
        if not self.raw_path.exists():
            return
        loaded = 0
        with open(self.raw_path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    if int(row["rc"]) != 0:
                        continue
                    key = (row["impl"], int(row["n"]), int(row["procs"]), int(row["threads"]))
                    rec = {
                        "wall_s": float(row["wall_s"]),
                        "region_s": float(row["region_s"]) if row["region_s"] else None,
                        "compute_s": float(row["compute_s"]) if row["compute_s"] else None,
                        "primes": int(row["primes"]) if row["primes"] else None,
                        "imbalance": float(row["imbalance"]) if row["imbalance"] else None,
                        "exp": row.get("experiment", ""),
                    }
                    self.store.setdefault(key, []).append(rec)
                    loaded += 1
                except Exception:
                    continue
        if loaded:
            say(f"  resumed {loaded} previous runs from {self.raw_path}")

    def close(self):
        self.raw_fh.close()

    # -- command construction ---------------------------------------------
    def command(self, impl, n, procs, threads):
        exe = str(self.bin[impl].resolve())
        if impl == "serial":
            return [exe, str(n)], {}
        if impl == "pthread":
            return [exe, str(n), str(threads)], {"OMP_NUM_THREADS": str(threads)}
        cmd = [self.cfg.mpirun] + list(self.cfg.mpi_flags)
        if impl == "hybrid":
            cmd += list(self.cfg.bind_none_flags)
        if self.cfg.hostfile:
            cmd += ["--hostfile", self.cfg.hostfile]
        extra = self.cfg.mpi_extra.split() if self.cfg.mpi_extra else []
        cmd += extra + ["-np", str(procs), exe, str(n)]
        env = {}
        if impl == "hybrid":
            cmd += [str(threads)]
            env["OMP_NUM_THREADS"] = str(threads)
        return cmd, env

    # -- a single timed execution -----------------------------------------
    def run_once(self, impl, n, procs, threads, experiment, rep):
        cmd, extra_env = self.command(impl, n, procs, threads)
        env = dict(os.environ)
        env.update(extra_env)
        cwd = self.scratch / impl
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd),
                               env=env, timeout=self.cfg.timeout)
            wall = time.perf_counter() - t0
            out, rc = p.stdout, p.returncode
            err = p.stderr
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - t0
            out, rc, err = "", 124, f"timeout after {self.cfg.timeout}s"
        self.run_count += 1
        self.spent += wall

        parsed = {"primes": None, "region_s": None, "compute_s": None, "imbalance": None}
        if rc == 0:
            parsed = parse_output(impl, out)
        else:
            self.failures.append((impl, n, procs, threads, rc, (err or out)[:200]))

        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment": experiment, "label": self.cfg.label, "impl": impl,
            "n": n, "procs": procs, "threads": threads, "workers": procs * threads,
            "rep": rep, "wall_s": f"{wall:.6f}",
            "region_s": "" if parsed["region_s"] is None else f"{parsed['region_s']:.6f}",
            "compute_s": "" if parsed["compute_s"] is None else f"{parsed['compute_s']:.6f}",
            "primes": "" if parsed["primes"] is None else parsed["primes"],
            "imbalance": "" if parsed["imbalance"] is None else f"{parsed['imbalance']:.4f}",
            "rc": rc,
        }
        self.raw_csv.writerow(row)
        self.raw_fh.flush()

        rec = {"wall_s": wall, "exp": experiment, **parsed}
        return rec, rc, (err or out)

    # -- repeated, cached measurement -------------------------------------
    def measure(self, impl, n, procs=1, threads=1, experiment="", reps=None, quiet=False):
        reps = reps or self.cfg.reps
        key = (impl, int(n), int(procs), int(threads))
        have = self.store.get(key, [])
        need = reps - len(have)
        for r in range(need):
            rec, rc, _ = self.run_once(impl, n, procs, threads, experiment, len(have) + r + 1)
            if rc == 0:
                self.store.setdefault(key, []).append(rec)
            else:
                say(f"    ! {impl} n={n} P={procs} T={threads} failed (rc={rc})")
                break
        runs = self.store.get(key, [])
        agg = aggregate(impl, n, procs, threads, runs, self.cfg.stat)
        if not quiet:
            wall = agg["wall"]
            say(f"    {impl:8s} n={n:<12d} P={procs:<3d} T={threads:<3d} "
                f"wall={wall:8.3f}s  compute={_f(agg['compute'])}  reps={len(runs)}")
        return agg


def _f(x):
    return "   n/a  " if x is None else f"{x:8.3f}s"


def parse_output(impl, out):
    d = {}
    m = RE_PRIMES.search(out)
    d["primes"] = int(m.group(1)) if m else None
    if impl == "serial":
        m = RE_SERIAL_TIME.search(out)
        t = float(m.group(1)) if m else None
        d["region_s"] = t
        d["compute_s"] = t
    elif impl == "pthread":
        m = RE_PTHREAD_TIME.search(out)
        d["region_s"] = float(m.group(1)) if m else None
        busies = [float(x) for x in RE_BUSY.findall(out)]
        d["compute_s"] = max(busies) if busies else d["region_s"]
    else:
        m = RE_MPI_REGION.search(out)
        d["region_s"] = float(m.group(1)) if m else None
        m = RE_MPI_COMPUTE.search(out)
        d["compute_s"] = float(m.group(1)) if m else None
    m = RE_IMBALANCE.search(out)
    d["imbalance"] = float(m.group(1)) if m else None
    return d


def aggregate(impl, n, procs, threads, runs, how):
    ok = bool(runs)
    return {
        "impl": impl, "n": n, "procs": procs, "threads": threads,
        "workers": procs * threads, "reps": len(runs), "ok": ok,
        "wall": stat_of([r["wall_s"] for r in runs], how) if ok else None,
        "wall_min": min([r["wall_s"] for r in runs]) if ok else None,
        "wall_sd": (statistics.pstdev([r["wall_s"] for r in runs]) if len(runs) > 1 else 0.0) if ok else None,
        "region": stat_of([r["region_s"] for r in runs], how) if ok else None,
        "compute": stat_of([r["compute_s"] for r in runs], how) if ok else None,
        "imbalance": stat_of([r["imbalance"] for r in runs], how) if ok else None,
        "primes": next((r["primes"] for r in runs if r["primes"] is not None), None) if ok else None,
    }


# --------------------------------------------------------------------------
# Experiment plan
# --------------------------------------------------------------------------

def worker_values(cores, max_workers):
    """1..cores in full, then a few oversubscribed points beyond cores."""
    vals = set()
    if cores <= 16:
        vals |= set(range(1, cores + 1))
    else:
        vals |= {1, 2, 4}
        vals |= {int(round(cores * f)) for f in (0.25, 0.5, 0.75, 1.0)}
    for f in (1.25, 1.5, 2.0):
        v = int(round(cores * f))
        if v <= max_workers:
            vals.add(v)
    vals.add(max_workers)
    return sorted(v for v in vals if 1 <= v <= max_workers)


def hybrid_configs(cores, max_workers, proc_list=None):
    """(procs, threads) pairs to sweep for the hybrid program."""
    proc_list = proc_list or [p for p in (1, 2, 4, 8) if p <= max(2, cores)]
    thread_list = [t for t in (1, 2, 3, 4, 6, 8, 12, 16) if t <= max_workers]
    pairs = set()
    for p in proc_list:
        for t in thread_list:
            if p * t <= max_workers:
                pairs.add((p, t))
    # Always include the "balanced" diagonal and the two extremes.
    k = 1
    while k * k <= max_workers:
        pairs.add((k, k))
        k += 1
    pairs.add((1, min(cores, max_workers)))
    pairs.add((min(cores, max_workers), 1))
    return sorted(pairs)


def calibrate(h: Harness, cfg, cores):
    """Pick problem sizes from a measured cost model T_serial(n) = c * n^1.5."""
    hr("Calibration: sizing n for this machine")
    probe = 2_000_000
    t = None
    for _ in range(4):
        agg = h.measure("serial", probe, experiment="calibrate", reps=2)
        t = agg["wall"]
        if t is None:
            sys.exit("error: the serial program failed during calibration")
        if t >= 0.25:
            break
        probe *= 3
    c = t / (probe ** WORK_EXPONENT)
    say(f"  serial({probe}) = {t:.3f}s  ->  T(n) = {c:.3e} * n^{WORK_EXPONENT}")

    def n_for(seconds):
        return int(round((seconds / c) ** (1.0 / WORK_EXPONENT)))

    cap_mem = int(mem_available_bytes() * 0.30 / 2)      # flags + gathered copy
    cap = min(cfg.n_cap, cap_mem, 2_000_000_000)
    cfg.n_cap_effective = cap        # every experiment must respect this

    # n is capped (10 million by default), so the top of the sweep and the fixed
    # size both sit at the cap and only the bottom of the sweep is calibrated.
    nmax = min(cfg.nmax or n_for(cfg.target_max_seconds), cap)
    nmin = cfg.nmin or min(n_for(cfg.target_min_seconds), max(1000, nmax // 4))
    n_fixed = min(cfg.fixed_n or n_for(cfg.target_fixed_seconds), cap)

    t_lo, t_hi = c * nmin ** WORK_EXPONENT, c * nmax ** WORK_EXPONENT
    say(f"  cap on n: {cap:,}  (--n-cap, memory, and the int in test_primes)")
    say(f"  n sweep : {nmin:,} .. {nmax:,}  ({cfg.npoints} points, "
        f"serial ~{t_lo:.2f}s .. ~{t_hi:.2f}s)")
    say(f"  fixed n : {n_fixed:,}  (serial ~{c * n_fixed ** WORK_EXPONENT:.2f}s) "
        f"for all scaling experiments")
    say(f"  largest n anywhere in the run: {max(nmax, n_fixed):,}  "
        f"(peak memory ~{2 * max(nmax, n_fixed) / 1024**3:.2f} GB)")
    if t_hi < 1.0:
        say("")
        say(f"  !! WARNING: even the LARGEST size runs in {t_hi:.2f}s on this machine.")
        say("     The brief warns against runtimes under a second: they are easily")
        say("     distorted by background processes and by mpirun's ~0.3s launch cost.")
        say(f"     Raise the ceiling (--n-cap 40000000) or average harder "
            f"(--reps {max(9, cfg.reps * 2)}).")
    elif t_lo < 1.0:
        short = sum(1 for n in geom_ints(nmin, nmax, cfg.npoints)
                    if c * n ** WORK_EXPONENT < 1.0)
        n_1s = n_for(1.0)
        say(f"  note: {short} of the {cfg.npoints} sizes run in under a second. With a "
            f"{cap:,} ceiling you cannot have both")
        say(f"        30+ sizes and every size above 1s -- the two requirements "
            f"collide. This default keeps the")
        say(f"        wider {nmax/nmin:.0f}x range so the trend (and the point where MPI "
            f"overtakes serial) is visible.")
        say(f"        For every size above 1s instead, use --nmin {n_1s} "
            f"(a narrower {nmax/n_1s:.1f}x range).")
    return c, geom_ints(nmin, nmax, cfg.npoints), n_fixed, n_for


def estimate_runtime(cfg, c, ns, n_fixed, cores, max_workers, hyb_cfgs):
    """Rough wall-clock estimate so the user knows what they are starting."""
    def tser(n):
        return c * n ** WORK_EXPONENT

    def tpar(n, w):
        eff = min(w, cores)                       # no gain past the core count
        return tser(n) / eff + 0.25 + 0.02 * w    # launch + per-worker overhead

    total = 0.0
    for n in ns:                                   # E1
        total += tser(n) + tpar(n, cores) * 3
    for w in worker_values(cores, max_workers):    # E2
        total += tpar(n_fixed, w) * 2
    for p, t in hyb_cfgs:                          # E3/E4
        total += tpar(n_fixed, p * t)
        total += tpar(n_fixed, p * t)              # matched pthread run
    total += tser(n_fixed) + tpar(n_fixed, 1) * 3  # E5 Amdahl decomposition
    return total * cfg.reps


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def experiment_validate(h: Harness, n_check=2_000_000):
    """All four programs must agree on the prime count and the output file."""
    hr("Validation: do the four implementations agree?")
    counts, digests = {}, {}
    plans = [("serial", 1, 1), ("pthread", 1, 2), ("mpi", 2, 1), ("hybrid", 2, 2)]
    for impl, p, t in plans:
        # Run fresh (not from cache) so that primes*.txt is on disk to hash.
        rec, rc, err = h.run_once(impl, n_check, p, t, "validate", 1)
        if rc != 0:
            say(f"  {impl:8s} FAILED rc={rc}: {err[:160]}")
        counts[impl] = rec.get("primes")
        fname = "primes1.txt" if impl in ("serial", "mpi") else "primes2.txt"
        f = h.scratch / impl / fname
        if f.exists():
            digests[impl] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    say("")
    ref = counts.get("serial")
    all_ok = True
    for impl in IMPLS:
        same_count = counts.get(impl) == ref
        same_file = digests.get(impl) == digests.get("serial")
        all_ok &= bool(same_count and same_file)
        say(f"  {impl:8s} primes={counts.get(impl)}  file_sha={digests.get(impl)}  "
            f"{'OK' if same_count and same_file else 'MISMATCH'}")
    if not all_ok:
        say("  WARNING: implementations disagree -- speed-up numbers are meaningless")
        say("           until the parallel versions produce the same primes.")
    return {"n": n_check, "counts": counts, "digests": digests, "agree": all_ok}


def experiment_n_sweep(h: Harness, ns, cores, hyb_pt):
    """Figures 1 and 2: runtime and speed-up against increasing n."""
    hr(f"E1  Increasing problem size n  ({len(ns)} sizes, all workers = {cores})")
    p_h, t_h = hyb_pt
    rows = []
    for i, n in enumerate(ns, 1):
        say(f"  [{i}/{len(ns)}] n = {n:,}")
        s = h.measure("serial", n, 1, 1, "n_sweep")
        pt = h.measure("pthread", n, 1, cores, "n_sweep")
        mp = h.measure("mpi", n, cores, 1, "n_sweep")
        hy = h.measure("hybrid", n, p_h, t_h, "n_sweep")
        rows.append({"n": n, "serial": s, "pthread": pt, "mpi": mp, "hybrid": hy})
    return rows


def experiment_scaling(h: Harness, n_fixed, cores, max_workers):
    """Figure 3: speed-up vs number of MPI processes / POSIX threads."""
    ws = worker_values(cores, max_workers)
    hr(f"E2  Increasing workers at fixed n = {n_fixed:,}   (workers: {ws})")
    base = h.measure("serial", n_fixed, 1, 1, "scaling")
    rows = []
    for w in ws:
        mp = h.measure("mpi", n_fixed, w, 1, "scaling")
        pt = h.measure("pthread", n_fixed, 1, w, "scaling")
        rows.append({"workers": w, "mpi": mp, "pthread": pt})
    return base, rows


def experiment_hybrid_threads(h: Harness, n_fixed, cores, max_workers, p0):
    """Figure 4: hybrid thread scaling at a fixed MPI process count."""
    ts = [t for t in worker_values(cores, max_workers) if p0 * t <= max_workers]
    hr(f"E3  Hybrid thread scaling with {p0} MPI process(es)  (threads: {ts})")
    rows = []
    for t in ts:
        hy = h.measure("hybrid", n_fixed, p0, t, "hybrid_threads")
        mp_equal = h.measure("mpi", n_fixed, p0 * t, 1, "hybrid_threads")
        rows.append({"threads": t, "hybrid": hy, "mpi_equal_workers": mp_equal})
    mp_fixed = h.measure("mpi", n_fixed, p0, 1, "hybrid_threads")
    return rows, mp_fixed


def experiment_hybrid_grid(h: Harness, n_fixed, cores, max_workers, cfgs):
    """Figures 5 and 7: the (processes x threads) grid, plus matched pthread runs."""
    hr(f"E4  Hybrid process x thread grid  ({len(cfgs)} configurations)")
    rows = []
    for p, t in cfgs:
        hy = h.measure("hybrid", n_fixed, p, t, "hybrid_grid")
        pt = h.measure("pthread", n_fixed, 1, p * t, "hybrid_grid")
        rows.append({"procs": p, "threads": t, "workers": p * t,
                     "hybrid": hy, "pthread": pt})
    return rows


def experiment_amdahl(h: Harness, n_fixed, cores):
    """
    Amdahl's law needs the serial fraction at a FIXED problem size.

    Two complementary measurements are taken:

    (a) Decomposition of the serial program.  Its own clock_gettime() region
        covers exactly the loop that the parallel versions replace, so
            T_par  = reported loop time
            T_ser  = wall clock - loop time   (process start, calloc, the
                     file write in report_primes(), exit)
            f      = T_ser / wall
        This is the textbook f and is implementation-independent.

    (b) Decomposition of each parallel program run on ONE worker.  This adds
        the cost that parallelisation itself introduces (mpirun launch,
        MPI_Init, the Gatherv, thread create/join).  The resulting model
            T(p) = T_ser_overhead + T_par / p
        predicts the achievable speed-up far better than (a) alone.
    """
    hr(f"E5  Amdahl decomposition at fixed n = {n_fixed:,}")
    out = {}
    ser = h.measure("serial", n_fixed, 1, 1, "amdahl")
    out["serial"] = ser
    if ser["wall"] and ser["compute"]:
        f = max(0.0, (ser["wall"] - ser["compute"]) / ser["wall"])
        out["f_serial"] = f
        say(f"    serial wall={ser['wall']:.3f}s  parallelisable loop="
            f"{ser['compute']:.3f}s  ->  f = {f:.4f}  (max speed-up {1/f if f>0 else float('inf'):.1f}x)")
    for impl, p, t in (("mpi", 1, 1), ("pthread", 1, 1), ("hybrid", 1, 1)):
        one = h.measure(impl, n_fixed, p, t, "amdahl")
        out[impl + "_1"] = one
        if one["wall"] and one["compute"]:
            ovh = one["wall"] - one["compute"]
            out["overhead_" + impl] = ovh
            out["f_eff_" + impl] = ovh / one["wall"]
            say(f"    {impl:8s} on 1 worker: wall={one['wall']:.3f}s  "
                f"compute={one['compute']:.3f}s  non-parallel part={ovh:.3f}s "
                f"(f_eff={ovh/one['wall']:.4f})")
    # Launch overhead of mpirun itself, measured with a trivial problem size.
    launch = {}
    for p in (1, 2, max(2, cores)):
        tiny = h.measure("mpi", 1000, p, 1, "amdahl_launch", reps=max(2, h.cfg.reps), quiet=True)
        launch[p] = tiny["wall"]
    out["launch"] = launch
    say("    mpirun launch overhead: " +
        ", ".join(f"{p} proc={v:.3f}s" for p, v in sorted(launch.items())))
    return out


# --------------------------------------------------------------------------
# Theory
# --------------------------------------------------------------------------

def amdahl(p, f):
    """Speed-up with serial fraction f on p workers."""
    return 1.0 / (f + (1.0 - f) / p)


def karp_flatt(speedup, p):
    """Experimentally determined serial fraction; rising with p means overhead."""
    if p <= 1 or not speedup or speedup <= 0:
        return None
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)


def overhead_model(p, t_overhead, t_parallel, t_serial_total):
    """Speed-up predicted by T(p) = overhead + parallel_work / p."""
    t = t_overhead + t_parallel / max(1, p)
    return t_serial_total / t if t > 0 else None


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (9.0, 5.6), "figure.dpi": 140,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
        "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9, "legend.framealpha": 0.9,
        "font.size": 10, "savefig.bbox": "tight",
    })
    return plt


def integer_xaxis(ax):
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def cap_yaxis(ax, values, headroom=1.18):
    """Keep the plot readable when the Amdahl ceiling is far above the data."""
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    top = max(vals) * headroom
    ax.set_ylim(0, top)
    return top


def annotate_cores(ax, cores, phys, max_x):
    if cores <= max_x:
        ax.axvline(cores, color="k", lw=0.9, ls="--", alpha=0.55)
        ax.text(cores, ax.get_ylim()[1] * 0.97, f" {cores} logical cores",
                rotation=90, va="top", ha="left", fontsize=8, alpha=0.75)
    if phys and phys != cores and phys <= max_x:
        ax.axvline(phys, color="k", lw=0.8, ls=":", alpha=0.45)
        ax.text(phys, ax.get_ylim()[1] * 0.97, f" {phys} physical cores",
                rotation=90, va="top", ha="left", fontsize=8, alpha=0.6)


def make_figures(res, cfg, sysinfo, figdir: Path):
    plt = setup_mpl()
    figdir.mkdir(parents=True, exist_ok=True)
    cores = sysinfo["logical_cores"]
    phys = sysinfo["physical_cores"]
    tag = f"{sysinfo['label']} | {cores} logical cores" + (f" / {phys} physical" if phys else "")
    saved = []

    def finish(fig, ax, name, title, xlabel, ylabel, legend_loc="best"):
        ax.set_title(f"{title}\n{tag}", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc=legend_loc)
        path = figdir / name
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)
        say(f"  wrote {path.name}")

    # ---- Figure 1: runtime vs n -------------------------------------------
    sweep = res.get("n_sweep") or []
    if sweep:
        fig, ax = plt.subplots()
        ns = [r["n"] for r in sweep]
        for impl in ("serial", "pthread", "mpi", "hybrid"):
            ys = [r[impl]["wall"] for r in sweep]
            if any(y is not None for y in ys):
                lbl = LABEL[impl]
                if impl == "hybrid":
                    lbl += f" [{res['hybrid_pt'][0]}x{res['hybrid_pt'][1]}]"
                elif impl != "serial":
                    lbl += f" [{cores} workers]"
                ax.plot(ns, ys, marker=MARKER[impl], color=COLOR[impl], lw=1.6,
                        ms=4, label=lbl)
        ax.set_xscale("log")
        ax.set_yscale("log")
        finish(fig, ax, "fig1_runtime_vs_n.png",
               "Fig 1  Wall-clock runtime vs problem size n",
               "n (upper bound on primes, log scale)", "wall-clock time (s, log scale)")

        # ---- Figure 2: empirical speed-up vs n ----------------------------
        fig, ax = plt.subplots()
        for impl in ("pthread", "mpi", "hybrid"):
            ys = [(r["serial"]["wall"] / r[impl]["wall"])
                  if r[impl]["wall"] else None for r in sweep]
            lbl = LABEL[impl] + (f" [{res['hybrid_pt'][0]}x{res['hybrid_pt'][1]}]"
                                 if impl == "hybrid" else f" [{cores} workers]")
            ax.plot(ns, ys, marker=MARKER[impl], color=COLOR[impl], lw=1.6, ms=4, label=lbl)
        ax.axhline(cores, color="k", ls="--", lw=0.9, alpha=0.6,
                   label=f"ideal = {cores} (linear speed-up)")
        ax.axhline(1.0, color="gray", ls=":", lw=0.9, alpha=0.7, label="no speed-up")
        ax.set_xscale("log")
        finish(fig, ax, "fig2_speedup_vs_n.png",
               "Fig 2  Empirical speed-up vs serial baseline, increasing n",
               "n (log scale)", "speed-up  T_serial / T_parallel")

    # ---- Figure 3: speed-up vs worker count -------------------------------
    scal = res.get("scaling")
    if scal:
        base, rows = scal
        fig, ax = plt.subplots()
        ws = [r["workers"] for r in rows]
        for impl, key in (("mpi", "mpi"), ("pthread", "pthread")):
            ys = [(base["wall"] / r[key]["wall"]) if r[key]["wall"] else None for r in rows]
            name = "MPI processes" if impl == "mpi" else "POSIX threads"
            ax.plot(ws, ys, marker=MARKER[impl], color=COLOR[impl], lw=1.7, ms=5,
                    label=f"{LABEL[impl]} ({name})")
        ax.plot(ws, ws, color="k", ls="--", lw=0.9, alpha=0.6, label="ideal (linear)")
        integer_xaxis(ax)
        annotate_cores(ax, cores, phys, max(ws))
        finish(fig, ax, "fig3_speedup_vs_workers.png",
               f"Fig 3  Speed-up vs number of workers (n = {res['n_fixed']:,})",
               "number of MPI processes / POSIX threads", "speed-up  T_serial / T_parallel")

    # ---- Figure 4: hybrid threads vs pure MPI -----------------------------
    ht = res.get("hybrid_threads")
    if ht:
        rows, mp_fixed = ht
        base = res["serial_fixed"]
        p0 = res["hybrid_p0"]
        fig, ax = plt.subplots()
        ts = [r["threads"] for r in rows]
        ys = [(base["wall"] / r["hybrid"]["wall"]) if r["hybrid"]["wall"] else None for r in rows]
        ax.plot(ts, ys, marker="D", color=COLOR["hybrid"], lw=1.7, ms=5,
                label=f"Hybrid Task 2: {p0} MPI process(es) x T threads")
        if mp_fixed["wall"]:
            ax.axhline(base["wall"] / mp_fixed["wall"], color=COLOR["mpi"], ls="-",
                       lw=1.7, label=f"Open MPI Task 1: {p0} process(es) (no threads)")
        ys2 = [(base["wall"] / r["mpi_equal_workers"]["wall"])
               if r["mpi_equal_workers"]["wall"] else None for r in rows]
        ax.plot(ts, ys2, marker="^", color=COLOR["mpi"], ls=":", lw=1.4, ms=5,
                label=f"Open MPI Task 1 with {p0}xT processes (equal worker count)")
        ax.plot(ts, [p0 * t for t in ts], color="k", ls="--", lw=0.9, alpha=0.6,
                label="ideal (linear in total workers)")
        integer_xaxis(ax)
        annotate_cores(ax, cores / p0, None, max(ts))
        finish(fig, ax, "fig4_hybrid_threads_vs_mpi.png",
               f"Fig 4  Hybrid speed-up vs threads per process (n = {res['n_fixed']:,})",
               "OpenMP threads per MPI process", "speed-up  T_serial / T_parallel")

    # ---- Figure 5: hybrid grid vs matched pthread -------------------------
    grid = res.get("hybrid_grid")
    if grid:
        base = res["serial_fixed"]
        fig, ax = plt.subplots()
        by_p = {}
        for r in grid:
            by_p.setdefault(r["procs"], []).append(r)
        palette = ["#2ca02c", "#7f2704", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
        plist = sorted(by_p)
        for i, p in enumerate(plist):
            rows = sorted(by_p[p], key=lambda r: r["workers"])
            xs = [r["workers"] for r in rows]
            ys = [(base["wall"] / r["hybrid"]["wall"]) if r["hybrid"]["wall"] else None
                  for r in rows]
            ax.plot(xs, ys, marker="D", ms=4.5, lw=1.5,
                    color=palette[i % len(palette)],
                    label=f"Hybrid, {p} MPI process(es)")
        seen = {}
        for r in grid:
            if r["pthread"]["wall"]:
                seen[r["workers"]] = base["wall"] / r["pthread"]["wall"]
        xs = sorted(seen)
        ax.plot(xs, [seen[x] for x in xs], marker="s", color=COLOR["pthread"],
                lw=1.8, ms=5, label="POSIX threads with the same total thread count")
        ax.plot(xs, xs, color="k", ls="--", lw=0.9, alpha=0.6, label="ideal (linear)")
        integer_xaxis(ax)
        annotate_cores(ax, cores, phys, max(xs) if xs else cores)
        finish(fig, ax, "fig5_hybrid_vs_pthread_total_workers.png",
               f"Fig 5  Hybrid (P x T) vs POSIX threads at equal total workers "
               f"(n = {res['n_fixed']:,})",
               "total workers  (MPI processes x threads per process)",
               "speed-up  T_serial / T_parallel")

    # ---- Figure 6: MPI empirical vs theoretical ---------------------------
    th = res.get("theory")
    if scal and th:
        base, rows = scal
        fig, ax = plt.subplots()
        ws = [r["workers"] for r in rows]
        emp = [(base["wall"] / r["mpi"]["wall"]) if r["mpi"]["wall"] else None for r in rows]
        ax.plot(ws, emp, marker="^", color=COLOR["mpi"], lw=1.8, ms=5,
                label="Open MPI Task 1 (measured)")
        f = th.get("f_serial")
        if f is not None:
            ax.plot(ws, [amdahl(w, f) for w in ws], color="#ff7f0e", ls="-", lw=1.5,
                    label=f"Amdahl, f = {f:.4f} from serial decomposition "
                          f"(ceiling {1/f:.1f}x)" if f > 0 else "Amdahl")
        ovh, one = th.get("overhead_mpi"), th.get("mpi_1")
        if ovh is not None and one and one["compute"]:
            ax.plot(ws, [overhead_model(w, ovh, one["compute"], base["wall"]) for w in ws],
                    color="#9467bd", ls="--", lw=1.5,
                    label="Amdahl + measured MPI overhead (1-process decomposition)")
        ax.plot(ws, ws, color="k", ls="--", lw=0.9, alpha=0.6, label="ideal (linear)")
        top = cap_yaxis(ax, [v for v in emp if v] + list(ws) +
                        ([amdahl(max(ws), f)] if f else []))
        if f and 0 < f and 1 / f < (top or 0):
            ax.axhline(1 / f, color="#ff7f0e", ls=":", lw=1.0, alpha=0.7,
                       label=f"Amdahl ceiling 1/f = {1/f:.1f}x")
        integer_xaxis(ax)
        annotate_cores(ax, cores, phys, max(ws))
        finish(fig, ax, "fig6_mpi_empirical_vs_theoretical.png",
               f"Fig 6  Open MPI: empirical vs theoretical speed-up (n = {res['n_fixed']:,})",
               "number of MPI processes", "speed-up")

    # ---- Figure 7: hybrid empirical vs theoretical ------------------------
    if grid and th:
        base = res["serial_fixed"]
        fig, ax = plt.subplots()
        best = {}
        for r in grid:
            if not r["hybrid"]["wall"]:
                continue
            s = base["wall"] / r["hybrid"]["wall"]
            w = r["workers"]
            if w not in best or s > best[w][0]:
                best[w] = (s, r["procs"], r["threads"])
        xs = sorted(best)
        ax.plot(xs, [best[x][0] for x in xs], marker="D", color=COLOR["hybrid"],
                lw=1.8, ms=5, label="Hybrid Task 2, best P x T at each worker count")
        for x in xs:
            s, p, t = best[x]
            ax.annotate(f"{p}x{t}", (x, s), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, alpha=0.85)
        f = th.get("f_serial")
        if f is not None:
            ax.plot(xs, [amdahl(x, f) for x in xs], color="#ff7f0e", lw=1.5,
                    label=f"Amdahl, f = {f:.4f} (ceiling {1/f:.1f}x)"
                          if f > 0 else "Amdahl")
        ovh, one = th.get("overhead_hybrid"), th.get("hybrid_1")
        if ovh is not None and one and one["compute"]:
            ax.plot(xs, [overhead_model(x, ovh, one["compute"], base["wall"]) for x in xs],
                    color="#9467bd", ls="--", lw=1.5,
                    label="Amdahl + measured hybrid overhead")
        ax.plot(xs, xs, color="k", ls="--", lw=0.9, alpha=0.6, label="ideal (linear)")
        top = cap_yaxis(ax, [best[x][0] for x in xs] + list(xs) +
                        ([amdahl(max(xs), f)] if f and xs else []))
        if f and 0 < f and 1 / f < (top or 0):
            ax.axhline(1 / f, color="#ff7f0e", ls=":", lw=1.0, alpha=0.7,
                       label=f"Amdahl ceiling {1/f:.1f}x")
        integer_xaxis(ax)
        annotate_cores(ax, cores, phys, max(xs) if xs else cores)
        finish(fig, ax, "fig7_hybrid_empirical_vs_theoretical.png",
               f"Fig 7  Hybrid: empirical vs theoretical speed-up (n = {res['n_fixed']:,})",
               "total workers (MPI processes x OpenMP threads)", "speed-up")

    # ---- Extra A: parallel efficiency -------------------------------------
    if scal:
        base, rows = scal
        fig, ax = plt.subplots()
        ws = [r["workers"] for r in rows]
        for impl in ("mpi", "pthread"):
            ys = [((base["wall"] / r[impl]["wall"]) / r["workers"] * 100)
                  if r[impl]["wall"] else None for r in rows]
            ax.plot(ws, ys, marker=MARKER[impl], color=COLOR[impl], lw=1.7, ms=5,
                    label=LABEL[impl])
        ax.axhline(100, color="k", ls="--", lw=0.9, alpha=0.6, label="ideal (100%)")
        integer_xaxis(ax)
        annotate_cores(ax, cores, phys, max(ws))
        finish(fig, ax, "figA_efficiency.png",
               f"Fig A  Parallel efficiency = speed-up / workers (n = {res['n_fixed']:,})",
               "number of workers", "efficiency (%)")

        # ---- Extra B: Karp-Flatt experimentally determined serial fraction
        fig, ax = plt.subplots()
        for impl in ("mpi", "pthread"):
            xs, ys = [], []
            for r in rows:
                if r["workers"] > 1 and r[impl]["wall"]:
                    e = karp_flatt(base["wall"] / r[impl]["wall"], r["workers"])
                    if e is not None:
                        xs.append(r["workers"])
                        ys.append(e)
            ax.plot(xs, ys, marker=MARKER[impl], color=COLOR[impl], lw=1.7, ms=5,
                    label=LABEL[impl])
        if th and th.get("f_serial") is not None:
            ax.axhline(th["f_serial"], color="#ff7f0e", ls="--", lw=1.2,
                       label=f"f from serial decomposition = {th['f_serial']:.4f}")
        integer_xaxis(ax)
        annotate_cores(ax, cores, phys, max(ws))
        finish(fig, ax, "figB_karp_flatt.png",
               "Fig B  Karp-Flatt metric: experimentally determined serial fraction\n"
               "(flat = pure Amdahl limit, rising = parallel overhead dominates)",
               "number of workers", "experimentally determined serial fraction e")

        # ---- Extra C: load imbalance --------------------------------------
        fig, ax = plt.subplots()
        for impl in ("mpi", "pthread"):
            xs = [r["workers"] for r in rows if r[impl]["imbalance"]]
            ys = [r[impl]["imbalance"] for r in rows if r[impl]["imbalance"]]
            if xs:
                ax.plot(xs, ys, marker=MARKER[impl], color=COLOR[impl], lw=1.7,
                        ms=5, label=LABEL[impl])
        if grid:
            xs = [r["workers"] for r in grid if r["hybrid"]["imbalance"]]
            ys = [r["hybrid"]["imbalance"] for r in grid if r["hybrid"]["imbalance"]]
            if xs:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                ax.plot([xs[i] for i in order], [ys[i] for i in order], marker="D",
                        color=COLOR["hybrid"], lw=1.2, ms=4, ls="none",
                        label=LABEL["hybrid"])
        ax.axhline(1.0, color="k", ls="--", lw=0.9, alpha=0.6, label="perfect balance")
        integer_xaxis(ax)
        annotate_cores(ax, cores, phys, max(ws))
        finish(fig, ax, "figC_load_imbalance.png",
               f"Fig C  Load imbalance (slowest worker / average) (n = {res['n_fixed']:,})",
               "number of workers", "imbalance ratio")

    return saved


# --------------------------------------------------------------------------
# CSV + report output
# --------------------------------------------------------------------------

def write_aggregate_csv(h: Harness, path: Path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["impl", "n", "procs", "threads", "workers", "reps",
                    "wall_s", "wall_min_s", "wall_sd_s", "region_s", "compute_s",
                    "primes", "imbalance"])
        for key in sorted(h.store):
            impl, n, p, t = key
            a = aggregate(impl, n, p, t, h.store[key], h.cfg.stat)
            w.writerow([impl, n, p, t, p * t, a["reps"],
                        f"{a['wall']:.6f}" if a["wall"] else "",
                        f"{a['wall_min']:.6f}" if a["wall_min"] else "",
                        f"{a['wall_sd']:.6f}" if a["wall_sd"] is not None else "",
                        f"{a['region']:.6f}" if a["region"] else "",
                        f"{a['compute']:.6f}" if a["compute"] else "",
                        a["primes"] or "", f"{a['imbalance']:.4f}" if a["imbalance"] else ""])


def write_theory_csv(res, path: Path):
    th = res.get("theory") or {}
    f = th.get("f_serial")
    rows = [["quantity", "value", "note"]]
    if f is not None:
        rows += [
            ["serial_fraction_f", f"{f:.6f}", "from serial program: (wall - loop) / wall"],
            ["parallel_fraction_1_minus_f", f"{1-f:.6f}", ""],
            ["amdahl_ceiling", f"{1/f:.3f}" if f > 0 else "inf", "1 / f"],
        ]
    for impl in ("mpi", "pthread", "hybrid"):
        if th.get("f_eff_" + impl) is not None:
            rows.append([f"f_effective_{impl}", f"{th['f_eff_'+impl]:.6f}",
                         "1-worker run: (wall - compute) / wall, includes parallel overhead"])
            rows.append([f"overhead_seconds_{impl}", f"{th['overhead_'+impl]:.6f}", ""])
    for p, v in (th.get("launch") or {}).items():
        rows.append([f"mpirun_launch_overhead_p{p}", f"{v:.6f}", "mpirun with n=1000"])
    rows.append([])
    rows.append(["p", "amdahl_speedup", "amdahl_plus_overhead", "empirical_mpi",
                 "karp_flatt_e"])
    scal = res.get("scaling")
    if scal and f is not None:
        base, srows = scal
        ovh, one = th.get("overhead_mpi"), th.get("mpi_1")
        for r in srows:
            p = r["workers"]
            emp = base["wall"] / r["mpi"]["wall"] if r["mpi"]["wall"] else None
            om = (overhead_model(p, ovh, one["compute"], base["wall"])
                  if ovh is not None and one and one["compute"] else None)
            kf = karp_flatt(emp, p)
            rows.append([p, f"{amdahl(p, f):.4f}",
                         f"{om:.4f}" if om else "",
                         f"{emp:.4f}" if emp else "",
                         f"{kf:.4f}" if kf is not None else ""])
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def plural(n, word, suffix="es"):
    return f"{n} {word}" + ("" if n == 1 else suffix)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def write_report(res, cfg, sysinfo, figures, path: Path):
    cores = sysinfo["logical_cores"]
    phys = sysinfo["physical_cores"] or "unknown"
    th = res.get("theory") or {}
    f = th.get("f_serial")
    L = []
    A = L.append

    A(f"# Parallel prime counting: empirical and theoretical evaluation\n")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on `{sysinfo['hostname']}` "
      f"(label `{sysinfo['label']}`).\n")

    A("## 1. Machine and toolchain\n")
    A(md_table(["Property", "Value"], [
        ["CPU", sysinfo["cpu_model"]],
        ["Logical cores (hardware threads)", cores],
        ["Physical cores", phys],
        ["Memory available", f"{sysinfo['mem_available_gb']} GB"],
        ["OS", sysinfo["os"]],
        ["Architecture", sysinfo["machine"]],
        ["C compiler", (sysinfo["cc"] or [""])[0]],
        ["MPI compiler wrapper (underlying cc)", (sysinfo["mpicc"] or [""])[0]],
        ["MPI runtime", (sysinfo["mpirun"] or [""])[0]],
        ["mpirun flags", " ".join(sysinfo["mpirun_flags"]) or "(none)"],
        ["Hostfile (multi-machine)", sysinfo["hostfile"] or "not used"],
        ["Compiler flags", "-O2 -std=gnu11 (-pthread / -fopenmp as required)"],
        ["Repetitions per data point", cfg.reps],
        ["Statistic reported", cfg.stat],
    ]))
    A("All speed-ups in this report are measured against the **serial Week 4 Task 1 "
      "program**, as required. Timing is the *overall wall-clock time* of the whole "
      "process as seen from outside (`perf_counter` around the process), so it "
      "includes process startup, `mpirun` launch, memory allocation, the parallel "
      "region, the `MPI_Gatherv`, and writing the prime list to disk.\n")

    val = res.get("validation")
    if not val:
        A("## 2. Correctness check\n")
        A("_Not performed in this invocation (`--plots-only` or `--no-validate`). "
          "Re-run without those flags to regenerate it._\n")
    if val:
        A("## 2. Correctness check\n")
        A(f"At n = {val['n']:,} every implementation must produce the same primes.\n")
        A(md_table(["Implementation", "Primes found", "SHA-256 of output file (16 hex)", "Verdict"],
                   [[LABEL[i], val["counts"].get(i), val["digests"].get(i, "-"),
                     "match" if val["counts"].get(i) == val["counts"].get("serial")
                     and val["digests"].get(i) == val["digests"].get("serial") else "MISMATCH"]
                    for i in IMPLS]))
        if not val["agree"]:
            A("> **Warning:** the implementations disagree. Fix correctness before "
              "quoting any speed-up.\n")

    A("## 3. Experimental design\n")
    A(md_table(["Experiment", "What is varied", "Fixed", "Figures"], [
        ["E1 problem size", f"{len(res.get('n_sweep') or [])} values of n "
         f"({res['ns'][0]:,} .. {res['ns'][-1]:,})", f"workers = {cores}", "1, 2"],
        ["E2 worker scaling", "MPI processes / POSIX threads, 1 .. "
         f"{cfg.max_workers}", f"n = {res['n_fixed']:,}", "3, 6, A, B, C"],
        ["E3 hybrid threads", f"threads per process, {res.get('hybrid_p0')} MPI "
         "processes fixed", f"n = {res['n_fixed']:,}", "4"],
        ["E4 hybrid grid", "processes x threads combinations",
         f"n = {res['n_fixed']:,}", "5, 7"],
        ["E5 Amdahl", "decomposition into serial and parallel time",
         f"n = {res['n_fixed']:,}", "6, 7"],
    ]))
    n_count = len(res.get("n_sweep") or [])
    if n_count >= 30:
        A(f"E1 uses {n_count} distinct values of n, satisfying the requirement of at "
          "least 30. ")
    else:
        A(f"**E1 used only {n_count} values of n here** (a reduced/`--quick` run); the "
          "brief requires at least 30, so re-run without `--quick` or with "
          "`--npoints 32` before submitting. ")
    sweep_rows = res.get("n_sweep") or []
    if sweep_rows:
        lo, hi = sweep_rows[0]["serial"]["wall"], sweep_rows[-1]["serial"]["wall"]
        short = sum(1 for r in sweep_rows if (r["serial"]["wall"] or 0) < 1.0)
        A(f"n is capped at {cfg.n_cap:,}, so the largest size is the cap itself and "
          f"only the bottom of the range is calibrated to this machine. Serial "
          f"runtime across the sweep spans {lo:.2f}s to {hi:.2f}s.")
        if short:
            A(f" {short} of the {len(sweep_rows)} sizes fall below one second: with a "
              f"{cfg.n_cap:,} ceiling the brief's two requirements (at least 30 sizes, "
              "runtimes above a second) cannot both be met, and the wider range was "
              "kept so the trend and the serial/parallel crossover stay visible. The "
              f"{cfg.stat} of {cfg.reps} repetitions damps the extra noise, and the "
              "conclusions should be read off the larger sizes.")
        A("\n")
    else:
        A(f"n is capped at {cfg.n_cap:,}.\n")

    # ---- results tables ---------------------------------------------------
    sweep = res.get("n_sweep") or []
    if sweep:
        A("## 4. Empirical results: increasing n (Figures 1 and 2)\n")
        rows = []
        step = max(1, len(sweep) // 12)
        for r in sweep[::step]:
            def sp(impl):
                return (f"{r['serial']['wall'] / r[impl]['wall']:.2f}x"
                        if r[impl]["wall"] else "-")
            rows.append([f"{r['n']:,}",
                         f"{r['serial']['wall']:.3f}",
                         f"{r['pthread']['wall']:.3f}", sp("pthread"),
                         f"{r['mpi']['wall']:.3f}", sp("mpi"),
                         f"{r['hybrid']['wall']:.3f}", sp("hybrid")])
        A(md_table(["n", "serial (s)", "pthread (s)", "speed-up",
                    "MPI (s)", "speed-up", "hybrid (s)", "speed-up"], rows))
        A(f"(Full data for all {len(sweep)} sizes is in `aggregate.csv`; this table is "
          "sampled for readability.)\n")

    scal = res.get("scaling")
    if scal:
        base, srows = scal
        A("## 5. Empirical results: increasing workers (Figure 3)\n")
        rows = []
        for r in srows:
            p = r["workers"]
            smpi = base["wall"] / r["mpi"]["wall"] if r["mpi"]["wall"] else None
            spt = base["wall"] / r["pthread"]["wall"] if r["pthread"]["wall"] else None
            rows.append([p,
                         f"{r['mpi']['wall']:.3f}" if r["mpi"]["wall"] else "-",
                         f"{smpi:.2f}x" if smpi else "-",
                         f"{smpi/p*100:.0f}%" if smpi else "-",
                         f"{r['mpi']['imbalance']:.3f}" if r["mpi"]["imbalance"] else "-",
                         f"{r['pthread']['wall']:.3f}" if r["pthread"]["wall"] else "-",
                         f"{spt:.2f}x" if spt else "-",
                         f"{spt/p*100:.0f}%" if spt else "-"])
        A(md_table(["workers", "MPI time (s)", "MPI speed-up", "MPI efficiency",
                    "MPI imbalance", "pthread time (s)", "pthread speed-up",
                    "pthread efficiency"], rows))

    A("## 6. Theoretical analysis (Amdahl's law)\n")
    A("### 6.1 Measuring the serial and parallel fractions\n")
    A("Amdahl's law assumes a **fixed problem size**. The serial program already "
      "brackets exactly the loop that the parallel versions replace with "
      "`clock_gettime`, so the split can be read off directly:\n")
    A("```\n"
      "T_total  = wall clock of the serial program        (measured externally)\n"
      "T_par    = the timed prime-testing loop            (measured internally)\n"
      "T_serial = T_total - T_par                         (startup, calloc,\n"
      "                                                    report_primes file I/O, exit)\n"
      "f        = T_serial / T_total\n"
      "S_amdahl(p) = 1 / ( f + (1 - f) / p ),   ceiling 1/f\n"
      "```\n")
    if f is not None:
        ser = th["serial"]
        A(md_table(["Quantity", "Value"], [
            ["Fixed problem size n", f"{res['n_fixed']:,}"],
            ["Total serial wall clock", f"{ser['wall']:.4f} s"],
            ["Parallelisable part (prime loop)", f"{ser['compute']:.4f} s"],
            ["Non-parallelisable part", f"{ser['wall'] - ser['compute']:.4f} s"],
            ["**Serial fraction f**", f"**{f:.4f}**"],
            ["**Parallel fraction 1 - f**", f"**{1 - f:.4f}**"],
            ["Amdahl ceiling 1/f", f"{1/f:.1f}x" if f > 0 else "unbounded"],
        ]))
    A("A second, stricter decomposition runs each *parallel* program on a single "
      "worker. The gap between its wall clock and its own compute region is the "
      "cost that parallelisation itself adds -- `mpirun` launch, `MPI_Init`, the "
      "`MPI_Gatherv`, thread create/join -- which Amdahl's plain f does not "
      "capture:\n")
    rows = []
    for impl in ("mpi", "pthread", "hybrid"):
        if th.get("f_eff_" + impl) is not None:
            one = th[impl + "_1"]
            rows.append([LABEL[impl], f"{one['wall']:.4f}", f"{one['compute']:.4f}",
                         f"{th['overhead_'+impl]:.4f}", f"{th['f_eff_'+impl]:.4f}",
                         f"{1/th['f_eff_'+impl]:.1f}x" if th["f_eff_" + impl] > 0 else "-"])
    if rows:
        A(md_table(["Implementation (1 worker)", "wall (s)", "compute (s)",
                    "non-parallel part (s)", "effective f", "its ceiling"], rows))
    launch = th.get("launch") or {}
    if launch:
        A("`mpirun` launch overhead measured with a trivial problem size "
          "(n = 1000): " + ", ".join(f"**{v:.3f} s** at {p} process(es)"
                                     for p, v in sorted(launch.items())) +
          ". This is a fixed cost that the shared-memory versions never pay, and "
          "it is the main reason MPI trails POSIX threads at small n.\n")

    A("### 6.2 Why Amdahl and not Gustafson\n")
    A("The brief allows either law. Amdahl's is the applicable one here because "
      f"the problem size in this study is **fixed** (capped at {cfg.n_cap:,}) and "
      "the same n is solved by 1 worker and by every larger worker count. That is "
      "precisely Amdahl's assumption: a constant workload divided over more "
      "processors, where the non-parallelisable part stops shrinking and sets a "
      "ceiling.\n")
    A("Gustafson's law assumes the opposite: that the problem grows with the "
      "machine so the time per worker stays constant, which describes how people "
      "actually use bigger clusters and gives the much more optimistic "
      "`S = s + p(1 - s)`. Measuring it correctly would mean re-running with n "
      "scaled up as p rises (work here grows as n^1.5, so p times the work needs "
      "n_p = n_1 * p^(2/3)) -- which the fixed ceiling in this study rules out. "
      "Quoting Gustafson numbers from fixed-size measurements would be a category "
      "error, so none are reported.\n")

    # ---- discussion -------------------------------------------------------
    A("## 7. Discussion\n")

    A("### How does the actual speed-up compare with the theoretical speed-up?\n")
    if scal and f is not None:
        base, srows = scal
        lines = []
        for r in srows:
            p = r["workers"]
            if r["mpi"]["wall"]:
                emp = base["wall"] / r["mpi"]["wall"]
                lines.append((p, emp, amdahl(p, f)))
        if lines:
            worst = max(lines, key=lambda x: x[2] - x[1])
            at_cores = min(lines, key=lambda x: abs(x[0] - cores))
            A(f"At {plural(at_cores[0], 'MPI process')} the measured speed-up is "
              f"**{at_cores[1]:.2f}x** against an Amdahl prediction of "
              f"**{at_cores[2]:.2f}x** "
              f"({at_cores[1]/at_cores[2]*100:.0f}% of the prediction). The largest "
              f"shortfall occurs at {plural(worst[0], 'process')} "
              f"({worst[1]:.2f}x measured vs {worst[2]:.2f}x predicted).\n")
            A("Amdahl's law is optimistic here because it only charges for the "
              "serial fraction. It assumes the parallel part divides perfectly and "
              "that parallelisation is free. Neither holds: `mpirun` startup, "
              "`MPI_Init`, the `MPI_Gatherv` of the whole flag array to rank 0, "
              "memory-bandwidth contention and any load imbalance all grow with p. "
              "The purple 'Amdahl + measured overhead' curve in Figures 6 and 7, "
              "which adds the measured one-worker overhead to the model, tracks the "
              "measurement much more closely. Figure B (Karp-Flatt) makes the same "
              "point from the data alone: if the only loss were the serial fraction, "
              "the experimentally determined fraction would be flat across p; where "
              "it rises, the extra loss is parallel overhead rather than serial "
              "code.\n")

    A("### Will more MPI processes always increase the speed-up?\n")
    if scal:
        base, srows = scal
        pts = [(r["workers"], base["wall"] / r["mpi"]["wall"])
               for r in srows if r["mpi"]["wall"]]
        if pts:
            bp, bs = max(pts, key=lambda x: x[1])
            last_p, last_s = pts[-1]
            A(f"No. On this machine the MPI speed-up peaks at **{bs:.2f}x with "
              f"{plural(bp, 'process')}** and is {last_s:.2f}x at "
              f"{plural(last_p, 'process')}. "
              f"There " + ("is " if cores == 1 else "are ") +
              plural(cores, "logical core", "s")
              + (f" ({phys} physical)" if phys != 'unknown' else "") + ", so:\n")
            A("- Up to the core count, extra processes get real cores and speed-up "
              "grows, though sub-linearly because of the serial fraction and "
              "communication.\n"
              "- If the logical core count exceeds the physical count "
              "(hyper-threading/SMT), the second thread on a core shares execution "
              "units, so those extra workers add much less than a full core would.\n"
              "- **Beyond the core count the machine is oversubscribed.** Processes "
              "time-share cores, the OS scheduler adds context switches, each rank "
              "still allocates its own buffers, and the gather grows. Speed-up flattens "
              "and then degrades. `mpirun` will refuse to launch more ranks than slots "
              "at all unless `--oversubscribe` is given, which this harness passes "
              "automatically.\n"
              "- Even with unlimited cores the fixed-size speed-up cannot exceed the "
              + (f"Amdahl ceiling of **{1/f:.1f}x**" if f else "Amdahl ceiling") +
              " for this n, because the file write and process startup do not shrink.\n")

    A("### How does workload distribution affect the speed-up?\n")
    A("Strongly, and it is visible in the data. Testing whether k is prime costs "
      "about sqrt(k) divisions, so candidates near n are far more expensive than "
      "candidates near 2, and a naive contiguous split would leave the last rank "
      "with several times the work of the first. Both implementations avoid that by "
      "cutting the range into roughly 64 chunks per worker and dealing them out "
      "round-robin, which is why the measured imbalance (slowest worker / average "
      "busy time) stays close to 1.0 in Figure C. ")
    if scal:
        base, srows = scal
        imb = [(r["workers"], r["mpi"]["imbalance"]) for r in srows if r["mpi"]["imbalance"]]
        if imb:
            worst = max(imb, key=lambda x: x[1])
            A(f"The worst MPI imbalance observed was **{worst[1]:.3f}** at "
              f"{plural(worst[0], 'process')}; because the wall clock is set by the *slowest* "
              f"worker, an imbalance of {worst[1]:.3f} alone caps efficiency at about "
              f"{100/worst[1]:.0f}%.\n")
    A("The hybrid version has an extra advantage here: `schedule(dynamic)` lets "
      "OpenMP hand the next chunk to whichever thread finishes first, so imbalance "
      "inside a rank is corrected at runtime, whereas the MPI split is decided "
      "statically before the run. The cost is that chunk size shrinks as "
      "processes x threads grows, so scheduling overhead and cache-line sharing rise.\n")

    A("### Will the speed-up be the same on different machines?\n")
    A(f"No. This run used {plural(cores, 'logical core', 's')}"
      + (f" / {plural(phys, 'physical core', 's')}" if phys != "unknown" else "") +
      f" on {sysinfo['cpu_model']}. Results depend on core count and SMT, per-core "
      "clock and turbo behaviour, cache and memory bandwidth per core, disk speed "
      "(the prime list write is part of the serial fraction), MPI implementation and "
      "its default process binding, compiler version and flags, and background load. "
      "A machine with more cores raises the linear part of the curve but does not "
      "raise the Amdahl ceiling, while a machine with a faster disk *lowers* the "
      "serial fraction and therefore raises the ceiling. The speed-up *shape* "
      "(linear, then knee at the core count, then flat or falling) should reproduce "
      "everywhere; the numbers will not.\n")
    A("To compare machines, run this script on each with the same "
      "`--fixed-n` and `--nmin/--nmax` (so the calibration does not choose different "
      "sizes) and a distinct `--label`, then combine with:\n")
    A("```\npython3 mpi_bench.py --merge results/laptop results/desktop results/vm\n```\n")
    A("For a genuinely distributed run across team members' machines, set up "
      "passwordless SSH between the hosts, put the identical binary at the same path "
      "on each, and pass a hostfile:\n")
    A("```\n# hosts.txt\n192.168.1.10 slots=8\n192.168.1.11 slots=4\n\n"
      "python3 mpi_bench.py --hostfile hosts.txt --label cluster\n```\n")
    A("Expect the network to hurt: the `MPI_Gatherv` moves one byte per candidate "
      "to rank 0, so at n = 10^8 that is ~100 MB across the link, which over gigabit "
      "Ethernet costs about a second regardless of how many cores the extra machine "
      "contributes.\n")

    A("## 8. Threats to validity and known limitations\n")
    A("- `test_primes()` in the Task 1 MPI program takes `int upper_bound`, so n "
      "above 2^31-1 would silently truncate; this harness caps n well below that.\n"
      "- Rank 0 allocates a full n-byte gather buffer, so memory is the binding "
      "constraint on n before time is.\n"
      "- Writing the prime list dominates the non-parallel part at large n. The "
      "serial fraction is therefore partly an I/O measurement; using a tmpfs "
      "scratch directory (`--scratch /dev/shm/bench`) isolates CPU scaling from "
      "disk speed.\n"
      "- Wall-clock timing includes `mpirun` startup, which is charged to MPI and "
      "not to the pthread version. This is deliberate: it is a real cost of the "
      "distributed design, and it is quantified separately in section 6.1.\n"
      f"- Each point is the {cfg.stat} of {cfg.reps} repetition(s). Run with a higher "
      "`--reps` on a quiet machine for publication-quality error bars.\n")

    A("## 9. Files produced\n")
    A(md_table(["File", "Contents"], [
        ["`raw_runs.csv`", "every individual run, one row each"],
        ["`aggregate.csv`", "per configuration statistics"],
        ["`theory.csv`", "serial/parallel fractions, Amdahl and Karp-Flatt curves"],
        ["`sysinfo.json`", "machine and toolchain description"],
        ["`report.md`", "this document"],
    ] + [[f"`figures/{p.name}`", ""] for p in figures]))

    path.write_text("\n".join(L))


# --------------------------------------------------------------------------
# Cross-machine merge
# --------------------------------------------------------------------------

def do_merge(dirs, out):
    plt = setup_mpl()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    found = 0
    for d in dirs:
        d = Path(d)
        agg = d / "aggregate.csv"
        info = d / "sysinfo.json"
        if not agg.exists():
            say(f"  skipping {d}: no aggregate.csv")
            continue
        si = json.loads(info.read_text()) if info.exists() else {}
        label = si.get("label", d.name)
        cores = si.get("logical_cores", "?")
        rows = list(csv.DictReader(open(agg)))
        by = {}
        for r in rows:
            if not r["wall_s"]:
                continue
            by.setdefault((r["impl"], int(r["n"])), {})[int(r["workers"])] = float(r["wall_s"])
        # Use the n with the most MPI worker counts measured.
        cands = [(k, v) for k, v in by.items() if k[0] == "mpi"]
        if not cands:
            continue
        (impl, n), series = max(cands, key=lambda kv: len(kv[1]))
        base = by.get(("serial", n), {}).get(1)
        if not base:
            continue
        ws = sorted(series)
        ax.plot(ws, [base / series[w] for w in ws], marker="o", lw=1.7, ms=5,
                label=f"{label} ({cores} cores, n={n:,})")
        ax2.plot(ws, [base / series[w] / w * 100 for w in ws], marker="o", lw=1.7,
                 ms=5, label=f"{label} ({cores} cores)")
        found += 1
    if not found:
        sys.exit("error: no usable result directories for --merge")
    for a, t, yl, name in ((ax, "Open MPI speed-up across machines", "speed-up",
                            "figE_cross_machine_speedup.png"),
                           (ax2, "Open MPI efficiency across machines", "efficiency (%)",
                            "figF_cross_machine_efficiency.png")):
        a.set_title(t)
        a.set_xlabel("number of MPI processes")
        a.set_ylabel(yl)
        a.legend()
        f = a.get_figure()
        f.savefig(out / name)
        say(f"  wrote {out / name}")
        plt.close(f)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    cores = os.cpu_count() or 1
    p = argparse.ArgumentParser(
        description="Benchmark serial / pthread / MPI / hybrid prime counters and "
                    "produce the seven required figures plus an Amdahl's law "
                    "analysis. Problem size is capped at --n-cap (default 10M).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("sources and build")
    g.add_argument("--src-dir", default=".", help="directory searched for .c files")
    g.add_argument("--serial", help="path to the Week 4 Task 1 serial source")
    g.add_argument("--pthread", help="path to the Week 4 Task 2 POSIX/OpenMP source")
    g.add_argument("--mpi", help="path to this week's Task 1 MPI source")
    g.add_argument("--hybrid", help="path to this week's Task 2 hybrid source")
    g.add_argument("--cc", default="gcc")
    g.add_argument("--mpicc", default="mpicc")
    g.add_argument("--mpirun", default="mpirun")
    g.add_argument("--skip-build", action="store_true",
                   help="reuse binaries already in <workdir>/bin")

    g = p.add_argument_group("problem sizes")
    g.add_argument("--npoints", type=int, default=32,
                   help="number of distinct n values (the brief requires >= 30)")
    g.add_argument("--nmin", type=int, help="override the smallest n")
    g.add_argument("--nmax", type=int, help="override the largest n")
    g.add_argument("--fixed-n", type=int, help="override n for the scaling experiments")
    g.add_argument("--target-min-seconds", type=float, default=1.0,
                   help="serial runtime targeted at the smallest n "
                        "(ignored if it would exceed n-cap/10)")
    g.add_argument("--target-max-seconds", type=float, default=12.0,
                   help="serial runtime targeted at the largest n "
                        "(in practice --n-cap binds first)")
    g.add_argument("--target-fixed-seconds", type=float, default=15.0,
                   help="serial runtime targeted for the fixed-n experiments "
                        "(in practice --n-cap binds first)")
    g.add_argument("--n-cap", type=int, default=10_000_000,
                   help="hard upper limit on n, applied to every experiment")

    g = p.add_argument_group("parallelism")
    g.add_argument("--max-workers", type=int, default=2 * cores,
                   help="highest process/thread count to test (oversubscription)")
    g.add_argument("--hybrid-config", default=None, metavar="P:T",
                   help="processes:threads used for the hybrid in the n sweep "
                        "(default: a balanced split of the core count)")
    g.add_argument("--hybrid-p0", type=int, default=None,
                   help="fixed MPI process count for the Figure 4 thread sweep")
    g.add_argument("--hostfile", help="mpirun hostfile for a multi-machine run")
    g.add_argument("--mpi-extra", default="", help="extra flags passed to mpirun")

    g = p.add_argument_group("measurement")
    g.add_argument("--reps", type=int, default=5,
                   help="repetitions per data point; the default is high because "
                        "a 10M ceiling makes individual runs short and noisy")
    g.add_argument("--stat", choices=("median", "min", "mean"), default="median")
    g.add_argument("--timeout", type=float, default=1800.0, help="seconds per run")
    g.add_argument("--no-validate", dest="validate", action="store_false")

    g = p.add_argument_group("output and control")
    g.add_argument("--label", default=platform.node().split(".")[0],
                   help="name for this machine, used in the results directory")
    g.add_argument("--outdir", default="results")
    g.add_argument("--workdir", default=".bench")
    g.add_argument("--scratch", default=None,
                   help="where the programs run and write primes*.txt "
                        "(tip: /dev/shm/bench on Linux removes disk noise)")
    g.add_argument("--only", default="all",
                   help="comma list of experiments: validate,n_sweep,scaling,"
                        "hybrid_threads,hybrid_grid,amdahl")
    g.add_argument("--plots-only", action="store_true",
                   help="rebuild figures and report from an existing results dir")
    g.add_argument("--merge", nargs="+", metavar="DIR",
                   help="combine result directories from several machines")
    g.add_argument("--no-resume", dest="resume", action="store_false",
                   help="ignore previously recorded runs and measure everything again")
    g.add_argument("--quick", action="store_true",
                   help="fast sanity run: 1 rep, 12 sizes, small n")
    g.add_argument("-y", "--yes", action="store_true", help="do not ask to confirm")

    a = p.parse_args(argv)
    if a.quick:
        # --quick lowers the defaults, but anything the user typed explicitly wins.
        given = set(argv if argv is not None else sys.argv[1:])
        def unset(*flags):
            return not any(f in given for f in flags)
        if unset("--reps"):
            a.reps = 2
        if unset("--npoints"):
            a.npoints = 12
        if unset("--target-min-seconds"):
            a.target_min_seconds = 0.2
        if unset("--n-cap"):
            a.n_cap = min(a.n_cap, 3_000_000)
        if unset("--max-workers"):
            a.max_workers = min(a.max_workers, max(2, cores))
    if a.scratch is None:
        a.scratch = str(Path(a.workdir) / "scratch")
    a.mpi_flags = []
    a.bind_none_flags = []
    return a


def main(argv=None):
    cfg = parse_args(argv)

    if cfg.merge:
        hr("Merging results from several machines")
        do_merge(cfg.merge, Path(cfg.outdir) / "merged")
        return 0

    outdir = Path(cfg.outdir) / cfg.label
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"

    cores = os.cpu_count() or 1
    phys = physical_cores()

    hr("Parallel prime counting benchmark")
    say(f"  machine       : {cpu_model()}")
    say(f"  logical cores : {cores}" + (f"   physical cores: {phys}" if phys else ""))
    say(f"  results       : {outdir}")

    # ---- tool checks ------------------------------------------------------
    for tool in (cfg.cc, cfg.mpicc, cfg.mpirun):
        if not which(tool):
            sys.exit(f"error: '{tool}' not found on PATH.\n"
                     "Install an MPI stack, e.g.\n"
                     "  Ubuntu/Debian : sudo apt install build-essential openmpi-bin libopenmpi-dev\n"
                     "  Fedora/RHEL   : sudo dnf install gcc openmpi openmpi-devel  (then module load mpi)\n"
                     "  macOS         : brew install open-mpi libomp")

    # ---- sources and build ------------------------------------------------
    hr("Sources and build")
    sources = discover_sources(cfg)
    for impl in IMPLS:
        say(f"  {impl:8s} {sources[impl]}")
    binaries = build_all(cfg, sources)

    cfg.mpi_flags, cfg.bind_none_flags = probe_mpi_flags(cfg)
    say(f"  mpirun flags: {' '.join(cfg.mpi_flags) or '(none)'}"
        f"   hybrid adds: {' '.join(cfg.bind_none_flags) or '(none)'}")

    sysinfo = collect_sysinfo(cfg)
    (outdir / "sysinfo.json").write_text(json.dumps(sysinfo, indent=2))

    h = Harness(cfg, binaries, outdir)
    only = {s.strip() for s in cfg.only.split(",")} if cfg.only != "all" else None

    def want(name):
        return (only is None or name in only) and not cfg.plots_only

    res = {"label": cfg.label}

    # ---- sizing -----------------------------------------------------------
    if cfg.plots_only:
        # Reconstruct the plan from what is already recorded.
        ns = sorted({n for (impl, n, p, t) in h.store if impl == "serial"})
        counts = {}
        for (impl, n, p, t) in h.store:
            counts[n] = counts.get(n, 0) + 1
        n_fixed = cfg.fixed_n or (max(counts, key=lambda k: counts[k]) if counts else 0)
        c = None
        def n_for(s):
            return n_fixed
        ns = [n for n in ns if n != 1000]
    else:
        c, ns, n_fixed, n_for = calibrate(h, cfg, cores)

    res["ns"] = ns or [0]
    res["n_fixed"] = n_fixed

    # hybrid split used for the n sweep
    if cfg.hybrid_config:
        p_h, t_h = (int(x) for x in cfg.hybrid_config.split(":"))
    else:
        p_h = 2 if cores >= 4 and cores % 2 == 0 else 1
        t_h = max(1, cores // p_h)
    res["hybrid_pt"] = (p_h, t_h)
    p0 = cfg.hybrid_p0 or (2 if cfg.max_workers >= 4 else 1)
    res["hybrid_p0"] = p0
    hyb_cfgs = hybrid_configs(cores, cfg.max_workers)

    # ---- time estimate ----------------------------------------------------
    if not cfg.plots_only and c:
        est = estimate_runtime(cfg, c, ns, n_fixed, cores, cfg.max_workers, hyb_cfgs)
        say("")
        say(f"  Estimated total benchmarking time: ~{fmt_hms(est)} "
            f"(rough; overheads vary)")
        say(f"  Hybrid split for the n sweep: {p_h} process(es) x {t_h} thread(s)")
        say(f"  Worker counts: {worker_values(cores, cfg.max_workers)}")
        if not cfg.yes:
            try:
                ans = input("  Proceed? [Y/n] ").strip().lower()
            except EOFError:
                ans = "y"
            if ans and ans not in ("y", "yes"):
                say("  aborted")
                return 1

    t_start = time.time()

    # ---- experiments ------------------------------------------------------
    if want("validate") and cfg.validate:
        res["validation"] = experiment_validate(h, min(2_000_000, max(100_000, n_fixed // 8)))

    if want("n_sweep"):
        res["n_sweep"] = experiment_n_sweep(h, ns, cores, (p_h, t_h))

    if want("scaling"):
        res["scaling"] = experiment_scaling(h, n_fixed, cores, cfg.max_workers)

    if want("amdahl"):
        res["theory"] = experiment_amdahl(h, n_fixed, cores)

    if not cfg.plots_only:
        res["serial_fixed"] = h.measure("serial", n_fixed, 1, 1, "baseline", quiet=True)

    if want("hybrid_threads"):
        res["hybrid_threads"] = experiment_hybrid_threads(h, n_fixed, cores,
                                                          cfg.max_workers, p0)
    if want("hybrid_grid"):
        res["hybrid_grid"] = experiment_hybrid_grid(h, n_fixed, cores,
                                                    cfg.max_workers, hyb_cfgs)
    # ---- reconstruct for --plots-only ------------------------------------
    if cfg.plots_only:
        res = rebuild_from_store(h, res, cores, cfg)

    # ---- outputs ----------------------------------------------------------
    hr("Writing results")
    write_aggregate_csv(h, outdir / "aggregate.csv")
    say(f"  wrote {outdir/'aggregate.csv'}")
    write_theory_csv(res, outdir / "theory.csv")
    say(f"  wrote {outdir/'theory.csv'}")

    try:
        figures = make_figures(res, cfg, sysinfo, figdir)
    except ImportError:
        say("  matplotlib/numpy missing -- install with: pip install matplotlib numpy")
        figures = []

    write_report(res, cfg, sysinfo, figures, outdir / "report.md")
    say(f"  wrote {outdir/'report.md'}")

    # ---- summary ----------------------------------------------------------
    hr("Summary")
    say(f"  runs executed   : {h.run_count}")
    say(f"  time in programs: {fmt_hms(h.spent)}")
    say(f"  total elapsed   : {fmt_hms(time.time() - t_start)}")
    if h.failures:
        say(f"  failures        : {len(h.failures)}")
        for fl in h.failures[:5]:
            say(f"     {fl}")
    th = res.get("theory") or {}
    if th.get("f_serial") is not None:
        fv = th["f_serial"]
        if fv > 0:
            say(f"  serial fraction f = {fv:.4f}  ->  Amdahl ceiling {1/fv:.1f}x")
        else:
            say("  serial fraction f ~ 0 (below timer resolution at this n)")
    scal = res.get("scaling")
    if scal:
        base, rows = scal
        pts = [(r["workers"], base["wall"] / r["mpi"]["wall"])
               for r in rows if r["mpi"]["wall"]]
        if pts:
            bp, bs = max(pts, key=lambda x: x[1])
            say(f"  best MPI speed-up: {bs:.2f}x at {bp} process(es) "
                f"on {cores} logical cores")
    say("")
    say(f"  Open {outdir/'report.md'} and {figdir} for the write-up.")
    h.close()
    return 0


def rebuild_from_store(h: Harness, res, cores, cfg):
    """Reconstruct the experiment structures from raw_runs.csv for --plots-only."""
    store = h.store
    agg = {k: aggregate(k[0], k[1], k[2], k[3], v, cfg.stat) for k, v in store.items()}

    def get(impl, n, p, t):
        return agg.get((impl, n, p, t))

    ns = sorted({n for (i, n, p, t) in store if i == "serial" and n > 10_000})
    p_h, t_h = res["hybrid_pt"]
    sweep = []
    for n in ns:
        row = {"n": n}
        ok = True
        for impl, p, t in (("serial", 1, 1), ("pthread", 1, cores),
                           ("mpi", cores, 1), ("hybrid", p_h, t_h)):
            a = get(impl, n, p, t)
            if a is None:
                ok = False
                break
            row[impl] = a
        if ok:
            sweep.append(row)
    if sweep:
        res["n_sweep"] = sweep

    n_fixed = res["n_fixed"]
    base = get("serial", n_fixed, 1, 1)
    if base:
        res["serial_fixed"] = base
        rows = []
        for w in sorted({p for (i, n, p, t) in store if i == "mpi" and n == n_fixed}):
            m = get("mpi", n_fixed, w, 1)
            pt = get("pthread", n_fixed, 1, w)
            if m and pt:
                rows.append({"workers": w, "mpi": m, "pthread": pt})
        if rows:
            res["scaling"] = (base, rows)

        grid = []
        for (i, n, p, t) in store:
            if i == "hybrid" and n == n_fixed:
                pt = get("pthread", n_fixed, 1, p * t)
                if pt:
                    grid.append({"procs": p, "threads": t, "workers": p * t,
                                 "hybrid": get("hybrid", n_fixed, p, t), "pthread": pt})
        if grid:
            res["hybrid_grid"] = sorted(grid, key=lambda r: (r["procs"], r["threads"]))

        p0 = res["hybrid_p0"]
        ht = []
        for t in sorted({t for (i, n, p, tt) in store
                         if i == "hybrid" and n == n_fixed and p == p0
                         for t in [tt]}):
            m_eq = get("mpi", n_fixed, p0 * t, 1)
            if m_eq:
                ht.append({"threads": t, "hybrid": get("hybrid", n_fixed, p0, t),
                           "mpi_equal_workers": m_eq})
        mf = get("mpi", n_fixed, p0, 1)
        if ht and mf:
            res["hybrid_threads"] = (ht, mf)

        ser1 = base
        th = {"serial": ser1}
        if ser1 and ser1["wall"] and ser1["compute"]:
            th["f_serial"] = max(0.0, (ser1["wall"] - ser1["compute"]) / ser1["wall"])
        for impl in ("mpi", "pthread", "hybrid"):
            one = get(impl, n_fixed, 1, 1)
            if one and one["wall"] and one["compute"]:
                th[impl + "_1"] = one
                th["overhead_" + impl] = one["wall"] - one["compute"]
                th["f_eff_" + impl] = (one["wall"] - one["compute"]) / one["wall"]
        launch = {}
        for (i, n, p, t) in store:
            if i == "mpi" and n == 1000:
                launch[p] = agg[(i, n, p, t)]["wall"]
        if launch:
            th["launch"] = launch
        res["theory"] = th

    return res


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\n  interrupted -- partial results are in raw_runs.csv "
            "(rerun without --no-resume to continue)")
        sys.exit(130)