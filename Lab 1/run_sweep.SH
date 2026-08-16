#!/usr/bin/env bash
#==============================================================================
# run_sweep.sh - FIT3143 Lab 1 experimental harness
#
# PURPOSE
#   Collects every measurement required by the eight graphs in the Lab 1
#   specification, in two experiments:
#
#     scaling_n.csv        Experiment A - n varies, thread count fixed
#                          -> graphs 1, 2, 5, 6, 7
#     scaling_threads.csv  Experiment B - thread count varies, n fixed
#                          -> graphs 3, 4, 8
#     raw_trials.csv       Every individual trial, unaggregated
#     run_metadata.txt     Machine, toolchain and parameter provenance
#
# METHODOLOGY
#   Four design decisions drive the measurement protocol. Each exists to
#   remove a specific source of systematic error:
#
#   1. INTERLEAVED TRIALS. Within one configuration the three implementations
#      are run round-robin (serial, pthreads, openmp, serial, pthreads, ...)
#      rather than in blocks of repeats. Thermal state and background load
#      drift over the minutes a sweep takes; block ordering would correlate
#      that drift with implementation, biasing the comparison. Round-robin
#      spreads any drift evenly across all three.
#
#   2. DISCARDED WARM-UP. The first execution at each new n pays cold
#      instruction cache, cold branch predictor and first-touch page-fault
#      costs on the freshly calloc'd flag array. It is run and thrown away
#      so those one-off costs do not contaminate the reported times.
#
#   3. MINIMUM REPORTED, DISPERSION ALSO RECORDED. Timing noise is one-sided:
#      an interrupt, page fault or preemption can only ever lengthen a run.
#      The minimum is therefore the best estimator of intrinsic cost. The
#      mean, sample standard deviation and coefficient of variation are also
#      written out, so the stability of every point can be demonstrated
#      rather than asserted.
#
#   4. IDENTICAL TOOLCHAIN AND FLAGS for all three binaries, so a measured
#      difference is attributable to the parallel construct and not to
#      code generation.
#
# USAGE
#   ./run_sweep.sh --quick     Short validation pass. Run this first.
#   ./run_sweep.sh             Full sweep.
#
#   Overridable:  CC=gcc-16  REPEATS=5  FIXED_THREADS=12  FIXED_N=10000000
#   e.g.  REPEATS=7 ./run_sweep.sh
#==============================================================================

set -euo pipefail
cd "$(dirname "$0")"

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
CC=${CC:-gcc-16}                     # must support -fopenmp; Apple clang does not
CFLAGS=${CFLAGS:--O2}
REPEATS=${REPEATS:-5}                # measured trials per implementation per config
FIXED_THREADS=${FIXED_THREADS:-12}   # thread count held constant in Experiment A
FIXED_N=${FIXED_N:-10000000}         # problem size held constant in Experiment B
N_MIN=${N_MIN:-100000}
N_MAX=${N_MAX:-20000000}
N_POINTS=${N_POINTS:-32}             # spec requires at least 30 distinct values
EXP=${EXP:-AB}                       # AB | A | B - which experiments to run.
                                     # EXP=A is used to re-measure individual
                                     # noisy points without repeating the
                                     # thread sweep as well.

QUICK=0
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=1; REPEATS=1; N_POINTS=5
fi

if command -v nproc >/dev/null 2>&1; then CORES=$(nproc); else CORES=$(sysctl -n hw.ncpu); fi

OUT_SUFFIX=${OUT_SUFFIX:-}           # e.g. OUT_SUFFIX=_rerun writes
                                     # scaling_n_rerun.csv instead of
                                     # overwriting the main sweep output
OUT_N=scaling_n${OUT_SUFFIX}.csv
OUT_T=scaling_threads${OUT_SUFFIX}.csv
OUT_RAW=raw_trials${OUT_SUFFIX}.csv
OUT_META=run_metadata${OUT_SUFFIX}.txt

#------------------------------------------------------------------------------
# Every output file is truncated when written. A targeted re-run of one
# configuration would therefore destroy a completed full sweep. Any existing
# file is copied to <name>.bak first, so a mistake costs one rename rather
# than a repeat of the whole experiment.
#------------------------------------------------------------------------------
backup_if_exists() {
    [ -f "$1" ] && { cp "$1" "$1.bak"; echo "  backed up $1 -> $1.bak"; }
    return 0
}
for f in "$OUT_N" "$OUT_T" "$OUT_RAW"; do backup_if_exists "$f"; done

#------------------------------------------------------------------------------
# Provenance. Results that cannot be reproduced cannot be defended, so the
# exact machine, toolchain and parameters are recorded alongside the data.
#------------------------------------------------------------------------------
{
    echo "FIT3143 Lab 1 - sweep provenance"
    echo "timestamp        : $(date -u '+%Y-%m-%dT%H:%M:%SZ') (UTC)"
    echo "host             : $(hostname)"
    echo "kernel           : $(uname -srm)"
    echo "logical cores    : $CORES"
    echo "compiler         : $($CC --version 2>/dev/null | head -1)"
    echo "cflags           : $CFLAGS"
    # Power management policy changes core clock frequencies, which rescales
    # the serial baseline and therefore every speedup ratio. It must be held
    # constant across a sweep and recorded, or results are not comparable.
    if command -v pmset >/dev/null 2>&1; then
        echo "power mode       : $(pmset -g 2>/dev/null | awk '/powermode/{print $2; f=1} END{if(!f) print "unknown"}')  (0=normal, 1=low, 2=high)"
        echo "power source     : $(pmset -g ps 2>/dev/null | head -1 | sed 's/.*drawing from //;s/[()]//g' | tr -d "'" )"
    else
        echo "power mode       : n/a (pmset not present)"
    fi
    echo "repeats          : $REPEATS (+1 discarded warm-up per configuration)"
    echo "experiment A     : n in [$N_MIN, $N_MAX], $N_POINTS log-spaced points, threads=$FIXED_THREADS"
    echo "experiment B     : n=$FIXED_N, thread sweep spanning and exceeding $CORES cores"
} > "$OUT_META"

echo "== Environment =="; cat "$OUT_META"; echo

#------------------------------------------------------------------------------
# Build. All three binaries share compiler and optimisation level.
#------------------------------------------------------------------------------
echo "== Build =="
$CC $CFLAGS task1.c -o task1 -lm
$CC $CFLAGS -pthread task2.c -o task2
$CC $CFLAGS -fopenmp task3.c -o task3
echo "task1, task2, task3 built with $CC $CFLAGS"; echo

#------------------------------------------------------------------------------
# Output parsing. Reads each program's existing stdout; no source changes are
# required beyond the argv handling all three already have.
#------------------------------------------------------------------------------
parse_serial_time()   { awk '/Time taken/{print $3; exit}'; }
parse_parallel_time() { awk -F'time=' '/^n=/{sub(/ *s.*/,"",$2); print $2; exit}'; }
parse_imbalance()     { awk -F'= *' '/^Imbalance/{split($2,a," "); print a[1]; exit}'; }

# Aggregates a whitespace-separated list of times to: min mean sd cv
# Sample standard deviation (n-1 denominator); cv expressed as a percentage.
stats() {
    awk '{ for (i=1;i<=NF;i++) { v[++n]=$i; s+=$i; if (n==1||$i<mn) mn=$i } }
         END {
             if (n==0) { print "NA NA NA NA"; exit }
             mean = s/n;
             if (n>1) { for (i=1;i<=n;i++) d+=(v[i]-mean)^2; sd=sqrt(d/(n-1)) } else sd=0;
             printf "%.6f %.6f %.6f %.2f\n", mn, mean, sd, (mean>0 ? 100*sd/mean : 0)
         }'
}

ratio() { awk -v a="$1" -v b="$2" 'BEGIN{ if (b>0) printf "%.4f", a/b; else printf "NA" }'; }

mean_of() { awk '{s=0;c=0;for(i=1;i<=NF;i++){s+=$i;c++} if(c) printf "%.4f", s/c; else printf "NA"}'; }

# Karp-Flatt metric: the experimentally determined serial fraction
#   e = (1/S - 1/p) / (1 - 1/p)
# Unlike raw speedup this separates the two causes of imperfect scaling. If e
# stays flat as p grows, the limit is an inherently sequential fraction
# (Amdahl); if e rises with p, parallel overhead is growing with thread count.
karp_flatt() {
    awk -v S="$1" -v p="$2" 'BEGIN{
        if (p<=1 || S<=0) printf "NA"; else printf "%.4f", (1/S - 1/p)/(1 - 1/p)
    }'
}

# Log-spaced problem sizes. Linear spacing would place almost every point in
# the large-n regime; log spacing preserves resolution at small n, where
# thread-management overhead is a large fraction of total work and speedup can
# fall below 1. That regime is itself a reportable result.
gen_log_values() {
    awk -v min="$1" -v max="$2" -v count="$3" 'BEGIN{
        # Single-point case: re-measuring one noisy n. Guard the step
        # calculation, which would otherwise divide by (count-1)=0.
        if (count <= 1) { printf "%d ", min; exit }
        step=(log(max)-log(min))/(count-1); prev=-1;
        for (i=0;i<count;i++) {
            v=int(exp(log(min)+i*step)+0.5);
            v=int(v/1000+0.5)*1000;
            if (v!=prev) { printf "%d ", v; prev=v }
        }
    }'
}

#------------------------------------------------------------------------------
# run_config <experiment> <n> <threads>
#
# Executes one configuration: warm-up, then $REPEATS interleaved rounds of
# (serial, pthreads, openmp). Appends every trial to raw_trials.csv and sets
# the aggregate globals consumed by the experiment loops.
#------------------------------------------------------------------------------
run_config() {
    local exp=$1 n=$2 threads=$3
    local st="" pt="" ot="" pimb="" oimb="" out t i

    # Warm-up, discarded (decision 2 above).
    ./task1 "$n" >/dev/null
    ./task2 "$n" "$threads" >/dev/null
    ./task3 "$n" "$threads" >/dev/null

    for trial in $(seq "$REPEATS"); do
        # Round-robin ordering (decision 1 above).
        out=$(./task1 "$n"); t=$(echo "$out" | parse_serial_time)
        [ -z "$t" ] && { echo "FATAL: no timing from task1 at n=$n" >&2; exit 1; }
        st+="$t "; echo "$exp,$n,$threads,serial,$trial,$t,NA" >> "$OUT_RAW"

        out=$(./task2 "$n" "$threads"); t=$(echo "$out" | parse_parallel_time)
        i=$(echo "$out" | parse_imbalance)
        [ -z "$t" ] && { echo "FATAL: no timing from task2 at n=$n p=$threads" >&2; exit 1; }
        pt+="$t "; pimb+="${i:-} "; echo "$exp,$n,$threads,pthreads,$trial,$t,${i:-NA}" >> "$OUT_RAW"

        out=$(./task3 "$n" "$threads"); t=$(echo "$out" | parse_parallel_time)
        i=$(echo "$out" | parse_imbalance)
        [ -z "$t" ] && { echo "FATAL: no timing from task3 at n=$n p=$threads" >&2; exit 1; }
        ot+="$t "; oimb+="${i:-} "; echo "$exp,$n,$threads,openmp,$trial,$t,${i:-NA}" >> "$OUT_RAW"
    done

    read -r S_MIN S_MEAN S_SD S_CV <<< "$(echo "$st" | stats)"
    read -r P_MIN P_MEAN P_SD P_CV <<< "$(echo "$pt" | stats)"
    read -r O_MIN O_MEAN O_SD O_CV <<< "$(echo "$ot" | stats)"
    P_IMB=$(echo "$pimb" | mean_of)
    O_IMB=$(echo "$oimb" | mean_of)

    SPD_P=$(ratio "$S_MIN" "$P_MIN"); SPD_O=$(ratio "$S_MIN" "$O_MIN")
    EFF_P=$(ratio "$SPD_P" "$threads"); EFF_O=$(ratio "$SPD_O" "$threads")
    MAX_CV=$(awk -v a="$S_CV" -v b="$P_CV" -v c="$O_CV" 'BEGIN{m=a; if(b>m)m=b; if(c>m)m=c; printf "%.1f", m}')
}

echo "experiment,n,threads,implementation,trial,time_s,imbalance" > "$OUT_RAW"

#==============================================================================
# EXPERIMENT A - runtime and speedup against problem size   (graphs 1,2,5,6,7)
#==============================================================================
if [[ "$EXP" == *A* ]]; then
N_VALUES=$(gen_log_values "$N_MIN" "$N_MAX" "$N_POINTS")
echo "== Experiment A: $(wc -w <<< "$N_VALUES" | tr -d ' ') values of n, threads=$FIXED_THREADS =="

echo "n,threads,serial_min_s,serial_mean_s,serial_sd_s,serial_cv_pct,pthreads_min_s,pthreads_mean_s,pthreads_sd_s,pthreads_cv_pct,openmp_min_s,openmp_mean_s,openmp_sd_s,openmp_cv_pct,speedup_pthreads,speedup_openmp,efficiency_pthreads,efficiency_openmp,imbalance_pthreads,imbalance_openmp" > "$OUT_N"

for n in $N_VALUES; do
    run_config A "$n" "$FIXED_THREADS"
    echo "$n,$FIXED_THREADS,$S_MIN,$S_MEAN,$S_SD,$S_CV,$P_MIN,$P_MEAN,$P_SD,$P_CV,$O_MIN,$O_MEAN,$O_SD,$O_CV,$SPD_P,$SPD_O,$EFF_P,$EFF_O,$P_IMB,$O_IMB" >> "$OUT_N"
    printf '  n=%-9s serial=%ss  pthreads=%ss (S=%s)  openmp=%ss (S=%s)  max cv=%s%%\n' \
        "$n" "$S_MIN" "$P_MIN" "$SPD_P" "$O_MIN" "$SPD_O" "$MAX_CV"
done
echo
fi

#==============================================================================
# EXPERIMENT B - runtime and speedup against thread count      (graphs 3,4,8)
#
# Resolution is fine up to the core count and continues well past it: the
# specification asks what happens when threads exceed available cores, so the
# oversubscribed regime must be measured rather than argued.
#==============================================================================
if [[ "$EXP" == *B* ]]; then
if [ "$QUICK" = "1" ]; then
    THREAD_LIST="1 2 4 8"
else
    THREAD_LIST="1 2 3 4 5 6 7 8 9 10 11 12 14 16 20 24"
fi
echo "== Experiment B: threads = {$THREAD_LIST}, n=$FIXED_N (cores=$CORES) =="

echo "threads,n,serial_min_s,serial_mean_s,serial_sd_s,serial_cv_pct,pthreads_min_s,pthreads_mean_s,pthreads_sd_s,pthreads_cv_pct,openmp_min_s,openmp_mean_s,openmp_sd_s,openmp_cv_pct,speedup_pthreads,speedup_openmp,efficiency_pthreads,efficiency_openmp,karp_flatt_pthreads,karp_flatt_openmp,imbalance_pthreads,imbalance_openmp" > "$OUT_T"

for t in $THREAD_LIST; do
    run_config B "$FIXED_N" "$t"
    KF_P=$(karp_flatt "$SPD_P" "$t"); KF_O=$(karp_flatt "$SPD_O" "$t")
    echo "$t,$FIXED_N,$S_MIN,$S_MEAN,$S_SD,$S_CV,$P_MIN,$P_MEAN,$P_SD,$P_CV,$O_MIN,$O_MEAN,$O_SD,$O_CV,$SPD_P,$SPD_O,$EFF_P,$EFF_O,$KF_P,$KF_O,$P_IMB,$O_IMB" >> "$OUT_T"
    printf '  p=%-3s pthreads=%ss (S=%s E=%s e=%s)  openmp=%ss (S=%s E=%s e=%s)\n' \
        "$t" "$P_MIN" "$SPD_P" "$EFF_P" "$KF_P" "$O_MIN" "$SPD_O" "$EFF_O" "$KF_O"
done
echo
fi

#==============================================================================
# CORRECTNESS VERIFICATION
#
# Equal runtimes are worthless if the parallel versions compute the wrong
# answer, so the three prime lists are compared byte-for-byte at the largest n
# measured. task1 writes primes.txt or primes1.txt depending on revision.
#==============================================================================
echo "== Correctness verification (n=$FIXED_N) =="
./task1 "$FIXED_N" >/dev/null
./task2 "$FIXED_N" "$FIXED_THREADS" >/dev/null
./task3 "$FIXED_N" "$FIXED_THREADS" >/dev/null

T1_OUT=""
for f in primes1.txt primes.txt; do [ -f "$f" ] && { T1_OUT=$f; break; }; done

if [ -z "$T1_OUT" ]; then
    echo "WARNING: no task1 output file found; skipping comparison."
elif cmp -s "$T1_OUT" primes2.txt && cmp -s "$T1_OUT" primes3.txt; then
    echo "PASS: serial, POSIX Threads and OpenMP produced byte-identical lists"
    echo "      ($(wc -l < "$T1_OUT" | tr -d ' ') primes below $FIXED_N, ascending, from $T1_OUT)"
else
    echo "FAIL: outputs differ - investigate before presenting."
    for f in "$T1_OUT" primes2.txt primes3.txt; do
        printf '      %-12s %s lines\n' "$f" "$(wc -l < "$f" | tr -d ' ')"
    done
    exit 1
fi

echo
echo "Wrote: $OUT_N, $OUT_T, $OUT_RAW, $OUT_META"