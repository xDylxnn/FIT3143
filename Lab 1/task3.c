#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define CHUNKS_PER_THREAD 64    /* higher = better balance, more loop overhead */
#define MIN_THREADS 1
#define MAX_THREADS 256
#define FILE_NAME "primes3.txt"

// Struct bundles several variables into one unit.
// One of these per thread, so the presentation can show the measured workload split.
typedef struct {
    long iters_done;
    double busy;
} worker_t;

/* Shared data. Written once before the threads start, then read-only. */
static long  upper_bound;
static char  *flags;
static int   nthreads;
static long  chunk_size;

/* Timestamp in seconds */
static double monotonic_seconds(void) {
    struct timespec t; 
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9; 
}

/*
 * same is_prime function as in task1.c and task2.c, 
 * which checks if a number is prime or not
 */
static bool is_prime(long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false; 
    for (long i = 5; i <= n / i; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

/*
 * Reads upper_bound and the thread count from argv, falling back to a 
 * prompt and to the core count. Returns false if either value is unusable.
 */
static bool read_configuration(int argc, char **argv) {
    if (argc > 1) {
        upper_bound = atol(argv[1]); // convert the first command line argument to a long integer
    } else {
        printf("Enter a number: ");
        if (scanf("%ld", &upper_bound) != 1) {
            fprintf(stderr, "Error: could not read an integer.\n");
            return false;
        }
    }
    // omp_get_max_threads() is OpenMP's own default, normally the number of cores.
    nthreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();

    if (nthreads < MIN_THREADS || nthreads > MAX_THREADS) {
        fprintf(stderr, "Error: thread count must be between %d and %d.\n", MIN_THREADS, MAX_THREADS);
        return false;
    }
    return true;
}

/*
 * Prints the primes found: to stdout for n <= 100, to FILE_NAME otherwise. 
 * Returns the count, or -1 on file error.
 */
static long report_primes(void) {
    long count = 0;
    if (upper_bound <= 100) {
        for (long i = 2; i < upper_bound; i++)
            if (flags[i]) { printf("%ld ", i); count++; }
        printf("\n");
    } else {
        FILE *fptr = fopen(FILE_NAME, "w");
        if (fptr == NULL) {
            fprintf(stderr, "Error: could not open %s for writing.\n", FILE_NAME);
            return -1;
        }
        for (long i = 2; i < upper_bound; i++)
            if (flags[i]) { fprintf(fptr, "%ld\n", i); count++; }
        fclose(fptr);
    }
    return count;
}

/*
 * Prints per-thread chunk counts and busy times, so the load balance is
 * measured. A balanced run sits near an imbalance of 1.0.
 */
static void report_threads(const worker_t *workers) {
    double sum = 0, max = 0;
    for (int t = 0; t < nthreads; t++) {
        sum += workers[t].busy;
        if (workers[t].busy > max) max = workers[t].busy;
        printf("  thread %-3d iters=%-10ld busy=%.6f s\n",
               t, workers[t].iters_done, workers[t].busy);
    }
    printf("Imbalance (slowest/average) = %.4f   (1.0 is perfect)\n",
           max / (sum / nthreads));
}

int main(int argc, char **argv) {
    struct timespec start, end;
    worker_t       *workers = NULL;
    long           count    = 0;

    if (!read_configuration(argc, argv)) return 1;

    if (upper_bound < 2) {
        printf("No primes are strictly less than %ld.\n", upper_bound);
        return 0;
    }

    omp_set_num_threads(nthreads); // API call, overrides the default and OMP_NUM_THREADS

    // Same granularity as  task2.c: aim for about 64chunks per thread. Enough chunks that the 
    // workload evens out, few enough that each chunk still covers many whole cache lines.
    chunk_size = upper_bound / ((long)nthreads * CHUNKS_PER_THREAD);
    if (chunk_size < CHUNKS_PER_THREAD) chunk_size = CHUNKS_PER_THREAD;

    flags = calloc((size_t)upper_bound, 1);
    if (flags == NULL) {
        fprintf(stderr, "Error: could not allocate memory for %ld candidates.\n", upper_bound);
        return 1;
    }

    workers = calloc((size_t)nthreads, sizeof(worker_t));
    if (workers == NULL) {
        fprintf(stderr, "Error: could not allocate memory for %d worker structs.\n", nthreads);
        free(flags);
        return 1;
    }

    // Phase 1: timed computation. Allocation and file output are outside the timed region, exactly 
    // as in task1.c and task2.c, so the three times measure the same thing and the ratios between them are valid speedups.
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Fork: one parallel region, the team is created once. flags, num and chunk are shared (declared outside); 
    // everything declared inside the region is private to each thread, so no false sharing.
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        long mine = 0;
        double t0 = monotonic_seconds();

        // Work-sharing: OpenMP hands each thread a chunk of iterations and gives it a new one as soon as it finishes. 
        // Cost per iteration grows with i (trial division runs to sqrt(i)), so a self-scheduling policy balances the load 
        // better than a fixed split. nowait removes the barrier at the end of the loop so busy measures this thread's own 
        // work and not its wait for the slowest thread.
        #pragma omp for schedule(dynamic, chunk_size) nowait
        for (long i = 2; i < upper_bound; i++) {
            if (is_prime(i)) flags[i] = 1; //only write on a hit: writing 0s;
            mine++;
        }

        workers[id].iters_done = mine;    // written once, after the loop
        workers[id].busy = monotonic_seconds() - t0;    // record the time spent in this thread
    } // Join: implicit barrier here, all threads finish before main continues.

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Phase 2: output in ascending order. Scanning flags[] by index gives a
    // sorted list for free - no sorting step is needed after the parallel phase.
    count = report_primes();
    if (count < 0) {
        free(flags);
        free(workers);
        return 1;
    }

    // Per-thread figures, so the presentation can show measured load balance rather than just claiming it. 
    // A well-balanced run gives near-identical busy times and an imbalance close to 1.0.
    report_threads(workers);

    printf("n=%ld  threads=%d  primes=%ld  time=%.6f s\n",
        upper_bound, nthreads, count, time_taken);

    free(flags);
    free(workers);
    return 0;
}