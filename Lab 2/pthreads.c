/* task2.c - counts the primes below n in parallel with POSIX threads. */

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>             /* sysconf, to detect the core count */
#include <pthread.h>

#define CACHE_LINE_BYTES 64  
#define CHUNKS_PER_THREAD 64    /* higher = better balance, more loop overhead */
#define MIN_THREADS 1
#define MAX_THREADS 256
#define FILE_NAME "primes2.txt"

/* One worker per thread, records thread working data. */
typedef struct {
    int id;
    long chunks_done;
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
 * Once multiples of 2 and 3 are ruled out, every remaining factor is 6k +/- 1, 
 * so the loop steps by 6 and tests two divisors at a time.
 * Returns true if n is prime, false otherwise.
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
 * Worker entry point. arg is this thread's worker_t. Created chunk range from
 * w.id and chunk size, marks the primes it finds in flags[], and
 * records its chunk count and busy time in *w.
 */
static void *test_primes(void *arg) {
    worker_t *w = (worker_t *)arg;
    double t0 = monotonic_seconds();

    for (long c = w->id; 2 + c * chunk_size < upper_bound; c += nthreads) {
        //turn the chunk number into a range of consecutive candidates
        long lo = 2 + c * chunk_size;
        long hi = lo + chunk_size;
        if (hi > upper_bound) hi = upper_bound;  //clamp the final, partial chunk

        for (long i = lo; i < hi; i++)
            if (is_prime(i)) flags[i] = 1;      //composites keep the calloc'd 0
        w->chunks_done++;
    }

    w->busy = monotonic_seconds() - t0;
    return NULL;
}

/*
 * Reads upper_bound and the thread count from argv, falling back to a 
 * prompt and to the core count. Returns false if either value is unusable.
 */
static bool read_configuration(int argc, char **argv)
{
    if (argc > 1) {
        upper_bound = atol(argv[1]);
    } else {
        printf("Enter a number: ");
        if (scanf("%ld", &upper_bound) != 1) {
            fprintf(stderr, "Error: could not read an integer.\n");
            return false;
        }
    }
    nthreads = (argc > 2) ? atoi(argv[2]) : (int)sysconf(_SC_NPROCESSORS_ONLN);

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
        printf("  thread %-3d chunks=%-6ld busy=%.6f s\n",
               workers[t].id, workers[t].chunks_done, workers[t].busy);
    }
    printf("Imbalance (slowest/average) = %.4f   (1.0 is perfect)\n",
           max / (sum / nthreads));
}


int main(int argc, char **argv) {
    struct timespec start, end;
    pthread_t      *tid     = NULL;
    worker_t       *workers = NULL;
    long           count    = 0;
    
    if (!read_configuration(argc, argv)) return 1;

    if (upper_bound < 2) {
        printf("No primes are strictly less than %ld.\n", upper_bound);
        return 0;
    }
 
    //Aim for ~64 chunks per thread: enough that the workload evens out, few
    //enough that each chunk still covers many whole cache lines.
    chunk_size = upper_bound / (nthreads * CHUNKS_PER_THREAD);
    if (chunk_size < CACHE_LINE_BYTES) {
        chunk_size = CACHE_LINE_BYTES;
    }

    flags = calloc(upper_bound, 1);
    if (flags == NULL) {
        fprintf(stderr, "Error: could not allocate memory for %ld candidates.\n", upper_bound);
        return 1;
    }

    tid = malloc((size_t)nthreads * sizeof(pthread_t));
    if (tid == NULL) {
        fprintf(stderr, "Error: could not allocate memory for %d thread handles.\n", nthreads);
        free(flags);
        return 1;
    }

    workers = calloc((size_t)nthreads, sizeof(worker_t));
    if (workers == NULL) {
        fprintf(stderr, "Error: could not allocate memory for %d worker structs.\n", nthreads);
        free(flags);
        free(tid);
        return 1;
    }
 
    //Time the threads only. Allocation and output sit outside the timed region,
    //as in task1.c, so the ratio of the two times is a valid speedup.
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int t = 0; t < nthreads; t++) {
        workers[t].id = t;
        if (pthread_create(&tid[t], NULL, test_primes, &workers[t]) != 0) {
            fprintf(stderr, "Error: could not create thread %d.\n", t);
            for (int j = 0; j < t; j++)
                pthread_join(tid[j], NULL);   //wait for the threads that did start
            return 1;   //threads already started are abandoned; results would be partial anyway
        }
    }
    for (int t = 0; t < nthreads; t++)
        pthread_join(tid[t], NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
 
    //Scanning flags[] by index gives an ascending list for free, so no sort
    //is needed after the parallel phase.
    count = report_primes();

    if (count < 0) {
        free(flags);
        free(tid);
        free(workers);
        return 1;
    }

    report_threads(workers);
 
    printf("n=%ld  threads=%d  primes=%ld  time=%.6f s\n",
           upper_bound, nthreads, count, time_taken);
 
    free(flags);
    free(tid);
    free(workers);
    return 0;
}
 