#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define CHUNKS_PER_THREAD   64
#define FILE_NAME "primes2.txt"

/*
 * Same as task1.c. Once multiples of 2 and 3 are ruled out, every remaining
 * factor is 6k +/- 1, so the loop steps by 6 and tests two divisors at a time.
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
 * Reads n from argv[1] and the thread count from argv[2] (defaults to the
 * OpenMP default, normally the core count). Rejects invalid or out-of-range input.
 */
static bool read_configuration(int argc, char **argv, long *upper_bound, int *threads) {
    if (argc < 2 || argc > 3) return false;
    char *end;
    errno = 0;
    *upper_bound = strtol(argv[1], &end, 10);
    if (errno || end == argv[1] || *end || *upper_bound < 0 || *upper_bound > INT_MAX)
        return false;
    long requested = omp_get_max_threads();
    if (argc == 3) {
        errno = 0;
        requested = strtol(argv[2], &end, 10);
        if (errno || end == argv[2] || *end) return false;
    }
    if (requested < 1 || requested > INT_MAX) return false;
    *threads = (int)requested;
    return true;
}

/*
 * Returns the number of candidates owned by this rank.
 */
static int get_individual_count(long candidates, long chunk_size, int rank, int size) {
    long all_chunks = candidates / chunk_size;
    long owned_chunks = all_chunks / size + (rank < all_chunks % size);
    long count = owned_chunks * chunk_size;
    if (rank == all_chunks % size) count += candidates % chunk_size;
    return (int)count;
}

/*
 * Same as task1.c. Root walks the chunks in global order, so the output is
 * already sorted - no sort step is needed.
 */
static long report_primes(long upper_bound, long chunk_size, int size, const char *flags, const int *indecies) {
    FILE *fptr = fopen(FILE_NAME, "w");
    if (fptr == NULL) {
        fprintf(stderr, "Error: could not open %s for writing.\n", FILE_NAME);
        return -1;
    }

    long candidates = upper_bound > 2 ? upper_bound - 2 : 0;
    long completed_chunks = candidates / chunk_size + (candidates % chunk_size != 0);
    long count = 0;
    bool valid = true;

    for (long c = 0; c < completed_chunks && valid; c++) {
        int rank = (c % size);
        long start = indecies[rank] + (c / size) * chunk_size;
        long lo = 2 + c * chunk_size;
        long length = upper_bound - lo;
        if (length > chunk_size) length = chunk_size;

        for (long offset = 0; offset < length; offset++) {
            if (flags[start + offset]) {
                if (fprintf(fptr, "%ld\n", lo + offset) < 0) {
                    valid = false;
                    break;
                }
                count++;
            }
        }
    }

    if (fclose(fptr) != 0) valid = false;
    if (!valid) {
        fprintf(stderr, "Error: failed writing %s.\n", FILE_NAME);
        return -1;
    }
    return count;
}

/*
 * Marks the primes in this rank's chunk range. Each thread marks its own
 * portion of the flags array, and the total number of chunks completed is
 * returned. The caller records the time spent in this function.
 * Each thread's local index is used to calculate where its chunk lives in
 * flags[], so that threads don't write to each other's memory.
 * The caller times this function and gathers busy times and chunk counts
 * for load-balance diagnostics after the overall timing interval ends.
 */
static long test_primes(long upper_bound, long chunk_size, int my_rank, int size, char *flags) {
    long candidates = upper_bound > 2 ? upper_bound - 2 : 0;
    long total_chunks = candidates / chunk_size + (candidates % chunk_size != 0);
    long completed_chunks = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:completed_chunks)
    for (long c = my_rank; c < total_chunks; c += size) {
        long local_index = (c - my_rank) / size;     // 0, 1, 2, ... within this rank
        long position = local_index * chunk_size;    // where this chunk lives in flags[]
        long lo = 2 + c * chunk_size;
        long length = upper_bound - lo;
        if (length > chunk_size) length = chunk_size;

        for (long offset = 0; offset < length; offset++)
            flags[position + offset] = is_prime(lo + offset);

        completed_chunks++;
    }

    return completed_chunks;
}

int main(int argc, char **argv) {
    int my_rank, size, provided;

    // FUNNELED: threads exist, but only the main thread makes MPI calls.
    // All MPI calls below are outside the parallel region, so this is enough.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Error: MPI does not provide MPI_THREAD_FUNNELED.\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // One root-clock interval: input/setup through successful file close.
    // MPI startup, this initial barrier, diagnostics and finalisation are excluded.
    MPI_Barrier(MPI_COMM_WORLD);
    double overall_start = MPI_Wtime();

    long upper_bound = 0, chunk_size = 1;
    int threads = 1;

    if (my_rank == 0) {
        if (!read_configuration(argc, argv, &upper_bound, &threads)) {
            fprintf(stderr, "Usage: %s <n> [threads]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Aim for ~64 chunks per thread across the whole job, as in task3.c
        chunk_size = upper_bound / ((long)size * threads * CHUNKS_PER_THREAD);
        if (chunk_size < 1) chunk_size = 1;
    }

    // Root owns configuration parsing and broadcasts the validated values.
    // Once in the process, n is a normal shared variable visible to every thread.
    MPI_Bcast(&upper_bound, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&chunk_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    long candidates = upper_bound > 2 ? upper_bound - 2 : 0;
    int count = get_individual_count(candidates, chunk_size, my_rank, size);
    char *flags = calloc(count > 0 ? count : 1, 1);   // shared by all threads in this process
    char *gathered_flags = NULL;
    int *candidates_per_process = NULL, *indecies = NULL;
    double *busy_times = NULL;
    long *chunk_counts = NULL;

    if (my_rank == 0) {
        gathered_flags = calloc(candidates > 0 ? candidates : 1, 1);
        candidates_per_process = calloc(size, sizeof(int));
        indecies = calloc(size, sizeof(int));
        busy_times = calloc(size, sizeof(double));
        chunk_counts = calloc(size, sizeof(long));

        if (!gathered_flags || !candidates_per_process || !indecies || !busy_times || !chunk_counts) {
            fprintf(stderr, "Error: root allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int offset = 0;
        for (int r = 0; r < size; r++) {
            candidates_per_process[r] = get_individual_count(candidates, chunk_size, r, size);
            indecies[r] = offset;
            offset += candidates_per_process[r];
        }
    }

    if (!flags) {
        fprintf(stderr, "Error: rank %d allocation failed.\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Setup includes waiting for all ranks to be ready for computation.
    MPI_Barrier(MPI_COMM_WORLD);
    double compute_start = MPI_Wtime();
    double setup_time = compute_start - overall_start;
    long chunks_done = test_primes(upper_bound, chunk_size, my_rank, size, flags);
    double gather_start = MPI_Wtime();
    double busy = gather_start - compute_start;

    MPI_Gatherv(flags, count, MPI_CHAR, gathered_flags, candidates_per_process,
                indecies, MPI_CHAR, 0, MPI_COMM_WORLD);
    double output_start = MPI_Wtime();
    double gather_time = output_start - gather_start;

    int status = 0;
    long primes = 0;
    double output_time = 0, overall_time = 0;
    if (my_rank == 0) {
        primes = report_primes(upper_bound, chunk_size, size, gathered_flags, indecies);
        double finished = MPI_Wtime();
        output_time = finished - output_start;
        overall_time = finished - overall_start;
        if (primes < 0) status = 1;
    }

    // Collect diagnostics AFTER the measured file close. These do not contribute
    // to overall_s. Per-rank durations use local clocks, never cross-host timestamps.
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&busy, 1, MPI_DOUBLE, busy_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&chunks_done, 1, MPI_LONG, chunk_counts, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (my_rank == 0 && status == 0) {
        double sum = 0, maximum = 0;
        for (int r = 0; r < size; r++) {
            sum += busy_times[r];
            if (busy_times[r] > maximum) maximum = busy_times[r];
            printf("  rank %-3d candidates=%-10d chunks=%-6ld busy=%.9f s\n",
                   r, candidates_per_process[r], chunk_counts[r], busy_times[r]);
        }
        double imbalance = sum > 0 ? maximum / (sum / size) : 0;
        printf("Imbalance (slowest/average) = %.6f\n", imbalance);
        printf("Overall wall-clock time: %.9f seconds\n", overall_time);
        printf("Sorted primes written to %s\n", FILE_NAME);
        // Root phases sum to overall_s. compute_max_s is a separate diagnostic;
        // gather_root_s can include waiting for slower ranks, so do not add both.
        printf("RESULT,hybrid,%ld,%d,%d,%ld,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
               upper_bound, size, threads, primes, overall_time, setup_time, busy,
               gather_time, output_time, maximum, imbalance);
    }

    free(flags);
    free(gathered_flags);
    free(candidates_per_process);
    free(indecies);
    free(busy_times);
    free(chunk_counts);
    MPI_Finalize();
    return status;
}
