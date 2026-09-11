#include <stdio.h>
#include <stdbool.h>
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
 * OpenMP default, normally the core count). Returns false if n is missing.
 */
static bool read_configuration(int argc, char **argv, long *upper_bound, int *threads) {
    if (argc < 2) return false;
    *upper_bound = atol(argv[1]);
    *threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    if (*threads < 1) *threads = 1;
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

    fclose(fptr);
    return count;
}

/*
 * Marks the primes in this rank's chunk range. Each thread marks its own
 * portion of the flags array, and the total number of chunks completed is
 * returned. The caller records the time spent in this function.
 * Each thread's local index is used to calculate where its chunk lives in
 * flags[], so that threads don't write to each other's memory.
 * Each rank's local chunk count is returned, so that the root can verify that
 * the partitioning was even.
 * Each rank's local busy time is returned, so that the root can report the
 * load imbalance.
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

    // Root read n and threads from argv; the others can't see argv, so broadcast.
    // Once in the process, n is a normal shared variable visible to every thread.
    MPI_Bcast(&upper_bound, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&chunk_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    omp_set_num_threads(threads);

    long candidates = upper_bound > 2 ? upper_bound - 2 : 0;
    int count = get_individual_count(candidates, chunk_size, my_rank, size);
    char *flags = calloc(count, 1);   // shared by all threads in this process
    char *gathered_flags = NULL;
    int *candidates_per_process = NULL, *indecies = NULL;
    double *busy_times = NULL;
    long *chunk_counts = NULL;

    if (my_rank == 0) {
        gathered_flags = calloc(candidates, 1);
        candidates_per_process = calloc(size, sizeof(int));
        indecies = calloc(size, sizeof(int));
        busy_times = calloc(size, sizeof(double));
        chunk_counts = calloc(size, sizeof(long));

        int offset = 0;
        for (int r = 0; r < size; r++) {
            candidates_per_process[r] = get_individual_count(candidates, chunk_size, r, size);
            indecies[r] = offset;
            offset += candidates_per_process[r];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();
    long chunks_done = test_primes(upper_bound, chunk_size, my_rank, size, flags);
    double busy = MPI_Wtime() - start;

    MPI_Gatherv(flags, count, MPI_CHAR, gathered_flags, candidates_per_process, indecies, MPI_CHAR, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start;
    double search_gather_time = 0;
    MPI_Reduce(&elapsed, &search_gather_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Gather(&busy, 1, MPI_DOUBLE, busy_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&chunks_done, 1, MPI_LONG, chunk_counts, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    int status = 0;
    if (my_rank == 0) {
        long primes = report_primes(upper_bound, chunk_size, size, gathered_flags, indecies);

        if (primes < 0) {
            status = 1;
        } else {
            double sum = 0, maximum = 0;
            for (int r = 0; r < size; r++) {
                sum += busy_times[r];
                if (busy_times[r] > maximum) maximum = busy_times[r];
                printf("  rank %-3d candidates=%-10d chunks=%-6ld busy=%.6f s\n",
                       r, candidates_per_process[r], chunk_counts[r], busy_times[r]);
            }
            if (sum > 0)
                printf("Imbalance (slowest/average) = %.4f\n", maximum / (sum / size));

            printf("n=%ld processes=%d threads/process=%d chunk_size=%ld primes=%ld\n",
                   upper_bound, size, threads, chunk_size, primes);
            printf("Search + gather time: %.6f seconds\n", search_gather_time);
            printf("Slowest local search: %.6f seconds\n", maximum);
            printf("Sorted primes written to %s\n", FILE_NAME);
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    free(flags);
    free(gathered_flags);
    free(candidates_per_process);
    free(indecies);
    free(busy_times);
    free(chunk_counts);

    MPI_Finalize();
    return status;
}