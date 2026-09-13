// task1.c - counts the primes below n in parallel using Message Passing Interface (MPI).

#include <stdio.h> // printf, scanf, fopen, fprintf, fclose
#include <stdbool.h> //Gives the 'bool' type with values 'true' and 'false'
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>   // calloc, free
#include <mpi.h> // MPI functions

#define CHUNKS_PER_THREAD   64 
#define CACHE_LINE_BYTES    64  
#define FILE_NAME "primes1.txt"

/*
 * Once multiples of 2 and 3 are ruled out, every remaining factor is 6k +/- 1, 
 * so the loop steps by 6 and tests two divisors at a time.
 * Returns true if n is prime, false otherwise.
 */
static bool is_prime(long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    //every prime number greater than 3 can be written in the form 6k ± 1, 
    //where k is a positive integer. This loop checks for factors of n in that form.
    for (long i = 5; i <= n / i; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false; //Checks for factors of n in the form 6k ± 1
    return true;
}

/*
 * Root reads and validates n from argv. Counts must fit MPI_Gatherv's int API.
 */
static bool read_configuration(int argc, char **argv, long *upper_bound) {
    if (argc != 2) return false;
    char *end;
    errno = 0;
    *upper_bound = strtol(argv[1], &end, 10);
    // MPI_Gatherv uses int counts/displacements in this implementation.
    return !errno && end != argv[1] && !*end && *upper_bound >= 0 && *upper_bound <= INT_MAX;
}

/*
 * Returns the number of candidates owned by this rank, given the total number
 */
static int get_individual_count(long candidates, long chunk_size, int rank, int size) {
    long all_chunks = candidates / chunk_size;
    long owned_chunks = all_chunks / size + (rank < all_chunks % size);
    long count = owned_chunks * chunk_size;
    if (rank == all_chunks % size) count += candidates % chunk_size;
    return (int)count;
}


/*
 * Writes the sorted primes to FILE_NAME; returns the count or -1 on file error.
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
 * Marks this rank's cyclic chunks in local order and returns its chunk count.
 */
static long test_primes(long upper_bound, long chunk_size, int my_rank, int size, char *flags) {
    
    long candidates = upper_bound > 2 ? upper_bound - 2 : 0;
    long total_chunks = candidates / chunk_size + (candidates % chunk_size != 0);
    long completed_chunks = 0;
    int position = 0;

    for (long c = my_rank; c < total_chunks; c += size) {
        //turn the chunk number into a range of consecutive candidates
        long lo = 2 + c * chunk_size;
        long length = upper_bound - lo;
        if (length > chunk_size) length = chunk_size;  //clamp the final, partial chunk

        for (long offset = 0; offset < length; offset++)
            flags[position++] = is_prime(lo + offset);

        completed_chunks++;
    }

    return completed_chunks;
}



int main(int argc, char **argv) {
    int my_rank, size;

    MPI_Init(&argc, &argv); //Start the runtime
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //Derive the rank of this process in the communicator
    MPI_Comm_size(MPI_COMM_WORLD, &size); //count how many of us

    // One root-clock interval: input/setup through successful file close.
    // MPI startup, this initial barrier, diagnostics and finalisation are excluded.
    MPI_Barrier(MPI_COMM_WORLD);
    double overall_start = MPI_Wtime();

    long upper_bound = 0, chunk_size = 1;
    //Set the default values for the upper bound and chunk size
    if (my_rank == 0) {
        if (!read_configuration(argc, argv, &upper_bound)) {
            fprintf(stderr, "Usage: %s <n: 0..INT_MAX>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    chunk_size = upper_bound / ((long)size * CHUNKS_PER_THREAD);
    if (chunk_size < 1) chunk_size = 1;
    
    // Root owns configuration parsing and broadcasts the validated values.
    MPI_Bcast(&upper_bound, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&chunk_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    //Calculate the number of candidates to be tested, excluding 0 and 1
    long candidates = upper_bound > 2 ? upper_bound - 2 : 0;
    int count = get_individual_count(candidates, chunk_size, my_rank, size);
    char *flags = calloc(count > 0 ? count : 1, 1);
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
        printf("RESULT,mpi,%ld,%d,1,%ld,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
               upper_bound, size, primes, overall_time, setup_time, busy,
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
