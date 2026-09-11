// task1.c - counts the primes below n in parallel using Message Passing Interface (MPI).

#include <stdio.h> // printf, scanf, fopen, fprintf, fclose
#include <stdbool.h> //Gives the 'bool' type with values 'true' and 'false'
#include <time.h>
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
 * Reads upper_bound and the thread count from argv, falling back to a 
 * prompt and to the core count. Returns false if either value is unusable.
 */
static bool read_configuration(int argc, char **argv, long *upper_bound) {
    if (argc > 1) {
        *upper_bound = atol(argv[1]); // convert the first command line argument to a long integer
    } else {
            return false;
    }

    return true;
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
 * Prints the primes found: to stdout for n <= 100, to FILE_NAME otherwise. 
 * Returns the count, or -1 on file error.
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
 * Worker entry point. arg is this thread's worker_t. Created chunk range from
 * w.id and chunk size, marks the primes it finds in flags[], and
 * records its chunk count and busy time in *w.
 */
static long test_primes(int upper_bound, int chunk_size, int my_rank, int size, char *flags) {
    
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

    long upper_bound = 0, chunk_size = 1;
    //Set the default values for the upper bound and chunk size
    if (my_rank == 0) {
        if (!read_configuration(argc, argv, &upper_bound)) {
            MPI_Finalize();
            return 1;
        }
    }

    chunk_size = upper_bound / (size * CHUNKS_PER_THREAD);
    //Root got n from argv, while others cant see it
    MPI_Bcast(&upper_bound, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&chunk_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    //Calculate the number of candidates to be tested, excluding 0 and 1
    long candidates = upper_bound > 2 ? upper_bound - 2 : 0;
    int count = get_individual_count(candidates, chunk_size, my_rank, size);
    char *flags = calloc(count, 1);
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

    //line up all processes before starting the timer
    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime(); //MPI's wall clock time
    long chunks_done = test_primes(upper_bound, chunk_size, my_rank, size, flags); //mark the primes in this thread's chunk range
    double busy = MPI_Wtime() - start; //record the time spent searching for primes
    //Concatenates every rank's local result buffer into root's buffer. The v variant is needed because ranks own different numbers of candidates, so it takes a counts array and a displacements array instead of one fixed count.
    MPI_Gatherv(flags, count, MPI_CHAR, gathered_flags, candidates_per_process, indecies, MPI_CHAR, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start;
    double search_gather_time = 0;
    // Slowest rank sets the wall clock, so MAX gives the true parallel time
    MPI_Reduce(&elapsed, &search_gather_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // Keep every rank's compute time so root can report the load imbalance
    MPI_Gather(&busy, 1, MPI_DOUBLE, busy_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Keep every rank's chunk count to verify the partitioning was even
    MPI_Gather(&chunks_done, 1, MPI_LONG, chunk_counts, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    int status = 0;
    //Root prints the results and writes the primes to a file
    if (my_rank == 0) {

        long count = report_primes(upper_bound, chunk_size, size, gathered_flags, indecies);

        if (count < 0) {
            status = 1;
        } 
        else {
            double sum = 0, maximum = 0;

            for (int r = 0; r < size; r++) {
                sum += busy_times[r];
                
                if (busy_times[r] > maximum) maximum = busy_times[r];
                printf("  rank %-3d candidates=%-10d chunks=%-6ld busy=%.6f s\n", r, candidates_per_process[r], chunk_counts[r], busy_times[r]);
            }

            if (sum > 0)
                printf("Imbalance (slowest/average) = %.4f\n", maximum / (sum / size));
            else
                printf("Imbalance: unavailable (below timer resolution).\n");

            printf("n=%ld processes=%d chunk_size=%ld primes=%ld\n", upper_bound, size, chunk_size, count);
            printf("Search + gather time: %.6f seconds\n", search_gather_time);
            printf("Slowest local search: %.6f seconds\n", maximum);
            printf("Sorted primes written to %s\n", FILE_NAME);
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD); //same exist code
    //Free the allocated memory for the flags, gathered_flags, candidates_per_process, indecies, busy_times, and chunk_counts arrays
    free(flags);
    free(gathered_flags);
    free(candidates_per_process);
    free(indecies);
    free(busy_times);
    free(chunk_counts);

    MPI_Finalize();

    return status;
}