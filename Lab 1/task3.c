#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

//same isPrime function as in task1.c and task2.c, which checks if a number is prime or not
bool isPrime(long n) {
    //Guard  clauses
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false; //Eliminates all even numbers and multiples of 3
    //every prime number greater than 3 can be written in the form 6k ± 1, where k is a positive integer. This loop checks for factors of n in that form.
    for (long i = 5; i <= n / i; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false; //Checks for factors of n in the form 6k ± 1
    return true;
}

//Struct bundles several variables into one unit.
//One of these per thread, so the presentation can show the measured workload split.
typedef struct {
    long iters_done;
    double busy;
} worker_t; //_t means is a type name

//the timing helper, identical to task2.c
//Returns the current time in seconds as a double, using the CLOCK_MONOTONIC clock (wall-clock seconds).
double now(void) {
    struct timespec t; //create a struct with two fields.
    clock_gettime(CLOCK_MONOTONIC, &t); //fill in the struct with the current time. & means "address of", so we are passing a pointer to the struct to the function, which fills in the fields.
    return t.tv_sec + t.tv_nsec / 1e9; //return the seconds plus the nanoseconds divided by 1e9 to convert to seconds.
}

int main(int argc, char **argv) {
    struct timespec start, end;
    long num;
    int nthreads;

    //Read n and the thread count from the command line so the experiment
    //script can sweep them without recompiling. Falls back to prompting.
    if (argc > 1) {
        num = atol(argv[1]); // convert the first command line argument to a long integer
    } else {
        printf("Enter a number: ");
        if (scanf("%ld", &num) != 1) { //%ld means long decimal, & means scanf needs to write into num
            fprintf(stderr, "Error: could not read an integer.\n");
            return 1;
        }
    }
    //omp_get_max_threads() is OpenMP's own default, normally the number of cores.
    nthreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();

    if (nthreads < 1 || nthreads > 256) {
        fprintf(stderr, "Error: thread count must be between 1 and 256.\n");
        return 1;
    }
    if (num < 2) {
        printf("No primes are strictly less than %ld.\n", num);
        return 0;
    }
    omp_set_num_threads(nthreads); //API call, overrides the default and OMP_NUM_THREADS

    //Same granularity as task2.c: aim for about 64 chunks per thread. Enough
    //chunks that the workload evens out, few enough that each chunk still
    //covers many whole cache lines.
    long chunk = num / ((long)nthreads * 64);
    if (chunk < 64) chunk = 64;

    char *flags = calloc((size_t)num, 1); //* means pointer and points at the first byte of the array.
    worker_t *w = calloc((size_t)nthreads, sizeof(worker_t));
    if (flags == NULL || w == NULL) {
        fprintf(stderr, "Error: could not allocate memory for %ld candidates.\n", num);
        return 1;
    }

    //Phase 1: timed computation. Allocation and file output are outside the
    //timed region, exactly as in task1.c and task2.c, so the three times
    //measure the same thing and the ratios between them are valid speedups.
    clock_gettime(CLOCK_MONOTONIC, &start);

    //Fork: one parallel region, the team is created once.
    //flags, num and chunk are shared (declared outside); everything declared
    //inside the region is private to each thread, so no false sharing.
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        long mine = 0;
        double t0 = now();

        //Work-sharing: OpenMP hands each thread a chunk of iterations and
        //gives it a new one as soon as it finishes. Cost per iteration grows
        //with i (trial division runs to sqrt(i)), so a self-scheduling policy
        //balances the load better than a fixed split.
        //nowait removes the barrier at the end of the loop so busy measures
        //this thread's own work and not its wait for the slowest thread.
        #pragma omp for schedule(dynamic, chunk) nowait
        for (long i = 2; i < num; i++) {
            if (isPrime(i)) flags[i] = 1; //only write on a hit: writing 0s;
            mine++;
        }

        w[id].iters_done = mine;    //written once, after the loop
        w[id].busy = now() - t0;    // record the time spent in this thread
    } //Join: implicit barrier here, all threads finish before main continues.

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    //Phase 2: output in ascending order. Scanning flags[] by index gives a
    //sorted list for free - no sorting step is needed after the parallel phase.
    long count = 0;
    if (num <= 100) {
        for (long i = 2; i < num; i++)
            if (flags[i]) { printf("%ld ", i); count++; }
        printf("\n");
    } else {
        FILE *fptr = fopen("primes3.txt", "w");
        if (fptr == NULL) {
            fprintf(stderr, "Error: could not open primes3.txt for writing.\n");
            free(flags);
            free(w);
            return 1;
        }
        for (long i = 2; i < num; i++)
            if (flags[i]) { fprintf(fptr, "%ld\n", i); count++; }
        fclose(fptr);
    }

    //Per-thread figures, so the presentation can show measured load balance
    //rather than just claiming it. A well-balanced run gives near-identical
    //busy times and an imbalance close to 1.0.
    double sum = 0, max = 0;
    for (int t = 0; t < nthreads; t++) {
        sum += w[t].busy;
        if (w[t].busy > max) max = w[t].busy;
        printf("  thread %-3d iters=%-10ld busy=%.6f s\n",
               t, w[t].iters_done, w[t].busy);
    }
    printf("Imbalance (slowest/average) = %.4f   (1.0 is perfect)\n",
           max / (sum / nthreads));

    printf("n=%ld  threads=%d  primes=%ld  time=%.6f s\n",
           num, nthreads, count, time_taken);

    free(flags);
    free(w);
    return 0;
}