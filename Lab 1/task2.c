#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>     /* sysconf, to detect the core count */
#include <pthread.h>

#define NUM_THREADS 8

//same isPrime function as in task1.c, which checks if a number is prime or not
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
//Global variables
//Shared data. Written once before the threads start, then read-only.
long num;
char *flags; //* means pointer and points at the first byte of the array.
int nthreads;
long chunk;
//Struct bundles several variables into one unit.
//One of these per thread, each thread. gets its own pointer, so nothing is shared between them.
typedef struct {
    int id;
    long chunks_done;
    double busy;
} worker_t; //_t means is a type name

//the timing helper
//Returns the current time in seconds as a double, using the CLOCK_MONOTONIC clock (wall-clock seconds).
double now(void) {
    struct timespec t; //create a struct with two fields.
    clock_gettime(CLOCK_MONOTONIC, &t); //fill in the struct with the current time. & means "address of", so we are passing a pointer to the struct to the function, which fills in the fields.
    return t.tv_sec + t.tv_nsec / 1e9; //return the seconds plus the nanoseconds divided by 1e9 to convert to seconds.
}

//The function that each thread runs. It takes a void pointer as an argument, which is cast to a worker_t pointer. Each thread computes prime numbers in chunks, based on its id and the total number of threads.
void *test_primes(void *arg) {
    worker_t *w = (worker_t *)arg;      // cast the void* back to our struct pointer
    double t0 = now();
 
    //Take chunk number id, then id+nthreads, id+2*nthreads, ... until the
    //work runs out. Each chunk is a run of consecutive numbers.
    for (long c = w->id; 2 + c * chunk < num; c += nthreads) {
        //convert a chunk number into an actual range
        long lo = 2 + c * chunk;
        long hi = lo + chunk;
        if (hi > num) hi = num;         //clamp the final, partial chunk
 
        for (long i = lo; i < hi; i++)
            if (isPrime(i)) flags[i] = 1;   //only write on a hit: writing 0s;
        w->chunks_done++;
    }
 
    w->busy = now() - t0; // record the time spent in this thread
    return NULL;
}


int main(int argc, char **argv) {
    struct timespec start, end;
 
    //Read n and the thread count from the command line so the experiment
    //script can sweep them without recompiling. Falls back to prompting.
    if (argc > 1) {
        num = atol(argv[1]); // convert the first command line argument to a long integer
    } else {
        printf("Enter a number: ");
        if (scanf("%ld", &num) != 1) { //%id means long decimal, & means scanf needs to write into num
            fprintf(stderr, "Error: could not read an integer.\n");
            return 1;
        }
    }
    nthreads = (argc > 2) ? atoi(argv[2]) : (int)sysconf(_SC_NPROCESSORS_ONLN); // convert the second command line argument to an integer, or use the number of available processors if not provided
 
    if (nthreads < 1 || nthreads > 256) {
        fprintf(stderr, "Error: thread count must be between 1 and 256.\n");
        return 1;
    }
    if (num < 2) {
        printf("No primes are strictly less than %ld.\n", num);
        return 0;
    }
 
    //Aim for about 64 chunks per thread: enough chunks that the workload
    //evens out, few enough that each chunk still covers many whole cache lines.
    chunk = num / (nthreads * 64);
    if (chunk < 64) chunk = 64;
 
    flags = calloc((size_t)num, 1);
    pthread_t *tid = malloc((size_t)nthreads * sizeof(pthread_t)); // allocate an array of thread IDs
    worker_t *w = calloc((size_t)nthreads, sizeof(worker_t));
    if (flags == NULL || tid == NULL || w == NULL) {
        fprintf(stderr, "Error: could not allocate memory for %ld candidates.\n", num);
        return 1;
    }
 
    //Phase 1: timed computation. Allocation and file output are outside the
    //timed region, exactly as in task1.c, so the two times measure the same
    //thing and the ratio between them is a valid speedup.
    clock_gettime(CLOCK_MONOTONIC, &start);
 
    for (int t = 0; t < nthreads; t++) {
        w[t].id = t;
        if (pthread_create(&tid[t], NULL, test_primes, &w[t]) != 0) { //phread_create takes 4 arguements: a pointer to a pthread_t variable, a pointer to a pthread_attr_t variable (NULL means default attributes), a pointer to the function to run, and a pointer to the argument to pass to the function. It returns 0 on success and an error code on failure.
            fprintf(stderr, "Error: could not create thread %d.\n", t);
            return 1;
        }
    }
    for (int t = 0; t < nthreads; t++)
        pthread_join(tid[t], NULL); //Join waits for the thread to finish. It takes two arguments: a pthread_t variable and a pointer to a void pointer (NULL means we don't care about the return value). It returns 0 on success and an error code on failure.
 
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
        FILE *fptr = fopen("primes2.txt", "w");
        if (fptr == NULL) {
            fprintf(stderr, "Error: could not open primes2.txt for writing.\n");
            free(flags);
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
        printf("  thread %-3d chunks=%-6ld busy=%.6f s\n",
               w[t].id, w[t].chunks_done, w[t].busy);
    }
    printf("Imbalance (slowest/average) = %.4f   (1.0 is perfect)\n",
           max / (sum / nthreads));
 
    printf("n=%ld  threads=%d  primes=%ld  time=%.6f s\n",
           num, nthreads, count, time_taken);
 
    free(flags);
    free(tid);
    free(w);
    return 0;
}
 