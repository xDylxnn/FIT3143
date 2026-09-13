/* task1.c - counts the primes below n in serial computation. */


#include <stdio.h> // printf, scanf, fopen, fprintf, fclose
#include <stdbool.h> //Gives the 'bool' type with values 'true' and 'false'
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>   // calloc, free

#define FILE_NAME "primes_serial.txt"

static long upper_bound;
static char *flags;  //array of flags indicating whether each number is prime

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

/* Always write the sorted list, including an empty file for n <= 2. */
static long report_primes(void) {
    FILE *fptr = fopen(FILE_NAME, "w");
    if (!fptr) {
        fprintf(stderr, "Error: could not open %s for writing.\n", FILE_NAME);
        return -1;
    }
    long count = 0;
    bool valid = true;
    for (long i = 2; i < upper_bound; i++) {
        if (flags[i]) {
            if (fprintf(fptr, "%ld\n", i) < 0) { valid = false; break; }
            count++;
        }
    }
    if (fclose(fptr) != 0) valid = false;
    if (!valid) {
        fprintf(stderr, "Error: failed writing %s.\n", FILE_NAME);
        return -1;
    }
    return count;
}

static double wall_time(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    // Same workload boundary as MPI: input/setup through file close.
    double overall_start = wall_time();
    char *end;
    errno = 0;
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n: 0..INT_MAX>\n", argv[0]);
        return 1;
    }
    upper_bound = strtol(argv[1], &end, 10);
    if (errno || end == argv[1] || *end || upper_bound < 0 || upper_bound > INT_MAX) {
        fprintf(stderr, "Error: n must be an integer from 0 to INT_MAX.\n");
        return 1;
    }
    flags = calloc(upper_bound > 0 ? (size_t)upper_bound : 1, sizeof(char));
    if (!flags) {
        fprintf(stderr, "Error: could not allocate memory for %ld candidates.\n", upper_bound);
        return 1;
    }
    double compute_start = wall_time();
    for (long i = 2; i < upper_bound; i++) flags[i] = is_prime(i);
    double output_start = wall_time();
    long count = report_primes();
    double finished = wall_time();
    if (count < 0) { free(flags); return 1; }
    double compute_time = output_start - compute_start;
    printf("n=%ld primes=%ld\n", upper_bound, count);
    printf("Overall wall-clock time: %.9f seconds\n", finished - overall_start);
    printf("Sorted primes written to %s\n", FILE_NAME);
    printf("RESULT,serial,%ld,1,1,%ld,%.9f,%.9f,%.9f,0.000000000,%.9f,%.9f,1.000000000\n",
           upper_bound, count, finished - overall_start, compute_start - overall_start,
           compute_time, finished - output_start, compute_time);
    free(flags);
    return 0;
}
