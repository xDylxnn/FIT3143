/* task1.c - counts the primes below n in serial computation. */


#include <stdio.h> // printf, scanf, fopen, fprintf, fclose
#include <stdbool.h> //Gives the 'bool' type with values 'true' and 'false'
#include <time.h>
#include <stdlib.h>   // calloc, free

#define FILE_NAME "primes1.txt"

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
 * Checks the amount of prime numbers below the user inputted upper_bound 
 * and prints them to stdout if upper_bound <= 100, or to a file otherwise.
 * Uses flags rather than inline printing to avoid the overhead of I/O in the timing measurement.
 */
int main() {
    struct timespec start, end;

    printf("Enter a number: ");
    if (scanf("%ld", &upper_bound) != 1) {
        fprintf(stderr, "Error: could not read an integer.\n");
        return 1;
    }

    if (upper_bound < 2) {
        printf("No primes are strictly less than %ld.\n", upper_bound);
        return 0;
    }
    
    flags = calloc((size_t)upper_bound, sizeof(char));
    if (flags == NULL) {
        fprintf(stderr, "Error: could not allocate memory for %ld candidates.\n", upper_bound);
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (long i = 2; i < upper_bound; i++)
        flags[i] = is_prime(i);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    long count = report_primes();
    if (count < 0) {
        free(flags);
        return 1;
    }

    printf("n=%ld  primes=%ld\n", upper_bound, count);
    printf("Time taken: %.6f seconds\n", time_taken);
    free(flags);

    return 0;
}