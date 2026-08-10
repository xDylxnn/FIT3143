#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 8

long num; 
char *flags;

bool isPrime(long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    for (long i = 5; i <= n / i; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

void *test_primes(void *arg) {
    long id = (long)arg;                                        // cast the void* argument back to long

    for (long i = id; i <= num; i += NUM_THREADS) {
        if (isPrime(i)) flags[i] = 1;
    }
    return NULL;
}

int main() {
    struct timespec start, end;

    pthread_t tid[NUM_THREADS];

    printf("Enter a number: ");
    scanf("%ld", &num);

    clock_gettime(CLOCK_MONOTONIC, &start);

    flags = calloc(num + 1, 1);

    for (long t = 0; t < NUM_THREADS; t++)
        pthread_create(&tid[t], NULL, test_primes, (void* )t);  // for the args passed it, in this case t, is it cast to void* because pthread_create expects a void* argument

    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(tid[t], NULL);

    if (num <= 100) {
        for (long i = 0; i <= num; i++)
            if (flags[i]) printf("%ld ", i);
    } else {
        FILE *fptr = fopen("primes2.txt", "w");
        for (long i = 0; i <= num; i++)
            if (flags[i]) fprintf(fptr, "%ld ", i);

        fclose(fptr);
    }

    free(flags);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("\nTime taken: %f seconds", time_taken);

    return 0;
}