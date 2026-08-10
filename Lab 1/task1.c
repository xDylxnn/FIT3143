#include <stdio.h>
#include <stdbool.h>
#include <time.h>

bool isPrime(long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    for (long i = 5; i <= n / i; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

int main() {
    struct timespec start, end;
    long num;
    printf("Enter a number: ");
    scanf("%ld", &num);

    clock_gettime(CLOCK_MONOTONIC, &start);

    if (num <= 100) {
        for (long i = 0; i <= num; i++)
            if (isPrime(i)) printf("%ld ", i);
    } else {
        FILE *fptr = fopen("primes.txt", "w");
        for (long i = 0; i <= num; i++)
            if (isPrime(i)) fprintf(fptr, "%ld ", i);
        fclose(fptr);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("\nTime taken: %f seconds\n", time_taken);

    return 0;
}