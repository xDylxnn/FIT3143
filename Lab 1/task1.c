#include <stdio.h>
#include <stdbool.h>

bool isPrime(long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;

    if (n % 2 == 0 || n % 3 == 0) return false;

    for (long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return 0;
        }
    }
    return 1;
}

int main() {
    FILE *fptr;

    long num;
    printf("Enter a number: ");
    scanf("%ld", &num);


    if (num <= 100) {
        for (long i = 0; i <= num; i++) {
            if (isPrime(i)) {
                printf("%ld ", i);
            }
        }
    }
    else {
        fptr = fopen("primes.txt", "w");

        for (long i = 0; i <= num; i++) {
            if (isPrime(i)) {
                fprintf(fptr, "%ld ", i);
            }
        }

        fclose(fptr);
    }

    return 0;
}