#include <stdio.h> // printf, scanf, fopen, fprintf, fclose
#include <stdbool.h> //Gives the 'bool' type with values 'true' and 'false'
#include <time.h>
#include <stdlib.h>   // calloc, free
//a function taking a long (a 64 bit integer, so it comfortably handles large numbers) and returning a bool (true or false)
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

int main() {
    struct timespec start, end;
    long num;
    printf("Enter a number: ");
    //errors handling for scanf, if the user does not enter a valid long integer, an error message is printed and the program exits with a non-zero status code.
    if (scanf("%ld", &num) != 1) {
        fprintf(stderr, "Error: could not read an integer.\n");
        return 1;
    }
    //If the number is less than 2, print a message and exit the program.
    if (num < 2) {
        printf("No primes are strictly less than %ld.\n", num);
        return 0;
    }
    
    char *flags = calloc((size_t)num, sizeof(char));
    if (flags == NULL) {
        fprintf(stderr, "Error: could not allocate memory for %ld candidates.\n", num);
        return 1;
    }

    //Phase 1: timed computation
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (long i = 2; i < num; i++)
        flags[i] = isPrime(i);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    // Phase 2: output in ascending order
    //If the number is less than or equal to 100, print the prime numbers to the console. Otherwise, write them to a file named "primes.txt".
    if (num <= 100) {
        for (long i = 2; i < num; i++)
            if (flags[i]) printf("%ld ", i);
    } else {
        FILE *fptr = fopen("primes.txt", "w");
        //If the file cannot be opened for writing, print an error message and exit the program.
        if (fptr == NULL) {
        fprintf(stderr, "Error: could not open primes.txt for writing.\n");
        free(flags);
        return 1;
        }
        //Write the prime numbers to the file, one per line.
        for (long i = 2; i < num; i++)
            if (flags[i]) fprintf(fptr, "%ld\n", i);
        fclose(fptr);
      
    }

    //Print the time taken for the computation in seconds with six decimal places.
    printf("\nTime taken: %.6f seconds\n", time_taken);
    free(flags);
    return 0;
}