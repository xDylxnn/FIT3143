#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include "wsn.h"
#include <unistd.h>

#define FREE_PORT 1
#define OCCUPIED_PORT 0

#define OCCUPY_CHANCE 0.5
#define RELEASE_CHANCE 0.5

int simulate_ports(int *ports, int k) {
    int free_ports = 0;

    #pragma omp parallel for num_threads(k) reduction(+:free_ports)
    for (int i = 0; i < k; i++) {
        if (ports[i] == FREE_PORT) {
            // Random chance to occupy a free port
            if ((double)rand() / RAND_MAX < OCCUPY_CHANCE) {
                ports[i] = OCCUPIED_PORT;
            } else {
                free_ports++;
            }
        } else {
            // Random chance to release an occupied port
            if ((double)rand() / RAND_MAX < RELEASE_CHANCE) {
                ports[i] = FREE_PORT;
                free_ports++;
            }
        }
    }
    return free_ports;
};


void run_node(MPI_Comm nodes, const SimulationConfig *config) {
    MPI_Comm cartesian_comm;
    int dimensions[2] = {config->m, config->n};
    int periods[2] = {0, 0}; // No wrap-around
    int rank, coords[2], neighbor_ranks[4];

    // Create a 2D Cartesian communicator for the grid of nodes and get the rank and coordinates of the current node
    // This is called by each node process to set up its own view of the grid and its neighbors.
    MPI_Cart_create(nodes, 2, dimensions, periods, 0, &cartesian_comm);
    MPI_Comm_rank(cartesian_comm, &rank);
    MPI_Cart_coords(cartesian_comm, rank, 2, coords);

    // Get the ranks of the neighboring nodes (up, down, left, right)
    MPI_Cart_shift(cartesian_comm, 0, 1, &neighbor_ranks[UP], &neighbor_ranks[DOWN]);
    MPI_Cart_shift(cartesian_comm, 1, 1, &neighbor_ranks[LEFT], &neighbor_ranks[RIGHT]);

    printf("Node %d (%d,%d): up=%d down=%d left=%d right=%d\n", rank, coords[0], coords[1], neighbor_ranks[UP], neighbor_ranks[DOWN], neighbor_ranks[LEFT], neighbor_ranks[RIGHT]);

    int k = config->k;
    int *ports = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) {
        ports[i] = FREE_PORT; // Initialize all ports as free
    }

    // TEMPORARY test: run 5 cycles and print availability
    for (int cycle = 1; cycle <= 5; cycle++) {
        int availability = simulate_ports(ports, k);
        printf("Node %d cycle %d: %d/%d free\n", rank, cycle, availability, k);
        sleep(config->cycle_time);
    }

    free(ports);

    MPI_Comm_free(&cartesian_comm);
}