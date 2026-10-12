#include <stdio.h>
#include <mpi.h>
#include "wsn.h"

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

    MPI_Comm_free(&cartesian_comm);
}