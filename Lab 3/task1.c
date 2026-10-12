#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 9) {
        printf("This program requires exactly 9 processes (got %d).\n", world_size);
        MPI_Finalize();
        return -1;
    }

    int ndims = 2;
    int dims[2] = {3, 3};      // 3 rows and 3 columns
    int periods[2] = {0, 0};   // Non-periodic in both dimensions (0 = false, 1 = true)
    int reorder = 1;           // Allow rank reordering

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cart_comm);

    int rank, coords[2];
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, ndims, coords);

    printf("Rank %d has coordinates (%d, %d)\n", rank, coords[0], coords[1]);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}