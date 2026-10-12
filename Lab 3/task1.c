#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include "wsn.h"
#include <unistd.h>
#include <time.h>

#define FREE_PORT 1
#define OCCUPIED_PORT 0

#define OCCUPY_CHANCE 0.5
#define RELEASE_CHANCE 0.5

void timestamp(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info);
}

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

void neighbour_communication(MPI_Comm cartesian_comm, int my_availability, const int neighbour_ranks[4], int neighbour_availability[4]) {
    MPI_Request requests[8];

    for (int i = 0; i < 4; i++) {
        neighbour_availability[i] = -1; // Initialize neighbour availability to -1 (no neighbour)
        // Send my availability to each neighbour
        MPI_Isend(&my_availability, 1, MPI_INT, neighbour_ranks[i], TAG_NEIGHBOUR, cartesian_comm, &requests[i]);
        // Receive neighbour availability from each neighbour
        MPI_Irecv(&neighbour_availability[i], 1, MPI_INT, neighbour_ranks[i], TAG_NEIGHBOUR, cartesian_comm, &requests[i + 4]);
        
    }
    MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
}

int find_most_available_neighbour(int neighbour_availability[4], int threshold) {
    int best_neighbour_index = -1;

    for (int i = 0; i < 4; i++) {
        if (neighbour_availability[i] > threshold && (best_neighbour_index == -1 || neighbour_availability[i] > neighbour_availability[best_neighbour_index])) {
            best_neighbour_index = neighbour_availability[i];
            best_neighbour_index = i;
        }
    }
    return best_neighbour_index;
}

Report create_report(int node_rank, int x, int y, int my_availability, const int neighbour_ranks[4], const int neighbour_availability[4]) {
    Report report;
    int real_neighbours = 0;

    for (int i = 0; i < 4; i++) {
        if (neighbour_ranks[i] != MPI_PROC_NULL) {
            real_neighbours++;
        }
    }

    report.node_rank = node_rank;
    report.x = x;
    report.y = y;
    report.my_availability = my_availability;
    for (int i = 0; i < 4; i++) {
        report.neighbour_ranks[i] = neighbour_ranks[i];
        report.neighbour_availability[i] = neighbour_availability[i];
    }

    report.msgs_exchanged = 2 * real_neighbours; // Each neighbour exchange consists of one send and one receive
    report.bytes_exchanges = 2 * real_neighbours * sizeof(int); // Each exchange consists of one int sent and one int received

    timestamp(report.timestamp, sizeof(report.timestamp));

    return report;
}

void handle_base_station_message(int rank, int base_station_rank, int *terminated, MPI_Status *status) {
    if (status->MPI_TAG == TAG_TERMINATE) {
        int terminate_signal;
        MPI_Recv(&terminate_signal, 1, MPI_INT, base_station_rank, TAG_TERMINATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *terminated = 1;
        printf("Node %d received terminate signal from base station\n", rank);

    } else if (status->MPI_TAG == TAG_REPLY) {
        BaseStationReply reply;
        MPI_Recv(&reply, sizeof(BaseStationReply), MPI_BYTE, base_station_rank, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if (reply.count > 0) {
            printf("Node %d received reply from base station: %d nodes, send_time=%.2f\n", rank, reply.count, reply.send_time);
            for (int i = 0; i < reply.count; i++) {
                printf("  Base Station suggests node: %d\n", reply.nodes[i]);
            }
        } else {
            printf("Node %d received reply from base station: no nodes available\n", rank);
        }
    }
}

void check_base_station(int rank, int base_station_rank, int *terminated) {
        MPI_Status status;
        int flag;
    
        MPI_Iprobe(base_station_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        while (flag) {
            handle_base_station_message(rank, base_station_rank, terminated, &status);
            MPI_Iprobe(base_station_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        }
}

void run_node(MPI_Comm nodes, const SimulationConfig *config) {
    MPI_Comm cartesian_comm;
    int dimensions[2] = {config->m, config->n};
    int periods[2] = {0, 0}; // No wrap-around
    int rank, coords[2], neighbour_ranks[4];

    // Create a 2D Cartesian communicator for the grid of nodes and get the rank and coordinates of the current node
    // This is called by each node process to set up its own view of the grid and its neighbours.
    MPI_Cart_create(nodes, 2, dimensions, periods, 0, &cartesian_comm);
    MPI_Comm_rank(cartesian_comm, &rank);
    MPI_Cart_coords(cartesian_comm, rank, 2, coords);

    // Get the ranks of the neighbouring nodes (up, down, left, right)
    MPI_Cart_shift(cartesian_comm, 0, 1, &neighbour_ranks[UP], &neighbour_ranks[DOWN]);
    MPI_Cart_shift(cartesian_comm, 1, 1, &neighbour_ranks[LEFT], &neighbour_ranks[RIGHT]);

    printf("Node %d (%d,%d): up=%d down=%d left=%d right=%d\n", rank, coords[0], coords[1], neighbour_ranks[UP], neighbour_ranks[DOWN], neighbour_ranks[LEFT], neighbour_ranks[RIGHT]);

    int k = config->k;
    int *ports = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) {
        ports[i] = FREE_PORT; // Initialize all ports as free
    }

    Report report;
    MPI_Request report_request = MPI_REQUEST_NULL;

    int terminated = 0;
    for (int cycle = 1; ; cycle++) {
        check_base_station(rank, config->base_station_rank, &terminated);

        int stop_all_nodes;
        MPI_Allreduce(&terminated, &stop_all_nodes, 1, MPI_INT, MPI_LOR, cartesian_comm);
        if (stop_all_nodes) {
            printf("Node %d cycle %d: terminating\n", rank, cycle);
            break;
        }

        int my_availability = simulate_ports(ports, k);

        int neighbour_availability[4];
        neighbour_communication(cartesian_comm, my_availability,
                                neighbour_ranks, neighbour_availability);

        printf("Node %d cycle %d: %d free | up=%d down=%d left=%d right=%d\n",
               rank, cycle, my_availability,
               neighbour_availability[UP], neighbour_availability[DOWN],
               neighbour_availability[LEFT], neighbour_availability[RIGHT]);

        if (my_availability < config->threshold) {
            int best_neighbour = find_most_available_neighbour(neighbour_availability, config->threshold);
            if (best_neighbour != -1) {
                printf("Node %d cycle %d: forwarding to neighbour %d\n", rank, cycle, neighbour_ranks[best_neighbour]);
                // Implement the logic to forward data to the best neighbour
            } else {
                printf("Node %d cycle %d: no neighbour above threshold\n", rank, cycle);

                if (report_request != MPI_REQUEST_NULL) {
                    MPI_Wait(&report_request, MPI_STATUS_IGNORE);
                }

                report = create_report(rank, coords[0], coords[1], my_availability, neighbour_ranks, neighbour_availability);
                report.sent_time = MPI_Wtime();

                MPI_Isend(&report, sizeof(Report), MPI_BYTE, config->base_station_rank, TAG_REPORT, MPI_COMM_WORLD, &report_request);
                printf("Node %d cycle %d: report sent to base station\n", rank, cycle);
            }
        }
        sleep(config->cycle_time);
    }

    // Handle any remaining messages from the base station after termination
    while (!terminated) {
        MPI_Status status;
        // Use Probe instead of IProbe to ensure CPU is not wasted in a busy wait loop. This will block until a message is available.
        MPI_Probe(config->base_station_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        handle_base_station_message(rank, config->base_station_rank, &terminated, &status);
    }

    if (report_request != MPI_REQUEST_NULL) MPI_Wait(&report_request, MPI_STATUS_IGNORE);

    MPI_Send(&rank, 1, MPI_INT, config->base_station_rank, TAG_COMPLETE, MPI_COMM_WORLD);
    printf("Node %d: sent complete signal to base station\n", rank);

    free(ports);
    MPI_Comm_free(&cartesian_comm);
}