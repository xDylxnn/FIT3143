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
}

int neighbour_communication(MPI_Comm cartesian_comm, int my_availability, int threshold,
                            const int neighbour_ranks[4], int neighbour_availability[4]) {
    MPI_Request requests[8], replies[4], cycle_barrier;
    int request_count = 0, reply_count = 0;
    int request_signal = 1;
    int barrier_started = 0, cycle_finished = 0;

    for (int i = 0; i < 4; i++) {
        neighbour_availability[i] = -1; // No neighbour, or availability not requested.
        if (my_availability <= threshold && neighbour_ranks[i] != MPI_PROC_NULL) {
            MPI_Irecv(&neighbour_availability[i], 1, MPI_INT, neighbour_ranks[i],
                      TAG_NEIGHBOUR_REPLY, cartesian_comm, &requests[request_count++]);
            MPI_Isend(&request_signal, 1, MPI_INT, neighbour_ranks[i],
                      TAG_NEIGHBOUR_REQUEST, cartesian_comm, &requests[request_count++]);
        }
    }

    // All nodes answer requests, including healthy nodes and nodes awaiting replies.
    // A node joins the barrier after its own replies arrive, but keeps answering
    // requests until every node has finished. This avoids unmatched requests.
    while (!cycle_finished) {
        MPI_Status status;
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_NEIGHBOUR_REQUEST, cartesian_comm, &flag, &status);
        while (flag) {
            int incoming_request;
            MPI_Recv(&incoming_request, 1, MPI_INT, status.MPI_SOURCE,
                     TAG_NEIGHBOUR_REQUEST, cartesian_comm, MPI_STATUS_IGNORE);
            MPI_Isend(&my_availability, 1, MPI_INT, status.MPI_SOURCE,
                      TAG_NEIGHBOUR_REPLY, cartesian_comm, &replies[reply_count++]);
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_NEIGHBOUR_REQUEST, cartesian_comm, &flag, &status);
        }

        if (!barrier_started) {
            int requests_finished;
            MPI_Testall(request_count, requests, &requests_finished, MPI_STATUSES_IGNORE);
            if (requests_finished) {
                MPI_Ibarrier(cartesian_comm, &cycle_barrier);
                barrier_started = 1;
            }
        }
        if (barrier_started) MPI_Test(&cycle_barrier, &cycle_finished, MPI_STATUS_IGNORE);
        if (!cycle_finished) usleep(1000);
    }

    MPI_Waitall(reply_count, replies, MPI_STATUSES_IGNORE);
    return request_count + 2 * reply_count; // Count sends and receives at this node.
}

int find_most_available_neighbour(int neighbour_availability[4], int threshold) {
    int best_neighbour_index = -1;

    for (int i = 0; i < 4; i++) {
        if (neighbour_availability[i] > threshold &&
            (best_neighbour_index == -1 ||
             neighbour_availability[i] > neighbour_availability[best_neighbour_index])) {
            best_neighbour_index = i;
        }
    }
    return best_neighbour_index;
}

// Each congested cycle represents one incoming EV asking for charging directions.
// Resolve the waiting requests by giving each EV one destination; no EV MPI
// process or reservation of the destination's ports is needed in this simulation.
void redirect_evs(int rank, int *waiting_evs, const int destinations[], int count) {
    for (int i = 0; i < count; i++) {
        int ev_count = *waiting_evs / count + (i < *waiting_evs % count);
        if (ev_count > 0) {
            printf("Node %d: directing %d waiting EV(s) to charging node %d\n",
                   rank, ev_count, destinations[i]);
        }
    }
    *waiting_evs = 0;
}

Report create_report(int node_rank, int x, int y, int my_availability,
                     const int neighbour_ranks[4], const int neighbour_availability[4],
                     int messages_exchanged) {
    Report report;

    report.node_rank = node_rank;
    report.x = x;
    report.y = y;
    report.my_availability = my_availability;
    for (int i = 0; i < 4; i++) {
        report.neighbour_ranks[i] = neighbour_ranks[i];
        report.neighbour_availability[i] = neighbour_availability[i];
    }

    report.msgs_exchanged = messages_exchanged;
    report.bytes_exchanges = messages_exchanged * sizeof(int); // Requests and replies each contain one int.

    timestamp(report.timestamp, sizeof(report.timestamp));

    return report;
}

void handle_base_station_message(int rank, int base_station_rank, int *terminated,
                                 int *waiting_evs, int *awaiting_reply, MPI_Status *status) {
    if (status->MPI_TAG == TAG_TERMINATE) {
        int terminate_signal;
        MPI_Recv(&terminate_signal, 1, MPI_INT, base_station_rank, TAG_TERMINATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *terminated = 1;
        printf("Node %d received terminate signal from base station\n", rank);

    } else if (status->MPI_TAG == TAG_REPLY) {
        BaseStationReply reply;
        MPI_Recv(&reply, sizeof(BaseStationReply), MPI_BYTE, base_station_rank, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *awaiting_reply = 0;

        if (reply.count > 0) {
            printf("Node %d received reply from base station: %d nodes, send_time=%.2f\n", rank, reply.count, reply.send_time);
            for (int i = 0; i < reply.count; i++) {
                printf("  Base Station suggests node: %d\n", reply.nodes[i]);
            }
            redirect_evs(rank, waiting_evs, reply.nodes, reply.count);
        } else {
            printf("Node %d: notifying %d waiting EV(s) that no nodes are available nearby\n",
                   rank, *waiting_evs);
            *waiting_evs = 0;
        }
    }
}

void check_base_station(int rank, int base_station_rank, int *terminated,
                        int *waiting_evs, int *awaiting_reply) {
    MPI_Status status;
    int flag;

    MPI_Iprobe(base_station_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
    while (flag) {
        handle_base_station_message(rank, base_station_rank, terminated,
                                    waiting_evs, awaiting_reply, &status);
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
    int waiting_evs = 0, awaiting_reply = 0;
    for (int cycle = 1; ; cycle++) {
        check_base_station(rank, config->base_station_rank, &terminated,
                           &waiting_evs, &awaiting_reply);

        int stop_all_nodes;
        MPI_Allreduce(&terminated, &stop_all_nodes, 1, MPI_INT, MPI_LOR, cartesian_comm);
        if (stop_all_nodes) {
            printf("Node %d cycle %d: terminating\n", rank, cycle);
            break;
        }

        int my_availability = simulate_ports(ports, k);

        int neighbour_availability[4];
        int messages_exchanged = neighbour_communication(cartesian_comm, my_availability,
                                  config->threshold, neighbour_ranks, neighbour_availability);

        printf("Node %d cycle %d: %d free\n", rank, cycle, my_availability);

        if (my_availability <= config->threshold) {
            waiting_evs++;
            printf("Node %d cycle %d: requested availability | up=%d down=%d left=%d right=%d\n",
                   rank, cycle, neighbour_availability[UP], neighbour_availability[DOWN],
                   neighbour_availability[LEFT], neighbour_availability[RIGHT]);
            int best_neighbour = find_most_available_neighbour(neighbour_availability, config->threshold);
            if (best_neighbour != -1) {
                int destination = neighbour_ranks[best_neighbour];
                redirect_evs(rank, &waiting_evs, &destination, 1);
            } else if (!awaiting_reply) {
                printf("Node %d cycle %d: no neighbour above threshold\n", rank, cycle);

                if (report_request != MPI_REQUEST_NULL) {
                    MPI_Wait(&report_request, MPI_STATUS_IGNORE);
                }

                report = create_report(rank, coords[0], coords[1], my_availability,
                                       neighbour_ranks, neighbour_availability, messages_exchanged);
                report.sent_time = MPI_Wtime();

                MPI_Isend(&report, sizeof(Report), MPI_BYTE, config->base_station_rank, TAG_REPORT, MPI_COMM_WORLD, &report_request);
                awaiting_reply = 1;
                printf("Node %d cycle %d: report sent to base station\n", rank, cycle);
            }
        } else if (waiting_evs > 0) {
            redirect_evs(rank, &waiting_evs, &rank, 1);
        }
        sleep(config->cycle_time);
    }

    // Handle any remaining messages from the base station after termination
    while (!terminated) {
        MPI_Status status;
        // Use Probe instead of IProbe to ensure CPU is not wasted in a busy wait loop. This will block until a message is available.
        MPI_Probe(config->base_station_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        handle_base_station_message(rank, config->base_station_rank, &terminated,
                                    &waiting_evs, &awaiting_reply, &status);
    }

    if (report_request != MPI_REQUEST_NULL) MPI_Wait(&report_request, MPI_STATUS_IGNORE);

    if (waiting_evs > 0) {
        printf("Node %d: cancelling %d waiting EV request(s) at shutdown\n", rank, waiting_evs);
    }

    MPI_Send(&rank, 1, MPI_INT, config->base_station_rank, TAG_COMPLETE, MPI_COMM_WORLD);
    printf("Node %d: sent complete signal to base station\n", rank);

    free(ports);
    MPI_Comm_free(&cartesian_comm);
}
