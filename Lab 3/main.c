/*
 * main.c - Entry point for the EV charging wireless sensor network simulation.
 *
 * FIT3143 Lab #3 - Distributed Wireless Sensor Network
 * Team: <Name 1> (<ID>, <email>), <Name 2> (<ID>, <email>)
 *
 * Responsibilities (and nothing more):
 *   1. Start MPI with thread support (OpenMP threads are used inside each node).
 *   2. Read and validate the command-line settings.
 *   3. Split MPI_COMM_WORLD into the charging nodes and the base station.
 *   4. Hand control to run_node() or run_base().
 *   5. Shut MPI down cleanly.
 *
 * Usage:
 *   mpirun -np <m*n+1> ./wsn m n k [threshold] [cycle_time] [simulation_time]
 *
 * Example (3x3 grid, 6 ports, full at <=1 free port, 2 s cycles, 30 s run):
 *   mpirun -np 10 ./wsn 3 3 6 1 2 30
 */
#include <stdio.h>
#include <stdlib.h>
#include "wsn.h"
#include <unistd.h>

/* Defaults for the optional arguments */
#define DEFAULT_THRESHOLD  1
#define DEFAULT_cycle_time  2
#define DEFAULT_simulation_time    30

/* Parse a positive integer argument; returns -1 if it is not one. */
static int parse_positive(const char *s)
{
    char *end;
    long v = strtol(s, &end, 10);
    if (*s == '\0' || *end != '\0' || v <= 0 || v > 1000000) return -1;
    return (int)v;
}

/* Parse a non-negative integer argument (threshold may be 0); -1 on error. */
static int parse_non_negative(const char *s)
{
    char *end;
    long v = strtol(s, &end, 10);
    if (*s == '\0' || *end != '\0' || v < 0 || v > 1000000) return -1;
    return (int)v;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: mpirun -np <m*n+1> %s m n k [threshold] [cycle_time] [simulation_time]\n"
        "  m, n       grid rows and columns (m*n charging nodes)\n"
        "  k          charging ports per node\n"
        "  threshold  node is full when free ports <= threshold (default %d)\n"
        "  cycle_time  seconds between port updates             (default %d)\n"
        "  simulation_time    simulation length before termination     (default %d)\n",
        prog, DEFAULT_THRESHOLD, DEFAULT_cycle_time, DEFAULT_simulation_time);
}

/*
 * Fill cfg from argv. Every rank parses the same arguments, so no broadcast is
 * needed. Only rank 0 prints errors. Returns 0 on success, -1 on any invalid input.
 */
static int parse_config(int argc, char **argv, int comm_size,
                        int rank, SimulationConfig *cfg)
{
    if (argc < 4 || argc > 7) return -1;

    cfg->m         = parse_positive(argv[1]);
    cfg->n         = parse_positive(argv[2]);
    cfg->k         = parse_positive(argv[3]);
    cfg->threshold = (argc > 4) ? parse_non_negative(argv[4]) : DEFAULT_THRESHOLD;
    cfg->cycle_time = (argc > 5) ? parse_positive(argv[5])     : DEFAULT_cycle_time;
    cfg->simulation_time   = (argc > 6) ? parse_positive(argv[6])     : DEFAULT_simulation_time;

    if (cfg->m < 0 || cfg->n < 0 || cfg->k < 0 ||
        cfg->threshold < 0 || cfg->cycle_time < 0 || cfg->simulation_time < 0)
        return -1;

    if (cfg->threshold >= cfg->k) {
        if (rank == 0) fprintf(stderr, "Error: threshold (%d) must be less than k (%d).\n",
                cfg->threshold, cfg->k);
        return -1;
    }

    if (comm_size != cfg->m * cfg->n + 1) {
        if (rank == 0) fprintf(stderr, "Error: %d x %d grid needs %d processes "
                        "(%d nodes + 1 base station), but got %d.\n",
                cfg->m, cfg->n, cfg->m * cfg->n + 1, cfg->m * cfg->n, comm_size);
        return -1;
    }

    /* The base station is always the last rank, so node ranks 0..m*n-1
     * match the grid numbering in the spec (Node 0 = (0,0), ...). */
    cfg->base_station_rank = comm_size - 1;
    return 0;
}

// TEMPORARY test base station. Your partner replaces this for Task 2.
void run_base(const SimulationConfig *config) {
    double start = MPI_Wtime();

    while (MPI_Wtime() - start < config->simulation_time) {
        int flag;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_REPORT, MPI_COMM_WORLD, &flag, &status);
        if (!flag) { usleep(1000); continue; }   // nothing yet, check again in 1 ms

        Report report;
        MPI_Recv(&report, sizeof(Report), MPI_BYTE, status.MPI_SOURCE, TAG_REPORT,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("BASE: report from node %d (%d,%d) at %s, %d free\n",
               report.node_rank, report.x, report.y, report.timestamp, report.my_availability);

        BaseStationReply reply = {0};   // count 0 = "no nodes available" (placeholder)
        reply.send_time = MPI_Wtime();
        MPI_Send(&reply, sizeof(reply), MPI_BYTE, report.node_rank, TAG_REPLY, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv)
{
    /* Only the master thread of each process calls MPI, so FUNNELED is enough. */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0)
            fprintf(stderr, "Error: MPI library lacks MPI_THREAD_FUNNELED support.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    SimulationConfig cfg;
    if (parse_config(argc, argv, comm_size, rank, &cfg) != 0) {
        /* Every rank reaches the same verdict, so all exit together cleanly. */
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        printf("WSN simulation: %d x %d grid, %d ports/node, threshold %d, "
               "cycle %d s, run %d s, base station = rank %d\n",
               cfg.m, cfg.n, cfg.k, cfg.threshold,
               cfg.cycle_time, cfg.simulation_time, cfg.base_station_rank);
        fflush(stdout);
    }

    /*
     * Split off the charging nodes. The base station passes MPI_UNDEFINED and
     * gets MPI_COMM_NULL back, because it never joins the grid communicator.
     * Key = rank keeps the node ranks in the same order as the world.
     */
    int is_base = (rank == cfg.base_station_rank);
    MPI_Comm node_comm;
    MPI_Comm_split(MPI_COMM_WORLD, is_base ? MPI_UNDEFINED : 0,
                   rank, &node_comm);

    if (is_base) {
        run_base(&cfg);
        // fprintf(stderr, "Base station functionality not implemented yet.\n");
    } else {
        run_node(node_comm, &cfg);
        MPI_Comm_free(&node_comm);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}