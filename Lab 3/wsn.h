#include <mpi.h>

#define TAG_NEIGHBOUR_REQUEST 1
#define TAG_REPORT 2
#define TAG_REPLY 3
#define TAG_TERMINATE 4
#define TAG_COMPLETE 5
#define TAG_NEIGHBOUR_REPLY 6

enum { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

typedef struct{
    int m, n;
    int k;
    int threshold;
    int cycle_time;
    int simulation_time;
    int period;      // base station: a report newer than this (seconds) is "recent"
    int base_mode;   // 1 = blocking, 2 = nonblocking, 3 = nonblocking + Testsome
    int base_station_rank;
} SimulationConfig;

typedef struct {
    int node_rank;
    int x, y;
    int my_availability; // amount of free ports
    int neighbour_ranks[4]; // Up, Down, Left, Right
    int neighbour_availability[4]; // neighbours free ports
    int msgs_exchanged, bytes_exchanges;
    char timestamp[32];
    double sent_time;
} Report;

typedef struct {
    int    count;          // 0 = no nodes available nearby
    int    nodes[4 * 2];   // nodes 2 hops away: at most 8
    double send_time;
} BaseStationReply;

// Set in main right after an MPI_Barrier. MPI_Wtime() starts at a different
// point in every process, so all times are measured from this common start.
extern double sim_start;

void run_node(MPI_Comm nodes, const SimulationConfig *config);
void run_base(const SimulationConfig *config);
