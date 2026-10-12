#include <mpi.h>

#define TAG_NEIGHBOUR 1
#define TAG_REPORT 2
#define TAG_REPLY 3
#define TAG_TERMINATE 4

enum { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

typedef struct{
    int m, n;
    int k;
    int threshold;
    int cycle_time;
    int simulation_time;
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

// THIS IS A PLACEHOLDER: TODO
typedef struct {
    int    count;                       
    int    nodes[4 * 2];    
    double send_time;   
} BaseStationReply;



void run_node(MPI_Comm nodes, const SimulationConfig *config);
void run_base(const SimulationConfig *config);