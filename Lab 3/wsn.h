typedef struct {
    int node_rank, x, y;
    int my_availability;
    int neighbour_ranks[4]; // Up, Down, Left, Right
    int neighbor_availability[4]; // -1 if neighbor does not exist, 0 if neighbor is unavailable, 1 if neighbor is available
    int msgs_exchanged, bytes_exchanges;
    char timestamp[32];
    double sent_time;
} Report;

typedef struct {
    //TODO
} BaseStation_Reply;

//TODO TAGS FOR MESSAGES BETWEEN NODES AND BASE STATION


