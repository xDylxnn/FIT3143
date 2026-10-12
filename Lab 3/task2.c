/*
 * base.c - Task 2: the base station (one MPI process, one thread).
 *
 * Follows the master/slave idea from the Week 11 prep (Task 3): the base
 * station is the master on MPI_COMM_WORLD and the charging nodes are slaves.
 *
 * Three versions of the receive loop, for comparing in the presentation:
 *   1 = blocking     MPI_Iprobe, then MPI_Recv / MPI_Send
 *   2 = nonblocking  one MPI_Irecv(MPI_ANY_SOURCE) + MPI_Test, reply with MPI_Isend
 *   3 = Testsome     one MPI_Irecv per node, MPI_Testsome handles all arrivals at once
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "wsn.h"

#define POLL_US 1000   // nothing arrived: wait 1 ms before checking again

static FILE *log_file;
static double *last_report;   // time each node last sent a report  (-1 = never)
static double *last_full;     // time each node was last known full (-1 = never)
static int alerts = 0, found = 0, late = 0;
static double total_delay = 0;

static double now(void) { return MPI_Wtime() - sim_start; }

/* Process one report and fill in the reply. Used by all three versions. */
static void handle_report(const Report *r, const SimulationConfig *cfg, BaseStationReply *reply)
{
    double t = now();
    double delay = t - r->sent_time;
    int node = r->node_rank;
    alerts++;
    total_delay += delay;

    // 1. Update what we know: the reporter is full, and so are its neighbours
    last_report[node] = last_full[node] = t;
    int neighbours = 0, recent = 0;
    for (int i = 0; i < 4; i++) {
        int nb = r->neighbour_ranks[i];
        if (nb < 0) continue;   // MPI_PROC_NULL: no neighbour on this side
        neighbours++;
        if (r->neighbour_availability[i] <= cfg->threshold) last_full[nb] = t;
        if (last_report[nb] >= 0 && t - last_report[nb] <= cfg->period) recent++;
    }

    // 2. If every neighbour reported recently, their reports already say the
    //    nodes 2 hops away are full -> no nodes available nearby.
    // 3. Otherwise suggest ALL nodes 2 hops away not known to be full
    //    (they are equally near, so all of them are reported).
    reply->count = 0;
    if (recent < neighbours) {
        for (int i = 0; i < cfg->m * cfg->n; i++) {
            int hops = abs(i / cfg->n - r->x) + abs(i % cfg->n - r->y);
            int known_full = last_full[i] >= 0 && t - last_full[i] <= cfg->period;
            if (hops == 2 && !known_full) reply->nodes[reply->count++] = i;
        }
    }
    if (reply->count > 0) found++;
    reply->send_time = now();

    // 4. Log the report and the decision
    fprintf(log_file, "[%7.3f s] Alert #%d from node %d (%d,%d) at %s: %d free, delay %.3f ms\n",
            t, alerts, node, r->x, r->y, r->timestamp, r->my_availability, delay * 1000);
    fprintf(log_file, "    neighbours (free ports):");
    for (int i = 0; i < 4; i++)
        if (r->neighbour_ranks[i] >= 0)
            fprintf(log_file, " %d(%d)", r->neighbour_ranks[i], r->neighbour_availability[i]);
    fprintf(log_file, " | %d messages, %d bytes exchanged\n", r->msgs_exchanged, r->bytes_exchanges);
    fprintf(log_file, "    %d/%d neighbours reported in the last %d s -> ",
            recent, neighbours, cfg->period);
    if (reply->count == 0) fprintf(log_file, "no nodes available nearby\n");
    else {
        fprintf(log_file, "suggest nodes");
        for (int i = 0; i < reply->count; i++) fprintf(log_file, " %d", reply->nodes[i]);
        fprintf(log_file, "\n");
    }
}

/* At the end, take back a receive that is still posted.
 * If a report already landed in it, it arrived too late to answer. */
static void cancel_recv(MPI_Request *request, const Report *r)
{
    MPI_Status status;
    int cancelled;
    MPI_Cancel(request);
    MPI_Wait(request, &status);
    MPI_Test_cancelled(&status, &cancelled);
    if (!cancelled) {
        late++;
        fprintf(log_file, "Late alert from node %d (not answered)\n", r->node_rank);
    }
}

/* Graceful shutdown: tell every node to stop, then wait for each node's
 * COMPLETE. Reports sent just before a node stopped arrive first and are
 * received here, so no message is left unmatched at MPI_Finalize. */
static void shutdown_nodes(const SimulationConfig *cfg)
{
    int num_nodes = cfg->m * cfg->n, stop = 0;
    fprintf(log_file, "[%7.3f s] Time is up: sending TERMINATE to all nodes\n", now());
    for (int i = 0; i < num_nodes; i++)
        MPI_Send(&stop, 1, MPI_INT, i, TAG_TERMINATE, MPI_COMM_WORLD);

    for (int i = 0; i < num_nodes; i++) {
        while (1) {
            MPI_Status status;
            MPI_Probe(i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_COMPLETE) {
                int done;
                MPI_Recv(&done, 1, MPI_INT, i, TAG_COMPLETE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            }
            Report r;
            MPI_Recv(&r, sizeof(Report), MPI_BYTE, i, TAG_REPORT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            late++;
            fprintf(log_file, "Late alert from node %d (not answered)\n", i);
        }
    }
}

/* Version 1: blocking send/receive */
static void run_blocking(const SimulationConfig *cfg)
{
    Report report;
    BaseStationReply reply;

    while (now() < cfg->simulation_time) {
        int flag;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_REPORT, MPI_COMM_WORLD, &flag, &status);
        if (!flag) { usleep(POLL_US); continue; }

        MPI_Recv(&report, sizeof(Report), MPI_BYTE, status.MPI_SOURCE, TAG_REPORT,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        handle_report(&report, cfg, &reply);
        MPI_Send(&reply, sizeof(reply), MPI_BYTE, status.MPI_SOURCE, TAG_REPLY, MPI_COMM_WORLD);
    }
    shutdown_nodes(cfg);
}

/* Version 2: nonblocking, one receive for any node */
static void run_nonblocking(const SimulationConfig *cfg)
{
    int num_nodes = cfg->m * cfg->n;
    Report report;
    BaseStationReply *replies = malloc(num_nodes * sizeof(BaseStationReply)); // one per node
    MPI_Request *send_req = malloc(num_nodes * sizeof(MPI_Request));
    MPI_Request recv_req;
    for (int i = 0; i < num_nodes; i++) send_req[i] = MPI_REQUEST_NULL;

    MPI_Irecv(&report, sizeof(Report), MPI_BYTE, MPI_ANY_SOURCE, TAG_REPORT,
              MPI_COMM_WORLD, &recv_req);
    while (now() < cfg->simulation_time) {
        int flag;
        MPI_Status status;
        MPI_Test(&recv_req, &flag, &status);
        if (!flag) { usleep(POLL_US); continue; }

        int src = status.MPI_SOURCE;
        MPI_Wait(&send_req[src], MPI_STATUS_IGNORE);   // last reply to src finished
        handle_report(&report, cfg, &replies[src]);
        MPI_Isend(&replies[src], sizeof(BaseStationReply), MPI_BYTE, src, TAG_REPLY,
                  MPI_COMM_WORLD, &send_req[src]);
        MPI_Irecv(&report, sizeof(Report), MPI_BYTE, MPI_ANY_SOURCE, TAG_REPORT,
                  MPI_COMM_WORLD, &recv_req);
    }
    cancel_recv(&recv_req, &report);
    shutdown_nodes(cfg);
    MPI_Waitall(num_nodes, send_req, MPI_STATUSES_IGNORE);
    free(replies);
    free(send_req);
}

/* Version 3: nonblocking, one receive per node, checked together with Testsome */
static void run_testsome(const SimulationConfig *cfg)
{
    int num_nodes = cfg->m * cfg->n;
    Report *reports = malloc(num_nodes * sizeof(Report));
    BaseStationReply *replies = malloc(num_nodes * sizeof(BaseStationReply));
    MPI_Request *recv_req = malloc(num_nodes * sizeof(MPI_Request));
    MPI_Request *send_req = malloc(num_nodes * sizeof(MPI_Request));
    int *arrived = malloc(num_nodes * sizeof(int));

    for (int i = 0; i < num_nodes; i++) {
        send_req[i] = MPI_REQUEST_NULL;
        MPI_Irecv(&reports[i], sizeof(Report), MPI_BYTE, i, TAG_REPORT,
                  MPI_COMM_WORLD, &recv_req[i]);
    }
    while (now() < cfg->simulation_time) {
        int count;
        MPI_Testsome(num_nodes, recv_req, &count, arrived, MPI_STATUSES_IGNORE);
        if (count == 0) { usleep(POLL_US); continue; }

        for (int k = 0; k < count; k++) {
            int i = arrived[k];
            MPI_Wait(&send_req[i], MPI_STATUS_IGNORE);
            handle_report(&reports[i], cfg, &replies[i]);
            MPI_Isend(&replies[i], sizeof(BaseStationReply), MPI_BYTE, i, TAG_REPLY,
                      MPI_COMM_WORLD, &send_req[i]);
            MPI_Irecv(&reports[i], sizeof(Report), MPI_BYTE, i, TAG_REPORT,
                      MPI_COMM_WORLD, &recv_req[i]);
        }
    }
    for (int i = 0; i < num_nodes; i++) cancel_recv(&recv_req[i], &reports[i]);
    shutdown_nodes(cfg);
    MPI_Waitall(num_nodes, send_req, MPI_STATUSES_IGNORE);
    free(reports); free(replies); free(recv_req); free(send_req); free(arrived);
}

void run_base(const SimulationConfig *cfg)
{
    int num_nodes = cfg->m * cfg->n;
    last_report = malloc(num_nodes * sizeof(double));
    last_full = malloc(num_nodes * sizeof(double));
    for (int i = 0; i < num_nodes; i++) last_report[i] = last_full[i] = -1;

    char filename[64];
    sprintf(filename, "base_log_%dx%d_p%d_v%d.txt", cfg->m, cfg->n, cfg->period, cfg->base_mode);
    log_file = fopen(filename, "w");
    if (log_file == NULL) { perror("fopen"); log_file = stdout; }

    fprintf(log_file, "Base station: %d x %d grid, k=%d, full when free <= %d, "
            "cycle %d s, run %d s, period %d s, version %d\n\n",
            cfg->m, cfg->n, cfg->k, cfg->threshold, cfg->cycle_time,
            cfg->simulation_time, cfg->period, cfg->base_mode);

    if (cfg->base_mode == 1)      run_blocking(cfg);
    else if (cfg->base_mode == 2) run_nonblocking(cfg);
    else                          run_testsome(cfg);

    // Summary: reports + replies + one TERMINATE and one COMPLETE per node
    int messages = alerts + late + alerts + 2 * num_nodes;
    fprintf(log_file, "\n===== SUMMARY =====\n");
    fprintf(log_file, "Total run time        : %.3f s (incl. shutdown)\n", now());
    fprintf(log_file, "Alerts answered       : %d (%d with suggestions, %d 'none nearby')\n",
            alerts, found, alerts - found);
    fprintf(log_file, "Late alerts           : %d\n", late);
    fprintf(log_file, "Messages at base      : %d\n", messages);
    fprintf(log_file, "Average report delay  : %.3f ms\n",
            alerts ? total_delay / alerts * 1000 : 0.0);

    printf("Base station finished: %d alerts, log written to %s\n", alerts, filename);
    if (log_file != stdout) fclose(log_file);
    free(last_report);
    free(last_full);
}