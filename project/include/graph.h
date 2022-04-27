#pragma once

typedef int node;

typedef struct {
    node from;
    node to;
    int weight;
} edge;

typedef struct {
    int num_nodes;
    int num_edges;
    edge *edges;
} graph;
