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

typedef node dense_edge;

typedef struct {
    int num_edges;
    dense_edge *edges;
} dense_node;

typedef struct {
    int num_nodes;
    int num_edges;
    dense_node *nodes;
} dense_graph;

dense_graph *to_dense(graph *graph);
graph *from_dense(dense_graph *graph);
graph free_dense(dense_graph *graph);
