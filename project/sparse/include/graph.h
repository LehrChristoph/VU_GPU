#pragma once
#include <stdbool.h>

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
void free_dense(dense_graph *graph);

typedef struct {
    int num_nodes;
    node *nodes;
} component;

typedef struct {
    unsigned int num_components;
    component *components;
    bool single_node_list;
} connected_components;

int compare_components(component first, component second);
int compare_connected_components(connected_components *first, connected_components *second);
void free_connected_components(connected_components *connected_components);
