#include "graph.h"
#include "io.h"
#include <stdlib.h>

dense_graph *to_dense(graph *graph) {
    dense_graph *out = malloc(sizeof(dense_graph));
    out->nodes = malloc(sizeof(dense_node) * graph->num_nodes);
    out->nodes->edges = malloc(sizeof(dense_edge) * graph->num_edges);
    out->num_nodes = graph->num_nodes;
    out->num_edges = graph->num_edges;
    dense_edge* edge_ptr = out->nodes->edges;
    dense_edge* old_ptr;
    for (int i = 0; i < graph->num_nodes; i++) {
        old_ptr = edge_ptr;
        out->nodes[i].edges = edge_ptr;
        for (int j = 0; j < graph->num_edges; j++) {
            if (graph->edges[j].from == i) {
                *(edge_ptr++) = graph->edges[j].to;
            }
        }
        out->nodes[i].num_edges = edge_ptr - old_ptr;
    }
    free_graph(graph);
    return out;
}

graph *from_dense(dense_graph *dgraph) {
    graph *out = malloc(sizeof(graph));
    out->num_nodes = dgraph->num_nodes;
    out->num_edges = dgraph->num_edges;
    out->edges = malloc(sizeof(edge) * dgraph->num_edges);
    edge* edge_ptr = out->edges;
    for (int i = 0; i < dgraph->num_nodes; i++) {
        for (int j = 0; j < dgraph->nodes[i].num_edges; j++) {
            edge_ptr->from = i;
            edge_ptr->to = dgraph->nodes[i].edges[j];
            edge_ptr->weight = 1; // we don't care about this
            edge_ptr++;
        }
    }
    free_dense(dgraph);
    return out;
}

graph free_dense(dense_graph *graph) {
    if (graph->nodes && graph->nodes->edges) {
        free(graph->nodes->edges);
    }
    if (graph->nodes) {
        free(graph->nodes);
    }
    free(graph);
}
