#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "generator.h"

dense_graph *generate(int num_nodes, float density, int min_weight, int max_weight) {
    if (num_nodes <= 0 || density < 0 || density > 1) {
        printf("Invalid input: num_nodes (%d) or density (%f) \n", num_nodes, density);
        return NULL;
    }
    int num_edges = (int) (num_nodes * (num_nodes - 1) * density / 2);
    dense_graph *graph = malloc(sizeof(dense_graph) + sizeof(dense_node) * num_nodes + sizeof(dense_edge) * num_edges * 2);
    graph->num_nodes = num_nodes;
    graph->num_edges = num_edges;
    graph->nodes = ((void*) graph) + sizeof(dense_graph);
    memset(graph->nodes, 0, sizeof(dense_node) * num_nodes);
    graph->nodes->edges = ((void*) graph) + sizeof(dense_graph) + sizeof(dense_node) * num_nodes;

    bool *adjacency_matrix = calloc(num_nodes * num_nodes, sizeof(bool));

    for (int i = 0; i < num_edges; i++) {
        int from = random() % num_nodes;
        int orig_to = random() % num_nodes;
        int to = orig_to;
        while (from == to || adjacency_matrix[from * num_nodes + to]) {
            to = (to + 1) % num_nodes;
            if (to == orig_to) {
                i--;
                break;
            }
        }
        if (from != to) {
          adjacency_matrix[from * num_nodes + to] = true;
          adjacency_matrix[to * num_nodes + from] = true;
        }
    }

    dense_edge* edge_ptr = graph->nodes->edges;
    int k = 0;
    for (int i = 0; i < num_nodes; i++) {
        graph->nodes[i].edges = edge_ptr;

        for (int j = 0; j < num_nodes; j++) {
            if (adjacency_matrix[i * num_nodes + j]) {
                graph->nodes[i].num_edges++;
                *(edge_ptr++) = j;
            }
        }
    }
    free(adjacency_matrix);
    return graph;
}
