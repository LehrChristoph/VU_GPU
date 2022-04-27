#include <stdlib.h>
#include <stdio.h>

#include "generator.h"

dense_graph *generate(int num_nodes, float density, int min_weight, int max_weight) {
    if (num_nodes <= 0 || density < 0 || density > 1) {
        printf("Invalid input\n");
        return NULL;
    }
    dense_graph *graph = malloc(sizeof(dense_graph));
    graph->num_nodes = num_nodes;
    graph->num_edges = (int) (num_nodes * (num_nodes - 1) * density / 2);
    graph->nodes = calloc(num_nodes, sizeof(dense_node));
    graph->nodes->edges = malloc(sizeof(dense_edge) * graph->num_edges);
    for (int i = 0; i < graph->num_edges; i++) {
        int from = random() % num_nodes;
        graph->nodes[from].num_edges++;
    }
    dense_edge* edge_ptr = graph->nodes->edges;
    for (int i = 0; i < num_nodes; i++) {
        graph->nodes[i].edges = edge_ptr;
        int blocked = 1 << i;
        for (int j = 0; j < graph->nodes[i].num_edges; j++) {
            int to = random() % num_nodes;
            while (blocked & (1 << to)) {
                if (++to == num_nodes) {
                    to = 0;
                }
            }
            blocked |= 1 << to;
            *(edge_ptr++) = to;
        }
    }
    return graph;
}
