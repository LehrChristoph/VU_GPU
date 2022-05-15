#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "impl.h"

clock_t calculate_connected_components(dense_graph *graph, connected_components** out) {
    clock_t start = clock();
    bool *used_nodes = calloc(graph->num_nodes, sizeof(bool));

    int num_components = 0;
    component *component_buffer = malloc(sizeof(component)*graph->num_nodes);

    int num_nodes;
    node *node_buffer = malloc(sizeof(node)*graph->num_nodes);
    int head;
    node *stack = malloc(sizeof(node)*graph->num_nodes);

    for (int i = 0; i < graph->num_nodes; i++) {
        if (!used_nodes[i]) {
            num_nodes = 0;
            head = 0;

            used_nodes[i] = true;
            stack[head++] = i;

            while (head > 0) {
                node cur = stack[--head];
                node_buffer[num_nodes++] = cur;
                dense_node cur_node = graph->nodes[cur];

                for (int j = 0; j < cur_node.num_edges; j++) {
                    node target_node = cur_node.edges[j];

                    if (used_nodes[target_node]) continue;

                    used_nodes[target_node] = true;
                    stack[head++] = target_node;
                }
            }

            size_t array_size = sizeof(node)*num_nodes;
            component_buffer[num_components].num_nodes = num_nodes;
            component_buffer[num_components].nodes = memcpy(malloc(array_size), node_buffer, array_size);
            num_components++;
        }
    }

    size_t array_size = sizeof(component)*num_components;
    connected_components *result = malloc(sizeof(connected_components));
    result->num_components = num_components;
    result->components = memcpy(malloc(array_size), component_buffer, array_size);

    free(component_buffer);

    *out = result;
    return clock() - start;
}
