#include <stdlib.h>
#include <string.h>

#include "graph.h"
#include "io.h"

dense_graph *to_dense(graph *graph) {
    if (!graph) return NULL;
    dense_graph *out = malloc(sizeof(dense_graph));
    out->nodes = malloc(sizeof(dense_node) * graph->num_nodes);
    out->nodes->edges = malloc(sizeof(dense_edge) * graph->num_edges * 2);
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
            } else if (graph->edges[j].to == i) {
                *(edge_ptr++) = graph->edges[j].from;
            }
        }
        out->nodes[i].num_edges = edge_ptr - old_ptr;
    }
    return out;
}

graph *from_dense(dense_graph *dgraph) {
    if (!dgraph) return NULL;
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
    return out;
}

void free_dense(dense_graph *graph) {
    if (!graph) return;

    if (graph->nodes && graph->nodes->edges) {
        free(graph->nodes->edges);
    }
    if (graph->nodes) {
        free(graph->nodes);
    }
    free(graph);
}

int compare_connected_components(connected_components *first, connected_components *second) {
    if (first->num_components != second->num_components) return 2;
    bool *used = calloc(first->num_components, sizeof(bool));

    for (int f = 0; f < first->num_components; f++) {
        component searching = first->components[f];

        bool found = false;
        for (int s = 0; s < second->num_components; s++) {
            if (used[s]) continue;

            component trying = second->components[s];
            if (searching.num_nodes != trying.num_nodes) continue;

            if (compare_components(trying, searching)) {
                used[s] = true;
                found = true;
                break;
            }
        }

        if (!found) {
            free(used);
            return -f;
        }
    }

    free(used);
    return 1;
}

int compare_components(component first, component second) {
    bool *used = calloc(first.num_nodes, sizeof(bool));
    for (int i = 0; i < first.num_nodes; i++) {
        bool found_node = false;
        for (int j = 0; j < second.num_nodes; j++) {
            if (used[j]) continue;
            if (first.nodes[i] == second.nodes[j]) {
                used[j] = true;
                found_node = true;
                break;
            }
        }
        if (!found_node) {
            free(used);
            return -i;
        }
    }

    free(used);
    return 1;
}

connected_components *calculate_connected_components(dense_graph *graph) {
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

    return result;
}

void free_connected_components(connected_components *connected_components) {
    for (int i = 0; i < connected_components->num_components; i++) {
        free(connected_components->components[i].nodes);
    }

    free(connected_components);
}
