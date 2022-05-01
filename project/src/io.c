#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "io.h"

#define FILE_ERR(...) {                         \
        printf(__VA_ARGS__);                    \
        out->num_nodes = 0;                     \
        fclose(fp);                             \
        free_graph(out);                        \
        return NULL;                            \
    }

graph *read_graph(char* filename) {
    char buf[64];
    char *line = buf;
    FILE* fp = fopen(filename, "r");
    graph *out = malloc(sizeof(graph));
    if (!fp) {
        FILE_ERR("Could not open file %s\n", filename);
    }
    fgets(buf, 64, fp);
    if (strcmp("H", strsep(&line, " "))) {
        FILE_ERR("Invalid input file (header line '%s')\n", buf);
    }
    out->num_nodes = atoi(strsep(&line, " "));
    out->num_edges = atoi(strsep(&line, " "));
    if (out->num_nodes == 0) {
        FILE_ERR("Invalid input file (header line '%s')\n", buf);
    }
    int undirected = atoi(strsep(&line, " "));
    if (undirected != 1) {
        FILE_ERR("Ivalid input file (undirected is set to %d, must be 1)\n", undirected);
    }
    out->edges = malloc(sizeof(edge) * out->num_edges);
    for (int i = 0; i < out->num_edges; i++) {
        if (!fgets(buf, 64, fp)) {
            FILE_ERR("Could not read edge %d\n", i);
        }
        line = buf;
        if (strcmp("E", strsep(&line, " "))) {
            FILE_ERR("Invalid input file (edge line '%s')\n", buf);
        }
        out->edges[i].from = atoi(strsep(&line, " "));
        out->edges[i].to = atoi(strsep(&line, " "));
        out->edges[i].weight = atoi(strsep(&line, " "));
    }
    return out;
}

void write_graph(graph *graph, FILE* file) {
    fprintf(file, "H %d %d 1\n", graph->num_nodes, graph->num_edges);
    for (int i = 0; i < graph->num_edges; i++) {
        edge edge = graph->edges[i];
        fprintf(file, "E %d %d %d\n", edge.from, edge.to, edge.weight);
    }
}

void free_graph(graph *graph) {
    if (!graph) return;

    if (graph->edges) {
        free(graph->edges);
    }
    free(graph);
}

void write_connected_components(connected_components *connected_components, FILE* file) {
    for (int i = 0; i < connected_components->num_components; i++) {
        fprintf(file, "C%d", i);

        component component = connected_components->components[i];
        for (int j = 0; j < component.num_nodes; j++) {
            fprintf(file, " %d", component.nodes[j]);
        }

        fprintf(file, "\n");
    }
}
