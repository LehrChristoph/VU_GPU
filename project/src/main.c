#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "io.h"
#include "generator.h"

int main(int argc, char** argv) {
    srandom(time(NULL));
    if (argc == 1) {
        printf("Usage: ...\n");
        return 1;
    }
    if (strcmp(argv[1], "generate") == 0) {
        if (argc != 6 && argc != 7) {
            printf("Usage: %s generate [filename] <#nodes> <density (0-1)> <min_weight> <max_weight>\n", argv[0]);
        }
        int num_nodes = atoi(argv[argc-4]);
        float density = atof(argv[argc-3]);
        int min_weight = atoi(argv[argc-2]);
        int max_weight = atoi(argv[argc-1]);
        FILE* fp = argc == 6 ? stdout : fopen(argv[2], "w");
        graph* graph = from_dense(generate(num_nodes, density, min_weight, max_weight));
        if (graph) {
            write_graph(graph, fp);
            free_graph(graph);
        }
        fclose(fp);
    } else if (strcmp(argv[1], "calculate") == 0) {
        if (argc != 3) {
            printf("Usage: %s calculate <filename>\n", argv[0]);
            return 1;
        }
        graph *graph = read_graph(argv[2]);
        if (graph) {
            write_graph(from_dense(to_dense(graph)), stdout);
        }
        free_graph(graph);
    } else {
        printf("Unknown command %s, available: generate, calculate\n", argv[0]);
    }
}
