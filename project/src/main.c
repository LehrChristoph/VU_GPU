#include <stdio.h>
#include <string.h>
#include "io.h"

int main(int argc, char** argv) {
    if (argc == 1) {
        printf("Usage: ...\n");
        return 1;
    }
    if (strcmp(argv[1], "generate") == 0) {
        // TODO generate
    } else if (strcmp(argv[1], "calculate") == 0) {
        if (argc != 3) {
            printf("Usage: %s calculate <filename>\n", argv[0]);
            return 1;
        }
        graph *graph = read_graph(argv[2]);
        if (graph) {
            write_graph(graph, stdout);
        }
        free_graph(graph);
    } else {
        printf("Unknown command %s, available: generate, calculate\n", argv[0]);
    }
}
