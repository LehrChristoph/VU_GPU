#pragma once

#include "graph.h"
#include <stdio.h>

graph *read_graph(char* filename);
void write_graph(graph *graph, FILE* filename);
void free_graph(graph *graph);
