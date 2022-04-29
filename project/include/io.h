#pragma once

#include "graph.h"
#include <stdio.h>

graph *read_graph(char* filename);
void write_graph(graph *graph, FILE* file);
void free_graph(graph *graph);

void write_connected_components(connected_components *connectedComponents, FILE* file);
