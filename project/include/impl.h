#pragma once

#include "graph.h"

typedef clock_t (*connected_components_function)(dense_graph*, connected_components**);

// CPU version
clock_t calculate_connected_components(dense_graph *graph, connected_components** out);
