#pragma once

#include "graph.h"

typedef connected_components* (*connected_components_function)(dense_graph*);

// CPU version
connected_components *calculate_connected_components(dense_graph *graph);
