#pragma once

#include "graph.h"

typedef clock_t (*connected_components_function)(dense_graph*, connected_components**);

// CPU version
clock_t calculate_connected_components(dense_graph *graph, connected_components** out);
clock_t connected_components_thread_per_cc(dense_graph *graph, connected_components** out);
clock_t connected_components_thread_per_cc_vector(dense_graph *graph, connected_components** out);
clock_t connected_components_pinned(dense_graph *graph, connected_components** out);
clock_t connected_components_vector_pinned(dense_graph *graph, connected_components** out);
clock_t connected_components_zerocopy(dense_graph *graph, connected_components** out);
clock_t connected_components_vector_zerocopy(dense_graph *graph, connected_components** out);

/* #define BENCH_INCL_MEMCPY */
