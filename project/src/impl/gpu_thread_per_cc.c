#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "impl.h"
#include "cuda_helpers.h"

__device__ void populateGraph(dense_graph *d_graph, dense_node *nodes, dense_edge *edges) {
}

__global__ void calculate(dense_graph *d_graph, dense_node *nodes, dense_edge *edges) {
    
}

clock_t connected_components_thread_per_cc(dense_graph *graph, connected_components** out) {
    dense_graph *d_graph;
    dense_node *d_nodes;
    dense_edge *d_edges;
    CHECK(cudaMalloc((void**) &d_graph, sizeof(dense_graph)));
    CHECK(cudaMalloc((void**) &d_nodes, sizeof(dense_node) * graph->num_nodes));
    CHECK(cudaMalloc((void**) &d_edges, sizeof(dense_edge) * graph->num_edges));
    CHECK(cudaMemcpy(d_nodes, graph->nodes, sizeof(dense_edge) * graph->num_edges, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_edges, graph->nodes->edges, sizeof(dense_edge) * graph->num_edges, cudaMemcpyHostToDevice));
    calculate(d_graph, d_nodes, d_edges);
    CHECK(cudaFree(d_edges));
    CHECK(cudaFree(d_nodes));
    return clock();
}
