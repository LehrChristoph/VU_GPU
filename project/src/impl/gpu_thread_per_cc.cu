#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern "C" {
#include "impl.h"
}
#include "cuda_helpers.h"

__global__ void populateGraph(dense_graph *d_graph, void* current_base, void* real_base) {
  int node = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y +
             threadIdx.y - 1;
  if (node > d_graph->num_nodes) return;
  d_graph->nodes[node].edges = (dense_edge*) ((uint64_t) d_graph->nodes[node].edges + (uint64_t) real_base - (uint64_t) current_base);
}

__global__ void calculate(dense_graph *d_graph) {}

extern "C" {
clock_t connected_components_thread_per_cc(dense_graph *graph,
                                           connected_components **out) {
  dense_graph *d_graph;
  CHECK(cudaMalloc((void **)&d_graph,
                   sizeof(dense_graph) + sizeof(dense_node) * graph->num_nodes +
                       sizeof(dense_edge) * graph->num_edges));
  dense_node* nodes = graph->nodes;
  graph->nodes = (dense_node*) (((uint8_t*) d_graph) + sizeof(dense_graph));
  CHECK(cudaMemcpy(d_graph, graph,
                   sizeof(dense_graph) + sizeof(dense_node) * graph->num_nodes +
                       sizeof(dense_edge) * graph->num_edges,
                   cudaMemcpyHostToDevice));
  graph->nodes = nodes;
  dim3 populateBlock, populateGrid;
  populateBlock.x = 32;
  populateBlock.y = 32;
  populateGrid.x = ceil((double) graph->num_nodes / populateBlock.x / populateBlock.y);
  populateGrid.y = 1;
  populateGraph<<<populateGrid, populateBlock>>>(d_graph, graph->nodes->edges, d_graph + sizeof(dense_graph) + sizeof(dense_node) * graph->num_nodes);
  clock_t start = clock();
  /* calculate(d_graph, d_nodes, d_edges); */
  clock_t end = clock();
  CHECK(cudaFree(d_graph));
  return end - start;
}
}
