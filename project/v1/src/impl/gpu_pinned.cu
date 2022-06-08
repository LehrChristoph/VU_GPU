#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern "C" {
#include "impl.h"
}
#include "cuda_helpers.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

/**
 * update references that still point to host memory
 */
__global__ void populateGraphPinned(dense_graph *d_graph, dense_node *d_nodes,
                              dense_edge *d_edges, void *current_base) {
  int node = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y +
             threadIdx.y;
  if (node >= d_graph->num_nodes)
    return;
  if (node == 0) {
    // update nodes ptr once
    d_graph->nodes = d_nodes;
  }

  ptrdiff_t idx = d_nodes[node].edges - d_nodes[0].edges;
  __syncthreads();
  // update edge ptr of node using same index as before (ptr - baseptr)
  d_nodes[node].edges = &d_edges[idx];
  __syncthreads();
}

__global__ void calculate_pinned(dense_graph *d_graph,
                          connected_components *d_components,
                          component *d_comps, int *d_nodes,
                          bool *d_used_nodes) {
  int node = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y +
             threadIdx.y;
  if (node >= d_graph->num_nodes)
    return;
  // calculate our section of arrays
  bool *used_nodes = d_used_nodes + node * d_graph->num_nodes;
  int *component_list = d_nodes + node * d_graph->num_nodes;

  d_comps[node].num_nodes = 0;
  int wl_len = 1, wl_pos = 0;
  component_list[0] = node;
  // nodes already part of this cc
  used_nodes[node] = true;
  while (wl_len) {
    // look for nodes connected to next node that wasn't yet processed
    int curr = component_list[wl_pos++];
    wl_len--;
    for (int i = 0; i < d_graph->nodes[curr].num_edges; i++) {
      int target = d_graph->nodes[curr].edges[i];
      if (target < node) {
        // not handled by this thread - thread with index of biggest node in cc
        // handles cc
        d_comps[node].num_nodes = 0;
        return;
      }
      // skip nodes already in cc
      if (used_nodes[target])
        continue;
      // add node to cc
      used_nodes[target] = true;
      component_list[wl_pos + wl_len++] = target;
    }
  }
  // save total number of nodes and node list, and increment number of ccs
  d_comps[node].num_nodes = wl_pos;
  d_comps[node].nodes = component_list;
  atomicInc(&d_components->num_components, d_graph->num_nodes);
}

extern "C" {
clock_t connected_components_pinned(dense_graph *graph,
                                           connected_components **out) {
  dense_graph *h_graph;
  dense_node *h_gnodes;
  dense_edge *h_edges;
  CHECK(cudaMallocHost((void **)&h_graph, sizeof(dense_graph)));
  CHECK(cudaMallocHost((void **)&h_gnodes, sizeof(dense_node) * graph->num_nodes));
  CHECK(cudaMallocHost((void **)&h_edges,
                   MAX(sizeof(dense_edge) * graph->num_edges * 2, 1)));
  memcpy(h_graph, graph, sizeof(dense_graph));
  memcpy(h_gnodes, graph->nodes, sizeof(dense_node) * graph->num_nodes);
  memcpy(h_edges, graph->nodes->edges, MAX(sizeof(dense_edge) * graph->num_edges * 2, 1));
  h_graph->nodes = h_gnodes;
  dense_graph *d_graph;
  dense_node *d_gnodes;
  dense_edge *d_edges;
  clock_t start = clock();
  // allocate graph on device
  CHECK(cudaMalloc((void **)&d_graph, sizeof(dense_graph)));
  CHECK(cudaMalloc((void **)&d_gnodes, sizeof(dense_node) * graph->num_nodes));
  CHECK(cudaMalloc((void **)&d_edges,
                   MAX(sizeof(dense_edge) * graph->num_edges * 2, 1)));
  // copy graph to device
  CHECK(
      cudaMemcpy(d_graph, h_graph, sizeof(dense_graph), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_gnodes, h_graph->nodes,
                   sizeof(dense_node) * h_graph->num_nodes,
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_edges, h_graph->nodes->edges,
                   sizeof(dense_edge) * h_graph->num_edges * 2,
                   cudaMemcpyHostToDevice));
  dim3 block, grid;
  block.x = 32;
  block.y = 32;
  grid.x = ceil((double)graph->num_nodes / block.x / block.y);
  grid.y = 1;
  populateGraphPinned<<<grid, block>>>(d_graph, d_gnodes, d_edges,
                                 graph->nodes->edges);
  cudaDeviceSynchronize();
  // allocate space for cc on device
  connected_components *d_components;
  component *d_comps;
  int *d_nodes;
  bool *d_used_nodes;
  CHECK(cudaMalloc((void **)&d_components, sizeof(connected_components)));
  CHECK(cudaMalloc((void **)&d_comps, sizeof(component) * graph->num_nodes));
  CHECK(cudaMalloc((void **)&d_nodes,
                   sizeof(int) * graph->num_nodes * graph->num_nodes));
  CHECK(cudaMalloc((void **)&d_used_nodes,
                   sizeof(bool) * graph->num_nodes * graph->num_nodes));
  // doing calculation
  clock_t calcStart = clock();
  calculate_pinned<<<grid, block>>>(d_graph, d_components, d_comps, d_nodes,
                             d_used_nodes);
  cudaDeviceSynchronize();
  clock_t calcEnd = clock();
  // copy result back
  connected_components *components;
  CHECK(cudaMallocHost((void **)&components, sizeof(connected_components)));
  CHECK(cudaMemcpy(components, d_components, sizeof(connected_components),
                   cudaMemcpyDeviceToHost));

  CHECK(cudaMallocHost((void **)&components->components,
                       sizeof(component) * graph->num_nodes));
  CHECK(cudaMemcpy(components->components, d_comps,
                   sizeof(component) * graph->num_nodes,
                   cudaMemcpyDeviceToHost));

  int ind = 0;
  for (int i = 0; i < graph->num_nodes; i++) {
    if (!components->components[i].num_nodes)
      continue;
    node *d_nodes = components->components[i].nodes;
    components->components[ind].nodes =
        (int *)malloc(sizeof(int) * components->components[i].num_nodes);
    CHECK(cudaMemcpy(components->components[ind].nodes, d_nodes,
                     sizeof(int) * components->components[i].num_nodes,
                     cudaMemcpyDeviceToHost));
    components->components[ind].num_nodes = components->components[i].num_nodes;
    ind++;
  }
  // free gpu memory
  CHECK(cudaFree(d_used_nodes));
  CHECK(cudaFree(d_nodes));
  CHECK(cudaFree(d_comps));
  CHECK(cudaFree(d_components));
  CHECK(cudaFree(d_edges));
  CHECK(cudaFree(d_gnodes));
  CHECK(cudaFree(d_graph));
  clock_t end = clock();
  CHECK(cudaFreeHost(h_graph));
  CHECK(cudaFreeHost(h_gnodes));
  CHECK(cudaFreeHost(h_edges));
  *out = components;
  // return end - start;
  return calcEnd - calcStart;
}
}
