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
__global__ void populateGraph_vec_zerocopy(dense_graph *d_graph, dense_node *d_nodes,
                                  dense_edge *d_edges, void *current_base,
                                  int *component_vector, int *base) {
  int node = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y +
             threadIdx.y;
  if (node >= d_graph->num_nodes)
    return;
  if (node == 0) {
    // update nodes ptr once
    d_graph->nodes = d_nodes;
    component_vector[d_graph->num_nodes] = 0;
  }

  component_vector[node] = -1;

  ptrdiff_t idx = d_nodes[node].edges - d_nodes[0].edges;
  __syncthreads();
  // update edge ptr of node using same index as before (ptr - baseptr)
  d_nodes[node].edges = &d_edges[idx];
  memset(&base[node * d_graph->num_nodes], 0, d_graph->num_nodes);
  __syncthreads();
}

__global__ void calculate_vec_zerocopy(dense_graph *d_graph, int *component_vector, int *base) {
  int node = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y +
             threadIdx.y;
  if (node >= d_graph->num_nodes)
    return;
  // processing stack
  base = &base[node * d_graph->num_nodes];
  int *stack = base;
  *(stack++) = node;

  while (base != stack) {
    int curr = *(--stack) & ~INT_MIN;
    // add node to cc
    int prev = atomicMax(&component_vector[curr], node);
    // already part of another, bigger cc
    if (prev > node) return;
    // skip nodes already in cc
    if (prev == node) continue;
    for (int i = 0; i < d_graph->nodes[curr].num_edges; i++) {
      int target = d_graph->nodes[curr].edges[i];
      if (target > node) {
        // not handled by this thread - thread with index of biggest node in
        // cc handles cc
        return;
      }
      // add node to worklist
      if (base[target] & INT_MIN) continue;
      *(stack++) = target | (*stack & INT_MIN);
      base[target] |= INT_MIN;
    }
  }
  // increment number of ccs
  atomicInc((unsigned int*) &component_vector[d_graph->num_nodes], d_graph->num_nodes);
}

extern "C" {
clock_t connected_components_vector_zerocopy(dense_graph *graph,
                                           connected_components **out) {
  dense_graph *h_graph;
  dense_node *h_gnodes;
  dense_edge *h_edges;
  CHECK(cudaMallocHost((void **)&h_graph, sizeof(dense_graph), cudaHostAllocMapped));
  CHECK(cudaMallocHost((void **)&h_gnodes, sizeof(dense_node) * graph->num_nodes, cudaHostAllocMapped));
  CHECK(cudaMallocHost((void **)&h_edges,
                   MAX(sizeof(dense_edge) * graph->num_edges * 2, 1), cudaHostAllocMapped));
  memcpy(h_graph, graph, sizeof(dense_graph));
  memcpy(h_gnodes, graph->nodes, sizeof(dense_node) * graph->num_nodes);
  memcpy(h_edges, graph->nodes->edges, MAX(sizeof(dense_edge) * graph->num_edges * 2, 1));
  h_graph->nodes = h_gnodes;
  dense_graph *d_graph;
  dense_node *d_gnodes;
  dense_edge *d_edges;
  clock_t start = clock();
  CHECK(cudaHostGetDevicePointer((void **)&d_graph, (void *)h_graph, 0));
  CHECK(cudaHostGetDevicePointer((void **)&d_gnodes, (void *)h_gnodes, 0));
  CHECK(cudaHostGetDevicePointer((void **)&d_edges, (void *)h_edges, 0));
  // allocate space for cc on device
  int *d_componentVector;
  int *result;
  // last is for num of components
  CHECK(cudaMallocHost((void **)&result, sizeof(int) * (graph->num_nodes + 1), cudaHostAllocMapped));
  CHECK(cudaHostGetDevicePointer((void **)&d_componentVector, (void *)result, 0));
  int *d_stack;
  CHECK(cudaMalloc((void **)&d_stack,
                   sizeof(int) * graph->num_nodes * graph->num_nodes));
  dim3 block, grid;
  block.x = 32;
  block.y = 32;
  grid.x = ceil((double)graph->num_nodes / block.x / block.y);
  grid.y = 1;
  populateGraph_vec_zerocopy<<<grid, block>>>(d_graph, d_gnodes, d_edges,
                                     graph->nodes->edges, d_componentVector,
                                     d_stack);
  cudaDeviceSynchronize();
  // doing calculation
  clock_t calcStart = clock();
  calculate_vec_zerocopy<<<grid, block>>>(d_graph, d_componentVector, d_stack);
  cudaDeviceSynchronize();
  clock_t calcEnd = clock();
  // copy result back
  connected_components *components =
      (connected_components *)malloc(sizeof(connected_components));
  clock_t end = clock();
  components->num_components = result[graph->num_nodes];
  components->components =
      (component *)malloc(sizeof(component) * components->num_components);
  for (int i = 0; i < components->num_components; i++) {
    for (int j = 0; j < graph->num_nodes; j++) {
      bool free = true;
      for (int k = 0; k < i; k++) {
        if (components->components[k].num_nodes == result[j]) {
          free = false;
          break;
        }
      }
      if (free) {
        components->components[i].num_nodes = result[j];
        break;
      }
    }
  }
  int *nodes = (int *)malloc(sizeof(int) * graph->num_nodes);
  for (int i = 0; i < components->num_components; i++) {
      int num_nodes = 0;
      components->components[i].nodes = nodes;
      for (int j = 0; j < graph->num_nodes; j++) {
          if (result[j] == components->components[i].num_nodes) {
              num_nodes++;
              *(nodes++) = j;
          }
      }
      components->components[i].num_nodes = num_nodes;
  }
  components->single_node_list = true;

  // free gpu memory
  CHECK(cudaFree(d_stack));
  CHECK(cudaFreeHost(result));
  *out = components;
  #ifdef BENCH_INCL_MEMCPY
  return end - start;
  #else
  return calcEnd - calcStart;
  #endif
}
}
