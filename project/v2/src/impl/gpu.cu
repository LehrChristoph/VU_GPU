#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "impl.h"
#include "cuda_helpers.h"


__global__ void init(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components, unsigned int* found_nodes)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int i = (iy * blockDim.x * gridDim.x + ix);

    if(i < num_nodes)
    {
        connected_components[i] = ~0;
        for(unsigned int j=0; j < num_nodes; j++)
        {
            found_nodes[i*num_nodes + j] = 0;
        }
        found_nodes[i*num_nodes + i] = 1;
    }
}

__global__ void calculate(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components, unsigned int* found_nodes_global)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int i = (iy * blockDim.x * gridDim.x + ix);

    if(i < num_nodes)
    {
        unsigned int nodes_changed = 1;
        unsigned int lowest_node_id =i;
        // find following neighbours
	while (nodes_changed > 0)
        {
            nodes_changed = 0;
            for(unsigned int j=0; j < num_nodes; j++)
            {
                //if(found_nodes_global[i*num_nodes + j] != 0)
                {
                    // iterate over connected nodes
                    for(unsigned int k=0; k < num_nodes; k++)
                    {
                        if(adjacency_matrix[j*num_nodes + k] != 0 && found_nodes_global[i*num_nodes + k] == 0)
                        {
                            found_nodes_global[i*num_nodes + k] = 1;
                            found_nodes_global[k*num_nodes + i] = 1;
                            nodes_changed++;
                            if(k < lowest_node_id)
                            {
                                lowest_node_id = k;
                            }
                        }
                    }
                }
            }
        }

        connected_components[i] = lowest_node_id;
    }
}


clock_t calculate_connected_components_gpu_simple(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components)
{
    clock_t start, end; 
    unsigned int *d_adjacency_matrix;
    unsigned int *d_connected_components;
    unsigned int *d_found_nodes;

    CHECK(cudaMalloc(&d_adjacency_matrix, sizeof(unsigned int) *num_nodes*num_nodes));
    CHECK(cudaMemcpy(d_adjacency_matrix, adjacency_matrix, sizeof(unsigned int) *num_nodes*num_nodes, cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_connected_components, sizeof(unsigned int) *num_nodes));
    CHECK(cudaMalloc(&d_found_nodes, sizeof(unsigned int) *num_nodes*num_nodes));

    dim3 block, grid;
    block.x = 1024;
    block.y = 1;
    grid.x = ceil((double)num_nodes / block.x );
    grid.y = 1;

    // init structures
    init<<<grid, block>>>(num_nodes, d_adjacency_matrix, d_connected_components, d_found_nodes);
    cudaDeviceSynchronize();

    start = clock();
    calculate<<<grid, block>>>(num_nodes, d_adjacency_matrix, d_connected_components, d_found_nodes);
    cudaDeviceSynchronize();
    end = clock();

    CHECK(cudaMemcpy(connected_components, d_connected_components, sizeof(unsigned int) *num_nodes, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_connected_components));
    CHECK(cudaFree(d_adjacency_matrix));

    return end - start;
}

