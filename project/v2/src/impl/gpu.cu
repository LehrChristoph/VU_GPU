#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "impl.h"
#include "cuda_helpers.h"

__global__ void init(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int i = (iy * blockDim.x * gridDim.x + ix);

    if(i < num_nodes)
    {
        connected_components[i] = ~0;
    }
}


__global__ void calculate(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int i = (iy * blockDim.x * gridDim.x + ix);

    if(i < num_nodes)
    {
        unsigned int found_nodes_cnt = 1;
        unsigned int first_index = 0;
        unsigned int last_index = 1;
        unsigned int lowest_node_id = i;
        unsigned int found_nodes[num_nodes];

        found_nodes[0]=i;
        // get inital node neightbours
        for(unsigned int j=0; j < num_nodes; j++)
        {
            if(adjacency_matrix[i*num_nodes + j] !=0)
            {
                found_nodes[last_index] = j;
                last_index++;
                found_nodes_cnt++;
                if(j < lowest_node_id)
                {
                    lowest_node_id = j;
                }
            }
        }
        
        // find following neighbours
        while (found_nodes_cnt > 0)
        {
            found_nodes_cnt =0;
            unsigned int new_first_index = last_index;
            unsigned int new_last_index = last_index;
        
            for(unsigned int j=first_index; j < last_index; j++)
            {
                // iterate over new found nodes
                for(unsigned int k=0; k < num_nodes; k++)
                {
                    // check if neighboruing relation exists 
                    if(adjacency_matrix[found_nodes[j]*num_nodes + k] !=0)
                    {
                        // check if neighbouring relation was already found
                        unsigned int node_already_found = 0;
                        for(unsigned int l=0; l < new_last_index; l++)
                        {
                            //printf("l Node %d\n", found_nodes[l]);
                            if(found_nodes[l] == k)
                            {
                                node_already_found = 1;
                                break;
                            }
                        }

                        if(node_already_found == 0)
                        {
                            found_nodes[new_last_index] =k;
                            new_last_index++;
                            found_nodes_cnt++;
                            if(k < lowest_node_id)
                            {
                                lowest_node_id = k;
                            }
                        }
                    }
                }
            }

            first_index = new_first_index;
            last_index = new_last_index;
        
        }
        connected_components[i] = lowest_node_id;
    }
}


clock_t calculate_connected_components_gpu_simple(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components)
{
    clock_t start, end; 
    unsigned int *d_adjacency_matrix,
    unsigned int *d_connected_components


    CHECK(cudaMalloc(&d_adjacency_matrix, sizeof(unsigned int) *num_nodes*num_nodes));
    CHECK(cudaMemcpy(d_adjacency_matrix, adjacency_matrix, sizeof(unsigned int) *num_nodes*num_nodes, cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_connected_components, sizeof(unsigned int) *num_nodes));

    dim3 block, grid;
    block.x = 1024;
    block.y = 1;
    grid.x = ceil((double)num_nodes / block.x );
    grid.y = 1;

    // init structures
    init<<<grid, block>>>(num_nodes, d_adjacency_matrix, d_connected_components);
    cudaDeviceSynchronize();

    start = clock();
    calculate<<<grid, block>>>(num_nodes, d_adjacency_matrix, d_connected_components);
    cudaDeviceSynchronize();
    end = clock();

    CHECK(cudaFree(d_adjacency_matrix));
    CHECK(cudaFree(d_connected_components));
    CHECK(cudaMemcpy(connected_components, d_connected_components, sizeof(unsigned int) *num_nodes, cudaMemcpyDeviceToHost));

    return end - start;
}

