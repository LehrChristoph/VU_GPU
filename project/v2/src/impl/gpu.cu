#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "impl.h"
#include "cuda_helpers.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

typedef struct
{
	unsigned int num_nodes;
	unsigned int *adjacency_matrix;
	unsigned int * connected_components;
	unsigned int* found_nodes;
}  kernel_args_t;

__global__ void init(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components, unsigned int* found_nodes)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int i = (iy * blockDim.x * gridDim.x + ix);

    if(i < num_nodes)
    {
        connected_components[i] = ~0;
        //for(unsigned int j=0; j < num_nodes; j++)
        //{
        //	found_nodes[i*num_nodes + j] = 0;
        //}
        found_nodes[i] = 0;
    }
}

__device__ unsigned int current_index;
__device__ unsigned int last_index;
__device__ unsigned int found_nodes_cnt;
__device__ unsigned int minimum_index;

//__global__ void calculate (unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components, unsigned int* found_nodes)
__global__ void calculate(kernel_args_t args)
{
	unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int tid = (tidy * blockDim.x * gridDim.x + tidx);
    grid_group grid = this_grid();
    
    unsigned int num_nodes = args.num_nodes;
    unsigned int *adjacency_matrix = args.adjacency_matrix;
    unsigned int *connected_components = args.connected_components;
    unsigned int *found_nodes = args.found_nodes;
    
    for (unsigned int i=0; i<num_nodes; i++)
    {
	if(connected_components[i] == ~0)
	{
	    if(tid == 0)
            {
                current_index = 1;
                found_nodes[0] = i;
                last_index =1;
                found_nodes_cnt =1;
                minimum_index = i;
            }
	    
	    grid.sync();
            // add all neighbours of node i	    
            if(tid < num_nodes && adjacency_matrix[i*num_nodes + tid] !=0)
            {
                unsigned int old= atomicAdd(&last_index, 1);
                found_nodes[old] = tid;
                atomicAdd(&found_nodes_cnt, 1);
                atomicMin(&minimum_index, tid);
            }
	    grid.sync();
            
	    while (current_index < last_index)
            {
                if(tid < num_nodes && adjacency_matrix[found_nodes[current_index]*num_nodes + tid] !=0)
                {
                    // check if neighbouring relation was already found
                    unsigned int node_already_found = 0;
                    for(unsigned int j=0; j < last_index; j++)
                    {
                        //printf("l Node %d\n", found_nodes[l]);
                        if(found_nodes[j] == tid)
                        {
                            node_already_found = 1;
                            break;
                        }
                    }

                    if(node_already_found == 0)
                    {
                        unsigned int old= atomicAdd(&last_index, 1);
                        found_nodes[old] = tid;
                        atomicAdd(&found_nodes_cnt, 1);
                        atomicMin(&minimum_index, tid);
                    }
                }

                if(tid ==0)
                {
                    atomicAdd(&current_index, 1);
                }

                grid.sync();
            }
            
            
            if(tid < last_index)
            {
                connected_components[found_nodes[tid]] = minimum_index;
            }

            grid.sync();
	   
        }
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
    CHECK(cudaMalloc(&d_found_nodes, sizeof(unsigned int) *num_nodes));

    dim3 block, grid;
    block.x = 512;//1024;
    block.y = 1;
    grid.x = ceil((double)num_nodes / block.x );
    grid.y = 1;

    kernel_args_t args;
    args.num_nodes = num_nodes;
    args.adjacency_matrix = d_adjacency_matrix;
    args.connected_components = d_connected_components;
    args.found_nodes = d_found_nodes;
    
    // init structures
    init<<<grid, block>>>(num_nodes, d_adjacency_matrix, d_connected_components, d_found_nodes);
    cudaDeviceSynchronize();
    void *kernelArgs[] = {&args};
    start = clock();
    //calculate<<<grid, block>>>(num_nodes, d_adjacency_matrix, d_connected_components, d_found_nodes);
    CHECK(cudaLaunchCooperativeKernel((void *)calculate, grid, block, kernelArgs));
    cudaDeviceSynchronize();
    end = clock();

    CHECK(cudaMemcpy(connected_components, d_connected_components, sizeof(unsigned int) *num_nodes, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_connected_components));
    CHECK(cudaFree(d_adjacency_matrix));
    CHECK(cudaFree(d_found_nodes));

    return end - start;
}

clock_t calculate_connected_components_gpu_simple_pinned(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components)
{
    clock_t start, end; 
    unsigned int *d_adjacency_matrix;
    unsigned int *d_connected_components;
    unsigned int *d_found_nodes;

    CHECK(cudaMallocHost(&d_adjacency_matrix, sizeof(unsigned int) *num_nodes*num_nodes));
    CHECK(cudaMemcpy(d_adjacency_matrix, adjacency_matrix, sizeof(unsigned int) *num_nodes*num_nodes, cudaMemcpyHostToDevice));

    CHECK(cudaMallocHost(&d_connected_components, sizeof(unsigned int) *num_nodes));
    CHECK(cudaMallocHost(&d_found_nodes, sizeof(unsigned int) *num_nodes));

    dim3 block, grid;
    block.x = 512;//1024;
    block.y = 1;
    grid.x = ceil((double)num_nodes / block.x );
    grid.y = 1;

    kernel_args_t args;
    args.num_nodes = num_nodes;
    args.adjacency_matrix = d_adjacency_matrix;
    args.connected_components = d_connected_components;
    args.found_nodes = d_found_nodes;
    
    // init structures
    init<<<grid, block>>>(num_nodes, d_adjacency_matrix, d_connected_components, d_found_nodes);
    cudaDeviceSynchronize();
    void *kernelArgs[] = {&args};
    start = clock();
    //calculate<<<grid, block>>>(num_nodes, d_adjacency_matrix, d_connected_components, d_found_nodes);
    CHECK(cudaLaunchCooperativeKernel((void *)calculate, grid, block, kernelArgs));
    cudaDeviceSynchronize();
    end = clock();

    CHECK(cudaMemcpy(connected_components, d_connected_components, sizeof(unsigned int) *num_nodes, cudaMemcpyDeviceToHost));
    CHECK(cudaFreeHost(d_connected_components));
    CHECK(cudaFreeHost(d_adjacency_matrix));
    CHECK(cudaFreeHost(d_found_nodes));

    return end - start;
}

clock_t calculate_connected_components_gpu_simple_zero_copy(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components)
{
    clock_t start, end; 
    unsigned int *h_adjacency_matrix;

    unsigned int *d_adjacency_matrix;
    unsigned int *d_connected_components;
    unsigned int *d_found_nodes;

    CHECK(cudaHostAlloc(&h_adjacency_matrix, sizeof(unsigned int) *num_nodes*num_nodes, cudaHostAllocMapped));
    memcpy(h_adjacency_matrix, adjacency_matrix, sizeof(unsigned int) *num_nodes*num_nodes, cudaMemcpyHostToDevice);
    CHECK(cudaHostGetDevicePointer((void **)&d_adjacency_matrix, (void *)h_adjacency_matrix, 0));

    CHECK(cudaMalloc(&h_connected_components, sizeof(unsigned int) *num_nodes));
    CHECK(cudaMalloc(&h_found_nodes, sizeof(unsigned int) *num_nodes));

    dim3 block, grid;
    block.x = 512;//1024;
    block.y = 1;
    grid.x = ceil((double)num_nodes / block.x );
    grid.y = 1;

    kernel_args_t args;
    args.num_nodes = num_nodes;
    args.adjacency_matrix = d_adjacency_matrix;
    args.connected_components = d_connected_components;
    args.found_nodes = d_found_nodes;
    
    // init structures
    init<<<grid, block>>>(num_nodes, d_adjacency_matrix, d_connected_components, d_found_nodes);
    cudaDeviceSynchronize();
    void *kernelArgs[] = {&args};
    start = clock();
    //calculate<<<grid, block>>>(num_nodes, d_adjacency_matrix, d_connected_components, d_found_nodes);
    CHECK(cudaLaunchCooperativeKernel((void *)calculate, grid, block, kernelArgs));
    cudaDeviceSynchronize();
    end = clock();

    CHECK(cudaMemcpy(connected_components, d_connected_components, sizeof(unsigned int) *num_nodes, cudaMemcpyDeviceToHost));

    CHECK(cudaFreeHost(h_adjacency_matrix));
    CHECK(cudaFree(d_adjacency_matrix));
    CHECK(cudaFree(d_found_nodes));

    return end - start;
}




