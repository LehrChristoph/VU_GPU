#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../inc/hist.h"

#define CHECK(call)                                                     \
    {                                                                   \
        const cudaError_t error = call;                                 \
        if (error != cudaSuccess)                                       \
            {                                                           \
                fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);  \
                fprintf(stderr, "code: %d, reason: %s\n", error,        \
                        cudaGetErrorString(error));                     \
                exit(1);                                                \
            }                                                           \
    }
    
__global__ void gpuNaive(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int i = (iy * gridDim.x + ix)*4;
    if (i < len) {
        // get wether rgb or alpha value 
        unsigned int color = i % 4;
        unsigned int entry =  colors[i];
	atomicAdd(&buckets[entry], 1);
	entry = 256   + colors[i+1];
	atomicAdd(&buckets[entry], 1);
	entry = 256*2 + colors[i+2];
	atomicAdd(&buckets[entry], 1);
	entry = 256*3 + colors[i+3];
	atomicAdd(&buckets[entry], 1);
	
    }

}

__global__ void gpuGood(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols) {
    // TODO
    int i = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * blockIdx.x + blockDim.x * blockDim.y * blockIdx.y * gridDim.x;
    if (i < len) {
        // get wether rgb or alpha value 
        unsigned int color = i % 4;
        unsigned int entry = 256*color + colors[i];
	    atomicAdd(&buckets[entry], 1);
    }
}

void runOnGpu(const unsigned char* colors, unsigned int* buckets, 
                unsigned int len, unsigned int rows, unsigned int cols, 
                unsigned char compute_function
    ) {
    if(compute_function > 2)
    {
        return;
    }

    unsigned char* d_colors;
    unsigned int* d_buckets;
    CHECK(cudaMalloc(&d_colors, sizeof(unsigned char) * len));
    CHECK(cudaMalloc(&d_buckets, sizeof(unsigned int) * 256*4));
    CHECK(cudaMemcpy(d_colors, colors, sizeof(unsigned char) * len, cudaMemcpyHostToDevice));
    dim3 grid, block;
    block.x = 32;
    block.y = 1;
    grid.x = ceil((double)(rows*cols)/ block.x);   //(cols*4)  / block.x + 1;
    grid.y = 1;// rows  / block.y + 1;
    
    //grid.x = ceil((double)(cols)/ block.x);   
    //grid.y = ceil((double)(rows*4)  / block.y);
    
    /* printf("%d - %d\n", grid.x, grid.y); */
    if(compute_function == 1)
    {
        printf("Using naive GPU implementation\n");
        gpuNaive<<<grid, block>>>(d_colors, d_buckets, len, rows, cols );
    }
    else
    {
        printf("Using good GPU implementation\n");
        gpuGood<<<grid, block>>>(d_colors, d_buckets, len, rows, cols);
    }
    CHECK(cudaMemcpy(buckets, d_buckets, sizeof(unsigned int) *256*4, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_colors));
    CHECK(cudaFree(d_buckets));
}


