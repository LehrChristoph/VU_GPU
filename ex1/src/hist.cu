#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
    
void runOnGpu(const unsigned char* colors, unsigned int* buckets, 
                unsigned int len, unsigned int rows, unsigned int cols, 
                void(*gpuFunc)(unsigned char*, unsigned int*, unsigned int, unsigned int, unsigned int)
    ) {
    unsigned char* d_colors;
    int* d_buckets;
    CHECK(cudaMalloc(&d_colors, sizeof(unsigned char) * len));
    CHECK(cudaMalloc(&d_buckets, sizeof(unsigned int) * 4* 256));
    CHECK(cudaMemcpy(d_colors, colors, sizeof(unsigned char) * len, cudaMemcpyHostToDevice));
    dim3 grid, block;
    block.x = 32;
    block.y = 32;
    grid.x = (rows - 1) / block.x + 1;
    grid.y = (cols - 1) / block.y + 1;
    /* printf("%d - %d\n", grid.x, grid.y); */
    (*gpuFunc)<<<grid, block>>>(d_colors, d_buckets, len, rows, cols);
    CHECK(cudaMemcpy(buckets, d_buckets, sizeof(unsigned int) * 256, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_colors));
    CHECK(cudaFree(d_buckets));
}


__global__ void gpuNaive(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols) {
    printf("Using naive GPU implementation\n");
    int i = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * blockIdx.x + blockDim.x * blockDim.y * blockIdx.y * gridDim.x;
    if (i < len) {
        // get wether rgb or alpha value 
        unsigned int color = i % 4;
        unsigned int entry = 256*color + colors[i];
	atomicAdd(&buckets[entry], 1);
    }
}

__global__ void gpuGood(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols) {
    printf("Using good GPU implementation\n");
    // TODO
    int i = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * blockIdx.x + blockDim.x * blockDim.y * blockIdx.y * gridDim.x;
    if (i < len) {
        // get wether rgb or alpha value 
        unsigned int color = i % 4;
        unsigned int entry = 256*color + colors[i];
	atomicAdd(&buckets[entry], 1);
    }
}
