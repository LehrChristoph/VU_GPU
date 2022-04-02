#include <stdio.h>
#include <stdlib.h>
#include "../inc/hist.h"

__global__ void gpuNaive(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols) {
    printf("Using naive GPU implementation\n");
    int i = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * blockIdx.x + blockDim.x * blockDim.y * blockIdx.y * gridDim.x;
    if (i < len) {
        // get wether rgb or alpha value 
        unsigned int color = i % 4;
        unsigned int entry = 256*i + colors[i];
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
        unsigned int entry = 256*i + colors[i];
        atomicAdd(&buckets[entry], 1);
    }
}
