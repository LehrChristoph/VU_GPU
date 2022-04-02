
#ifndef GUARD_CUDA_HIST_H
#define GUARD_CUDA_HIST_H

#include <cuda.h>
#include <cuda_runtime.h>

void runOnGpu(const unsigned char* colors, unsigned int* buckets, 
                unsigned int len, unsigned int rows, unsigned int cols, 
                void(*gpuFunc)(unsigned char*, unsigned int*, unsigned int, unsigned int, unsigned int)
    );

__global__ void gpuNaive(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols);

__global__ void gpuGood(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols);

#endif
