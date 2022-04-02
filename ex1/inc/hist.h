
#ifndef GUARD_CUDA_HIST_H
#define GUARD_CUDA_HIST_H


__global__ void gpuNaive(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols);

__global__ void gpuGood(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols);

#endif
