
#ifndef GUARD_CUDA_HIST_H
#define GUARD_CUDA_HIST_H


void gpuNaive(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols);

void gpuGood(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols);

#endif