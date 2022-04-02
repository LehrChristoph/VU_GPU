
#ifndef GUARD_CUDA_HIST_H
#define GUARD_CUDA_HIST_H

#include <cuda.h>
#include <cuda_runtime.h>

void runOnGpu(const unsigned char* colors, unsigned int* buckets, 
                unsigned int len, unsigned int rows, unsigned int cols, 
                unsigned char compute_function)
    );

#endif
