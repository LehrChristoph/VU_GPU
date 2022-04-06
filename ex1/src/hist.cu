#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../inc/hist.h"
#include <sys/time.h>

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
    
__global__ void gpuNaive(unsigned char* colors, unsigned int* buckets, unsigned int len) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int i = (iy * blockDim.x * gridDim.x + ix);
   
    if (i < len)
    {
        unsigned int entry = (i%4)*256 + colors[i];
		atomicAdd(&buckets[entry], 1);
    }
}

__global__ void gpuGood(unsigned char* colors, unsigned int* buckets, unsigned int len) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int i = (iy * blockDim.x * gridDim.x + ix);

    __shared__ unsigned int local_buckets [4*256];
    // check whether the dimension of the block exceeds the size of the block
    if(blockDim.x * blockDim.y >= 4*256)
    {
        // use first 4*256 threads to init shared array
        unsigned int j = blockDim.x* threadIdx.y + threadIdx.x;
        if(j < 4*256)
        {
            local_buckets[j] = 0;
        }
    }
    else
    {
        // each thread has to init more than 1 bucket
        for(unsigned int j = blockDim.x* threadIdx.y + threadIdx.x; j < 4*256; j += blockDim.x * blockDim.y)
        {
            local_buckets[j] = 0;
        }
    }

    __syncthreads();
    
    // add value to bucket
    if (i < len)
    {
        unsigned int entry = (i%4)*256 + colors[i];
        atomicAdd(&local_buckets[entry], 1);
    }
    
    __syncthreads();

    if(blockDim.x * blockDim.y >= 4*256)
    {
        // use first 4*256 buckets to add up buckets
        unsigned int j = blockDim.x* threadIdx.y + threadIdx.x;
        if(j < 4*256)
        {
            atomicAdd(&buckets[j], local_buckets[j]);
        }
    }
    else
    {
        // each thread has to add more than one bucket
        for(unsigned int j = blockDim.x* threadIdx.y + threadIdx.x; j < 4*256; j += blockDim.x * blockDim.y)
        {
            atomicAdd(&buckets[j], local_buckets[j]);
        }
    }
}

double runOnGpu(const unsigned char* colors, unsigned int* buckets, 
                unsigned int len, unsigned int rows, unsigned int cols, 
                unsigned char compute_function
    ) {
    if(compute_function > 2)
    {
        return -1 ;
    }

    struct timeval start, end;
    gettimeofday(&start,NULL);

    unsigned char* d_colors;
    unsigned int* d_buckets;
    CHECK(cudaMalloc(&d_colors, sizeof(unsigned char) * len));
    CHECK(cudaMemcpy(d_colors, colors, sizeof(unsigned char) * len, cudaMemcpyHostToDevice));
        
    
    if(compute_function == 1)
    {
        dim3 grid, block;
        block.x = 256;
        block.y = 4;
        grid.x = ceil((double)(rows*cols)/ block.x); 
        grid.y = 1;
        
        CHECK(cudaMalloc(&d_buckets, sizeof(unsigned int) * 256*4));
        gettimeofday(&start,NULL);
        gpuNaive<<<grid, block>>>(d_colors, d_buckets, len);
        cudaDeviceSynchronize();
        gettimeofday(&end,NULL);
    
        CHECK(cudaMemcpy(buckets, d_buckets, sizeof(unsigned int) *256*4, cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_buckets));
    }
    else
    {
        dim3 grid, block;
        block.x = 256;
        block.y = 4;
        grid.x = ceil((double)(rows*cols)/ block.x ); 
        grid.y = 1;
        
        CHECK(cudaMalloc(&d_buckets, sizeof(unsigned int) * 256*4 ));
        gettimeofday(&start,NULL);
    
        gpuGood<<<grid, block>>>(d_colors, d_buckets, len);
        cudaDeviceSynchronize();
        gettimeofday(&end,NULL);

        CHECK(cudaMemcpy(buckets, d_buckets, sizeof(unsigned int) *256*4, cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_buckets));
    }

    CHECK(cudaFree(d_colors));

    double start_seconds = ((double)start.tv_sec + (double)start.tv_usec*1.e-6);
    double end_seconds = ((double)end.tv_sec + (double)end.tv_usec*1.e-6);

    return end_seconds - start_seconds;
}

