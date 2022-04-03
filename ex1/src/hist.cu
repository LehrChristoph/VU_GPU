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
    
__global__ void gpuNaive(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int i = (iy * rows + ix)*4;
    if (i < len) {
        // get wether rgb or alpha value 
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

__global__ void gpuGood_Block(unsigned char* colors, unsigned int* buckets, unsigned int len, unsigned int rows, unsigned int cols) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int i = (iy * blockDim.x * gridDim.x + ix);

    if (i < len) {
        //unsigned int offset = blockIdx.y * gridDim.x + blockIdx.x;
        //offset *= 4*256;
        // get wether rgb or alpha value 
        //unsigned int entry = offset+ colors[i];
        unsigned int entry = (i%4)*256 + colors[i];
	atomicAdd(&buckets[entry], 1);
        //entry = offset + 256   + colors[i+1];
        //atomicAdd(&buckets[entry], 1);
        //entry = offset + 256*2 + colors[i+2];
        //atomicAdd(&buckets[entry], 1);
        //entry = offset + 256*3 + colors[i+3];
        //atomicAdd(&buckets[entry], 1);
    }
}

__global__ void gpuGood_MergeBlocks(unsigned int* buckets, unsigned int blockcnt) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    //unsigned int i = iy * 256 + ix;
    unsigned int i = (iy * blockDim.x * gridDim.x + ix); 
    
    /*
    if (i < 4*256) {

        //unsigned int offset = blockIdx.y * gridDim.x + blockIdx.x;
        //offset *= 4*256;
        for(unsigned int j=1; j < blockcnt; j++)
        {
            unsigned int entry = i+ j *4*256;
            //atomicAdd(&buckets[i], buckets[entry]);
	    buckets[i] += buckets[entry];
        }
    }*/
    
	    atomicAdd(&buckets[i%(256*4)], buckets[i+(256*4)]);
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
        block.y = 1;
        grid.x = ceil((double)(rows*cols)/ block.x); 
        grid.y = 1;
        
	CHECK(cudaMalloc(&d_buckets, sizeof(unsigned int) * 256*4));
        printf("Using naive GPU implementation\n");
        gettimeofday(&start,NULL);
        gpuNaive<<<grid, block>>>(d_colors, d_buckets, len, rows, cols );
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
        
	CHECK(cudaMalloc(&d_buckets, sizeof(unsigned int) * 256*4 * grid.x ));
        printf("Using good GPU implementation\n");
        gettimeofday(&start,NULL);
    
        gpuGood_Block<<<grid, block>>>(d_colors, d_buckets, len, rows, cols);
        unsigned int blockCnt = grid.x;
        //block.x = 256;
        //block.y = 4;
        grid.x -= 1; 
        //grid.y = 1;
       	//cudaDeviceSynchronize(); 
        gpuGood_MergeBlocks<<<grid, block>>>(d_buckets, blockCnt);
        gettimeofday(&end,NULL);

        CHECK(cudaMemcpy(buckets, d_buckets, sizeof(unsigned int) *256*4, cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_buckets));
    }
    CHECK(cudaFree(d_colors));

    double start_seconds = ((double)start.tv_sec + (double)start.tv_usec*1.e-6);
    double end_seconds = ((double)end.tv_sec + (double)end.tv_usec*1.e-6);
    return end_seconds - start_seconds;

}


