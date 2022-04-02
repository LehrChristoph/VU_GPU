/**
 * compile with `nvcc hist.cu -o hist -lopencv_imgcodecs -lopencv_core`
 * run with `./hist image.png <impl>`
 * where impl in
 * 0: cpu only
 * 1: naive gpu (from lecture)
 * 2: intelligent gpu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../inc/lodepng.h"
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


void cpuOnly(const unsigned char* colors, unsigned int* buckets, unsigned int len) {
    printf("Using CPU implementation\n");
    for (unsigned int i = 0; i < len; i++) {
        // get wether rgb or alpha value 
        unsigned int color = i % 4;
        unsigned int entry = 256*color + colors[i];
        buckets[entry]++;
    }
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

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <image> <implementation (0-2)>\n", argv[0]);
        printf("Implementations:\n");
        printf("\t0: cpu only\n");
        printf("\t1: naive gpu (from lecture)\n");
        printf("\t2: intelligent gpu\n");
        return 1;
    }
    int impl = atoi(argv[2]);

    if (impl < 0 || impl > 2){
        printf("Unknown implementation %d\n", impl);
        return 1;
    }

    // Read the arguments
    const char* input_file = argv[1];

    std::vector<unsigned char> in_image;
    unsigned int width, height;

    // Load the data
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if(error) 
    {
        printf("decoder error %u : %s\n", error, lodepng_error_text(error));
    }
    // convert vector to array
    unsigned char* colors = &in_image[0];
    unsigned int* buckets = (unsigned int*) calloc(256*4, sizeof(unsigned int));
    if (impl == 0) {
        cpuOnly(colors, buckets, in_image.size());
    } else if (impl == 1) {
        runOnGpu(colors, buckets, in_image.size(), height, width, gpuNaive);
    } else if (impl == 2) {
        runOnGpu(colors, buckets, in_image.size(), height, width, gpuGood);
    } 
    for (int i = 0; i < 256; i++) {
        printf("%4u | %6d | %6d | %6d | %6d \n", i, buckets[i], buckets[i+256], buckets[i+(256*2)], buckets[i+(256*3)] );
    }
    
    free(buckets);
}
