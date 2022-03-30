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
#include <opencv2/opencv.hpp>

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


void cpuOnly(const uchar* colors, int* buckets, int len) {
    printf("Using CPU implementation\n");
    for (int i = 0; i < len; i++) {
        buckets[colors[i]]++;
    }
}

__global__ void gpuNaive(uchar* colors, int* buckets, int len, int rows, int cols) {
    printf("Using naive GPU implementation\n");
    int i = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * blockIdx.x + blockDim.x * blockDim.y * blockIdx.y * gridDim.x;
    if (i < len) {
        int c = colors[i];
        atomicAdd(&buckets[c], 1);
    }
}

__global__ void gpuGood(uchar* colors, int* buckets, int len, int rows, int cols) {
    printf("Using good GPU implementation\n");
    // TODO
    int i = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * blockIdx.x + blockDim.x * blockDim.y * blockIdx.y * gridDim.x;
    if (i < len) {
        int c = colors[i];
        atomicAdd(&buckets[c], 1);
    }
}

void runOnGpu(const uchar* colors, int* buckets, int len, int rows, int cols, void(*gpuFunc)(uchar*, int*, int, int, int)) {
    uchar* d_colors;
    int* d_buckets;
    CHECK(cudaMalloc(&d_colors, sizeof(uchar) * len));
    CHECK(cudaMalloc(&d_buckets, sizeof(int) * 256));
    CHECK(cudaMemcpy(d_colors, colors, sizeof(uchar) * len, cudaMemcpyHostToDevice));
    dim3 grid, block;
    block.x = 32;
    block.y = 32;
    grid.x = (rows - 1) / block.x + 1;
    grid.y = (cols - 1) / block.y + 1;
    /* printf("%d - %d\n", grid.x, grid.y); */
    gpuFunc<<<grid, block>>>(d_colors, d_buckets, len, rows, cols);
    CHECK(cudaMemcpy(buckets, d_buckets, sizeof(int) * 256, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_colors));
    CHECK(cudaFree(d_buckets));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <image> <implementation (0-2)>\n", argv[0]);
    }
    int impl = atoi(argv[2]);

    cv::Mat im = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    const uchar* colors = im.datastart;
    int imlen = im.dataend - im.datastart;
    int* buckets = (int*) calloc(256, sizeof(int));
    if (impl == 0) {
        cpuOnly(colors, buckets, imlen);
    } else if (impl == 1) {
        runOnGpu(colors, buckets, imlen, im.rows, im.cols, gpuNaive);
    } else if (impl == 2) {
        runOnGpu(colors, buckets, imlen, im.rows, im.cols, gpuGood);
    } else {
        printf("Unknown implementation %d\n", impl);
        free(buckets);
        return 1;
    }
    for (int i = 0; i < 256; i++) {
        printf("|%d", buckets[i]);
    }
    printf("|\n");
    free(buckets);
}
