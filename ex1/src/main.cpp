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
#include "../inc/lodepng.h"
#include "../inc/hist.h"
#include <sys/time.h>

double cpuOnly(const unsigned char* colors, unsigned int* buckets, unsigned int len) {
    struct timeval start, end;
    gettimeofday(&start,NULL);
    
    for (unsigned int i = 0; i < len; i++)
    {
        // get wether rgb or alpha value 
        unsigned int color = i % 4;
        unsigned int entry = 256*color + colors[i];
	    buckets[entry]++;
    }

    gettimeofday(&end,NULL);
    double start_seconds = ((double)start.tv_sec + (double)start.tv_usec*1.e-6);
    double end_seconds = ((double)end.tv_sec + (double)end.tv_usec*1.e-6);

    return end_seconds - start_seconds;
}

int main(int argc, char** argv) {
    if (argc != 4)
    {
        printf("Usage: %s <implementation (0-2)> <repetitions> <image>\n", argv[0]);
        printf("Implementations:\n");
        printf("\t0: cpu only\n");
        printf("\t1: naive gpu (from lecture)\n");
        printf("\t2: intelligent gpu\n");
        return 1;
    }

    unsigned char impl = atoi(argv[1]);
    unsigned int repetitions = atoi(argv[2]);
    if (impl > 2)
    {
        printf("Unknown implementation %d\n", impl);
        return 1;
    }

    // Read the arguments
    const char* input_file = argv[3];

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
    long double runtime = 0;

    for(unsigned int i = 0; i < repetitions; i++)
    {
        memset(buckets, 0, 256*4 *sizeof(unsigned int));
        if (impl == 0)
        {
            runtime += (long double) cpuOnly(colors, buckets, in_image.size());
        }
        else
        {
            runtime += (long double) runOnGpu(colors, buckets, in_image.size(), height, width, impl);
        } 
    }

    for(int i = 0; i < 256; i++)
    {
        printf("%4u | %6d | %6d | %6d | %6d \n", i, buckets[i], buckets[i+256], buckets[i+(256*2)], buckets[i+(256*3)]);
    }

    printf("Runtime total:%Lf, avg:%Lf\n", runtime, runtime/repetitions);  
    free(buckets);
}
