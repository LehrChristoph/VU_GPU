#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "graph.h"
#include "impl.h"

#define n_functions 2

int main(int argc, char** argv) {
    if (argc == 1) {
        printf("Usage: [generate|bench|calculate] ...\n");
        return 1;
    }

    unsigned int seed = time(NULL);
    seed = 1654506507;
    printf("seeding with %d\n", seed);
    srandom(seed);

    if (strcmp(argv[1], "generate") == 0) {
        if (argc != 6 && argc != 7) {
            printf("Usage: %s generate [filename] <#nodes> <density (0-1)> <min_weight> <max_weight>\n", argv[0]);
            return 1;
        }

        unsigned int num_nodes = atoi(argv[argc-4]);
        float density = atof(argv[argc-3]);
        unsigned int min_weight = atoi(argv[argc-2]);
        unsigned int max_weight = atoi(argv[argc-1]);

        printf("density: %f\n", density);
        unsigned int * adjacency_matrix = (unsigned int *) malloc(sizeof(unsigned int ) * num_nodes * num_nodes);
        graph_generate(adjacency_matrix, num_nodes, density, min_weight, max_weight);

        for(unsigned int i=0; i < num_nodes;  i++) {
            for(unsigned int j=0; j < num_nodes; j++) {
                printf("%d ", adjacency_matrix[i * num_nodes + j]);
            }
            printf("\n");
        }

        graph_write(num_nodes, adjacency_matrix, argv[2]);
        free(adjacency_matrix);

    } else if (strcmp(argv[1], "calculate") == 0) {
        if (argc != 4 && (argc != 6 || strcmp(argv[3], "--generate"))) {
            printf("Usage: %s calculate <impl (0-0)> <filename|--generate <#nodes> <density (0-1)>>\n", argv[0]);
            return 1;
        }

        int impl = atoi(argv[2]);
        if (impl < 0 || impl >= n_functions) {
            printf("Invalid implementation %d\n", impl);
            return 1;
        }

        unsigned int *adjacency_matrix;
        unsigned int num_nodes;
        if (argc == 4) {
            if (!strcmp(argv[3], "--generate")) {
                printf("Usage: %s calculate <impl (0-%d)> <filename|--generate <#nodes> <density (0-1)>>\n", argv[0], n_functions-1);
                return 1;
            }

            adjacency_matrix = graph_read(argv[3],&num_nodes);
        } else {
            num_nodes = atoi(argv[argc-2]);
            float density = atof(argv[argc-1]);
            adjacency_matrix = (unsigned int *) malloc(sizeof(unsigned int ) * num_nodes * num_nodes);
            graph_generate(adjacency_matrix, num_nodes, density, 1, 1);
        }
        
        clock_t runtime;

        unsigned int * connected_components = (unsigned int *) malloc(sizeof(unsigned int) * num_nodes);
        if(impl == 0) {
            runtime = calculate_connected_components_cpu(num_nodes, adjacency_matrix, connected_components);
        } else if(impl == 1) {
            runtime = calculate_connected_components_gpu_thrust(num_nodes, adjacency_matrix, connected_components);
        }

        for (int i = 0; i < num_nodes; i++) {
            printf("%d: %d\n", i, connected_components[i]);
        }
        
        double runtime_secs = ((double) runtime) / CLOCKS_PER_SEC;
        printf("Runtime %lf\n", runtime_secs);

        free(connected_components);
        free(adjacency_matrix);

    } else if (strcmp(argv[1], "bench") == 0) {
        if (argc != 5) {
            printf("Usage: %s bench <rounds> <#nodes> <do-checking (0/1)>\n", argv[0]);
            return 1;
        }
        
        int rounds = atoi(argv[2]);
        int num_nodes = atoi(argv[3]);
        int do_checking = atoi(argv[4]);

        double avg_runtime_cpu_secs, avg_runtime_gpu_thrust_secs;
        
        unsigned int * connected_components_cpu = (unsigned int *) malloc(sizeof(unsigned int) * num_nodes);
        unsigned int * connected_components_gpu_thrust = (unsigned int *) malloc(sizeof(unsigned int) * num_nodes);
        unsigned int * adjacency_matrix = (unsigned int *) malloc(sizeof(unsigned int ) * num_nodes * num_nodes);
	
        for (unsigned int round = 1; round < rounds-1; round++) {
            float density = (float)round/(float)rounds/10;
            printf("%d nodes, %f density\n", num_nodes, density);

            graph_generate(adjacency_matrix, num_nodes, density, 1, 1);

            double runtime_cpu = calculate_connected_components_cpu(num_nodes, adjacency_matrix, connected_components_cpu);
            double runtime_cpu_secs= ((double) runtime_cpu) / CLOCKS_PER_SEC;
            avg_runtime_cpu_secs += runtime_cpu_secs;

            double runtime_gpu_thrust = calculate_connected_components_gpu_thrust(num_nodes, adjacency_matrix, connected_components_gpu_thrust);
            double runtime_gpu_thrust_secs= ((double) runtime_gpu_thrust) / CLOCKS_PER_SEC;
            avg_runtime_gpu_thrust_secs += runtime_gpu_thrust_secs;

            if(do_checking != 0) {
                for (unsigned int i = 0; i < num_nodes; i++) {
                    if(connected_components_cpu[i] != connected_components_gpu_thrust[i]) {
                        printf("Connected components of algorithms do not match\n");
                        return -1;
                    }
                }
            }
            
            printf("Runtime CPU %lf, GPU thrust %lf \n", runtime_cpu_secs, runtime_gpu_thrust_secs);
        }

        printf("Total runtime CPU %lf, GPU thrust %lf \n", avg_runtime_cpu_secs, avg_runtime_gpu_thrust_secs);
        printf("Average runtime CPU %lf, GPU thrust %lf \n", avg_runtime_cpu_secs/num_nodes, avg_runtime_gpu_thrust_secs/num_nodes);
        
        free(connected_components_cpu);
        free(connected_components_gpu_thrust);
        free(adjacency_matrix);

    } else {
        printf("Unknown command %s, available: generate, bench, calculate\n", argv[0]);
    }
}
