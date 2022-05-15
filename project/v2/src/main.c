#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "graph.h"
#include "impl.h"

#define n_functions 2

int main(int argc, char** argv) {
    srandom(time(NULL));
    if (argc == 1) {
        printf("Usage: [generate|bench|calculate] ...\n");
        return 1;
    }
    if (strcmp(argv[1], "generate") == 0) {
        if (argc != 6 && argc != 7) {
            printf("Usage: %s generate [filename] <#nodes> <density (0-1)> <min_weight> <max_weight>\n", argv[0]);
            return 1;
        }

        unsigned int num_nodes = atoi(argv[argc-4]);
        float density = atof(argv[argc-3]);
        unsigned int min_weight = atoi(argv[argc-2]);
        unsigned int max_weight = atoi(argv[argc-1]);

        unsigned int * adjacency_matrix = malloc(sizeof(unsigned int ) * num_nodes * num_nodes);
        graph_generate(adjacency_matrix, num_nodes, density, min_weight, max_weight);
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
            int num_nodes = atoi(argv[argc-2]);
            float density = atof(argv[argc-1]);
            adjacency_matrix = malloc(sizeof(unsigned int ) * num_nodes * num_nodes);
            graph_generate(adjacency_matrix, num_nodes, density, 1, 10);
        }
        
        clock_t runtime;

        unsigned int * connected_components = malloc(sizeof(unsigned int) * num_nodes);
        if(impl ==0 )
        {
            runtime = calculate_connected_components_cpu(num_nodes, adjacency_matrix, connected_components);
        }
        
        for(unsigned int i=0; i < num_nodes;  i++)
        {
            for(unsigned int j=0; j < num_nodes; j++)
            {
                printf("%d ", adjacency_matrix[i * num_nodes + j]);
            }
            printf("\n");
        }

        for(unsigned int i=0; i < num_nodes; i++)
        {
            printf("%u: ", i);
            for(unsigned int j=0; j < num_nodes; j++)
            {
                if(connected_components[j] == i)
                {
                    printf(" %d", j);
                }
            }
            printf("\n");
        }

        free(connected_components);
        free(adjacency_matrix);
        /*
        connected_components *connected_components;
        functions[impl](dense_graph, &connected_components);
        write_connected_components(connected_components, stdout);

        free_connected_components(connected_components);
        free_dense(dense_graph);
        free_graph(graph);
        */
    } else if (strcmp(argv[1], "bench") == 0) {
        if (argc != 5) {
            printf("Usage: %s bench <rounds> <#nodes> <do-checking (0/1)>\n", argv[0]);
            return 1;
        }
        /*
        int rounds = atoi(argv[2]);
        int num_nodes = atoi(argv[3]);
        int do_checking = atoi(argv[4]);

        arr(double, mins, n_functions, {[0 ... n_functions-1] = DBL_MAX});
        arr(double, maxs, n_functions, {[0 ... n_functions-1] = 0});
        arr(double, sums, n_functions, {[0 ... n_functions-1] = 0});

        arr(char*, function_names, n_functions, {"CPU", "GPU / Thread per CC"});

        clock_t dur;
        for (int round = 0; round < rounds; round++) {
            printf("%d nodes, %f density\n", num_nodes, (float)round/(float)rounds);
            dense_graph *dense_graph = generate(num_nodes, (float)round/(float)rounds, 1, 1);
            connected_components *true_components;
            calculate_connected_components(dense_graph, &true_components);

            for (int i = 0; i < n_functions; i++) {
                connected_components *calculated;
                dur = functions[i](dense_graph, &calculated);

                if (do_checking) {
                    int compare_result = compare_connected_components(true_components, calculated);
                    if (compare_result != 1) {
                        printf("Function %s failed for graph:\n", function_names[i]);
                        graph *out_graph = from_dense(dense_graph);
                        write_graph(out_graph, stdout);
                        free_graph(out_graph);
                        printf("\n\nWith result:\n");
                        write_connected_components(calculated, stdout);
                        printf("\n\nShould have been:\n");
                        write_connected_components(true_components, stdout);
                        return compare_result;
                    }
                }

                double duration = (double) dur / (double) CLOCKS_PER_SEC;
                //printf("Graph with %d nodes and %d edges took %f seconds for %s\n", dense_graph->num_nodes, dense_graph->num_edges, duration, function_names[i]);
                if (mins[i] > duration) mins[i] = duration;
                if (maxs[i] < duration) maxs[i] = duration;
                sums[i] += duration;
            }

            free_dense(dense_graph);
        }

        for (int i = 0; i < n_functions; i++) {
            printf("Function %s took min %f, avg %f, max %f (total %f), speedup factor %f\n", function_names[i], mins[i], sums[i]/(double)rounds, maxs[i], sums[i], sums[0] / sums[i]);
        }
        */
    } else {
        printf("Unknown command %s, available: generate, bench, calculate\n", argv[0]);
    }
}
