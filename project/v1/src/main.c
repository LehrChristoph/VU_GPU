#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "io.h"
#include "generator.h"
#include "impl.h"

#define arr(type, name, size, ...) type name[] = __VA_ARGS__; _Static_assert(sizeof(name) / sizeof(*name) == size, #name " not initialized correctly");

#define n_functions 7

arr(connected_components_function, functions, n_functions, {calculate_connected_components, connected_components_thread_per_cc, connected_components_thread_per_cc_vector, connected_components_pinned, connected_components_vector_pinned, connected_components_zerocopy, connected_components_vector_zerocopy});

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

        int num_nodes = atoi(argv[argc-4]);
        float density = atof(argv[argc-3]);
        int min_weight = atoi(argv[argc-2]);
        int max_weight = atoi(argv[argc-1]);
        FILE* fp = argc == 6 ? stdout : fopen(argv[2], "w");
        dense_graph *dense_graph = generate(num_nodes, density, min_weight, max_weight);
        graph* graph = from_dense(dense_graph);
        if (graph) {
            write_graph(graph, fp);
        }
        free_graph(graph);
        free_dense(dense_graph);
        fclose(fp);

    } else if (strcmp(argv[1], "calculate") == 0) {
        if (argc != 4 && (argc != 6 || strcmp(argv[3], "--generate"))) {
            printf("Usage: %s calculate <impl (0-%d)> <filename|--generate <#nodes> <density (0-1)>>\n", argv[0], n_functions-1);
            return 1;
        }

        int impl = atoi(argv[2]);
        if (impl < 0 || impl >= n_functions) {
            printf("Invalid implementation %d\n", impl);
            return 1;
        }

        graph *graph;
        dense_graph *dense_graph;
        if (argc == 4) {
            if (!strcmp(argv[3], "--generate")) {
                printf("Usage: %s calculate <impl (0-%d)> <filename|--generate <#nodes> <density (0-1)>>\n", argv[0], n_functions-1);
                return 1;
            }
            graph = read_graph(argv[3]);
            dense_graph = to_dense(graph);
        } else {
            int num_nodes = atoi(argv[argc-2]);
            float density = atof(argv[argc-1]);
            dense_graph = generate(num_nodes, density, 1, 10);
        }
        if (!dense_graph) {
            return 1;
        }

        connected_components *connected_components;
        functions[impl](dense_graph, &connected_components);
        write_connected_components(connected_components, stdout);

        free_connected_components(connected_components);
        free_dense(dense_graph);
        free_graph(graph);
    } else if (strcmp(argv[1], "bench") == 0) {
        if (argc != 5) {
            printf("Usage: %s bench <rounds> <#nodes> <do-checking (0/1)>\n", argv[0]);
            return 1;
        }

        int rounds = atoi(argv[2]);
        int num_nodes = atoi(argv[3]);
        int do_checking = atoi(argv[4]);

        arr(double, mins, n_functions, {[0 ... n_functions-1] = DBL_MAX});
        arr(double, maxs, n_functions, {[0 ... n_functions-1] = 0});
        arr(double, sums, n_functions, {[0 ... n_functions-1] = 0});

        arr(char*, function_names, n_functions, {"CPU", "GPU / Thread per CC", "GPU / Thread per CC (Vector)", "GPU / Pinned", "GPU / Pinned (Vector)", "GPU / Zero copy", "GPU / Zero copy (Vector)"});

        clock_t dur;
        for (int round = 0; round < rounds; round++) {
            printf("%d nodes, %f density\n", num_nodes, (float)round/(float)rounds);
            dense_graph *dense_graph = generate(num_nodes, (float)round/(float)rounds, 1, 1);
            connected_components *true_components;
            calculate_connected_components(dense_graph, &true_components);

	    printf("Runtime ");
            for (int i = 0; i < n_functions; i++) {
                if (i != 0) printf(", ");
                connected_components *calculated;
                dur = functions[i](dense_graph, &calculated);
                printf("%s %f", function_names[i], (double) dur / (double) CLOCKS_PER_SEC);

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
            printf("\n");
        }

        for (int i = 0; i < n_functions; i++) {
            printf("Function %s took min %f, avg %f, max %f (total %f), speedup factor %f\n", function_names[i], mins[i], sums[i]/(double)rounds, maxs[i], sums[i], sums[0] / sums[i]);
        }

    } else {
        printf("Unknown command %s, available: generate, bench, calculate\n", argv[0]);
    }
}
