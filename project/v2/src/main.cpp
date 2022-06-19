#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <dirent.h>
#include <libgen.h>
#include <linux/limits.h>

#include "graph.h"
#include "impl.h"

#define n_functions 4

typedef clock_t (*connected_components_function)(unsigned int, unsigned int *, unsigned int *);
char* function_names[n_functions] = {"CPU", "GPU", "GPU Pinned", "GPU Zero Copy"};
connected_components_function functions[n_functions] = { calculate_connected_components_cpu, calculate_connected_components_gpu_simple, calculate_connected_components_gpu_simple_pinned, calculate_connected_components_gpu_simple_zero_copy };

void evaluate(char *filename) {
    printf("%s", basename(filename));
    fprintf(stderr, "Graph: %s\n", basename(filename));

    clock_t runtime;
    for (int impl = 0; impl < n_functions; impl++) {
        fprintf(stderr, "- v2 %s\n", function_names[impl]);
        unsigned int num_nodes;
        unsigned int *adjacency_matrix = graph_read(filename, &num_nodes);
        unsigned int * connected_components = (unsigned int *) malloc(sizeof(unsigned int) * num_nodes);
        runtime = functions[impl](num_nodes, adjacency_matrix, connected_components);
        printf(";%f", ((double) runtime) / CLOCKS_PER_SEC);
        free(connected_components);
        free(adjacency_matrix);
    }

    printf("\n");
}

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

        unsigned int * adjacency_matrix = (unsigned int *) malloc(sizeof(unsigned int ) * num_nodes * num_nodes);
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
            adjacency_matrix = (unsigned int *) malloc(sizeof(unsigned int ) * num_nodes * num_nodes);
            graph_generate(adjacency_matrix, num_nodes, density, 1, 10);
        }
        
        clock_t runtime;

        unsigned int * connected_components = (unsigned int *) malloc(sizeof(unsigned int) * num_nodes);
        if(impl ==0 )
        {
            runtime = calculate_connected_components_cpu(num_nodes, adjacency_matrix, connected_components);
        }
        else if(impl == 1 )
        {
            runtime = calculate_connected_components_gpu_simple(num_nodes, adjacency_matrix, connected_components);
        }
        else if(impl == 2 )
        {
            runtime = calculate_connected_components_gpu_simple_pinned(num_nodes, adjacency_matrix, connected_components);
        }
        else if(impl == 3 )
        {
            runtime = calculate_connected_components_gpu_simple_zero_copy(num_nodes, adjacency_matrix, connected_components);
        }
        double runtime_secs= ((double) runtime) / CLOCKS_PER_SEC;
         
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
	
        printf("Runtime %lf\n", runtime_secs);

        free(connected_components);
        free(adjacency_matrix);

    } else if (strcmp(argv[1], "bench") == 0) {
        if (argc != 6) {
            printf("Usage: %s bench <rounds> <#nodes> <do-checking (0/1) <data-file>\n", argv[0]);
            return 1;
        }
        
        int rounds = atoi(argv[2]);
        int num_nodes = atoi(argv[3]);
        int do_checking = atoi(argv[4]);
        char* file_name = argv[5];

        double avg_runtime_cpu_secs, avg_runtime_gpu_simple_secs, avg_runtime_gpu_simple_pinned_secs, avg_runtime_gpu_simple_zero_copy_secs;
        
        unsigned int * connected_components_cpu = (unsigned int *) malloc(sizeof(unsigned int) * num_nodes);
        unsigned int * connected_components_gpu_simple = (unsigned int *) malloc(sizeof(unsigned int) * num_nodes);
        unsigned int * connected_components_gpu_simple_pinned = (unsigned int *) malloc(sizeof(unsigned int) * num_nodes);
        unsigned int * connected_components_gpu_simple_zero_copy = (unsigned int *) malloc(sizeof(unsigned int) * num_nodes);
        unsigned int * adjacency_matrix = (unsigned int *) malloc(sizeof(unsigned int ) * num_nodes * num_nodes);
        if(adjacency_matrix == NULL)
        {
            printf("Unable to allocate memory for graph\n");
            return -1;
        }	

        FILE *fp = fopen(file_name, "w+");
        fprintf(fp,"cnt;nodes;density;CPU-min;CPU-avg;CPU-max;GPU-min;GPU-avg;GPU-max;GPU-Pinned-min;GPU-Pinned-avg;GPU-Pinned-max;GPU-Zero-Copy-min;GPU-Zero-Copy-avg;GPU-Zero-Copy-max;\n");
        fclose(fp);
	clock_t start = clock();

        unsigned int cnt=0;
        for(int i=3; i>0;i--)
        {
            for(unsigned int j=1; j<10;j++)
            {
                float density =  pow(10, -i)*j;
                double runtime_cpu_secs_min, runtime_cpu_secs_avg, runtime_cpu_secs_max;
                double runtime_gpu_simple_secs_min, runtime_gpu_simple_secs_avg, runtime_gpu_simple_secs_max;
                double runtime_gpu_simple_pinned_secs_min, runtime_gpu_simple_pinned_secs_avg, runtime_gpu_simple_pinned_secs_max;
                double runtime_gpu_simple_zero_copy_secs_min, runtime_gpu_simple_zero_copy_secs_avg, runtime_gpu_simple_zero_copy_secs_max;
        
                //float density = (float)round/(float)rounds;
                cnt++;
                for (unsigned int round = 0; round < rounds; round++) {
                    graph_generate(adjacency_matrix, num_nodes, density, 1, 1);
                        
                    double runtime_cpu = calculate_connected_components_cpu(num_nodes, adjacency_matrix, connected_components_cpu);
                    double runtime_cpu_secs= ((double) runtime_cpu) / CLOCKS_PER_SEC;
                    runtime_cpu_secs_avg += runtime_cpu;

                    if(runtime_cpu_secs < runtime_cpu_secs_min || round==0)
                    {
                        runtime_cpu_secs_min=runtime_cpu_secs;
                    }

                    if(runtime_cpu_secs > runtime_cpu_secs_max || round==0)
                    {
                        runtime_cpu_secs_max=runtime_cpu_secs;
                    }

                    double runtime_gpu_simple = calculate_connected_components_gpu_simple(num_nodes, adjacency_matrix, connected_components_gpu_simple);
                    double runtime_gpu_simple_secs= ((double) runtime_gpu_simple) / CLOCKS_PER_SEC;
                    runtime_gpu_simple_secs_avg += runtime_gpu_simple;

                    if(runtime_gpu_simple_secs < runtime_gpu_simple_secs_min || round==0)
                    {
                        runtime_gpu_simple_secs_min=runtime_gpu_simple_secs;
                    }

                    if(runtime_gpu_simple_secs > runtime_gpu_simple_secs_max || round==0)
                    {
                        runtime_gpu_simple_secs_max=runtime_gpu_simple_secs;
                    }

                    double runtime_gpu_simple_pinned = calculate_connected_components_gpu_simple(num_nodes, adjacency_matrix, connected_components_gpu_simple_pinned);
                    double runtime_gpu_simple_pinned_secs= ((double) runtime_gpu_simple_pinned) / CLOCKS_PER_SEC;
                    runtime_gpu_simple_pinned_secs_avg += runtime_gpu_simple_pinned;

                    if(runtime_gpu_simple_pinned < runtime_gpu_simple_pinned_secs_min || round==0)
                    {
                        runtime_gpu_simple_pinned_secs_min=runtime_gpu_simple_pinned_secs;
                    }

                    if(runtime_gpu_simple_pinned > runtime_gpu_simple_pinned_secs_max || round==0)
                    {
                       runtime_gpu_simple_pinned_secs_max=runtime_gpu_simple_pinned_secs;
                    }

                    double runtime_gpu_simple_zero_copy = calculate_connected_components_gpu_simple(num_nodes, adjacency_matrix, connected_components_gpu_simple_zero_copy);
                    double runtime_gpu_simple_zero_copy_secs= ((double) runtime_gpu_simple_zero_copy) / CLOCKS_PER_SEC;
                    runtime_gpu_simple_zero_copy_secs_avg += runtime_gpu_simple_zero_copy;

                    if(runtime_gpu_simple_zero_copy_secs < runtime_gpu_simple_zero_copy_secs_min || round==0)
                    {
                        runtime_gpu_simple_zero_copy_secs_min=runtime_gpu_simple_zero_copy_secs;
                    }

                    if(runtime_gpu_simple_zero_copy_secs > runtime_gpu_simple_zero_copy_secs_max || round==0)
                    {
                        runtime_gpu_simple_zero_copy_secs_max=runtime_gpu_simple_zero_copy_secs;
                    }

                    if(do_checking != 0)
                    {
                        for (unsigned int i = 0; i < num_nodes; i++) {
                        
                            if( connected_components_cpu[i] != connected_components_gpu_simple[i] ||
                                connected_components_cpu[i] != connected_components_gpu_simple_pinned[i] ||
                                connected_components_cpu[i] != connected_components_gpu_simple_zero_copy[i] 
                            )
                            {
                                printf("Connected components of algorithms do not match\n");
                                break;
                            }

                        }
                    }

                    printf("Density %lf, Runtime CPU %lf, GPU Simple %lf, GPU Pinned: %lf, GPU Zero-Copy %lf \n", density, runtime_cpu_secs, runtime_gpu_simple_secs, runtime_gpu_simple_pinned_secs, runtime_gpu_simple_zero_copy_secs);
		        }

                runtime_cpu_secs_avg = runtime_cpu_secs_avg / rounds / CLOCKS_PER_SEC;
                runtime_gpu_simple_secs_avg = runtime_gpu_simple_secs_avg / rounds / CLOCKS_PER_SEC;
                runtime_gpu_simple_pinned_secs_avg = runtime_gpu_simple_pinned_secs_avg / rounds / CLOCKS_PER_SEC;
                runtime_gpu_simple_zero_copy_secs_avg = runtime_gpu_simple_zero_copy_secs_avg / rounds / CLOCKS_PER_SEC;
                
                fp = fopen(file_name, "a");

                fprintf(fp,"%d;%d;%f;", cnt, num_nodes, density);
                fprintf(fp,"%f;%f;%f;", runtime_cpu_secs_min, runtime_cpu_secs_avg, runtime_cpu_secs_max);
                fprintf(fp,"%f;%f;%f;", runtime_gpu_simple_secs_min, runtime_gpu_simple_secs_avg, runtime_gpu_simple_secs_max);
                fprintf(fp,"%f;%f;%f;", runtime_gpu_simple_pinned_secs_min, runtime_gpu_simple_pinned_secs_avg, runtime_gpu_simple_pinned_secs_max);
                fprintf(fp,"%f;%f;%f;", runtime_gpu_simple_zero_copy_secs_min, runtime_gpu_simple_zero_copy_secs_avg, runtime_gpu_simple_zero_copy_secs_max);
                fprintf(fp,"\n");

                fclose(fp);
            }
        }

        free(connected_components_cpu);
        free(connected_components_gpu_simple);
        free(adjacency_matrix);
        
	double runtime = (clock() - start)/CLOCKS_PER_SEC;
	printf("Total Runtime %lf\n", runtime );

    } else if (strcmp(argv[1], "evaluate") == 0) {
        printf("graph");
        for (int impl = 0; impl < n_functions; impl++) {
            printf(";v1 %s", function_names[impl]);
        }
        printf("\n");

        char path[PATH_MAX];
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(argv[2])) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                if (ent->d_type != DT_REG) continue;
                sprintf(path, "%s/%s", argv[2], ent->d_name);
                if (basename(path)[0] != '_') evaluate(path);
            }
            closedir(dir);
        } else {
            perror("could not open evaluation folder");
            return 2;
        }

        return 0;
    } else {
        printf("Unknown command %s, available: generate, bench, calculate\n", argv[0]);
    }
}
