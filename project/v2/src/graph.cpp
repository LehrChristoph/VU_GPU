#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "graph.h"

void graph_generate(unsigned int * adjacency_matrix,unsigned int num_nodes, float density, unsigned int  min_weight, unsigned int  max_weight) {
    if (num_nodes <= 0 || density < 0 || density > 1 || max_weight < min_weight ) {
        printf("Invalid input: num_nodes (%d) or density (%f) \n", num_nodes, density);
        return;
    }
    
    // allocat matrix to store graph data
    unsigned int num_edges = (unsigned int) (num_nodes * (num_nodes - 1) * density / 2);
     
    // init adjeceny matrix
    for (unsigned int i=0; i < num_nodes; i++)
    {
        for (unsigned int j=0; j < num_nodes; j++)
        {
            adjacency_matrix[i * num_nodes + j] = 0;
        }
    }
     
    // fill adjecency matrix with weights
    for (int i = 0; i < num_edges; i++) {
        unsigned int from = random() % num_nodes;
        unsigned int to = random() % num_nodes;
        unsigned int  weight = min_weight;
	if(max_weight> min_weight)
	{
 		weight += random() % (max_weight - min_weight);
 	}
	
        while (from == to || adjacency_matrix[from * num_nodes + to] != 0) {
            to = random() % num_nodes;
        }
	
        adjacency_matrix[from * num_nodes + to] = weight;
        adjacency_matrix[to * num_nodes + from] = weight;
        
    }
    
    return;
}

unsigned int * graph_read(char* filename, unsigned int *number_of_nodes) {
    char buf[64];
    char *line = buf;
    FILE* fp = fopen(filename, "r");
    
    if (!fp) {
        printf("Could not open file %s\n", filename);
        return 0;
    }
    fgets(buf, 64, fp);
    
    if (strcmp(strsep(&line, " "), "H")) {
        printf("Invalid input file (header line '%s')\n", buf);
        return 0;
    }
    
    unsigned int num_nodes = atoi(strsep(&line, " "));
    unsigned int *adjacency_matrix = (unsigned int *) malloc(sizeof(unsigned int ) * num_nodes * num_nodes);
    if(adjacency_matrix == NULL)
    {
	    printf("Error: unable to allocate space for graph");
	    return NULL;
	}
	
    // init adjeceny matrix
    for (unsigned int i=0; i < num_nodes; i++)
    {
        for (unsigned int j=0; j < num_nodes; j++)
        {
            adjacency_matrix[i * num_nodes + j] = 0;
        }
    }
    
    while (fgets(buf, 64, fp))
    {
        line = buf;
        if (strcmp(strsep(&line, " "), "E")) {
            printf("Invalid input file (header line '%s')\n", buf);
            free(adjacency_matrix);

            return 0;
        }
        unsigned int i = atoi(strsep(&line, " "));
        unsigned int j = atoi(strsep(&line, " "));
        unsigned int weight = atoi(strsep(&line, " "));
        adjacency_matrix[i * num_nodes + j] = weight;
        adjacency_matrix[j * num_nodes + i] = weight;
    }
    fclose(fp);

    *number_of_nodes = num_nodes;

    return adjacency_matrix;
}

void graph_write(unsigned int num_nodes, unsigned int *adjacency_matrix, char* filename) {

    char buf[64];
    char *line = buf;
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("Could not open file %s\n", filename);
    }

    fprintf(fp, "H %d \n", num_nodes);
    for (unsigned int i = 0; i < num_nodes; i++) {
        for(unsigned int j=0; j < num_nodes; j++)
        {
            if(adjacency_matrix[i * num_nodes + j] > 0)
            {
                fprintf(fp, "E %d %d %d\n", i, j, adjacency_matrix[i * num_nodes + j]);        
            }
        }
    }

    fclose(fp);
}
