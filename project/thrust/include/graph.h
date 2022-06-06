#ifndef _GRAPH_H_
#define _GRAPH_H_

void graph_generate(unsigned int * adjacency_matrix, unsigned int num_nodes, float density, unsigned int  min_weight, unsigned int  max_weight);

unsigned int *graph_read(char* filename, unsigned int * number_of_nodes);

void graph_write(unsigned int num_nodes, unsigned int *adjacency_matrix, char* filename);

#endif