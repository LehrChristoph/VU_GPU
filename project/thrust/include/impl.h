#ifndef _IMPL_H_
#define _IMPL_H_

#include <time.h>

// CPU version
clock_t calculate_connected_components_cpu(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int *connected_components);
clock_t calculate_connected_components_gpu_thrust(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int *connected_components);

#endif