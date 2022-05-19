#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "impl.h"

clock_t calculate_connected_components_cpu(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components)
{

    unsigned int found_nodes[num_nodes];

    for (unsigned int i=0; i<num_nodes; i++)
    {
        connected_components[i] = ~0;
        found_nodes[i]=0;
    }

    clock_t start = clock();
    
    for (unsigned int i=0; i<num_nodes; i++)
    {
        if(connected_components[i] == ~0)
        {
            unsigned int found_nodes_cnt = 1;
            unsigned int current_index = 0;
            unsigned int last_index = 1;
            unsigned int lowest_node_id = i;

            // find following neighbours
            while (current_index < last_index)
            {
                // iterate over new found nodes
                for(unsigned int k=0; k < num_nodes; k++)
                {
                    // check if neighboruing relation exists 
                    if(adjacency_matrix[found_nodes[current_index]*num_nodes + k] !=0)
                    {
                        // check if neighbouring relation was already found
                        unsigned int node_already_found = 0;
                        for(unsigned int l=0; l < last_index; l++)
                        {
                            if(found_nodes[l] == k)
                            {
                                node_already_found = 1;
                                break;
                            }
                        }

                        if(node_already_found == 0)
                        {
                            found_nodes[last_index] =k;
                            last_index++;
                            if(k < lowest_node_id)
                            {
                                lowest_node_id = k;
                            }
                        }
                    }
                }        
                current_index++;
            }
            // set connected components ids
            for (unsigned int j=0; j < last_index; j++)
            {
                unsigned int node= found_nodes[j];
                connected_components[node] = lowest_node_id;
            }
        }
    }
    
    return clock() - start;
}
