#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "impl.h"

clock_t calculate_connected_components_cpu(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int * connected_components)
{
    clock_t start = clock();

    for (unsigned int i=0; i<num_nodes; i++)
    {
        connected_components[i] = ~0;
    }

    for (unsigned int i=0; i<num_nodes; i++)
    {
        if(connected_components[i] == ~0)
        {
            unsigned int found_nodes_cnt = 1;
            unsigned int first_index = 0;
            unsigned int last_index = 1;
            unsigned int lowest_node_id = i;
            unsigned int found_nodes[num_nodes];

            for(unsigned int j=0; j < num_nodes; j++)
            {
                found_nodes[j]=0;
            }

            found_nodes[0]=i;
            // get inital node neightbours
            for(unsigned int j=0; j < num_nodes; j++)
            {
                if(adjacency_matrix[i*num_nodes + j] !=0)
                {
                    found_nodes[last_index] = j;
                    last_index++;
                    found_nodes_cnt++;
                    if(j < lowest_node_id)
                    {
                        lowest_node_id = j;
                    }
                }
            }
            
            // find following neighbours
            while (found_nodes_cnt > 0)
            {
                found_nodes_cnt =0;
                unsigned int new_first_index = last_index;
                unsigned int new_last_index = last_index;
            
                for(unsigned int j=first_index; j < last_index; j++)
                {
                    // iterate over new found nodes
                    for(unsigned int k=0; k < num_nodes; k++)
                    {
                        // check if neighboruing relation exists 
                        if(adjacency_matrix[found_nodes[j]*num_nodes + k] !=0)
                        {
                            // check if neighbouring relation was already found
                            unsigned int node_already_found = 0;
                            for(unsigned int l=0; l < new_last_index; l++)
                            {
                                //printf("l Node %d\n", found_nodes[l]);
                                if(found_nodes[l] == k)
                                {
                                    node_already_found = 1;
                                    break;
                                }
                            }

                            if(node_already_found == 0)
                            {
                                found_nodes[new_last_index] =k;
                                new_last_index++;
                                found_nodes_cnt++;
                                if(k < lowest_node_id)
                                {
                                    lowest_node_id = k;
                                }
                            }
                        }
                    }
                }

                first_index = new_first_index;
                last_index = new_last_index;
            
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
