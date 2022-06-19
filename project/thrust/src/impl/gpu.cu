#define CUB_IGNORE_DEPRECATED_CPP_DIALECT 1
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT 1

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <unistd.h>
#include <time.h>

//#define DEBUG 1


struct which_row : thrust::unary_function<unsigned int, unsigned int> {
    unsigned int row_length;

    __host__ __device__
    which_row(unsigned int row_length_) : row_length(row_length_) {}

    __host__ __device__
    unsigned int operator()(unsigned int idx) const {
        return (idx / row_length) + 1;
    }
};

struct which_column : thrust::unary_function<unsigned int, unsigned int> {
    unsigned int row_length;

    __host__ __device__
    which_column(unsigned int row_length_) : row_length(row_length_) {}

    __host__ __device__
    int operator()(unsigned int idx) const {
        return (idx % row_length) + 1;
    }
};

struct make_connection : thrust::unary_function<thrust::tuple<unsigned int, int>, unsigned int> {
    __host__ __device__
    unsigned int operator()(thrust::tuple<unsigned int, int> x) const {
        if (thrust::get<0>(x) == 0) return 0;
        return thrust::get<1>(x);
    }
};


struct apply_pos : public thrust::binary_function<unsigned int, unsigned int,unsigned int> {
    __host__ __device__
    unsigned int operator()(unsigned int val, unsigned int pos) const {
        if (val == 0) return 0;
        return pos;
    }
};

void calculate(unsigned int num_nodes, thrust::device_vector<unsigned int> adjacency_matrix, unsigned int* connected_components) {
    unsigned int num_entries = num_nodes*num_nodes;

    thrust::equal_to<unsigned int> eq;
    thrust::maximum<unsigned int> max;
    thrust::device_vector<unsigned int> output_keys(num_nodes);
    thrust::device_vector<unsigned int> output_values(num_nodes+1);
    thrust::device_vector<unsigned int> old_values(num_nodes+1);

    thrust::transform_iterator<which_column, thrust::counting_iterator<unsigned int>> columns(thrust::counting_iterator<unsigned int>(0), which_column(num_nodes));
    thrust::transform(adjacency_matrix.begin(), adjacency_matrix.begin()+num_entries, columns, adjacency_matrix.begin(), apply_pos());

    thrust::device_vector<unsigned int> tmp_values(num_entries);

    thrust::transform_iterator<which_row, thrust::counting_iterator<unsigned int>> rows(thrust::counting_iterator<unsigned int>(0), which_row(num_nodes));
    auto connections = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(adjacency_matrix.begin(), columns)), make_connection());

#ifdef DEBUG
    printf("\n");
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            printf("%d ", connections[i * num_nodes + j]);
        }
        printf("\n");
    }
#endif // DEBUG

    for (int n = 0; n < num_nodes*num_nodes; n++) {
#ifdef DEBUG
        printf("\n\n%d:\n\n", n);
#endif // DEBUG
        auto new_ends = thrust::reduce_by_key(
                rows, rows + num_entries,
                adjacency_matrix.begin(),
                output_keys.begin(),
                output_values.begin() + 1,
                eq, max
        );

        // in theory it should be sorted, but as far as I can tell it was always sorted from reduce_by_key, and not doing it is faster
        //thrust::sort_by_key(output_keys.begin(), thrust::get<0>(new_ends), output_values.begin() + 1);

#ifdef DEBUG
        for (int i = 0; i < num_nodes; i++) {
            printf("%d: %d\n", (unsigned int) output_keys[i], (unsigned int) output_values[i+1]);
        }
#endif // DEBUG

        auto mismatch = thrust::mismatch(output_values.begin(), thrust::get<1>(new_ends), old_values.begin());
        if (thrust::get<0>(mismatch) == thrust::get<1>(new_ends)) {
            thrust::host_vector<unsigned int> res(output_values.begin()+1, output_values.begin()+num_nodes);
            for (int i = 0; i < num_nodes; i++) {
                if(res[i] == 0) {
                    connected_components[i] = i;
                } else {
                    connected_components[i] = res[i]-1;
                }
            }
            return;
        }

        thrust::copy(output_values.begin(), thrust::get<1>(new_ends), old_values.begin());

        thrust::gather(
            connections, connections + num_entries,
            output_values.begin(),
            tmp_values.begin()
        );

        thrust::transform(
            adjacency_matrix.begin(), adjacency_matrix.begin() + num_entries,
            tmp_values.begin(),
            adjacency_matrix.begin(),
            max
        );

#ifdef DEBUG
        for (int i = 0; i < num_nodes; i++) {
            for (int j = 0; j < num_nodes; j++) {
                printf("%d ", (unsigned int) adjacency_matrix[i * num_nodes + j]);
            }
            printf("\n");
        }
#endif // DEBUG
    }

    printf("FAILED\n");
}

clock_t calculate_connected_components_gpu_thrust(unsigned int num_nodes, unsigned int *adjacency_matrix, unsigned int *connected_components) {
    thrust::host_vector<unsigned int> h_matrix(adjacency_matrix, adjacency_matrix + num_nodes*num_nodes);
    thrust::device_vector<unsigned int> d_matrix = h_matrix;

    clock_t start = clock();

    calculate(num_nodes, d_matrix, connected_components);

    return clock() - start;
}
