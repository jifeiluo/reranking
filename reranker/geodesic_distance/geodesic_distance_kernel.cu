#include <torch/extension.h>

#define TILE_WIDTH 16

template <typename scalar_t>
__global__ void bellman_ford_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> adjacency_matrix,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> register_matrix,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> distance_matrix
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int vertex_num = adjacency_matrix.size(0);

    if (i>=vertex_num || j>=vertex_num) return;
    float min_distance = register_matrix[i][j];
    for (int k=0; k<vertex_num; k++){
        min_distance = fminf(min_distance, register_matrix[i][k]+adjacency_matrix[k][j]);
    }
    distance_matrix[i][j] = min_distance;
}

torch::Tensor bellman_ford_cu(
    const int iter_num,
    const torch::Tensor adjacency_matrix
){
    const int vertex_num = adjacency_matrix.size(0);
    torch::Tensor distance_matrix = torch::clone(adjacency_matrix); // Initialize the shortest distance matrix with the original adjacency matrix
    torch::Tensor register_matrix = torch::clone(distance_matrix);  // Register the shortest distance matrix obtained from the previous step
    
    const dim3 threads(16, 16);
    const dim3 blocks((vertex_num+threads.x-1)/threads.x, (vertex_num+threads.y-1)/threads.y);

    for (int iter=0; iter<iter_num; iter++){
        AT_DISPATCH_FLOATING_TYPES(adjacency_matrix.type(), "bellman_ford_cu", 
        ([&] {
            bellman_ford_kernel<scalar_t><<<blocks, threads>>>(
                adjacency_matrix.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                register_matrix.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                distance_matrix.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
            );
        }));
        register_matrix = torch::clone(distance_matrix);
    }    

    return distance_matrix;
}


template <typename scalar_t>
__global__ void square_bellman_ford_base_kernel(
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> register_matrix,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> distance_matrix
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int vertex_num = register_matrix.size(0);

    // Global Memory
    if (i>=vertex_num || j>=vertex_num) return;
    float min_distance = register_matrix[i][j];
    for (int k=0; k<vertex_num; k++){
        min_distance = fminf(min_distance, register_matrix[i][k]+register_matrix[k][j]);
    }
    distance_matrix[i][j] = min_distance;
}

torch::Tensor square_bellman_ford_base_cu(
    const int iter_num,
    const torch::Tensor adjacency_matrix
){
    const int vertex_num = adjacency_matrix.size(0);
    torch::Tensor distance_matrix = torch::clone(adjacency_matrix); // Initialize the shortest distance matrix with the original adjacency matrix
    torch::Tensor register_matrix = torch::clone(adjacency_matrix);  // Register the shortest distance matrix obtained from the previous step
    
    const dim3 threads(16, 16);
    const dim3 blocks((vertex_num+threads.x-1)/threads.x, (vertex_num+threads.y-1)/threads.y);

    for (int iter=0; iter<iter_num; iter++){
        AT_DISPATCH_FLOATING_TYPES(adjacency_matrix.type(), "square_bellman_ford_base_cu", 
        ([&] {
            square_bellman_ford_base_kernel<scalar_t><<<blocks, threads>>>(
                register_matrix.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                distance_matrix.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
            );
        }));
        register_matrix = torch::clone(distance_matrix);
    }    

    return distance_matrix;
}