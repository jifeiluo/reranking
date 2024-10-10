#include <torch/extension.h>

template <typename scalar_t>
__global__ void rowwise_deepcopy_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> source_matrix,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> row_indexes,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> target_matrix
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int dimension = source_matrix.size(1);

    if (i>=row_indexes.size(0) || j>=dimension) return;
    target_matrix[i][j] = source_matrix[row_indexes[i]][j];
}


torch::Tensor rowwise_deepcopy_cu(
    const torch::Tensor source_matrix,
    const torch::Tensor row_indexes
){
    const int n1 = row_indexes.size(0), n2 = source_matrix.size(1);
    
    torch::Tensor target_matrix = torch::empty({n1, n2}, source_matrix.options());

    const dim3 threads(16, 16);
    const dim3 blocks((n1+threads.x-1)/threads.x, (n2+threads.y-1)/threads.y);

    rowwise_deepcopy_kernel<<<blocks, threads>>>(
        source_matrix.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        row_indexes.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
        target_matrix.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>()
    );

    return target_matrix;
}