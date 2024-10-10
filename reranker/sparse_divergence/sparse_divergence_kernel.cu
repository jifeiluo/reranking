#include <torch/extension.h>

#define TILE_WIDTH 16

template <typename scalar_t>
__global__ void jensen_shannon_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sparse_feats_1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sparse_feats_2,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> divergence
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int dimension = sparse_feats_1.size(1);

    // Shared Memory
    __shared__ scalar_t shared_tile_1[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t shared_tile_2[TILE_WIDTH][TILE_WIDTH];
    int numStrides = (dimension + TILE_WIDTH - 1) / TILE_WIDTH;

    scalar_t shared_div = 0;
    for (int stride = 0; stride < numStrides; stride++) {
        
        int tile_1_idx = TILE_WIDTH*stride + threadIdx.y;
        int tile_2_idx = TILE_WIDTH*stride + threadIdx.x;
        if (i<sparse_feats_1.size(0) && tile_1_idx<dimension){
            shared_tile_1[threadIdx.x][threadIdx.y] = sparse_feats_1[i][tile_1_idx];
        }
        else{
            shared_tile_1[threadIdx.x][threadIdx.y] = 0;
        }
        if (j<sparse_feats_2.size(0) && tile_2_idx<dimension){
            shared_tile_2[threadIdx.y][threadIdx.x] = sparse_feats_2[j][tile_2_idx];
        }
        else{
            shared_tile_2[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();


        for(int k = 0; k < TILE_WIDTH; k++) {
            if (shared_tile_1[threadIdx.x][k]>0){
                shared_div += shared_tile_1[threadIdx.x][k] * log2f(2*shared_tile_1[threadIdx.x][k]/(shared_tile_1[threadIdx.x][k]+shared_tile_2[threadIdx.y][k]));
            }
            if (shared_tile_2[threadIdx.y][k]>0){
                shared_div += shared_tile_2[threadIdx.y][k] * log2f(2*shared_tile_2[threadIdx.y][k]/(shared_tile_2[threadIdx.y][k]+shared_tile_1[threadIdx.x][k]));
            }
        }
        __syncthreads();
    }

    if (i<sparse_feats_1.size(0) && j<sparse_feats_2.size(0)){
        divergence[i][j] = shared_div;
    }
}


torch::Tensor jensen_shannon_cu(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
){
    const int n1 = sparse_feats_1.size(0), n2 = sparse_feats_2.size(0);
    
    torch::Tensor divergence = torch::empty({n1, n2}, sparse_feats_1.options());

    const dim3 threads(16, 16);
    const dim3 blocks((n1+threads.x-1)/threads.x, (n2+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(sparse_feats_1.type(), "jensen_shannon_cu", 
    ([&] {
        jensen_shannon_kernel<scalar_t><<<blocks, threads>>>(
            sparse_feats_1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sparse_feats_2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            divergence.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    divergence = 0.5 * divergence;

    return divergence;
}


template <typename scalar_t>
__global__ void jensen_shannon_base_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sparse_feats_1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sparse_feats_2,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> divergence
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int dimension = sparse_feats_1.size(1);

    // Global Memory
    if (i>=sparse_feats_1.size(0) || j>=sparse_feats_2.size(0)) return;
    divergence[i][j] = 0;
    for (int k=0; k<dimension; k++){
        if (sparse_feats_1[i][k]>0){
            divergence[i][j] += sparse_feats_1[i][k] * log2f(2*sparse_feats_1[i][k]/(sparse_feats_1[i][k]+sparse_feats_2[j][k]));
        }
        if (sparse_feats_2[j][k]>0){
            divergence[i][j] += sparse_feats_2[j][k] * log2f(2*sparse_feats_2[j][k]/(sparse_feats_2[j][k]+sparse_feats_1[i][k]));
        }
    }
}


torch::Tensor jensen_shannon_base_cu(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
){
    const int n1 = sparse_feats_1.size(0), n2 = sparse_feats_2.size(0);
    
    torch::Tensor divergence = torch::empty({n1, n2}, sparse_feats_1.options());

    const dim3 threads(16, 16);
    const dim3 blocks((n1+threads.x-1)/threads.x, (n2+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(sparse_feats_1.type(), "jensen_shannon_base_cu", 
    ([&] {
        jensen_shannon_base_kernel<scalar_t><<<blocks, threads>>>(
            sparse_feats_1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sparse_feats_2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            divergence.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    divergence = 0.5 * divergence;

    return divergence;
}


template <typename scalar_t>
__global__ void jaccard_base_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sparse_feats_1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sparse_feats_2,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> divergence
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int dimension = sparse_feats_1.size(1);

    // Global Memory
    if (i>=sparse_feats_1.size(0) || j>=sparse_feats_2.size(0)) return;
    float intersection_min = 0;
    for (int k=0; k<dimension; k++){
        intersection_min += fminf(sparse_feats_1[i][k], sparse_feats_2[j][k]);
    }
    divergence[i][j] = 1.0 - intersection_min/(2.0-intersection_min);
}


torch::Tensor jaccard_base_cu(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
){
    const int n1 = sparse_feats_1.size(0), n2 = sparse_feats_2.size(0);
    
    torch::Tensor divergence = torch::empty({n1, n2}, sparse_feats_1.options());

    const dim3 threads(16, 16);
    const dim3 blocks((n1+threads.x-1)/threads.x, (n2+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(sparse_feats_1.type(), "jaccard_base_cu", 
    ([&] {
        jaccard_base_kernel<scalar_t><<<blocks, threads>>>(
            sparse_feats_1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sparse_feats_2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            divergence.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return divergence;
}


template <typename scalar_t>
__global__ void intersection_base_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sparse_feats_1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sparse_feats_2,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> divergence
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int dimension = sparse_feats_1.size(1);

    // Global Memory
    if (i>=sparse_feats_1.size(0) || j>=sparse_feats_2.size(0)) return;
    float intersection_min = 0;
    for (int k=0; k<dimension; k++){
        intersection_min += fminf(sparse_feats_1[i][k], sparse_feats_2[j][k]);
    }
    divergence[i][j] = 1 - intersection_min;
}


torch::Tensor intersection_base_cu(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
){
    const int n1 = sparse_feats_1.size(0), n2 = sparse_feats_2.size(0);
    
    torch::Tensor divergence = torch::empty({n1, n2}, sparse_feats_1.options());

    const dim3 threads(16, 16);
    const dim3 blocks((n1+threads.x-1)/threads.x, (n2+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(sparse_feats_1.type(), "intersection_base_cu", 
    ([&] {
        intersection_base_kernel<scalar_t><<<blocks, threads>>>(
            sparse_feats_1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sparse_feats_2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            divergence.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return divergence;
}