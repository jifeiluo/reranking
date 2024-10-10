#include <torch/extension.h>

template <typename scalar_t>
__global__ void compute_anchor_vector_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> trans_matrix,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> confident_non_zero_start_num,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> confident_non_zero_end_num,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> confident_non_zero_all_index,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> anchor_vector,
    int option
){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int all_num = trans_matrix.size(0);

    if (idx>=all_num) return;
    if (confident_non_zero_end_num[idx]-confident_non_zero_start_num[idx]>1){
        anchor_vector[idx] = 0.0;
        if (option==1){
            for (int i=confident_non_zero_start_num[idx]; i<confident_non_zero_end_num[idx]; i++){
                anchor_vector[idx] += trans_matrix[confident_non_zero_all_index[i]][confident_non_zero_all_index[i]];
            }
            anchor_vector[idx] = anchor_vector[idx]/(confident_non_zero_end_num[idx]-confident_non_zero_start_num[idx]);
        }
        else{
            anchor_vector[idx] = 1.0;
        }
    }
    else{
        anchor_vector[idx] = 1.0;
    }
}

torch::Tensor compute_anchor_vector_cu(
    const torch::Tensor trans_matrix,
    const torch::Tensor confident_non_zero_start_num,
    const torch::Tensor confident_non_zero_end_num,
    const torch::Tensor confident_non_zero_all_index,
    int option=1
){
    const int all_num = trans_matrix.size(0);
    torch::Tensor anchor_vector = torch::empty({all_num}, trans_matrix.options());

    const dim3 threads(256);
    const dim3 blocks((all_num+threads.x-1)/threads.x);

    compute_anchor_vector_kernel<<<blocks, threads>>>(
        trans_matrix.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        confident_non_zero_start_num.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
        confident_non_zero_end_num.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
        confident_non_zero_all_index.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
        anchor_vector.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>(),
        option
    );

    return anchor_vector;
}


template <typename scalar_t>
__global__ void compute_primary_transition_weight_matrix_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> trans_matrix,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> confident_non_zero_start_num,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> confident_non_zero_end_num,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> confident_non_zero_all_index,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> semi_candidate_non_zero_start_num,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> semi_candidate_non_zero_end_num,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> transition_weight_matrix
){
    const int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    const int all_num = trans_matrix.size(0);

    if (idx1>=all_num || idx2>=all_num) return;
    if ((semi_candidate_non_zero_end_num[idx1]-semi_candidate_non_zero_start_num[idx1]>1)&&(confident_non_zero_end_num[idx1]-confident_non_zero_start_num[idx1]>1)){
        float transition_weight = 0.0;
        for (int i=confident_non_zero_start_num[idx1]; i<confident_non_zero_end_num[idx1]; i++){
            transition_weight += trans_matrix[idx2][confident_non_zero_all_index[i]];
        }
        transition_weight_matrix[idx1][idx2] = transition_weight/(confident_non_zero_end_num[idx1]-confident_non_zero_start_num[idx1]);
    }
}

torch::Tensor compute_transition_weight_matrix_cu(
    const torch::Tensor trans_matrix,
    const torch::Tensor confident_non_zero_start_num,
    const torch::Tensor confident_non_zero_end_num,
    const torch::Tensor confident_non_zero_all_index,
    const torch::Tensor semi_candidate_non_zero_start_num,
    const torch::Tensor semi_candidate_non_zero_end_num,
    const torch::Tensor confident_matrix,
    const torch::Tensor semi_candidate_matrix,
    const torch::Tensor anchor_vector
){
    const int n1 = trans_matrix.size(0), n2 = trans_matrix.size(1);
    torch::Tensor transition_weight_matrix = torch::ones({n1, n2}, trans_matrix.options());

    const dim3 threads(16, 16);
    const dim3 blocks((n1+threads.x-1)/threads.x, (n2+threads.y-1)/threads.y);

    compute_primary_transition_weight_matrix_kernel<<<blocks, threads>>>(
        trans_matrix.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        confident_non_zero_start_num.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
        confident_non_zero_end_num.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
        confident_non_zero_all_index.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
        semi_candidate_non_zero_start_num.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
        semi_candidate_non_zero_end_num.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
        transition_weight_matrix.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>()
    );
    auto anchor_vector_unsqueezed = anchor_vector.unsqueeze(1);
    transition_weight_matrix = torch::mul(transition_weight_matrix, semi_candidate_matrix) + torch::mul(confident_matrix, anchor_vector_unsqueezed);
    transition_weight_matrix = torch::min(transition_weight_matrix, anchor_vector_unsqueezed);

    return transition_weight_matrix;
}