#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor compute_anchor_vector_cu(
    const torch::Tensor trans_matrix,
    const torch::Tensor confident_non_zero_start_num,
    const torch::Tensor confident_non_zero_end_num,
    const torch::Tensor confident_non_zero_all_index,
    int option=1
);

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
);