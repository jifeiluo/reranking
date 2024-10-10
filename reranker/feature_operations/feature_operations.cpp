#include "include/utils.h"


torch::Tensor compute_anchor_vector(
    const torch::Tensor trans_matrix,
    const torch::Tensor confident_non_zero_start_num,
    const torch::Tensor confident_non_zero_end_num,
    const torch::Tensor confident_non_zero_all_index,
    int option=1
){
    CHECK_INPUT(trans_matrix);
    CHECK_INPUT(confident_non_zero_start_num);
    CHECK_INPUT(confident_non_zero_end_num);
    CHECK_INPUT(confident_non_zero_all_index);

    return compute_anchor_vector_cu(trans_matrix, 
                                    confident_non_zero_start_num, 
                                    confident_non_zero_end_num, 
                                    confident_non_zero_all_index,
                                    option);
}


torch::Tensor compute_transition_weight_matrix(
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
    CHECK_INPUT(trans_matrix);
    CHECK_INPUT(confident_non_zero_start_num);
    CHECK_INPUT(confident_non_zero_end_num);
    CHECK_INPUT(confident_non_zero_all_index);
    CHECK_INPUT(semi_candidate_non_zero_start_num);
    CHECK_INPUT(semi_candidate_non_zero_end_num);
    CHECK_INPUT(confident_matrix);
    CHECK_INPUT(semi_candidate_matrix);
    CHECK_INPUT(anchor_vector);

    return compute_transition_weight_matrix_cu(trans_matrix,
                                               confident_non_zero_start_num,
                                               confident_non_zero_end_num,
                                               confident_non_zero_all_index,
                                               semi_candidate_non_zero_start_num,
                                               semi_candidate_non_zero_end_num,
                                               confident_matrix,
                                               semi_candidate_matrix,
                                               anchor_vector);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("compute_anchor_vector", &compute_anchor_vector);
    m.def("compute_transition_weight_matrix", &compute_transition_weight_matrix);
}