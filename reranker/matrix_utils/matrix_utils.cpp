#include "include/utils.h"


torch::Tensor rowwise_deepcopy(
    const torch::Tensor source_matrix,
    const torch::Tensor row_indexes
){
    CHECK_INPUT(source_matrix);
    CHECK_INPUT(row_indexes);

    return rowwise_deepcopy_cu(source_matrix, row_indexes);
}

torch::Tensor rowwise_multiply(
    const torch::Tensor source_matrix,
    const torch::Tensor source_vector
){
    CHECK_INPUT(source_matrix);
    CHECK_INPUT(source_vector);

    auto source_vector_unsqueezed = source_vector.unsqueeze(1);

    return torch::mul(source_matrix, source_vector_unsqueezed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("rowwise_deepcopy", &rowwise_deepcopy, "Row-wise copy of matrix");
    m.def("rowwise_multiply", &rowwise_multiply, "Row-wise multiply matrix by vector");
}