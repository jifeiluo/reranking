#include "include/utils.h"


torch::Tensor bellman_ford(
    const int iter_num,
    const torch::Tensor adjacency_matrix
){
    CHECK_INPUT(adjacency_matrix);

    return bellman_ford_cu(iter_num, adjacency_matrix);
}

torch::Tensor square_bellman_ford(
    const int iter_num,
    const torch::Tensor adjacency_matrix
){
    CHECK_INPUT(adjacency_matrix);

    return square_bellman_ford_cu(iter_num, adjacency_matrix);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("bellman_ford", &bellman_ford);
    m.def("square_bellman_ford", &square_bellman_ford);
}