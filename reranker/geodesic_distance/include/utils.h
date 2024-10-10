#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor bellman_ford_cu(
    const int iter_num,
    const torch::Tensor adjacency_matrix
);

torch::Tensor square_bellman_ford_cu(
    const int iter_num,
    const torch::Tensor adjacency_matrix
);