#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor jensen_shannon_cu(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
);

torch::Tensor jensen_shannon_base_cu(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
);

torch::Tensor jaccard_base_cu(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
);

torch::Tensor intersection_base_cu(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
);