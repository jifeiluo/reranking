#include "include/utils.h"


torch::Tensor jensen_shannon(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
){
    CHECK_INPUT(sparse_feats_1);
    CHECK_INPUT(sparse_feats_2);

    return jensen_shannon_cu(sparse_feats_1, sparse_feats_2);
}

torch::Tensor jensen_shannon_base(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
){
    CHECK_INPUT(sparse_feats_1);
    CHECK_INPUT(sparse_feats_2);

    return jensen_shannon_base_cu(sparse_feats_1, sparse_feats_2);
}

torch::Tensor jaccard_base(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
){
    CHECK_INPUT(sparse_feats_1);
    CHECK_INPUT(sparse_feats_2);

    return jaccard_base_cu(sparse_feats_1, sparse_feats_2);
}

torch::Tensor intersection_base(
    const torch::Tensor sparse_feats_1,
    const torch::Tensor sparse_feats_2
){
    CHECK_INPUT(sparse_feats_1);
    CHECK_INPUT(sparse_feats_2);

    return intersection_base_cu(sparse_feats_1, sparse_feats_2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("jensen_shannon", &jensen_shannon);
    m.def("jensen_shannon_base", &jensen_shannon_base);
    m.def("jaccard_base", &jaccard_base);
    m.def("intersection_base", &intersection_base);
}