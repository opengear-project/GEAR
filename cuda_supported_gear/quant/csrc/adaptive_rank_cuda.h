#pragma once
#include <torch/extension.h>

torch::Tensor get_adaptive_rank_cuda(
    torch::Tensor input,
    double energy_threshold,
    int max_rank);
