#include <algorithm>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "adaptive_rank_cuda.h"

constexpr int kAdaptiveRankMin = 1;

__global__ void compute_rank_from_eigenvalues_kernel(
    const float* __restrict__ eigenvalues,
    int* __restrict__ ranks,
    int batch_size,
    int k,
    float energy_threshold,
    int max_rank) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size) {
    return;
  }

  const float* evals = eigenvalues + static_cast<int64_t>(b) * k;

  float total = 0.f;
  for (int i = 0; i < k; ++i) {
    total += evals[i];
  }

  if (total <= 0.f) {
    ranks[b] = kAdaptiveRankMin;
    return;
  }

  float cumsum = 0.f;
  int rank = 0;
  for (int i = k - 1; i >= 0; --i) {
    cumsum += evals[i];
    float energy = cumsum / total;
    if (energy < energy_threshold) {
      ++rank;
    }
  }

  if (rank < kAdaptiveRankMin) {
    rank = kAdaptiveRankMin;
  }
  if (rank > max_rank) {
    rank = max_rank;
  }
  ranks[b] = rank;
}

namespace {

void check_cublas(cublasStatus_t status, const char* msg) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, msg);
}

void check_cusolver(cusolverStatus_t status, const char* msg) {
  TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS, msg);
}

void check_cuda(cudaError_t status, const char* msg) {
  TORCH_CHECK(status == cudaSuccess, msg, ": ", cudaGetErrorString(status));
}

cusolverDnHandle_t get_cusolver_handle() {
  static thread_local cusolverDnHandle_t handle = nullptr;
  if (handle == nullptr) {
    check_cusolver(cusolverDnCreate(&handle), "cusolverDnCreate failed");
  }
  check_cusolver(
      cusolverDnSetStream(handle, at::cuda::getCurrentCUDAStream()),
      "cusolverDnSetStream failed");
  return handle;
}

}  // namespace

torch::Tensor get_adaptive_rank_cuda(
    torch::Tensor input,
    double energy_threshold,
    int max_rank) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 4, "input must have shape [B, nh, seq_len, head_dim]");
  TORCH_CHECK(energy_threshold > 0.0 && energy_threshold < 1.0,
              "energy_threshold must be in (0, 1)");
  TORCH_CHECK(max_rank >= kAdaptiveRankMin, "max_rank must be >= ", kAdaptiveRankMin);

  c10::cuda::CUDAGuard device_guard(input.device());

  torch::Tensor input_f = input.scalar_type() == at::kHalf
      ? input.to(at::kFloat)
      : input.contiguous();
  if (input.scalar_type() != at::kHalf) {
    input_f = input_f.contiguous();
  }

  const int64_t batch_size = input_f.size(0);
  const int64_t num_head = input_f.size(1);
  const int64_t seq_len = input_f.size(2);
  const int64_t head_dim = input_f.size(3);
  const int64_t feature_dim = num_head * head_dim;

  // Combined-head layout: [B, seq_len, num_head * head_dim]
  torch::Tensor matrix =
      input_f.permute({0, 2, 1, 3}).contiguous().view({batch_size, seq_len, feature_dim});

  const int64_t m = seq_len;
  const int64_t n = feature_dim;
  const int64_t k = std::min(m, n);
  const bool use_aat = (m <= n);

  torch::Tensor gram = torch::empty({batch_size, k, k}, input_f.options());
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int64_t stride_a = m * n;
  const int64_t stride_g = k * k;
  float* a_ptr = matrix.data_ptr<float>();
  float* g_ptr = gram.data_ptr<float>();

  if (use_aat) {
    check_cublas(
        cublasSgemmStridedBatched(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            static_cast<int>(m),
            static_cast<int>(m),
            static_cast<int>(n),
            &alpha,
            a_ptr,
            static_cast<int>(m),
            stride_a,
            a_ptr,
            static_cast<int>(m),
            stride_a,
            &beta,
            g_ptr,
            static_cast<int>(m),
            stride_g,
            static_cast<int>(batch_size)),
        "cublasSgemmStridedBatched failed for A @ A^T");
  } else {
    check_cublas(
        cublasSgemmStridedBatched(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(n),
            static_cast<int>(m),
            &alpha,
            a_ptr,
            static_cast<int>(m),
            stride_a,
            a_ptr,
            static_cast<int>(m),
            stride_a,
            &beta,
            g_ptr,
            static_cast<int>(n),
            stride_g,
            static_cast<int>(batch_size)),
        "cublasSgemmStridedBatched failed for A^T @ A");
  }

  cusolverDnHandle_t solver_handle = get_cusolver_handle();
  torch::Tensor eigenvalues = torch::empty({batch_size, k}, input_f.options());

  int lwork = 0;
  check_cusolver(
      cusolverDnSsyevd_bufferSize(
          solver_handle,
          CUSOLVER_EIG_MODE_NOVECTOR,
          CUBLAS_FILL_MODE_UPPER,
          static_cast<int>(k),
          g_ptr,
          static_cast<int>(k),
          eigenvalues.data_ptr<float>(),
          &lwork),
      "cusolverDnSsyevd_bufferSize failed");

  TORCH_CHECK(lwork > 0, "invalid syev workspace size");
  torch::Tensor workspace =
      torch::empty({lwork}, torch::dtype(torch::kFloat32).device(input.device()));
  torch::Tensor dev_info =
      torch::empty({batch_size}, torch::dtype(torch::kInt32).device(input.device()));

  for (int64_t b = 0; b < batch_size; ++b) {
    float* gram_b = g_ptr + b * stride_g;
    float* evals_b = eigenvalues.data_ptr<float>() + b * k;
    int* info_b = dev_info.data_ptr<int>() + b;

    check_cusolver(
        cusolverDnSsyevd(
            solver_handle,
            CUSOLVER_EIG_MODE_NOVECTOR,
            CUBLAS_FILL_MODE_UPPER,
            static_cast<int>(k),
            gram_b,
            static_cast<int>(k),
            evals_b,
            workspace.data_ptr<float>(),
            lwork,
            info_b),
        "cusolverDnSsyevd failed");
  }

  torch::Tensor ranks =
      torch::empty({batch_size}, torch::dtype(torch::kInt32).device(input.device()));

  const int threads = 256;
  const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
  compute_rank_from_eigenvalues_kernel<<<blocks, threads>>>(
      eigenvalues.data_ptr<float>(),
      ranks.data_ptr<int>(),
      static_cast<int>(batch_size),
      static_cast<int>(k),
      static_cast<float>(energy_threshold),
      max_rank);
  check_cuda(cudaGetLastError(), "compute_rank_from_eigenvalues_kernel launch failed");

  return ranks;
}
