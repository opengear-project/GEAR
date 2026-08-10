#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "adaptive_rank_cuda.h"
#include "gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("gemv_forward_cuda", &gemv_forward_cuda);
  m.def("gemv_forward_cuda_outer_dim", &gemv_forward_cuda_outer_dim);
  m.def("get_adaptive_rank_cuda", &get_adaptive_rank_cuda, py::arg("input"),
        py::arg("energy_threshold") = 0.5, py::arg("max_rank") = 16);
}