#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>


torch::Tensor lut_forward(torch::Tensor x,
                          torch::Tensor a1, torch::Tensor a2, torch::Tensor a3,
                          torch::Tensor a4, torch::Tensor a5, torch::Tensor a6,
                          torch::Tensor w,
                          torch::Tensor given_x_indices_of_y_start,
                          torch::Tensor given_x_indices_of_y);

torch::Tensor lut_backward_x(torch::Tensor x,
                             torch::Tensor a1, torch::Tensor a2, torch::Tensor a3,
                             torch::Tensor a4, torch::Tensor a5, torch::Tensor a6,
                             torch::Tensor w, torch::Tensor grad_out,
                             torch::Tensor given_x_indices_of_y_start,
                             torch::Tensor given_x_indices_of_y);

torch::Tensor lut_backward_w(torch::Tensor x,
                             torch::Tensor a1, torch::Tensor a2, torch::Tensor a3,
                             torch::Tensor a4, torch::Tensor a5, torch::Tensor a6,
                             torch::Tensor grad_out);
