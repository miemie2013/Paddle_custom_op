#include <paddle/extension.h>
#include <vector>
#define PADDLE_WITH_CUDA
#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

std::vector<paddle::Tensor> tanh_forward_cuda(const paddle::Tensor &input);

std::vector<paddle::Tensor> tanh_backward_cuda(const paddle::Tensor &input,
                                               const paddle::Tensor &output,
                                               const paddle::Tensor &output_grad);

std::vector<paddle::Tensor> tanh_forward(const paddle::Tensor& input) {
  CHECK_INPUT(input);

  return tanh_forward_cuda(input);
}

std::vector<paddle::Tensor> tanh_backward(const paddle::Tensor& input,
                                          const paddle::Tensor& output,
                                          const paddle::Tensor& output_grad) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(output_grad);

  return tanh_backward_cuda(input, output, output_grad);
}

PD_BUILD_OP(tanh_op)
    .Inputs({"input"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(tanh_forward));

PD_BUILD_GRAD_OP(tanh_op)
    .Inputs({"input", "output", paddle::Grad("output")})
    .Outputs({paddle::Grad("input")})
    .SetKernelFn(PD_KERNEL(tanh_backward));
