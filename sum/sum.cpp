#include <paddle/extension.h>
#include <vector>
#define PADDLE_WITH_CUDA
#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

// cuda实现声明
std::vector<paddle::Tensor> sum_cuda_forward(const paddle::Tensor& x, const int64_t& axis);

std::vector<paddle::Tensor> sum_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& dy);

std::vector<paddle::Tensor> sum_cuda_double_backward(const paddle::Tensor& y,
                                                      const paddle::Tensor& ddx);

// 决定调用cpu或者gpu实现。暂时只提供了gpu实现
std::vector<paddle::Tensor> SumForward(const paddle::Tensor& x, const int64_t& axis) {
  CHECK_INPUT(x);

  return sum_cuda_forward(x, axis);
}

std::vector<paddle::Tensor> SumBackward(const paddle::Tensor& x,
                                         const paddle::Tensor& dy) {
  CHECK_INPUT(x);
  CHECK_INPUT(dy);

  return sum_cuda_backward(x, dy);
}

std::vector<paddle::Tensor> SumDoubleBackward(const paddle::Tensor& y,
                                               const paddle::Tensor& ddx) {
  CHECK_INPUT(y);
  CHECK_INPUT(ddx);

  return sum_cuda_double_backward(y, ddx);
}

// 形状推断函数
std::vector<std::vector<int64_t>> sum_forward_InferShape(
    const std::vector<int64_t>& x_shape,
    const int64_t& axis) {
  x_shape[axis] = 1;
  return {x_shape};
}

std::vector<std::vector<int64_t>> sum_backward_InferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& dy_shape) {
  return {x_shape};
}

std::vector<std::vector<int64_t>> sum_double_backward_InferShape(
    const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& ddx_shape) {
  return {y_shape};
}

PD_BUILD_OP(sum_op)
    .Inputs({"X"})
    .Outputs({"Y"})
    .Attrs({"axis: int64_t"})
    .SetKernelFn(PD_KERNEL(SumForward))
    .SetInferShapeFn(PD_INFER_SHAPE(sum_forward_InferShape));

PD_BUILD_GRAD_OP(sum_op)
    .Inputs({"X", paddle::Grad("Y")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(SumBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(sum_backward_InferShape));

PD_BUILD_DOUBLE_GRAD_OP(sum_op)
    .Inputs({"Y", paddle::Grad(paddle::Grad("X"))})
    .Outputs({paddle::Grad(paddle::Grad("Y"))})
    .SetKernelFn(PD_KERNEL(SumDoubleBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(sum_double_backward_InferShape));
