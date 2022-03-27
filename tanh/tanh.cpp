#include <paddle/extension.h>
#include <vector>
#define PADDLE_WITH_CUDA
#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

// cuda实现声明
std::vector<paddle::Tensor> tanh_cuda_forward(const paddle::Tensor &x);

std::vector<paddle::Tensor> tanh_cuda_backward(const paddle::Tensor &x,
                                               const paddle::Tensor &y,
                                               const paddle::Tensor &dy);

std::vector<paddle::Tensor> tanh_cuda_double_backward(const paddle::Tensor &y,
                                                      // const paddle::Tensor &dy,
                                                      const paddle::Tensor &ddx);

// 决定调用cpu或者gpu实现。暂时只提供了gpu实现
std::vector<paddle::Tensor> TanhForward(const paddle::Tensor& x) {
  CHECK_INPUT(x);

  return tanh_cuda_forward(x);
}

std::vector<paddle::Tensor> TanhBackward(const paddle::Tensor& x,
                                         const paddle::Tensor& y,
                                         const paddle::Tensor& dy) {
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  CHECK_INPUT(dy);

  return tanh_cuda_backward(x, y, dy);
}

std::vector<paddle::Tensor> TanhDoubleBackward(const paddle::Tensor& y,
                                               // const paddle::Tensor& dy,
                                               const paddle::Tensor& ddx) {
  CHECK_INPUT(y);
  // CHECK_INPUT(dy);
  CHECK_INPUT(ddx);

//  return tanh_cuda_double_backward(y, dy, ddx);
  return tanh_cuda_double_backward(y, ddx);
}

// 形状推断函数
std::vector<std::vector<int64_t>> tanh_forward_InferShape(
    const std::vector<int64_t>& x_shape) {
  return {x_shape};
}

std::vector<std::vector<int64_t>> tanh_backward_InferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& dy_shape) {
  return {x_shape};
}

std::vector<std::vector<int64_t>> tanh_double_backward_InferShape(
    const std::vector<int64_t>& y_shape,
    // const std::vector<int64_t>& dy_shape,
    const std::vector<int64_t>& ddx_shape) {
  return {y_shape};
}

PD_BUILD_OP(tanh_op)
    .Inputs({"X"})
    .Outputs({"Y"})
    .SetKernelFn(PD_KERNEL(TanhForward))
    .SetInferShapeFn(PD_INFER_SHAPE(tanh_forward_InferShape));

PD_BUILD_GRAD_OP(tanh_op)
    .Inputs({"X", "Y", paddle::Grad("Y")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(TanhBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(tanh_backward_InferShape));

PD_BUILD_DOUBLE_GRAD_OP(tanh_op)
//    .Inputs({"Y", paddle::Grad("Y"), paddle::Grad(paddle::Grad("X"))})
    .Inputs({"Y", paddle::Grad(paddle::Grad("X"))})
    .Outputs({paddle::Grad(paddle::Grad("Y"))})
    .SetKernelFn(PD_KERNEL(TanhDoubleBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(tanh_double_backward_InferShape));
