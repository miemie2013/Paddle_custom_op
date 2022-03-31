#include <paddle/extension.h>
#include <vector>
#define PADDLE_WITH_CUDA
#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

// cuda实现声明
std::vector<paddle::Tensor> gather_cuda_forward(const paddle::Tensor& input, const paddle::Tensor& index);

std::vector<paddle::Tensor> gather_cuda_backward(const paddle::Tensor& input, const paddle::Tensor& index, const paddle::Tensor& doutput);

//std::vector<paddle::Tensor> gather_cuda_double_backward(const paddle::Tensor& y,
//                                                      const paddle::Tensor& dy,
//                                                      const paddle::Tensor& ddx);

// 决定调用cpu或者gpu实现。暂时只提供了gpu实现
std::vector<paddle::Tensor> GatherForward(const paddle::Tensor& input, const paddle::Tensor& index) {
  CHECK_INPUT(input);
  CHECK_INPUT(index);
  return gather_cuda_forward(input, index);
}

std::vector<paddle::Tensor> GatherBackward(const paddle::Tensor& input, const paddle::Tensor& index, const paddle::Tensor& doutput) {
  CHECK_INPUT(input);
  CHECK_INPUT(index);
  CHECK_INPUT(doutput);
  return gather_cuda_backward(input, index, doutput);
}

//std::vector<paddle::Tensor> GatherDoubleBackward(const paddle::Tensor& y,
//                                               const paddle::Tensor& dy,
//                                               const paddle::Tensor& ddx) {
//  CHECK_INPUT(y);
//  CHECK_INPUT(dy);
//  CHECK_INPUT(ddx);
//
//  return gather_cuda_double_backward(y, dy, ddx);
//}

// 形状推断函数
std::vector<std::vector<int64_t>> gather_forward_InferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& index_shape) {
    std::vector<int64_t> output_shape;
    output_shape.push_back(index_shape[0]);
    for (int i = 1; i < input_shape.size(); i++) {
        output_shape.push_back(input_shape[i]);
    }
    return {output_shape};
}

std::vector<std::vector<int64_t>> gather_backward_InferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& index_shape,
    const std::vector<int64_t>& doutput_shape) {
    return {input_shape};
}

//std::vector<std::vector<int64_t>> gather_double_backward_InferShape(
//    const std::vector<int64_t>& y_shape,
//    const std::vector<int64_t>& dy_shape,
//    const std::vector<int64_t>& ddx_shape) {
//  return {y_shape};
//}


// 类型推断函数
std::vector<paddle::DataType> gather_forward_InferDtype(paddle::DataType input_dtype, paddle::DataType index_dtype) {
    return {input_dtype};
}

std::vector<paddle::DataType> gather_backward_InferDtype(paddle::DataType input_dtype, paddle::DataType index_dtype, paddle::DataType doutput_dtype) {
    return {input_dtype};
}


PD_BUILD_OP(gather_op)
    .Inputs({"Input", "Index"})
    .Outputs({"Output"})
    .SetKernelFn(PD_KERNEL(GatherForward))
    .SetInferShapeFn(PD_INFER_SHAPE(gather_forward_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(gather_forward_InferDtype));

PD_BUILD_GRAD_OP(gather_op)
    .Inputs({"Input", "Index", paddle::Grad("Output")})
    .Outputs({paddle::Grad("Input")})
    .SetKernelFn(PD_KERNEL(GatherBackward));
//    .SetInferShapeFn(PD_INFER_SHAPE(gather_backward_InferShape))
//    .SetInferDtypeFn(PD_INFER_DTYPE(gather_backward_InferDtype));

//PD_BUILD_DOUBLE_GRAD_OP(gather_op)
//    .Inputs({"Y", paddle::Grad("Y"), paddle::Grad(paddle::Grad("X"))})
//    .Outputs({paddle::Grad(paddle::Grad("Y")), paddle::Grad("Y")})
//    .SetKernelFn(PD_KERNEL(GatherDoubleBackward))
//    .SetInferShapeFn(PD_INFER_SHAPE(gather_double_backward_InferShape));
