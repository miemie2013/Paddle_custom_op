#include <paddle/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#define CHECK_GPU_INPUT(x) \
  PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")
#define BLOCK 512

template<typename data_t>
__global__ void tanh_cuda_forward_kernel(const data_t* x,
                                         data_t* y,
                                         int num){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
        y[i] = std::tanh(x[i]);
    }
}

template<typename data_t>
__global__ void tanh_cuda_backward_kernel(const data_t* x,
                                          const data_t* dy,
                                          data_t* dx,
                                          int num){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
        dx[i] = dy[i] * (1 - std::pow(std::tanh(x[i]), 2));
    }
}



std::vector<paddle::Tensor> tanh_cuda_forward(const paddle::Tensor& x){
    auto y = paddle::Tensor(paddle::PlaceType::kGPU, x.shape());

    int numel = x.size();
    int grid = (numel + BLOCK - 1) / BLOCK;

    PD_DISPATCH_FLOATING_TYPES(
        x.type(), "tanh_cuda_forward_kernel", ([&] {
            tanh_cuda_forward_kernel<data_t><<<grid, BLOCK, 0, x.stream()>>>(
                x.data<data_t>(),
                y.mutable_data<data_t>(x.place()),
                numel
            );
        })
    );

    return {y};
}

std::vector<paddle::Tensor> tanh_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& y,
                                               const paddle::Tensor& dy){
    auto dx = paddle::Tensor(paddle::PlaceType::kGPU, x.shape());

    int numel = y.size();
    int grid = (numel + BLOCK - 1) / BLOCK;

    PD_DISPATCH_FLOATING_TYPES(
        x.type(), "tanh_cuda_backward_kernel", ([&] {
            tanh_cuda_backward_kernel<data_t><<<grid, BLOCK, 0, x.stream()>>>(
                x.data<data_t>(),
                dy.data<data_t>(),
                dx.mutable_data<data_t>(x.place()),
                numel
            );
        })
    );

    return {dx};
}





