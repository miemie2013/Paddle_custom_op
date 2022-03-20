#include <paddle/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK 512

template<typename data_t>
__global__ void tanh_forward_cuda_kernel(const data_t* input_data,
                                    data_t* output_data,
                                    int input_numel){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=gid; i<input_numel; i+=blockDim.x*gridDim.x){
        output_data[i] = std::tanh(input_data[i]);
    }
}

template<typename data_t>
__global__ void tanh_backward_cuda_kernel(const data_t* input_data,
                                    const data_t* output_grad_data,
                                    data_t* input_grad_data,
                                    int output_numel){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=gid; i<output_numel; i+=blockDim.x*gridDim.x){
        input_grad_data[i] = output_grad_data[i] * (1 - std::pow(std::tanh(input_data[i]), 2));
    }
}

std::vector<paddle::Tensor> tanh_forward_cuda(const paddle::Tensor &input){
    auto output = paddle::Tensor(paddle::PlaceType::kGPU, input.shape());

    int input_numel = input.size();
    int grid = (input_numel + BLOCK - 1) / BLOCK;

    PD_DISPATCH_FLOATING_TYPES(
        input.type(), "tanh_forward_cuda_kernel", ([&] {
            tanh_forward_cuda_kernel<data_t><<<grid, BLOCK, 0, input.stream()>>>(
                input.data<data_t>(), 
                output.mutable_data<data_t>(input.place()), 
                input_numel
            );
        })
    );

    return {output};
}

std::vector<paddle::Tensor> tanh_backward_cuda(const paddle::Tensor &input,
                                               const paddle::Tensor &output,
                                               const paddle::Tensor &output_grad){
    auto input_grad = paddle::Tensor(paddle::PlaceType::kGPU, input.shape());

    int output_numel = output.size();
    int grid = (output_numel + BLOCK - 1) / BLOCK;

    PD_DISPATCH_FLOATING_TYPES(
        input.type(), "tanh_backward_cuda_kernel", ([&] {
            tanh_backward_cuda_kernel<data_t><<<grid, BLOCK, 0, input.stream()>>>(
                input.data<data_t>(), 
                output_grad.data<data_t>(), 
                input_grad.mutable_data<data_t>(input.place()), 
                output_numel
            );
        })
    );

    return {input_grad};
}