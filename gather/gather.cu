#include <paddle/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#define CHECK_GPU_INPUT(x) \
  PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")


// template<typename data_t>
// __global__ void gather_cuda_forward_kernel(const data_t* x,
//                                          data_t* y,
//                                          int num){
//     int gid = blockIdx.x * blockDim.x + threadIdx.x;
//     for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
//         y[i] = std::tanh(x[i]);
//     }
// }


#define CUDA_KERNEL_LOOP(i, n)                            \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x, \
               step = blockDim.x * gridDim.x;             \
       i < (n); i += step)



template <typename T, typename IndexT = int>
__global__ void gather_cuda_forward_kernel(const T* params, const IndexT* indices,
                                 T* output, size_t index_size,
                                 size_t slice_size) {
  CUDA_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT gather_i = indices[indices_i];
    IndexT params_i = gather_i * slice_size + slice_i;
    *(output + i) = *(params + params_i);
  }
}




// template<typename data_t>
// __global__ void gather_cuda_backward_kernel(const data_t* x,
//                                           const data_t* dy,
//                                           data_t* dx,
//                                           int num){
//     int gid = blockIdx.x * blockDim.x + threadIdx.x;
//     for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
//         dx[i] = dy[i] * (1 - std::pow(std::tanh(x[i]), 2));
//     }
// }

// template<typename data_t>
// __global__ void gather_cuda_double_backward_kernel(const data_t* y,
//                                                  const data_t* dy,
//                                                  const data_t* ddx,
//                                                  data_t* ddy,
//                                                  data_t* dy_new,
//                                                  int num){
//     int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//     for (int64_t i = num; i < num; i += blockDim.x * gridDim.x) {
//         ddy[i] = ddx[i] * (1 - std::pow(y[i], 2));
//         dy_new[i] = ddx[i] * dy[i] * static_cast<data_t>(-2.) * y[i];
//     }
// }


template <typename T, typename IndexT = int>
std::vector<paddle::Tensor> gather_cuda_forward(const paddle::Tensor& input, const paddle::Tensor& index){
    std::vector<int64_t> input_shape = input.shape();
    std::vector<int64_t> index_shape = index.shape();
    std::vector<int64_t> output_shape;
    output_shape.push_back(index_shape[0]);
    for (int i = 1; i < input_shape.size(); i++) {
        output_shape.push_back(input_shape[i]);
    }
    int index_size = index_shape[0];

    auto output = paddle::Tensor(paddle::PlaceType::kGPU, output_shape);

    // slice size
    int slice_size = 1;
    for (int i = 1; i < input_shape.size(); ++i) {
        slice_size *= input_shape[i];
    }

//     T* p_output = output.data<T>();
//     T* p_output = output.mutable_data<T>(input.place());

    int block = 512;
    int n = slice_size * index_size;
    int grid = (n + block - 1) / block;


    PD_DISPATCH_FLOATING_TYPES(
        input.type(), "gather_cuda_forward_kernel", ([&] {
            gather_cuda_forward_kernel<T, IndexT><<<grid, block, 0, input.stream()>>>(
                input.data<T>(), index.data<IndexT>(), output.mutable_data<T>(input.place()), index_size, slice_size
            );
        })
    );

    return {output};
}

// std::vector<paddle::Tensor> gather_cuda_backward(const paddle::Tensor& x,
//                                                const paddle::Tensor& y,
//                                                const paddle::Tensor& dy){
//     auto dx = paddle::Tensor(paddle::PlaceType::kGPU, x.shape());
//
//     int numel = y.size();
//     int grid = (numel + BLOCK - 1) / BLOCK;
//
//     PD_DISPATCH_FLOATING_TYPES(
//         x.type(), "gather_cuda_backward_kernel", ([&] {
//             gather_cuda_backward_kernel<data_t><<<grid, BLOCK, 0, x.stream()>>>(
//                 x.data<data_t>(),
//                 dy.data<data_t>(),
//                 dx.mutable_data<data_t>(x.place()),
//                 numel
//             );
//         })
//     );
//
//     return {dx};
// }

// std::vector<paddle::Tensor> gather_cuda_double_backward(const paddle::Tensor& y,
//                                                       const paddle::Tensor& dy,
//                                                       const paddle::Tensor& ddx){
//     CHECK_GPU_INPUT(y);
//     CHECK_GPU_INPUT(dy);
//     CHECK_GPU_INPUT(ddx);
//     auto ddy = paddle::Tensor(paddle::PlaceType::kGPU, y.shape());
//     auto dy_new = paddle::Tensor(paddle::PlaceType::kGPU, y.shape());
//
//     int numel = y.size();
//     int grid = (numel + BLOCK - 1) / BLOCK;
//
//     PD_DISPATCH_FLOATING_TYPES(
//         y.type(), "gather_cuda_double_backward_kernel", ([&] {
//             gather_cuda_double_backward_kernel<data_t><<<grid, BLOCK, 0, y.stream()>>>(
//                 y.data<data_t>(),
//                 dy.data<data_t>(),
//                 ddx.data<data_t>(),
//                 ddy.mutable_data<data_t>(y.place()),
//                 dy_new.mutable_data<data_t>(y.place()),
//                 numel
//             );
//         })
//     );
//
//     return {ddy, dy_new};
// }




