#include <paddle/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>


#define CUDA_KERNEL_LOOP(i, n)                            \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x, \
               step = blockDim.x * gridDim.x;             \
       i < (n); i += step)


// 暂时不支持定义2个泛型？
// template<typename data_t, typename index_t>
template<typename data_t>
__global__ void GatherCUDAKernel(const data_t* params, const int64_t* indices,
                                 data_t* output, size_t index_size,
                                 size_t slice_size) {
  CUDA_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    int64_t gather_i = indices[indices_i];
    int64_t params_i = gather_i * slice_size + slice_i;
    *(output + i) = *(params + params_i);
  }
}


template<typename data_t>
__global__ void ScatterInitCUDAKernel(const int64_t* indices, data_t* output,
                                      size_t index_size, size_t slice_size,
                                      bool overwrite) {
  CUDA_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    int64_t scatter_i = indices[indices_i];
    int64_t out_i = scatter_i * slice_size + slice_i;
    *(output + out_i) = static_cast<data_t>(0);
  }
}

template<typename data_t>
__global__ void ScatterCUDAKernel(const data_t* params, const int64_t* indices,
                                  data_t* output, size_t index_size,
                                  size_t slice_size, bool overwrite) {
  CUDA_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    int64_t scatter_i = indices[indices_i];
    int64_t out_i = scatter_i * slice_size + slice_i;
    if (overwrite) {
      *(output + out_i) = *(params + i);
    } else {
      // paddle::platform::CudaAtomicAdd(output + out_i, *(params + i));
      *(output + out_i) += *(params + i);
    }
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

    int block = 512;
    int n = slice_size * index_size;
    int grid = (n + block - 1) / block;

    PD_DISPATCH_FLOATING_TYPES(
        input.type(), "GatherCUDAKernel", ([&] {
            GatherCUDAKernel<data_t><<<grid, block, 0, input.stream()>>>(
                input.data<data_t>(),
                index.data<int64_t>(),
                output.mutable_data<data_t>(input.place()),
                index_size, slice_size
            );
        })
    );

    return {output};
}



template<typename data_t>
__global__ void fill_constant_kernel(data_t* x, int value,
                                         int num){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
        x[i] = static_cast<data_t>(value);
    }
}

std::vector<paddle::Tensor> fill_constant(paddle::Tensor& x, int value, const paddle::Tensor& shape_as_x){
    int block = 512;
    int numel = x.size();
    int grid = (numel + block - 1) / block;

    PD_DISPATCH_FLOATING_TYPES(
        shape_as_x.type(), "fill_constant_kernel", ([&] {
            fill_constant_kernel<data_t><<<grid, block, 0, shape_as_x.stream()>>>(
                x.mutable_data<data_t>(shape_as_x.place()), value,
                numel
            );
        })
    );

    return {x};
}

std::vector<paddle::Tensor> gather_cuda_backward(const paddle::Tensor& input, const paddle::Tensor& index, const paddle::Tensor& doutput){
    std::vector<int64_t> input_shape = input.shape();
    std::vector<int64_t> index_shape = index.shape();
    std::vector<int64_t> doutput_shape = doutput.shape();
    int index_size = index_shape[0];

    paddle::Tensor dinput = paddle::Tensor(paddle::PlaceType::kGPU, input_shape);
    // dinput初始化为全0。
    dinput = fill_constant(dinput, 0, input)[0];

    // slice size
    int slice_size = 1;
    for (int i = 1; i < doutput_shape.size(); ++i) {
        slice_size *= doutput_shape[i];
    }

    int block = 512;
    int n = slice_size * index_size;
    int grid = (n + block - 1) / block;

    // 为 true 时表示覆盖写；为 false 时表示累加。一定要是false。
    bool overwrite = false;

    // 累加模式，被填写的位置要初始化为全0。
    if (!overwrite) {
        PD_DISPATCH_FLOATING_TYPES(
            input.type(), "ScatterInitCUDAKernel", ([&] {
                ScatterInitCUDAKernel<data_t><<<grid, block, 0, input.stream()>>>(
                    index.data<int64_t>(),
                    dinput.mutable_data<data_t>(input.place()),
                    index_size, slice_size, overwrite
                );
            })
        );
    }

    PD_DISPATCH_FLOATING_TYPES(
        input.type(), "ScatterCUDAKernel", ([&] {
            ScatterCUDAKernel<data_t><<<grid, block, 0, input.stream()>>>(
                doutput.data<data_t>(),
                index.data<int64_t>(),
                dinput.mutable_data<data_t>(input.place()),
                index_size, slice_size, overwrite
            );
        })
    );

    return {dinput};
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




