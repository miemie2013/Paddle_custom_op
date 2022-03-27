from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='custom_sum',
    ext_modules=CUDAExtension(
        sources=['sum.cpp', 'sum.cu']
    )
)