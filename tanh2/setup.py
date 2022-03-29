from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='custom_ops',
    ext_modules=CUDAExtension(
        sources=['tanh.cpp', 'tanh.cu']
    )
)