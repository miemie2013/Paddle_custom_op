
目前只有develop分支支持二阶导数：
python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

Paddle中tanh二阶导的源码位于：
paddle/phi/kernels/funcs/activation_functor.h
TanhGradGradFunctor

Paddle中gather一阶导的源码位于：
paddle/phi/kernels/funcs/gather.h
GatherV2GradFunction



cd ~/work/tanh


rm -rf *.cpp
rm -rf *.cu


rm -rf build
rm -rf custom_ops.egg-info
python setup.py install
python test2_16_elementwise_grad_paddle_custom.py


rm -rf *


rm -rf *.npz
rm -rf *.pdparams


























