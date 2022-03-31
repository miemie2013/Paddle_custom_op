
目前只有develop分支支持二阶导数：
python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

Paddle中tanh二阶导的源码位于：
paddle/phi/kernels/funcs/activation_functor.h
TanhGradGradFunctor

Paddle中gather一阶导的源码位于：
paddle/phi/kernels/funcs/gather.h
GatherV2GradFunction


可变形卷积的源码位于飞桨2.0版本的
paddle/fluid/operators/deformable_conv_op.cu


gather op的源码位于飞桨2.0版本的
paddle/fluid/operators/gather_op.cu
会跳转到paddle/fluid/operators/gather.cu.h的
GatherV2CUDAFunction()、GatherV2GradCUDAFunction()函数




cd ~/work/tanh


rm -rf *.cpp
rm -rf *.cu


python setup.py install如果报错
```
The following error occurred while trying to add or remove files in the
installation directory:

    [Errno 13] Permission denied: '/home/miemie2013/anaconda3/lib/python3.9/site-packages/test-easy-install-5206.write-test'

The installation directory you specified (via --install-dir, --prefix, or
the distutils default setting) was:

    /home/miemie2013/anaconda3/lib/python3.9/site-packages/

Perhaps your account does not have write access to this directory?
```
就输入
sudo chown -R $USER:$USER ~/anaconda3



rm -rf build
rm -rf custom_ops.egg-info
python setup.py install
python test2_16_elementwise_grad_paddle_custom.py


cat /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/custom_ops-0.0.0-py3.7-linux-x86_64.egg/custom_ops.py



rm -rf *


rm -rf *.npz
rm -rf *.pdparams


























