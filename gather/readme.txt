
目前只有develop分支支持二阶导数：
python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html


gather op的源码位于飞桨2.0版本的
paddle/fluid/operators/gather_op.cu
会跳转到paddle/fluid/operators/gather.cu.h的
GPUGather()、
paddle/fluid/operators/scatter.cu.h的
GPUScatterAssign()函数。


=====================================================
paddle::Tensor& x   的常见属性：


int numel = x.size();    // x的元素个数
data_t* x_data = x.data<data_t>();    // 返回x里的数据的指针
int input_size = x.numel();   // 返回x里的元素个数
auto input_dim = x.dims();    // 返回一个int数组，表示x的形状。比如若x的形状是[2, 512]，返回数组[2, 512]。相当于python里x.shape


auto ddy = paddle::Tensor(paddle::PlaceType::kGPU, y.shape());   // 创建一个GPU上的新的张量，形状是y.shape()
data_t* ddy_data = ddy.mutable_data<data_t>(y.place());    // 返回ddy里的数据的指针，可写



=====================================================






cd ~/work/gather




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




rm -rf *.cpp
rm -rf *.cu
rm -rf build
rm -rf custom_*.egg-info


rm -rf *.cpp && rm -rf *.cu && rm -rf build && rm -rf custom_*.egg-info


python setup.py install


python test2_16_elementwise_grad_paddle_custom.py


cat /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/custom_ops-0.0.0-py3.7-linux-x86_64.egg/custom_ops.py



rm -rf *


rm -rf *.npz
rm -rf *.pdparams


























