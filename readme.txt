
目前只有develop分支支持二阶导数：
python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html



cd ~/work/tanh


rm -rf *.cpp
rm -rf *.cu


rm -rf build
rm -rf custom_ops.egg-info
python setup.py install


rm -rf *


rm -rf *.npz
rm -rf *.pdparams


























