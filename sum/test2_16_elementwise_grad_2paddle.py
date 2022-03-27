import paddle
import numpy as np
from tanh.networks_paddle import FullyConnectedLayer
import os
import torch


w_dim = 512
in_channels = 256
activation = 'linear'
batch_size = 2
lr = 0.1


model = FullyConnectedLayer(w_dim, in_channels, activation=activation, bias_init=1)
model.train()

use_gpu = True
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

def copy(name, w, std):
    value2 = paddle.to_tensor(w, place=place)
    value = std[name]
    value = value * 0 + value2
    std[name] = value

model_std = model.state_dict()

ckpt_file = '16.pth'
save_name = '16.pdparams'
state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))


model_dic = {}
for key, value in state_dict.items():
    model_dic[key] = value.data.numpy()

for key in model_dic.keys():
    name2 = key
    w = model_dic[key]
    if '.linear.weight' in key:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    if '.noise_strength' in key:
        print()
        w = np.reshape(w, [1, ])
    print(key)
    copy(name2, w, model_std)
model.set_state_dict(model_std)

paddle.save(model_std, save_name)

