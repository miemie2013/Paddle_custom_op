import torch
import numpy as np
from tanh.networks_pytorch import FullyConnectedLayer



w_dim = 512
in_channels = 256
activation = 'linear'
batch_size = 2
lr = 0.1


model = FullyConnectedLayer(w_dim, in_channels, activation=activation, bias_init=1)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
model.load_state_dict(torch.load("16.pth", map_location="cpu"))



class MyGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index):
        y = x[index]
        ctx.save_for_backward(x, index)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, index = ctx.saved_tensors
        dx = MyGatherGrad.apply(x, dy, index)
        dindex = None
        return dx, dindex

class MyGatherGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dy, index):
        dx = torch.zeros_like(x)
        dx = dx.scatter_add_(0, index=index.unsqueeze(1), src=dy)
        # dx[index] = dy
        ctx.save_for_backward(index)
        return dx

    @staticmethod
    def backward(ctx, ddx):
        index, = ctx.saved_tensors
        dy_new = ddx[index]
        return None, dy_new, None



# https://blog.csdn.net/peng_pi/article/details/123413701
src = torch.arange(1, 11).reshape((2, 5))
index = torch.tensor([[0, 1, 2, 0]])
aaaaaaaa = torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
index = torch.tensor([[0, 1, 1, 0]])
aaaaaaaa2 = torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)


dic2 = np.load('16.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.zero_grad(set_to_none=True)

    dloss_dx_pytorch = dic2['batch_%.3d.dloss_dx'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    index = dic2['batch_%.3d.index'%batch_idx]
    x = torch.Tensor(x)
    x.requires_grad_(True)
    index = torch.Tensor(index).to(torch.int64)
    index.requires_grad_(False)

    y = model(x)
    y = MyGather.apply(y, index)
    loss = torch.tanh(y)
    dloss_dx = torch.autograd.grad(outputs=[loss.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    y_paddle = y.cpu().detach().numpy()
    ddd = np.mean((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dx_paddle = dloss_dx.cpu().detach().numpy()
    ddd = np.mean((dloss_dx_pytorch - dloss_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)
    print()

    loss = dloss_dx.sum() + loss.sum()
    loss.backward()
    optimizer.step()
print()
