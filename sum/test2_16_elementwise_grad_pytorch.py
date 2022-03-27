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



class MySum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        ctx.dim = dim
        ctx.dim_len = x.shape[dim]
        y = torch.sum(x, dim=dim)
        return y

    @staticmethod
    def backward(ctx, dy):
        dim = ctx.dim
        dim_len = ctx.dim_len
        dx = MySumGrad.apply(dy, dim, dim_len)
        return dx, None


class MySumGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dy, dim, dim_len):
        ctx.dim = dim
        dx = dy.unsqueeze(dim).repeat([1, dim_len])
        return dx

    @staticmethod
    def backward(ctx, ddx):
        dim = ctx.dim
        ddy = torch.sum(ddx, dim=dim)
        return ddy, None, None


dic2 = np.load('16.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.zero_grad(set_to_none=True)

    dloss_dx_pytorch = dic2['batch_%.3d.dloss_dx'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    x = torch.Tensor(x)
    x.requires_grad_(True)

    y = model(x)
    loss = MySum.apply(y, 1)
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
