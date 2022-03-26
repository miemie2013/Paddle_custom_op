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



class MyTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        y, = ctx.saved_tensors
        # dloss_dx = dloss_dy * (1 - torch.square(y))
        dx = MyTanhGrad.apply(dy, y)
        return dx

# class MyTanhGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, w):
#         y = x * (1 - torch.square(w))
#         ctx.save_for_backward(x, w)
#         return y
#
#     @staticmethod
#     def backward(ctx, dy):
#         x, w = ctx.saved_tensors
#         dx = dy * (1 - torch.square(w))
#         dw = dy * x * -2 * w
#         return dx, dw

class MyTanhGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dy, y):
        dx = dy * (1 - torch.square(y))
        ctx.save_for_backward(dy, y)
        return dx

    @staticmethod
    def backward(ctx, ddx):
        dy, y = ctx.saved_tensors
        ddy = ddx * (1 - torch.square(y))
        dy2 = ddx * dy * -2 * y
        # dy2 = None
        return ddy, dy2

# class MyTanhGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, dloss_dy, y):
#         dloss_dx = dloss_dy * (1 - torch.square(y))
#         ctx.save_for_backward(dloss_dy, y)
#         return dloss_dx
#
#     @staticmethod
#     def backward(ctx, d2loss_dxdx):
#         dloss_dy, y = ctx.saved_tensors
#         ddy = d2loss_dxdx * (1 - torch.square(y))
#         dy2 = d2loss_dxdx * dloss_dy * -2 * y
#         return ddy, dy2



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
    loss = MyTanh.apply(y)
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
