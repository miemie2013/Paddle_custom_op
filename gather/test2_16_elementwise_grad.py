import torch
import numpy as np
from tanh.networks_pytorch import FullyConnectedLayer



w_dim = 512
in_channels = 256
activation = 'linear'
batch_size = 8
lr = 0.1


model = FullyConnectedLayer(w_dim, in_channels, activation=activation, bias_init=1)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
torch.save(model.state_dict(), "16.pth")

dic = {}
for batch_idx in range(8):
    optimizer.zero_grad(set_to_none=True)

    x = torch.randn([batch_size, w_dim])
    x.requires_grad_(True)
    index = [0, 1, 3, 3, 3, 5]
    index = torch.Tensor(index).to(torch.int64)
    index.requires_grad_(False)

    y = model(x)
    y = y[index]
    loss = torch.tanh(y)

    dloss_dx = torch.autograd.grad(outputs=[loss.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dloss_dx'%batch_idx] = dloss_dx.cpu().detach().numpy()
    dic['batch_%.3d.y'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
    dic['batch_%.3d.index'%batch_idx] = index.cpu().detach().numpy()

    loss = dloss_dx.sum() + loss.sum()
    loss.backward()
    optimizer.step()
np.savez('16', **dic)
print()
