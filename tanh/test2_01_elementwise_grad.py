
import torch
import numpy as np

import torch.nn.functional as F







dic = {}
for batch_idx in range(8):
    x = torch.randn([2, 6, 16, 16])
    x.requires_grad_(True)
    batch_size = 2

    z = torch.square(x)
    y = torch.sigmoid(z)
    # y = torch.sigmoid(x)

    dydx = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]

    dic['batch_%.3d.dydx'%batch_idx] = dydx.cpu().detach().numpy()
    dic['batch_%.3d.out'%batch_idx] = y.cpu().detach().numpy()
    dic['batch_%.3d.x'%batch_idx] = x.cpu().detach().numpy()
np.savez('01_grad', **dic)
print()
