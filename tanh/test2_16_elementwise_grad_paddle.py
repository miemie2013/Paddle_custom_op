import paddle
import numpy as np
from tanh.networks_paddle import FullyConnectedLayer



w_dim = 512
in_channels = 256
activation = 'linear'
batch_size = 2
lr = 0.1


model = FullyConnectedLayer(w_dim, in_channels, activation=activation, bias_init=1)
model.train()
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=lr, momentum=0.9)
model.set_state_dict(paddle.load("16.pdparams"))


dic2 = np.load('16.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.clear_gradients()

    dloss_dx_pytorch = dic2['batch_%.3d.dloss_dx'%batch_idx]
    y_pytorch = dic2['batch_%.3d.y'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False

    y = model(x)
    loss = paddle.tanh(y)
    dloss_dx = paddle.grad(outputs=[loss.sum()], inputs=[x], create_graph=True)[0]

    y_paddle = y.numpy()
    ddd = np.mean((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dloss_dx_paddle = dloss_dx.numpy()
    ddd = np.mean((dloss_dx_pytorch - dloss_dx_paddle) ** 2)
    print('ddd=%.6f' % ddd)
    print()

    loss = dloss_dx.sum() + loss.sum()
    loss.backward()
    optimizer.step()
print()
