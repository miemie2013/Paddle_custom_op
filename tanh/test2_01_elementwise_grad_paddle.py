
import paddle
import numpy as np
import paddle.nn.functional as F





dic2 = np.load('01_grad.npz')
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)

    dydx_pytorch = dic2['batch_%.3d.dydx'%batch_idx]
    y_pytorch = dic2['batch_%.3d.out'%batch_idx]
    x = dic2['batch_%.3d.x'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False

    batch_size = 2

    z = paddle.square(x)
    y = F.sigmoid(z)
    # y = F.sigmoid(x)

    dydx = paddle.grad(outputs=[y.sum()], inputs=[x], create_graph=True)[0]

    loss = dydx.sum()
    loss.backward()
    double_grad_paddle = x.gradient()

    # double_grad_paddle2 = (1-2*y)*y*(1-y)
    # double_grad_paddle2 = double_grad_paddle2.numpy()

    double_grad_paddle2 = y*(1-y)
    double_grad_paddle2 = double_grad_paddle2 * 2 * x
    double_grad_paddle2 = double_grad_paddle2.numpy()


    y_paddle = y.numpy()
    ddd = np.mean((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    dydx_paddle = dydx.numpy()
    ddd = np.mean((dydx_pytorch - dydx_paddle) ** 2)
    print('ddd=%.6f' % ddd)
    print()
print()
