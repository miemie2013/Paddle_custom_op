import paddle
import numpy as np
from custom_gather import gather_op


class FullyConnectedLayer(paddle.nn.Layer):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = self.create_parameter([out_features, in_features],
                                            default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0 / lr_multiplier))
        self.bias = self.create_parameter([out_features], is_bias=True,
                                          default_initializer=paddle.nn.initializer.Constant(bias_init)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = paddle.cast(self.weight, dtype=x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = paddle.cast(b, dtype=x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        out = paddle.matmul(x, w, transpose_y=True) + b.unsqueeze(0)
        return out


w_dim = 512
in_channels = 256
activation = 'linear'
batch_size = 8
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
    index = dic2['batch_%.3d.index'%batch_idx]
    x = paddle.to_tensor(x)
    x.stop_gradient = False
    index = paddle.to_tensor(index)
    index.stop_gradient = True

    y = model(x)
    y = gather_op(y, index)
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
