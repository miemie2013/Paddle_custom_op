'''
PD_BUILD_GRAD_OP(tanh_op)
    .Inputs({"X", "Y", paddle::Grad("Y")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(TanhBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(tanh_backward_InferShape));
'''
import os
import sys
import types
import paddle

cur_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(cur_dir, "custom_ops_pd_.so")

def inject_ext_module(module_name, api_names):
    if module_name in sys.modules:
        return sys.modules[module_name]

    new_module = types.ModuleType(module_name)
    for api_name in api_names:
        setattr(new_module, api_name, eval(api_name))

    return new_module

def __bootstrap__():
    assert os.path.exists(so_path)

    # load custom op shared library with abs path
    new_custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)
    m = inject_ext_module(__name__, new_custom_ops)

__bootstrap__()

import paddle.fluid.core as core
from paddle.fluid.core import VarBase, CustomOpKernelContext
from paddle.fluid.framework import _non_static_mode, _dygraph_tracer, _in_legacy_dygraph, in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper

def tanh_op(x):
    # prepare inputs and outputs
    ins = {'X' : x}
    attrs = {}
    outs = {}
    out_names = ['Y']

    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dygraph_mode():
        ctx = CustomOpKernelContext()
        for i in [x]:
            ctx.add_inputs(i)
        for j in []:
            ctx.add_attr(j)
        for out_name in out_names:
            outs[out_name] = core.eager.Tensor()
            ctx.add_outputs(outs[out_name])
        core.eager._run_custom_op(ctx, "tanh_op", True)
    else:
        if _in_legacy_dygraph():
            for out_name in out_names:
                outs[out_name] = VarBase()
            _dygraph_tracer().trace_op(type="tanh_op", inputs=ins, outputs=outs, attrs=attrs)
        else:
            helper = LayerHelper("tanh_op", **locals())
            for out_name in out_names:
                outs[out_name] = helper.create_variable(dtype='float32')

            helper.append_op(type="tanh_op", inputs=ins, outputs=outs, attrs=attrs)

    res = [outs[out_name] for out_name in out_names]

    return res[0] if len(res)==1 else res

