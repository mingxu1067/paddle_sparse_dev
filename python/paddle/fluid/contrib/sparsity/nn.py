from paddle.fluid.data_feeder import check_type, check_dtype, convert_dtype
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import Variable

from functools import reduce
import warnings

__all__ = ['fc_sparse']

def fc_sparse(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       act=None,
       enable_cache=False,
       name=None):
    helper = LayerHelper("fc", **locals())
    check_type(input, 'input', (list, tuple, Variable), 'fc_sparse')
    if isinstance(input, (list, tuple)):
        for i, input_x in enumerate(input):
            check_type(input_x, 'input[' + str(i) + ']', Variable, 'fc_sparse')
    dtype = helper.input_dtype()
    check_dtype(dtype, 'input', ['float16', 'float32', 'float64'], 'fc_sparse')
    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        if num_flatten_dims == -1:
            num_flatten_dims = len(input_shape) - 1
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1)
        ] + [size]

        w = helper.create_parameter(
            attr=param_attr, shape=param_shape, dtype=dtype, is_bias=False)
        tmp = helper.create_variable_for_type_inference(dtype)

        if convert_dtype(dtype) in ['float32', 'float64']:
            warnings.warn("fc_sparse: sparse mul op only supports float16 on GPUs. Insert dense mul op.")
            helper.append_op(
                type="mul",
                inputs={"X": input_var,
                        "Y": w},
                outputs={"Out": tmp},
                attrs={"x_num_col_dims": num_flatten_dims,
                    "y_num_col_dims": 1})
            mul_results.append(tmp)
        else:
            layer_attr = {"x_num_col_dims": num_flatten_dims,
                          "y_num_col_dims": 1,
                          "is_col_major": True,
                          "m":param_shape[1],
                          "k":param_shape[0],
                          "lda":param_shape[1],
                          "ldb":param_shape[0],
                          "ldc":param_shape[1],
                          "is_transpose_Y": True,
                          "switch_XY": True}
            if enable_cache:
                layer_attr["param_name"]=w.name
            helper.append_op(
                type="mul_sparse",
                inputs={"X": input_var,
                        "Y": w},
                outputs={"Out": tmp},
                attrs=layer_attr)
            mul_results.append(tmp)

    if len(mul_results) == 1:
        pre_bias = mul_results[0]
    else:
        pre_bias = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="sum",
            inputs={"X": mul_results},
            outputs={"Out": pre_bias},
            attrs={"use_mkldnn": False})
    # add bias
    pre_activation = helper.append_bias_op(pre_bias, dim_start=num_flatten_dims)
    # add activation
    return helper.append_activation(pre_activation)