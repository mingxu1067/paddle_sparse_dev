import numpy as np
from paddle.fluid import framework, global_scope, program_guard, layers
from paddle.fluid.contrib import sparsity

__all__ = ['ASPHelper']

class ASPHelper(object):

    MASKE_APPENDDED_NAME = '_asp_mask'
    SUPPORTED_LAYERS = {'fc':'w_0'}

    __mask_vars = {}
    __masks = {}

    @staticmethod
    def is_supported_layer(param_name):
        for name in ASPHelper.SUPPORTED_LAYERS:
            if name in param_name and \
               ASPHelper.SUPPORTED_LAYERS[name] in param_name:
               return True
        return False

    @staticmethod
    def get_mask_name(param_name):
        return param_name + ASPHelper.MASKE_APPENDDED_NAME

    @classmethod
    def initialize_asp_training(cls, main_program, start_program, exe):
        exe.run(start_program)
        with program_guard(main_program, start_program):
            for param in main_program.all_parameters():
                if ASPHelper.is_supported_layer(param.name):
                    mask_param = layers.create_parameter(
                                 name=param.name + ASPHelper.MASKE_APPENDDED_NAME,
                                 shape=param.shape, dtype=param.dtype)
                    cls.__mask_vars[param.name] = mask_param
        exe.run(start_program)

    @classmethod
    def prune_model(cls, main_program, start_program, place):
        for param in main_program.global_block().all_parameters():
            if ASPHelper.is_supported_layer(param.name) and \
               ASPHelper.MASKE_APPENDDED_NAME not in param.name:
                weight_param = global_scope().find_var(param.name).get_tensor()
                weight_mask_param = global_scope().find_var(ASPHelper.get_mask_name(param.name)).get_tensor()
                weight_tensor = np.array(weight_param)
                weight_sparse_mask = sparsity.create_mask(weight_tensor)
                weight_pruned_tensor = np.multiply(weight_tensor, weight_sparse_mask)
                weight_param.set(weight_pruned_tensor, place)
                weight_mask_param.set(weight_sparse_mask, place)
                assert sparsity.check_mask_2d(weight_pruned_tensor, m=4, n=2), \
                        "Pruning {} weight matrix failure!!!".format(param.name)
                cls.__masks[param.name] = weight_sparse_mask

    @classmethod
    def insert_grads_mask(cls, main_program, start_program, optimizer_type, param_grads):
        block = main_program.global_block()
        ops = main_program.global_block().ops
        for idx in len(ops):
            if ops[idx].type == optimizer_type:
                for param_grad in param_grads:
                    if param_grad[0].name in cls.__mask_vars:
                        block._insert_op(
                            idx,
                            type='elementwise_mul',
                            inputs={"X": param_grad[1],
                                    'Y': cls.__mask_vars[param_grad[0].name]},
                            outputs={'Out': param_grad[1]},
                            attrs={'axis': -1,
                                    'use_mkldnn': False},
                            stop_gradient=True)
                break