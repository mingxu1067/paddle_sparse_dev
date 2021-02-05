import numpy as np
from paddle.fluid import framework, global_scope, program_guard, layers
from paddle.fluid.initializer import ConstantInitializer
from paddle.fluid.contrib import sparsity

__all__ = ['ASPHelper']

class ASPHelper(object):

    MASKE_APPENDDED_NAME = '_asp_mask'
    SUPPORTED_LAYERS = {'fc':'w_0', 'linear':'w_0', 'conv2d':'w_0'}

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

    @staticmethod
    def get_vars(main_program):
        var_list = []
        for param in main_program.global_block().all_parameters():
            if ASPHelper.MASKE_APPENDDED_NAME not in param.name:
                var_list.append(param)
        return var_list

    @classmethod
    def minimize(cls, loss, optimizer, place, main_program, start_program):
        optimizer_ops, params_and_grads = optimizer.minimize(loss)
        cls.create_mask_variables(main_program, start_program, params_and_grads)
        cls.insert_sparse_mask_ops(main_program, start_program, optimizer.type, params_and_grads)
        return optimizer_ops, params_and_grads

    @classmethod
    def create_mask_variables(cls, main_program, start_program, params_and_grads):
        with program_guard(main_program, start_program):
            for param_and_grad in params_and_grads:
                if ASPHelper.is_supported_layer(param_and_grad[0].name):
                    mask_param = layers.create_parameter(
                                 name=param_and_grad[0].name + ASPHelper.MASKE_APPENDDED_NAME,
                                 shape=param_and_grad[0].shape, dtype=param_and_grad[0].dtype,
                                 default_initializer=ConstantInitializer(value=1.0))
                    mask_param.stop_gradient=True
                    mask_param.trainable=False
                    cls.__mask_vars[param_and_grad[0].name] = mask_param

    @classmethod
    def prune_model(cls, main_program, start_program, place, func_name="get_mask_2d_greedy", with_mask=True):
        for param in main_program.global_block().all_parameters():
            if ASPHelper.is_supported_layer(param.name) and \
               ASPHelper.MASKE_APPENDDED_NAME not in param.name:
                weight_param = global_scope().find_var(param.name).get_tensor()
                weight_tensor = np.array(weight_param)
                weight_sparse_mask = sparsity.create_mask(weight_tensor, func_name=func_name)
                weight_pruned_tensor = np.multiply(weight_tensor, weight_sparse_mask)
                weight_param.set(weight_pruned_tensor, place)
                assert sparsity.check_sparsity(weight_pruned_tensor, m=4, n=2), \
                        "Pruning {} weight matrix failure!!!".format(param.name)
                if with_mask:
                    weight_mask_param = global_scope().find_var(ASPHelper.get_mask_name(param.name)).get_tensor()
                    assert weight_mask_param is not None, \
                        "Cannot find {} parameter, please call ASPHelper.minimize" \
                        " or ASPHelper.initialize_asp_training first!".format(ASPHelper.get_mask_name(param.name))
                    weight_mask_param.set(weight_sparse_mask, place)
                cls.__masks[param.name] = weight_sparse_mask
        return cls.__masks.copy()

    @classmethod
    def insert_sparse_mask_ops(cls, main_program, start_program, optimizer_type, param_grads):
        block = main_program.global_block()
        for param_grad in param_grads:
            if param_grad[0].name in cls.__mask_vars:
                block.append_op(
                    type='elementwise_mul',
                    inputs={"X": param_grad[0],
                            'Y': cls.__mask_vars[param_grad[0].name]},
                    outputs={'Out': param_grad[0]},
                    attrs={'axis': -1,
                            'use_mkldnn': False}
                )
        # ops = main_program.global_block().ops
        # for idx in range(len(ops)):
        #     if ops[idx].type == optimizer_type:
        #         for param_grad in param_grads:
        #             if param_grad[0].name in cls.__mask_vars:
        #                 block._insert_op(
        #                     idx,
        #                     type='elementwise_mul',
        #                     inputs={"X": param_grad[1],
        #                             'Y': cls.__mask_vars[param_grad[0].name]},
        #                     outputs={'Out': param_grad[1]},
        #                     attrs={'axis': -1,
        #                             'use_mkldnn': False})
        #         break


