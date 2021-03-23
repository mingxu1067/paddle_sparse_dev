import numpy as np
from paddle.fluid import framework, global_scope, program_guard, layers
from paddle.fluid.initializer import ConstantInitializer
from paddle.fluid.contrib import sparsity
from paddle.fluid import core

__all__ = ['ASPHelper']

class OpRelacementInfo(object):
    def __init__(self, source_type, target_type,
                param_shape_related_attrs={}, constant_attrs={},
                source_param_input_name='Y', source_param_idx=0,
                source_data_input_name='X', source_data_idx=0):
        self.source_type = source_type
        self.target_type = target_type
        self.param_shape_related_attrs = param_shape_related_attrs
        self.constant_attrs = constant_attrs
        self.source_param_input_name = source_param_input_name
        self.source_param_idx = source_param_idx
        self.source_data_input_name = source_data_input_name
        self.source_data_idx = source_data_idx

    def is_executable(self, block, op):
        return True

class MulSparseOpRelacementInfo(OpRelacementInfo):
    def __init__(self, source_type,
                source_param_input_name='Y', source_param_idx=0,
                source_data_input_name='X', source_data_idx=0):

        param_shape_related_attrs = {'m':1, 'k':0, 'lda':1, 'ldb':0, 'ldc':1}
        constant_attrs={'is_col_major':True, 'is_transpose_Y':True, 'switch_XY':True}

        super(MulSparseOpRelacementInfo, self).__init__(
            source_type=source_type, target_type='mul_sparse',
            param_shape_related_attrs=param_shape_related_attrs,
            constant_attrs=constant_attrs,
            source_param_input_name=source_param_input_name,
            source_param_idx=source_param_idx,
            source_data_input_name=source_data_input_name,
            source_data_idx=source_data_idx
        )

    def is_executable(self, block, op):
        param_name = op.input(self.source_param_input_name)[self.source_param_idx]
        param = block.var(param_name)

        data_name = op.input(self.source_data_input_name)[self.source_data_idx]
        data = block.var(data_name)

        if param is None or data is None:
            return False
        if param.dtype != core.VarDesc.VarType.FP16 and \
           data.dtype != core.VarDesc.VarType.FP16:
           return False
        if (param.shape[1] % 8) or \
           (param.shape[0] % 32):
           return False

        return True


class ASPHelper(object):

    MASKE_APPENDDED_NAME = '_asp_mask'
    SUPPORTED_LAYERS = {'fc':'w_0', 'linear':'w_0', 'conv2d':'w_0'}

    DENSE_SPARSE_OP_MAP = {
        'mul':MulSparseOpRelacementInfo(source_type='mul'),
        'matmul':MulSparseOpRelacementInfo(source_type='matmul')
    }

    __mask_vars = {}
    __masks = {}
    __compressed_cache = {}

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
    def prune_model(cls, main_program, start_program, place, func_name='get_mask_1d_greedy', with_mask=True):
        checked_func_name = 'check_mask_1d' if '1d' in func_name else 'check_mask_2d'

        for param in main_program.global_block().all_parameters():
            if ASPHelper.is_supported_layer(param.name) and \
               ASPHelper.MASKE_APPENDDED_NAME not in param.name:
                weight_param = global_scope().find_var(param.name).get_tensor()
                weight_tensor = np.array(weight_param)
                weight_sparse_mask = sparsity.create_mask(weight_tensor, func_name=func_name)
                weight_pruned_tensor = np.multiply(weight_tensor, weight_sparse_mask)
                weight_param.set(weight_pruned_tensor, place)
                assert sparsity.check_sparsity(weight_pruned_tensor, m=4, n=2, func_name=checked_func_name), \
                        'Pruning {} weight matrix failure!!!'.format(param.name)
                if with_mask:
                    weight_mask_param = global_scope().find_var(ASPHelper.get_mask_name(param.name)).get_tensor()
                    assert weight_mask_param is not None, \
                        'Cannot find {} parameter, please call ASPHelper.minimize' \
                        ' or ASPHelper.initialize_asp_training first!'.format(ASPHelper.get_mask_name(param.name))
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

    @classmethod
    def compress_model(cls, main_program, place):
        assert not main_program in cls.__compressed_cache, \
               'One program only need to compress model once. Called more than one would make errors'

        fake_batch_size = 128
        for param in main_program.global_block().all_parameters():
            if ASPHelper.is_supported_layer(param.name) and \
               ASPHelper.MASKE_APPENDDED_NAME not in param.name:
                shape = param.shape
                param_tensor = global_scope().find_var(param.name).get_tensor()
                core.compress_parameter(place, param_tensor, shape[1], fake_batch_size, shape[0],
                                        shape[1], shape[0], shape[1], True)

        cls.__compressed_cache[main_program] = True

        block = main_program.global_block()
        for op in block.global_block():
            replacement_info = ASPHelper.DENSE_SPARSE_OP_MAP.get(op.type, None)
            if (replacement_info is not None) and \
               (replacement_info.source_param_input_name in op.input_names):
               for var_name in op.input(replacement_info.source_param_input_name):
                   if ASPHelper.is_supported_layer(var_name):
                       op._set_attr("is_sparse_compressed", True)

    @classmethod
    def replace_dense_to_sparse_op(cls, main_program):
        is_compressed = main_program in cls.__compressed_cache

        block = main_program.global_block()
        for op in block.ops:
            replacement_info = ASPHelper.DENSE_SPARSE_OP_MAP.get(op.type, None)
            if (replacement_info is not None) and \
               (replacement_info.source_param_input_name in op.input_names) and \
               (replacement_info.is_executable(block, op)):
                param_name = op.input(replacement_info.source_param_input_name)[replacement_info.source_param_idx]
                param = block.var(param_name)
                if (ASPHelper.is_supported_layer(param_name)) and \
                   (param is not None):
                    op.desc.set_type(replacement_info.target_type)
                    # TODO Need to be more general for future sparse conv ops.
                    op._set_attr("param_name", param.name)
                    op._set_attr("is_sparse_compressed", is_compressed)
                    for key, val in replacement_info.param_shape_related_attrs.items():
                       op._set_attr(key, param.shape[val])
                    for key, val in replacement_info.constant_attrs.items():
                       op._set_attr(key, val)

