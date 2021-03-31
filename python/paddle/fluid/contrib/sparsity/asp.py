# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Functions for Auto SParsity (ASP) training and inference.
"""

import copy
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
    r"""
    ASPHelper is a collection of Auto SParsity (ASP) functions to enable 

    1. training models with weights in 2:4 sparse pattern from scratch.
    2. pruning well-trained models into 2:4 sparse pattern for fine-tuning.
    """

    MASKE_APPENDDED_NAME = '_asp_mask'
    SUPPORTED_LAYERS = {'fc':'w_0', 'linear':'w_0', 'conv2d':'w_0'}

    DENSE_SPARSE_OP_MAP = {
        'mul':MulSparseOpRelacementInfo(source_type='mul'),
        'matmul':MulSparseOpRelacementInfo(source_type='matmul')
    }

    SPARSE_OP_PARAM_INPUT_NAME_MAP = {
        'mul_sparse':'Y'
    }

    __mask_vars = {}
    __masks = {}
    __compressed_cache = {}
    __excluded_layers = []

    @staticmethod
    def get_mask_name(param_name):
        r"""
        Return mask name by given parameter name :attr:`param_name`.

        Args:
            param_name (string): The name of parameter.
        Returns:
            string: The mask name of :attr:`param_name`.
        """
        return param_name + ASPHelper.MASKE_APPENDDED_NAME

    @staticmethod
    def get_vars(main_program):
        r"""
        Get all parameters in :attr:`main_program` excluded ASP mask Variables.

        Args:
            main_program (Program): Program with model definition and its parameters.
        Returns:
            list: A list of parameter Variables in :attr:`main_program` (excluded ASP mask Variables).
        """
        var_list = []
        for param in main_program.global_block().all_parameters():
            if ASPHelper.MASKE_APPENDDED_NAME not in param.name:
                var_list.append(param)
        return var_list

    @classmethod
    def set_excluded_layers(cls, param_names):
        r"""
        Set parameter name of layers which would not be pruned as sparse weights.

        Args:
            param_names (list): A list contains names of parameters.
        """
        cls.__excluded_layers = copy.deepcopy(param_names)

    @classmethod
    def is_supported_layer(cls, param_name):
        r"""
        Verify if given :attr:`param_name` is supported by ASP.

        Args:
            param_name (string): The name of parameter.
        Returns:
            bool: True if it is supported, else False.
        Examples:
            .. code-block:: python

              import paddle.fluid as fluid
              from paddle.fluid.contrib.sparsity import ASPHelper

              main_program = fluid.Program()
              startup_program = fluid.Program()

              with fluid.program_guard(main_program, startup_program):
                  input_data = fluid.layers.data(name='data', shape=[None, 128])
                  fc = fluid.layers.fc(input=input_data, num_flatten_dims=-1, size=32, act=None)

              for param in main_program.global_block().all_parameters():
                  ASPHelper.is_supported_layer(param.name)
              # fc_0.w_0 -> True
              # fc_0.b_0 -> False
        """
        if ASPHelper.MASKE_APPENDDED_NAME in param_name:
            return False

        for layer in cls.__excluded_layers:
            if layer in param_name:
                return False

        for name in ASPHelper.SUPPORTED_LAYERS:
            if name in param_name and \
               ASPHelper.SUPPORTED_LAYERS[name] in param_name:
               return True
        return False

    @classmethod
    def minimize(cls, loss, optimizer, main_program, start_program):
        r"""
        This function is a decorator of `minimize` function in `Optimizer`.
        There are three steps:

        1. Call :attr:`optimizer`.minimize(:attr:`loss`)
        2. Create sparse mask Tensors according to supported layers in :attr:`main_program`.
        3. Insert masking ops in the end of parameters update.

        Args:
            loss (Variable): A Variable containing the value to minimize.
            optimizer (Optimizer): A Optimizer used for training.
            main_program (Program): Program with model definition and its parameters.
            start_program (Program): Program for initializing parameters.
        Returns:
            list: operators from :attr:`optimizer`.minimize(:attr:`loss`).
            list: pairs of parameters and their gradients.
        Examples:
            .. code-block:: python

              import paddle.fluid as fluid
              from paddle.fluid.contrib.sparsity import ASPHelper

              main_program = fluid.Program()
              start_program = fluid.Program()

              with fluid.program_guard(main_program, start_program):
                    input_data = fluid.layers.data(name='data', shape=[None, 128])
                    label = fluid.layers.data(name='label', shape=[None, 10])
                    hidden = fluid.layers.fc(input=input_data, num_flatten_dims=-1, size=32, act=None)
                    prob = fluid.layers.fc(input=hidden, num_flatten_dims=-1, size=10, act=None)
                    loss = fluid.layers.mean(fluid.layers.square_error_cost(prob, label))

                    optimizer = fluid.optimizer.SGD(learning_rate=0.1)
                    ASPHelper.minimize(loss, optimizer, main_program, start_program)
        """
        optimizer_ops, params_and_grads = optimizer.minimize(loss)
        cls.create_mask_variables(main_program, start_program, params_and_grads)
        cls.insert_sparse_mask_ops(main_program, start_program, params_and_grads)
        return optimizer_ops, params_and_grads

    @classmethod
    def create_mask_variables(cls, main_program, start_program, params_and_grads):
        r"""
        Create sparse mask Tensors according to supported layers in :attr:`main_program`.
        This function is called in second step of `ASPHelper.minimize`

        Args:
            main_program (Program): Program with model definition and its parameters.
            start_program (Program): Program for initializing parameters.
            params_and_grads (list): Variable pairs of parameters and their gradients.
        Examples:
            .. code-block:: python

              import paddle.fluid as fluid
              from paddle.fluid.contrib.sparsity import ASPHelper

              main_program = fluid.Program()
              start_program = fluid.Program()

              with fluid.program_guard(main_program, start_program):
                    input_data = fluid.layers.data(name='data', shape=[None, 128])
                    label = fluid.layers.data(name='label', shape=[None, 10])
                    hidden = fluid.layers.fc(input=input_data, num_flatten_dims=-1, size=32, act=None)
                    prob = fluid.layers.fc(input=hidden, num_flatten_dims=-1, size=10, act=None)
                    loss = fluid.layers.mean(fluid.layers.square_error_cost(prob, label))

                    optimizer = fluid.optimizer.SGD(learning_rate=0.1)

                    # Equal to ASPHelper.minimize(loss, optimizer, main_program, start_program)
                    optimizer_ops, params_and_grads = optimizer.minimize(loss)
                    ASPHelper.create_mask_variables(main_program, start_program, params_and_grads)
                    ASPHelper.insert_sparse_mask_ops(main_program, start_program, params_and_grads)
        """
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
        r"""
        Pruning parameters of supported layers in :attr:`main_program` via 
        specified mask generation function given by :attr:`func_name`. This 
        function supports both training and inference controlled by :attr:`with_mask`.
        If :attr:`with_mask` is True, it would also prune parameter related ASP mask Variables,
        else only prunes parameters.

        *Note*: If calling this function with :attr:`with_mask`, it should call `ASPHelper.minimize` 
        and initialization (`exe.run(startup_program`)) before.

        Args:
            main_program (Program): Program with model definition and its parameters.
            start_program (Program): Program for initializing parameters.
            place (fluid.CPUPlace()|fluid.CUDAPlace(N)): Device place for pruned parameter and mask Variables.
            func_name (string, optional): The name of function to generate spase masks. Defalut is `get_mask_1d_greedy`.
            with_mask (bool, optional): To prune mask Variables related to parameters or not. Ture is purning also, False is not. Defalut is True.
        Returns:
            dictionary: A dictionary with key: `parameter name` (string) and value: its corresponding mask Variable.
        Examples:
            .. code-block:: python

              import paddle.fluid as fluid
              from paddle.fluid.contrib.sparsity import ASPHelper

              main_program = fluid.Program()
              start_program = fluid.Program()

              place = fluid.CUDAPlace(0)

              with fluid.program_guard(main_program, start_program):
                    input_data = fluid.layers.data(name='data', shape=[None, 128])
                    label = fluid.layers.data(name='label', shape=[None, 10])
                    hidden = fluid.layers.fc(input=input_data, num_flatten_dims=-1, size=32, act=None)
                    prob = fluid.layers.fc(input=hidden, num_flatten_dims=-1, size=10, act=None)
                    loss = fluid.layers.mean(fluid.layers.square_error_cost(prob, label))

                    optimizer = fluid.optimizer.SGD(learning_rate=0.1)
                    ASPHelper.minimize(loss, optimizer, main_program, start_program)

              exe = fluid.Executor(place)
              exe.run(start_program)

              ASPHelper.prune_model(train_prog, start_prog, place, func_name="get_mask_2d_greedy")
        """
        checked_func_name = 'check_mask_1d' if '1d' in func_name else 'check_mask_2d'

        for param in main_program.global_block().all_parameters():
            if ASPHelper.is_supported_layer(param.name):
                weight_param = global_scope().find_var(param.name).get_tensor()
                weight_tensor = np.array(weight_param)
                weight_sparse_mask = sparsity.create_mask(weight_tensor.T, func_name=func_name).T
                weight_pruned_tensor = np.multiply(weight_tensor, weight_sparse_mask)
                weight_param.set(weight_pruned_tensor, place)
                assert sparsity.check_sparsity(weight_pruned_tensor.T, m=4, n=2, func_name=checked_func_name), \
                        'Pruning {} weight matrix failure!!!'.format(param.name)
                if with_mask:
                    weight_mask_param = global_scope().find_var(ASPHelper.get_mask_name(param.name))
                    assert weight_mask_param is not None, \
                        'Cannot find {} variable, please call ASPHelper.minimize' \
                        'initialization (exe.run(startup_program)) first!'.format(ASPHelper.get_mask_name(param.name))
                    weight_mask_param = weight_mask_param.get_tensor()
                    weight_mask_param.set(weight_sparse_mask, place)
                cls.__masks[param.name] = weight_sparse_mask
        return cls.__masks.copy()

    @classmethod
    def insert_sparse_mask_ops(cls, main_program, start_program, param_grads):
        r"""
        Insert masking ops in the end of parameters update.
        This function is called in third step of `ASPHelper.minimize`

        Args:
            main_program (Program): Program with model definition and its parameters.
            start_program (Program): Program for initializing parameters.
            params_and_grads (list): Variable pairs of parameters and their gradients.
        Examples:
            .. code-block:: python

              import paddle.fluid as fluid
              from paddle.fluid.contrib.sparsity import ASPHelper

              main_program = fluid.Program()
              start_program = fluid.Program()

              with fluid.program_guard(main_program, start_program):
                    input_data = fluid.layers.data(name='data', shape=[None, 128])
                    label = fluid.layers.data(name='label', shape=[None, 10])
                    hidden = fluid.layers.fc(input=input_data, num_flatten_dims=-1, size=32, act=None)
                    prob = fluid.layers.fc(input=hidden, num_flatten_dims=-1, size=10, act=None)
                    loss = fluid.layers.mean(fluid.layers.square_error_cost(prob, label))

                    optimizer = fluid.optimizer.SGD(learning_rate=0.1)

                    # Equal to ASPHelper.minimize(loss, optimizer, main_program, start_program)
                    optimizer_ops, params_and_grads = optimizer.minimize(loss)
                    ASPHelper.create_mask_variables(main_program, start_program, params_and_grads)
                    ASPHelper.insert_sparse_mask_ops(main_program, start_program, params_and_grads)
        """
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
            if ASPHelper.is_supported_layer(param.name):
                shape = param.shape
                param_tensor = global_scope().find_var(param.name).get_tensor()
                core.compress_parameter(place, param_tensor, shape[1], fake_batch_size, shape[0],
                                        shape[1], shape[0], shape[1], True)

        cls.__compressed_cache[main_program] = True

        block = main_program.global_block()
        for op in block.ops:
            param_input_name = ASPHelper.SPARSE_OP_PARAM_INPUT_NAME_MAP.get(op.type, None)
            if param_input_name in op.input_names:
               for var_name in op.input(param_input_name):
                   if ASPHelper.is_supported_layer(var_name):
                       op._set_attr("is_sparse_compressed", True)
                       break

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

