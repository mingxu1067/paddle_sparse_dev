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
Utilities of Auto SParsity (ASP).
"""

from __future__ import print_function

import sys
import math
import collections
import numpy as np
from itertools import permutations

__all__ = [
    'density', 'reshape_1d', 'check_mask_1d', 'get_mask_1d_greedy',
    'get_mask_1d_best',
    'check_mask_2d', 'get_mask_2d_greedy', 'get_mask_2d_best', 'create_mask',
    'check_sparsity'
]


def density(x):
    """
    Calculate density of input matrix.

    Args:
        x (nparray): The input matrix.
    Returns:
        float: The density of x.
    """
    x_flattened = x.flatten()
    return float(np.nonzero(x_flattened)[0].size) / x_flattened.size


def reshape_1d(mat, m):
    """
    Reshape input matrix to shape (-1, m)
    If the second dimension of mat is not a multiples of m,
    then this function would pad the remainder with 0 before reshaping.

    Args:
        mat (nparray): The input matrix.
        m (int): The second dimension of reshaped matrix.
    Returns:
        tuple: (The reshaped mat, The shape of padded mat).
    """
    remainder = mat.shape[1] % m
    if mat.shape[1] % m > 0:
        mat_padded = np.zeros((mat.shape[0], mat.shape[1] + (m - remainder)))
        mat_padded[:, :mat.shape[1]] = mat
        shape = mat_padded.shape
        return mat_padded.reshape(-1, m), shape
    else:
        return mat.reshape(-1, m), mat.shape


def check_mask_1d(mat, m, n):
    """
    Check if the input matrix is in 1D n:m sparse pattern.
    1D n:m sparse pattern: There are must n zeros in every 1xm block.

    Args:
        mat (nparray): The input matrix.
        m (int): m of n:m sparse pattern.
        n (int): n of n:m sparse pattern.
    Returns:
        bool: True if mat in 1D n:m sparse pattern, otherwise False.
    """
    if len(mat.shape) <= 1:
        mat_flattern, shape = reshape_1d(mat.reshape(1, mat.shape[0]), m)
    else:
        mat_flattern, shape = reshape_1d(mat, m)

    for sub_mat in mat_flattern:
        if np.nonzero(sub_mat)[0].size > n:
            return False
    return True


def get_mask_1d_greedy(mat, m, n):
    """
    Compute 1D n:m sparse pattern mask for the input matrix mat in greedy way.

    Args:
        mat (nparray): The input matrix.
        m (int): m of n:m sparse pattern.
        n (int): n of n:m sparse pattern.
    Returns:
        nparray: The 1D n:m sparse mask of mat.
    """
    mat_flattern, shape = reshape_1d(mat, m)

    mask_flattern = np.ones_like(mat_flattern)
    mask = np.ones_like(mat)
    for i in range(mat_flattern.shape[0]):
        sub_mat = mat_flattern[i]
        min_order_indices = np.argsort(np.absolute(sub_mat))
        mask_flattern[i, min_order_indices[:n].tolist()] = 0
    mask_flattern = mask_flattern.reshape(shape)
    mask[:, :] = mask_flattern[:, :mat.shape[1]]
    return mask


valid_1d_patterns = {}
def compute_valid_1d_patterns(m, n):
    """
    Compute all vaild 1D n:m sparse pattern.

    Args:
        m (int): m of n:m sparse pattern.
        n (int): n of n:m sparse pattern.
    Returns:
        map: A map from m_n (string) to vaild 1D n:m sparse patterns.
    """
    global valid_1d_patterns

    valid_key = '{}_{}'.format(m, n)
    if valid_key in valid_1d_patterns:
        return valid_1d_patterns[valid_key]
    else:
        patterns = np.zeros(m)
        patterns[:n] = 1
        valid_patterns = np.asarray(list(set(permutations(patterns.tolist()))))
        valid_1d_patterns[valid_key] = valid_patterns
        return valid_patterns


def get_mask_1d_best(mat, m, n):
    """
    Compute 1D n:m sparse pattern mask for the input matrix mat in best way.

    Args:
        mat (nparray): The input matrix.
        m (int): m of n:m sparse pattern.
        n (int): n of n:m sparse pattern.
    Returns:
        nparray: The 1D n:m sparse mask of mat.
    """
    patterns = compute_valid_1d_patterns(m, n)

    mat_flattern, shape = reshape_1d(mat, m)
    mask_flattern = np.ones_like(mat_flattern)
    pmax = np.argmax(np.matmul(np.abs(mat_flattern), patterns.T), axis=1)
    mask_flattern[:] = patterns[pmax[:]]
    mask = mask_flattern.reshape(mat.shape)
    return mask


def reshape_2d(mat, m):
    """
    Reshape input matrix to shape (-1, m*m)
    If the first and second dimension of mat is not a multiples of m,
    then this function would pad the remainders with 0 before reshaping.

    Args:
        mat (nparray): The input matrix.
        m (int): The second dimension of reshaped matrix.
    Returns:
        tuple: (The reshaped mat, The shape of padded mat).
    """
    remainder_0 = mat.shape[0] % m
    remainder_1 = mat.shape[1] % m

    new_shape = (mat.shape[0] if remainder_0 == 0 \
                 else mat.shape[0] + (m - remainder_0),
                 mat.shape[1] if remainder_1 == 0 \
                 else mat.shape[1] + (m - remainder_1))
    mat_padded = np.zeros(new_shape)
    mat_padded[:mat.shape[0], :mat.shape[1]] = mat

    mat_flattern = np.empty(new_shape).reshape(-1, m * m)
    curr_idx = 0
    for row_start in range(0, mat_padded.shape[0], m):
        row_end = row_start + m
        for col_start in range(0, mat_padded.shape[1], m):
            col_end = col_start + m
            sub_mat = np.squeeze(mat_padded[row_start:row_end, \
                                            col_start:col_end] \
                                            .reshape(-1))
            mat_flattern[curr_idx] = sub_mat
            curr_idx += 1
    return mat_flattern, mat_padded.shape


def check_mask_2d(mat, m, n):
    """
    Check if the input matrix is in 2D n:m sparse pattern.
    2D n:m sparse pattern: There are must n*n zeros in every mxm block,
    under the constraint of selecting exactly n elements for each row and column.

    Args:
        mat (nparray): The input matrix.
        m (int): m of n:m sparse pattern.
        n (int): n of n:m sparse pattern.
    Returns:
        bool: True if mat in 2D n:m sparse pattern, otherwise False.
    """
    mat_padded, shape = reshape_2d(mat, m)
    for sub_mat in mat_padded:
        sub_mask = np.absolute(np.squeeze(sub_mat.reshape(m, m))) > 0
        if (np.sum(np.sum(sub_mask, axis=1) > n) != 0) and \
            (np.sum(np.sum(sub_mask, axis=0) > n) != 0):
            return False
    return True


def get_mask_2d_greedy(mat, m, n):
    """
    Compute 2D n:m sparse pattern mask for the input matrix mat in greedy way.

    Args:
        mat (nparray): The input matrix.
        m (int): m of n:m sparse pattern.
        n (int): n of n:m sparse pattern.
    Returns:
        nparray: The 2D n:m sparse mask of mat.
    """
    mat_padded, shape = reshape_2d(mat, m)
    mask_padded = np.zeros_like(mat_padded).reshape(-1, m, m)

    for idx in range(len(mat_padded)):
        sub_mat = np.absolute(np.squeeze(mat_padded[idx]))
        sub_mask = np.squeeze(mask_padded[idx])

        min_order_1d_indices = np.argsort(sub_mat)
        min_order_2d_indices = [(int(x / m), x % m)
                                for x in min_order_1d_indices]
        row_counter = collections.Counter()
        col_counter = collections.Counter()

        for i in range(len(min_order_1d_indices) - 1, -1, -1):
            matrix_entry = min_order_2d_indices[i]
            if (row_counter[matrix_entry[0]] == n) or \
               (col_counter[matrix_entry[1]] == n):
                continue

            sub_mask[matrix_entry[0], matrix_entry[1]] = 1.0
            row_counter[matrix_entry[0]] += 1
            col_counter[matrix_entry[1]] += 1

    mask = np.empty(shape)
    curr_idx = 0
    for row_start in range(0, shape[0], m):
        row_end = row_start + m
        for col_start in range(0, shape[1], m):
            col_end = col_start + m
            mask[row_start:row_end, col_start:col_end] = mask_padded[curr_idx]
            curr_idx += 1
    return mask[:mat.shape[0], :mat.shape[1]]


valid_2d_patterns = {}
def compute_valid_2d_patterns(m, n):
    """
    Compute all vaild 2D n:m sparse pattern.

    Args:
        m (int): m of n:m sparse pattern.
        n (int): n of n:m sparse pattern.
    Returns:
        map: A map from m_n (string) to vaild 2D n:m sparse patterns.
    """
    global valid_2d_patterns

    valid_key = '{}_{}'.format(m, n)
    if valid_key in valid_2d_patterns:
        return valid_2d_patterns[valid_key]
    else:
        patterns = np.zeros(m)
        patterns[:n] = 1
        patterns = list(set(permutations(patterns.tolist())))
        patterns = patterns + patterns
        patterns = np.asarray(list(set(permutations(patterns, m))))

        valid = ((patterns.sum(axis=1) <= n).sum(axis=1) == m).nonzero()[0].reshape(-1)
        valid_patterns = np.empty((valid.shape[0], m, m))
        valid_patterns[:] = patterns[valid[:]]
        valid_2d_patterns[valid_key] = valid_patterns
        return valid_patterns


def get_mask_2d_best(mat, m, n):
    """
    Compute 2D n:m sparse pattern mask for the input matrix mat in best way.

    Args:
        mat (nparray): The input matrix.
        m (int): m of n:m sparse pattern.
        n (int): n of n:m sparse pattern.
    Returns:
        nparray: The 2D n:m sparse mask of mat.
    """
    patterns = compute_valid_2d_patterns(m, n)

    mat_flattern, shape = reshape_2d(mat, m)
    mask_flattern = np.ones_like(mat_flattern).reshape(-1, m, m)
    pmax = np.argmax(
        np.matmul(mat_flattern, patterns.reshape(patterns.shape[0], m * m).T),
        axis=1)

    mask_flattern[:] = patterns[pmax[:]]
    mask = np.empty(shape)

    curr_idx = 0
    for row_start in range(0, shape[0], m):
        row_end = row_start + m
        for col_start in range(0, shape[1], m):
            col_end = col_start + m
            mask[row_start:row_end, col_start:col_end] = mask_flattern[curr_idx]
            curr_idx += 1
    return mask[:mat.shape[0], :mat.shape[1]]


def create_mask(tensor, func_name="get_mask_1d_greedy", m=4, n=2):
    """
    Create n:m sparse pattern mask of input tensor via function given by func_name.
    Currently only support tensor with dimension less than or equal to 4.

    Args:
        tensor (nparray): The input matrix.
        func_name (string): The function name to generate spase mask.
        m (int): m of n:m sparse pattern.
        n (int): n of n:m sparse pattern.
    Returns:
        nparray: The n:m sparse mask of mat generated by func_name.
    """
    shape = tensor.shape
    ttype = tensor.dtype
    t = tensor.astype(float)

    func = getattr(sys.modules[__name__], func_name, None)
    if len(shape) == 1:
        t = t.reshape(1, shape[0])
        mask = func(t, m, n)
        return mask.reshape(shape).astype(ttype)
    elif len(shape) == 2:
        t = t.reshape(shape[0], shape[1])
        mask = func(t, m, n)
        return mask.reshape(shape).astype(ttype)
    elif len(shape) == 3:
        t = t.reshape(shape[0] * shape[1], shape[2])
        mask = func(t, m, n)
        return mask.reshape(shape).astype(ttype)
    # 4d-tensor conv (out, in, h, w) -> (out, in*h*w) in GemmConvKernel Op
    elif len(shape) == 4:
        t = t.reshape(shape[0], shape[1] * shape[2] * shape[3])
        mask = func(t, m, n)
        return mask.reshape(shape).astype(ttype)
    else:
        assert True, "The dimension of input tensor is not supported, " \
                     "Only dimension < 4 is supported but got {}".format(len(shape))


def check_sparsity(tensor, func_name="check_mask_1d", m=4, n=2):
    """
    Check if input tensor is in  n:m sparse pattern via function given by func_name.
    Currently only support tensor with dimension less than or equal to 4.

    Args:
        tensor (nparray): The input matrix.
        func_name (string): The function name to check spase mask.
        m (int): m of n:m sparse pattern.
        n (int): n of n:m sparse pattern.
    Returns:
        bool: True if tensor pass checking via func_name, otherwise False.
    """
    shape = tensor.shape
    t = tensor.astype(float)

    func = getattr(sys.modules[__name__], func_name, None)
    if len(shape) == 1:
        t = t.reshape(1, shape[0])
        return func(t, m, n)
    elif len(shape) == 2:
        t = t.reshape(shape[0], shape[1])
        return func(t, m, n)
    elif len(shape) == 3:
        t = t.reshape(shape[0] * shape[1], shape[2])
        return func(t, m, n)
    # 4d-tensor conv (out, in, h, w) -> (out, in*h*w) in GemmConvKernel Op
    elif len(shape) == 4:
        t = t.reshape(shape[0], shape[1] * shape[2] * shape[3])
        return func(t, m, n)

    return False