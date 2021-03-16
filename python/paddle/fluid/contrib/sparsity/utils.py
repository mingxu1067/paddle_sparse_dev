from __future__ import print_function

import sys
import math
import collections
import numpy as np
from itertools import permutations

__all__ = ['density', 'reshape_1d', 'check_mask_1d', 'get_mask_1d_greedy',
           'check_mask_2d', 'get_mask_2d_greedy', 'create_mask', 'check_sparsity']

def density(x):
    x_flattened = x.flatten()
    return float(np.nonzero(x_flattened)[0].size) / x_flattened.size

def reshape_1d(mat, m):
    remainder = mat.shape[1] % m
    if mat.shape[1] % m > 0:
        mat_padded = np.zeros((mat.shape[0], mat.shape[1]+(m-remainder)))
        mat_padded[:, :mat.shape[1]] = mat
        shape = mat_padded.shape
        return mat_padded.reshape(-1, m), shape
    else:
        return mat.reshape(-1, m), mat.shape

def check_mask_1d(mat, m, n):
    if len(mat.shape) <= 1:
        mat_flattern, shape = reshape_1d(mat.reshape(1, mat.shape[0]), m)
    else:
        mat_flattern, shape = reshape_1d(mat, m)

    for sub_mat in mat_flattern:
        if np.nonzero(sub_mat)[0].size > n:
            return False
    return True

def get_mask_1d_greedy(mat, m, n):
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
    patterns = compute_valid_1d_patterns(m, n)

    mat_flattern, shape = reshape_1d(mat, m)
    mask_flattern = np.ones_like(mat_flattern)
    pmax = np.argmax(np.matmul(np.abs(mat_flattern), patterns.T), axis=1)
    mask_flattern[:] = patterns[pmax[:]]
    mask = mask_flattern.reshape(mat.shape)
    return mask

def reshape_2d(mat, m):
    remainder_0 = mat.shape[0] % m
    remainder_1 = mat.shape[1] % m

    new_shape = (mat.shape[0] if remainder_0 == 0 else mat.shape[0]+(m-remainder_0),
                 mat.shape[1] if remainder_1 == 0 else mat.shape[1]+(m-remainder_1))
    mat_padded = np.zeros(new_shape)
    mat_padded[:mat.shape[0], :mat.shape[1]] = mat

    mat_flattern = np.empty(new_shape).reshape(-1, m*m)
    curr_idx = 0
    for row_start in range(0, mat_padded.shape[0], m):
        row_end = row_start + m
        for col_start in range(0, mat_padded.shape[1], m):
            col_end = col_start + m
            sub_mat = np.squeeze(mat_padded[row_start:row_end, col_start:col_end].reshape(-1))
            mat_flattern[curr_idx] = sub_mat
            curr_idx += 1
    return mat_flattern, mat_padded.shape

def check_mask_2d(mat, m, n):
    row_count = int(mat.shape[0]/m) * m
    col_count = int(mat.shape[1]/m) * m

    for row_start in range(0, row_count, m):
        row_end = row_start + m
        for col_start in range(0, col_count, m):
            col_end = col_start + m

            sub_mask = np.absolute(np.squeeze(mat[row_start:row_end, col_start:col_end])) >0
            if (np.sum(np.sum(sub_mask, axis=1) > n) != 0) and \
               (np.sum(np.sum(sub_mask, axis=0) > n) != 0):
               return False
    return True

def get_mask_2d_greedy(mat, m, n):
    mask = np.ones_like(mat)

    row_count = int(mat.shape[0]/m) * m
    col_count = int(mat.shape[1]/m) * m

    for row_start in range(0, row_count, m):
        row_end = row_start + m
        for col_start in range(0, col_count, m):
            col_end = col_start + m
            sub_mat = np.absolute(np.squeeze(mat[row_start:row_end, col_start:col_end]))
            sub_mask = np.squeeze(mask[row_start:row_end, col_start:col_end]) 
            sub_mask.fill(0.0)

            sub_mat_flatten = sub_mat.reshape(-1)
            min_order_1d_indices = np.argsort(sub_mat_flatten)
            min_order_2d_indices = [(int(x/m), x % m) for x in min_order_1d_indices]
            row_counter = collections.Counter()
            col_counter = collections.Counter()

            for i in range(len(min_order_1d_indices) - 1, -1, -1):
                matrix_entry = min_order_2d_indices[i]
                if (row_counter[matrix_entry[0]] == n) or (col_counter[matrix_entry[1]] == n):
                    continue

                sub_mask[matrix_entry[0], matrix_entry[1]] = 1.0
                row_counter[matrix_entry[0]] += 1
                col_counter[matrix_entry[1]] += 1
    return mask

valid_2d_patterns = {}
def compute_valid_2d_patterns(m,n):
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
    patterns = compute_valid_2d_patterns(m, n)

    mat_flattern, shape = reshape_2d(mat, m)
    mask_flattern = np.ones_like(mat_flattern).reshape(-1, m, m)
    pmax = np.argmax(np.matmul(mat_flattern, patterns.reshape(patterns.shape[0],m*m).T), axis=1)

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

def create_mask(tensor, func_name="get_mask_2d_greedy", m=4, n=2):
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
        t = t.reshape(shape[0]*shape[1], shape[2])
        mask = func(t, m, n)
        return mask.reshape(shape).astype(ttype)
    # 4d-tensor conv (out, in, h, w) -> (out, in*h*w) in GemmConvKernel Op
    elif len(shape) == 4:
        t = t.reshape(shape[0], shape[1]*shape[2]*shape[3])
        mask = func(t, m, n)
        return mask.reshape(shape).astype(ttype)

def check_sparsity(tensor, func_name="check_mask_2d", m=4, n=2):
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
        t = t.reshape(shape[0]*shape[1], shape[2])
        return func(t, m, n)
    # 4d-tensor conv (out, in, h, w) -> (out, in*h*w) in GemmConvKernel Op
    elif len(shape) == 4:
        t = t.reshape(shape[0], shape[1]*shape[2]*shape[3])
        return func(t, m, n)

    return False

if __name__ == "__main__":

    x = np.array([3.00, 6.00, 7.00, 5.00, 3.00, 5.00, 6.00, 2.00, 9.00, 1.00, 2.00, 7.00, 0.00, 9.00, 3.00, 6.00, 0.00, 6.00, 2.00, 6.00, 1.00, 8.00, 7.00, 9.00, 2.00, 0.00, 1.00, 3.00, 7.00, 5.00, 9.00, 2.00, 2.00, 8.00, 9.00, 7.00, 3.00, 6.00, 1.00, 2.00, 9.00, 3.00, 1.00, 9.00, 4.00, 7.00, 8.00, 4.00, 5.00, 0.00, 3.00, 6.00, 1.00, 0.00, 6.00, 3.00, 2.00, 0.00, 6.00, 1.00, 5.00, 3.00, 4.00, 7.00, 6.00, 5.00, 5.00, 9.00, 3.00, 7.00, 4.00, 5.00, 2.00, 5.00, 4.00, 7.00, 4.00, 4.00, 3.00, 0.00, 7.00, 8.00, 6.00, 8.00, 8.00, 4.00, 3.00, 1.00, 4.00, 9.00, 2.00, 0.00, 6.00, 8.00, 9.00, 2.00, 6.00, 3.00, 4.00, 9.00, 5.00, 0.00, 4.00, 8.00, 7.00, 1.00, 7.00, 2.00, 7.00, 2.00, 2.00, 6.00, 1.00, 0.00, 6.00, 0.00, 5.00, 9.00, 4.00, 9.00, 0.00, 9.00, 1.00, 7.00, 7.00, 1.00, 1.00, 5.00, 9.00, 7.00, 5.00, 6.00, 7.00, 3.00, 6.00, 5.00, 6.00, 3.00, 9.00, 4.00, 8.00, 1.00, 2.00, 9.00, 3.00, 9.00, 0.00, 8.00, 8.00, 5.00, 0.00, 9.00, 6.00, 3.00, 8.00, 5.00, 6.00, 1.00, 1.00, 5.00, 9.00, 8.00, 4.00, 7.00, 1.00, 0.00, 3.00, 0.00,4.00, 4.00, 2.00, 1.00, 7.00, 6.00, 3.00, 1.00, 7.00, 5.00, 9.00, 6.00, 2.00, 1.00, 7.00, 8.00, 5.00, 7.00, 4.00, 1.00, 8.00, 5.00, 9.00, 7.00, 5.00, 3.00, 8.00, 8.00, 3.00, 1.00, 8.00, 9.00, 6.00, 4.00, 3.00, 3.00, 3.00, 8.00, 6.00, 0.00, 4.00, 8.00, 8.00, 7.00, 9.00, 7.00, 5.00, 6.00, 4.00, 3.00, 0.00, 2.00, 0.00, 9.00, 2.00, 5.00, 4.00, 0.00, 5.00, 9.00, 4.00, 6.00, 9.00, 2.00, 2.00, 4.00, 7.00, 7.00, 5.00, 4.00, 8.00, 1.00, 2.00, 8.00, 9.00, 3.00, 6.00, 8.00, 0.00, 2.00, 1.00, 0.00, 5.00, 0.00, 1.00, 0.00, 8.00, 5.00])
    x_ref = np.array([0.00, 6.00, 7.00, 0.00, 0.00, 5.00, 6.00, 0.00, 9.00, 0.00, 0.00, 7.00, 0.00, 9.00, 0.00, 6.00, 0.00, 6.00, 0.00, 6.00, 0.00, 8.00, 0.00, 9.00, 2.00, 0.00, 0.00, 3.00, 7.00, 0.00, 9.00, 0.00, 0.00, 8.00, 9.00, 0.00, 3.00, 6.00, 0.00, 0.00, 9.00, 0.00, 0.00, 9.00, 0.00, 7.00, 8.00, 0.00, 5.00, 0.00, 0.00, 6.00, 0.00, 0.00, 6.00, 3.00, 2.00, 0.00, 6.00, 0.00, 5.00, 0.00, 0.00, 7.00, 6.00, 0.00, 0.00, 9.00, 0.00, 7.00, 0.00, 5.00, 0.00, 5.00, 0.00, 7.00, 4.00, 4.00, 0.00, 0.00, 0.00, 8.00, 0.00, 8.00, 8.00, 4.00, 0.00, 0.00, 4.00, 9.00, 0.00, 0.00, 0.00, 8.00, 9.00, 0.00, 6.00, 0.00, 0.00, 9.00, 5.00, 0.00, 0.00, 8.00, 7.00, 0.00, 7.00, 0.00, 7.00, 0.00, 0.00, 6.00, 1.00, 0.00, 6.00, 0.00, 0.00, 9.00, 0.00, 9.00, 0.00, 9.00, 0.00, 7.00, 7.00, 0.00, 0.00, 5.00, 9.00, 7.00, 0.00, 0.00, 7.00, 0.00, 6.00, 0.00, 6.00, 0.00, 9.00, 0.00, 8.00, 0.00, 0.00, 9.00, 0.00, 9.00, 0.00, 8.00, 8.00, 0.00, 0.00, 9.00, 6.00, 0.00, 8.00, 0.00, 6.00, 0.00, 0.00, 5.00, 9.00, 8.00, 0.00, 0.00, 1.00, 0.00, 3.00, 0.00, 4.00, 4.00, 0.00, 0.00, 7.00, 6.00, 0.00, 0.00, 7.00, 0.00, 9.00, 0.00, 0.00, 0.00, 7.00, 8.00, 5.00, 7.00, 0.00, 0.00, 8.00, 0.00, 9.00, 0.00, 0.00, 0.00, 8.00, 8.00, 0.00, 0.00, 8.00, 9.00, 6.00, 4.00, 0.00, 0.00, 0.00, 8.00, 6.00, 0.00, 0.00, 8.00, 8.00, 0.00, 9.00, 7.00, 0.00, 0.00, 4.00, 3.00, 0.00, 0.00, 0.00, 9.00, 0.00, 5.00, 0.00, 0.00, 5.00, 9.00, 0.00, 6.00, 9.00, 0.00, 0.00, 0.00, 7.00, 7.00, 5.00, 0.00, 8.00, 0.00, 0.00, 8.00, 9.00, 0.00, 6.00, 8.00, 0.00, 0.00, 1.00, 0.00, 5.00, 0.00, 0.00, 0.00, 8.00, 5.00])
    x_2d = x.reshape((32, 8))

    print("Density of X: ", "density: {:.2f}".format(density(x)*100))
    print("Density of X_ref: ", "density: {:.2f}".format(density(x_ref)*100))

    mask = get_mask_1d_greedy(x_2d, 4, 2)
    x_pruned = np.multiply(x_2d, mask)
    print("Checking non_pruned X_1D:", check_mask_1d(x_2d, 4, 2))
    print("Checking pruned X_1D:", check_mask_1d(x_pruned, 4, 2))

    mask = get_mask_1d_best(x_2d, 4, 2)
    x_pruned = np.multiply(x_2d, mask)
    print("Checking non_pruned X_1D:", check_mask_1d(x_2d, 4, 2))
    print("Checking pruned X_1D (BEST):", check_mask_1d(x_pruned, 4, 2))

    x_not_in_4 = np.random.randint(5, size=(11, 11))
    mask = get_mask_1d_greedy(x_not_in_4, 4, 2)
    x_pruned = np.multiply(x_not_in_4, mask)
    print("Checking non_pruned X_1D:", check_mask_1d(x_not_in_4, 4, 2))
    print("Checking pruned X_1D:", check_mask_1d(x_pruned, 4, 2))

    mask_2d = get_mask_2d_greedy(x_2d, 4, 2)
    x_2d_pruned = np.multiply(x_2d, mask_2d)
    check_mask_2d(x_2d_pruned, 4, 2)
    print("Checking non_pruned X_2D:", check_mask_2d(x_2d.transpose(), 4, 2))
    print("Checking pruned X_2D:", check_mask_2d(x_2d_pruned, 4, 2))

    mask_2d = get_mask_2d_best(x_2d, 4, 2)
    x_2d_pruned = np.multiply(x_2d, mask_2d)
    check_mask_2d(x_2d_pruned, 4, 2)
    print("Checking non_pruned X_2D:", check_mask_2d(x_2d.transpose(), 4, 2))
    print("Checking pruned X_2D (BEST):", check_mask_2d(x_2d_pruned, 4, 2))

    created_mask_1d = create_mask(x, func_name="get_mask_1d_greedy")
    created_mask_2d = create_mask(x_2d, func_name="get_mask_2d_greedy")
    print("Checking created_mask 1D:", check_mask_1d(created_mask_1d, 4, 2))
    print("Checking created_mask 2D:", check_mask_2d(created_mask_2d, 4, 2))

    pass_verification = True
    for i in range(30):
        x_2d = np.random.rand(16, 16)
        mask_greedy = get_mask_1d_greedy(x_2d, 4, 2)
        mask_best = get_mask_1d_best(x_2d, 4, 2)
        if (mask_best == mask_greedy).all() == False:
            print(i, "Mask1D greedy != best")
            pass_verification = False

        mask2d_greedy = get_mask_2d_greedy(x_2d, 4, 2)
        mask2d_best = get_mask_2d_best(x_2d, 4, 2)
        x_pruned_greedy = np.multiply(x_2d, mask2d_greedy)
        x_pruned_best = np.multiply(x_2d, mask2d_best)
        if (x_pruned_best.sum() < x_pruned_greedy.sum()):
            print(i, "Mask2D greedy > best")
            pass_verification = False

        if not check_mask_1d(mask_best, 4, 2):
            print(i, "1D best donot pass checking!")
        if not check_mask_2d(mask2d_best, 4, 2):
            print(i, "2D best donot pass checking!")
    print("Best masking checking:", pass_verification)


