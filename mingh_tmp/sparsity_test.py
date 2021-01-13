import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib import sparsity
import numpy as np

paddle.enable_static()

def main():
    train_program = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(train_program, startup_prog):
        input_data = fluid.layers.data(
            name='test_data', shape=[None, 32], dtype='float16')
        fc = fluid.layers.fc(input=input_data, size=8, act=None)
        fc_sparse = fluid.layers.fc_sparse(input=input_data, size=8, act=None)

    for param in train_program.global_block().all_parameters():
        print(param.name)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[input_data])

    exe.run(startup_prog)

    fcw = fluid.global_scope().find_var('fc_0.w_0')
    fcsw = fluid.global_scope().find_var('fc_sparse_0.w_0')
    fcw_param = fcw.get_tensor()
    fcsw_param = fcsw.get_tensor()

    # fcw_array = np.transpose(np.array(fcw_param)).flatten()
    # pruned_w = np.transpose(sparsity.prune_matrix(fcw_array).reshape((8,32)))
    fcw_array = np.array(fcw_param)
    sparse_mask = sparsity.create_mask(fcw_array)
    pruned_w = np.multiply(fcw_array, sparse_mask)
    assert sparsity.check_mask_2d(pruned_w, m=4, n=2), "Pruning FC weight matrix failure!!!"

    fcw_param.set(pruned_w, place)
    fcsw_param.set(pruned_w, place)

    fcb = fluid.global_scope().find_var('fc_0.b_0')
    fcsb = fluid.global_scope().find_var('fc_sparse_0.b_0')

    fcb_param = fcb.get_tensor()
    fcsb_param = fcsb.get_tensor()
    fcb_array = np.array(fcb_param)
    fcsb_param.set(fcb_array, place)

    data = np.random.randint(9, size=(8, 32))
    fc_result, fc_sparse_result = exe.run(
        train_program, feed=feeder.feed([(data,)]), fetch_list=[fc, fc_sparse])
    print(fc_result.shape, fc_sparse_result.shape)

    for i in range(8):
        for j in range(8):
            if fc_result[i][j] != fc_sparse_result[i][j]:
                print(i, j, "::", fc_result[i][j], "-" ,fc_sparse_result[i][j])

if __name__ == "__main__":
    main()