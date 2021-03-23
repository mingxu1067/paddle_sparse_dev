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
        fc_sparse_compressed = sparsity.fc_sparse(input=input_data, num_flatten_dims=-1, size=32, act=None,
                                                  is_param_compressed=True)
        fc_sparse_compressed2 = sparsity.fc_sparse(input=input_data, num_flatten_dims=-1, size=32, act=None,
                                                  is_param_compressed=True)

    for param in train_program.global_block().all_parameters():
        print(param.name)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[input_data])

    for idx in range(1, 17):

        exe.run(startup_prog)

        fcscw = fluid.global_scope().find_var('fc_0.w_0')
        fcscw2 = fluid.global_scope().find_var('fc_1.w_0')

        fcscw_param = fcscw.get_tensor()
        fcscw_param2 = fcscw2.get_tensor()

        fcw_array = np.array(fcscw_param)
        sparse_mask = sparsity.create_mask(fcw_array, func_name='get_mask_2d_greedy')
        pruned_w = np.multiply(fcw_array, sparse_mask)
        assert sparsity.check_mask_2d(pruned_w, m=4, n=2), "Pruning FC weight matrix failure!!!"

        fcscw_param.set(pruned_w, place)
        fcscw_param2.set(pruned_w, place)

        fluid.core.compress_parameter(place, fcscw_param, 32, 32, 32, 32, 32, 32, True)
        fluid.core.compress_parameter(place, fcscw_param2, 32, idx*32, 32, 32, 32, 32, True)


        data = np.random.randint(9, size=(32, 32))
        fc_sparse_compressed_result, fc_sparse_compressed2_result = exe.run(
                train_program, feed=feeder.feed([(data,)]), fetch_list=[fc_sparse_compressed, fc_sparse_compressed2])

        for i in range(32):
            for j in range(32):
                if fc_sparse_compressed_result[i][j] != fc_sparse_compressed2_result[i][j]:
                            print(i, j, "::", fc_sparse_compressed2_result[i][j], "-" ,fc_sparse_compressed2_result[i][j])


if __name__ == "__main__":
    main()
