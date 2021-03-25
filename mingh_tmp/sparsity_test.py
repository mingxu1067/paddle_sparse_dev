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
            name='test_data', shape=[None, 4, 32], dtype='float16')
        fc = fluid.layers.fc(input=input_data, num_flatten_dims=-1, size=16, act=None)
        fc_sparse = sparsity.fc_sparse(input=input_data, num_flatten_dims=-1, size=16, act=None)

    for param in train_program.global_block().all_parameters():
        print(param.name)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[input_data])

    exe.run(startup_prog)

    sparsity.ASPHelper.prune_model(train_program, startup_prog, place, func_name="get_mask_1d_greedy", with_mask=False)

    fcw = fluid.global_scope().find_var('fc_0.w_0')
    fcsw = fluid.global_scope().find_var('fc_1.w_0')
    fcw_param = fcw.get_tensor()
    fcsw_param = fcsw.get_tensor()

    fcw_array = np.array(fcw_param)
    fcsw_param.set(fcw_array, place)

    fcb = fluid.global_scope().find_var('fc_0.b_0')
    fcsb = fluid.global_scope().find_var('fc_1.b_0')

    fcb_param = fcb.get_tensor()
    fcsb_param = fcsb.get_tensor()
    fcb_array = np.array(fcb_param)
    fcsb_param.set(fcb_array, place)

    for _ in range(20):
        data = np.random.randint(9, size=(8, 4, 32))
        fc_result, fc_sparse_result = exe.run(
            train_program, feed=feeder.feed([(data,)]), fetch_list=[fc, fc_sparse])
        print(fc_result.shape, fc_sparse_result.shape)

        for i in range(8):
            for j in range(4):
                for k in range(16):
                    if fc_result[i][j][k] != fc_sparse_result[i][j][k]:
                        print(i, j, "::", fc_result[i][j][k], "-" ,fc_sparse_result[i][j][k])

if __name__ == "__main__":
    main()