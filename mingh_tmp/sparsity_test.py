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
            name='test_data', shape=[None, 8], dtype='float16')
        fc = fluid.layers.fc(input=input_data, size=64, act=None)
        fc_sparse = fluid.layers.fc_sparse(input=input_data, size=64, act=None)

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

    fcw_array = np.array(fcw_param).flatten()
    pruned_w = sparsity.prune_matrix(fcw_array).reshape((8,64))

    fcw_param.set(pruned_w, place)
    fcsw_param.set(pruned_w, place)

    data = np.random.randint(9, size=(32, 8))
    fc_result, fc_sparse_result = exe.run(
        train_program, feed=feeder.feed([(data,)]), fetch_list=[fc, fc_sparse])
    print(fc_result.shape, fc_sparse_result.shape)


    for i in range(32):
        for j in range(64):
            if fc_result[i][j] != fc_sparse_result[i][j]:
                print(i, j, "::", fc_result[i][j], "-" ,fc_sparse_result[i][j])

if __name__ == "__main__":
    main()