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
        fc = fluid.layers.fc(input=input_data, num_flatten_dims=-1, size=32, act=None)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[input_data])

    exe.run(startup_prog)

    fcw = fluid.global_scope().find_var('fc_0.w_0')

    fcw_param = fcw.get_tensor()

    fcw_array = np.array(fcw_param)
    sparse_mask = sparsity.create_mask(fcw_array)
    pruned_w = np.multiply(fcw_array, sparse_mask)
    assert sparsity.check_mask_2d(pruned_w, m=4, n=2), "Pruning FC weight matrix failure!!!"

    fcw_param.set(pruned_w, place)

    data = np.random.randint(9, size=(32, 32))
    fc_result = exe.run(train_program, feed=feeder.feed([(data,)]), fetch_list=[fc])[0]

    sparsity.ASPHelper.compress_model(train_program, 32, place)
    sparsity.ASPHelper.replace_dense_to_sparse_op(train_program, is_compressed=True)

    fc_sparse_compress_result = exe.run(train_program, feed=feeder.feed([(data,)]), fetch_list=[fc])[0]

    for i in range(32):
        for j in range(32):
            if fc_result[i][j] != fc_sparse_compress_result[i][j]:
                        print(i, j, "::", fc_result[i][j], "-" ,fc_sparse_compress_result[i][j])


if __name__ == "__main__":
    main()
