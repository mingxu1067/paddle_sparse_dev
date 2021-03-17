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
        fc_sparse = sparsity.fc_sparse(input=input_data, num_flatten_dims=-1, size=32, act=None)

    for param in train_program.global_block().all_parameters():
        print(param.name)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[input_data, label])

    exe.run(startup_prog)

    fcw = fluid.global_scope().find_var('fc_0.w_0')
    fcsw = fluid.global_scope().find_var('fc_1.w_0')
    fcw_param = fcw.get_tensor()
    fcsw_param = fcsw.get_tensor()

    fcw_array = np.array(fcw_param)
    sparse_mask = sparsity.create_mask(fcw_array)
    pruned_w = np.multiply(fcw_array, sparse_mask)
    assert sparsity.check_mask_2d(pruned_w, m=4, n=2), "Pruning FC weight matrix failure!!!"

    fcw_param.set(pruned_w, place)
    fcsw_param.set(pruned_w, place)

    dev_ctx = paddle.core.DeviceContext.create(place)
    fcs_compressed_param = paddle.core.compress_parameter(dev_ctx, fcsw_param, 32, 32, 32, 32, 32, 32, True)


if __name__ == "__main__":
    main()