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
        label = fluid.layers.data(
            name='test_label', shape=[None, 4, 32], dtype='float32')
        fc = fluid.layers.fc(input=input_data, num_flatten_dims=-1, size=32, act=None)
        fc_32= fluid.layers.cast(x=fc, dtype="float32")
        fc_loss = fluid.layers.mean(fluid.layers.square_error_cost(fc_32, label))

    with fluid.program_guard(train_program, startup_prog):
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
        sgd_optimizer.minimize(fc_loss)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[input_data, label])

    exe.run(startup_prog)

    fcw = fluid.global_scope().find_var('fc_0.w_0')
    fcw_param = fcw.get_tensor()

    fcw_array = np.array(fcw_param)
    sparse_mask = sparsity.create_mask(fcw_array, func_name='get_mask_2d_greedy')
    pruned_w = np.multiply(fcw_array, sparse_mask)
    assert sparsity.check_mask_2d(pruned_w, m=4, n=2), "Pruning FC weight matrix failure!!!"

    fcw_param.set(pruned_w, place)

    fcb = fluid.global_scope().find_var('fc_0.b_0')
    fcb_param = fcb.get_tensor()
    fcb_array = np.array(fcb_param)

    data = np.random.randint(9, size=(8, 4, 32))
    fc_result, fc_loss_result , fc_grad = exe.run(
        train_program, feed=feeder.feed([(data, data)]), fetch_list=[fc, fc_loss, 'fc_0.w_0@GRAD',])

    fcw_param.set(pruned_w, place)
    fcb_param.set(fcb_array, place)
    sparsity.ASPHelper.replace_dense_to_sparse_op(train_program)
    fc_sparse_result, fc_sparse_loss_result , fcs_grad = exe.run(
        train_program, feed=feeder.feed([(data, data)]), fetch_list=[fc, fc_loss, 'fc_0.w_0@GRAD'])

    print(fc_result.shape, fc_sparse_result.shape)
    print(fc_grad.shape, fcs_grad.shape)

    print("FC Loss: {:.3f}, FC_Sparse Loss: {:.3f}".format(fc_loss_result[0], fc_sparse_loss_result[0]))
    
    print("Checking forwarding results")
    is_pass = True
    for i in range(8):
        for j in range(4):
            for k in range(32):
                if fc_result[i][j][k] != fc_sparse_result[i][j][k]:
                    is_pass = False
                    print(i, j, "::", fc_result[i][j][k], "-" ,fc_sparse_result[i][j][k])
    print("Checking forwarding results ---> ", is_pass)

    print("Checking gradients")
    is_pass = True
    for i in range(32):
        for j in range(32):
            if fc_grad[i][j] != fcs_grad[i][j]:
                is_pass = False
                print(i, j, "::", fc_grad[i][j], "-" ,fcs_grad[i][j])
    print("Checking gradients results ---> ", is_pass)

if __name__ == "__main__":
    main()