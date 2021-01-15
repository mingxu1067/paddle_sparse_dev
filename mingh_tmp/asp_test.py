import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.sparsity import ASPHelper, check_mask_2d
import numpy as np

paddle.enable_static()

def network():
    input_data = fluid.layers.data(
            name='test_data', shape=[None, 32], dtype='float32')
    fc = fluid.layers.fc(input=input_data, size=8, act=None)
    fc_sparse = fluid.layers.fc(input=input_data, size=8, act=None)
    tmp_loss = fluid.layers.mean(fluid.layers.square_error_cost(fc, fc_sparse))
    return input_data, tmp_loss

def main():

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    train_program = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(train_program, startup_prog):
        sgd_input_data, sgd_loss = network()
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
        # _, param_grads = sgd_optimizer.minimize(sgd_loss)
        ASPHelper.minimize(sgd_loss, sgd_optimizer, place, train_program, startup_prog)

    exe.run(startup_prog)
    # ASPHelper.initialize_asp_training(train_program, startup_prog, exe)
    # ASPHelper.insert_grads_mask_ops(train_program, startup_prog,
    #                             sgd_optimizer.type, param_grads)
    ASPHelper.prune_model(train_program, startup_prog, place)

    sgd_feeder = fluid.DataFeeder(place=place, feed_list=[sgd_input_data])

    for _ in range(5):
        data = np.random.random_sample((8, 32))
        exe.run(train_program, feed=sgd_feeder.feed([(data,)]))

    for param in train_program.global_block().all_parameters():
         if ASPHelper.is_supported_layer(param.name):
            mat = np.array(fluid.global_scope().find_var(param.name).get_tensor())
            valid = check_mask_2d(mat, 4, 2)
            if valid:
                print(param.name, "Sparsity Validation:", valid)
            else:
                print("!!!!!!!!!!", param.name, "Sparsity Validation:", valid)

if __name__ == "__main__":
    main()