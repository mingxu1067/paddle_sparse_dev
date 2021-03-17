import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib import sparsity, ASPHelper
import numpy as np

paddle.enable_static()

def main():
    train_program = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(train_program, startup_prog):
        input_data = fluid.layers.data(
            name='test_data', shape=[None, 10240], dtype='float16')
        # fc = input_data
        fc_sparse = input_data
        for _ in range(10):
            # fc = fluid.layers.fc(input=fc, size=10240, act=None)
            fc_sparse = sparsity.fc_sparse(input=fc_sparse, size=10240, act=None, enable_cache=True)

    for param in train_program.global_block().all_parameters():
        print(param.name)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[input_data])

    exe.run(startup_prog)

    # print("-------------------- Sparsity Pruning --------------------")
    ASPHelper.prune_model(train_program, startup_prog, place, with_mask=False)

    # SAVE_DIR="./fc_sparse_cache_model_10/"
    # fluid.io.save_params(exe, dirname=SAVE_DIR, main_program=train_program)
    # print("Saved dense model weights to", SAVE_DIR)

    # LOAD_DIR="/workspace/data/fc_sparse_cache_model_10"
    # print("-------------------- Loading model --------------------")
    # fluid.io.load_vars(exe, LOAD_DIR, test_program,  vars=ASPHelper.get_vars(test_program))
    # print("Loaded model from", LOAD_DIR)

    import time
    for _ in range(20):
        data = np.random.randint(9, size=(10240, 10240))
        # fc_time = time.time()
        # exe.run(
        #     train_program, feed=feeder.feed([(data,)]), fetch_list=[fc])
        # fc_time = time.time()-fc_time

        fc_sparse_time = time.time()
        exe.run(
            train_program, feed=feeder.feed([(data,)]), fetch_list=[fc_sparse])
        fc_sparse_time = time.time()-fc_sparse_time
        print(fc_time, fc_sparse_time)

    print("-------------------- Sparsity Checking --------------------")
    for param in train_prog.global_block().all_parameters():
         if ASPHelper.is_supported_layer(param.name):
            mat = np.array(fluid.global_scope().find_var(param.name).get_tensor())
            valid = check_mask_2d(mat, 4, 2)
            if valid:
                print(param.name, "Sparsity Validation:", valid)
            else:
                print("!!!!!!!!!!", param.name, "Sparsity Validation:", valid)

if __name__ == "__main__":
    main()