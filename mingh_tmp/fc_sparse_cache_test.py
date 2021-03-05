import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.sparsity import ASPHelper, fc_sparse
import numpy as np
import time

def build_model():
    img = fluid.data(name='img', shape=[None, 1, 28, 28], dtype='float16')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    hidden = fluid.layers.fc(input=img, size=1536, act='relu')
    hidden = fluid.layers.fc(input=hidden, size=1536, act='relu')
    hidden = fluid.layers.fc(input=hidden, size=1536, act='relu')
    hidden = fluid.layers.fc(input=hidden, size=1536, act='relu')
    # hidden = fc_sparse(input=hidden, size=1536, act='relu', enable_cache=True)
    # hidden = fc_sparse(input=hidden, size=1536, act='relu', enable_cache=True)
    # hidden = fc_sparse(input=hidden, size=1536, act='relu', enable_cache=True)
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return img, label, prediction

def test(test_program, test_reader, test_feader, exe, fetch_list):
    test_acc_set = []
    test_avg_loss_set = []
    for test_data in test_reader():
        acc_np, avg_loss_np = exe.run(
                program=test_program,
                feed=test_feader.feed(test_data),
                fetch_list=fetch_list)
        test_acc_set.append(float(acc_np))
        test_avg_loss_set.append(float(avg_loss_np))

    return np.array(test_acc_set).mean(), np.array(test_avg_loss_set).mean()

def main():
    BATCH_SIZE = 512

    test_program = fluid.Program()
    start_prog = fluid.Program()

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    with fluid.program_guard(test_program, start_prog):
        img, label, predict = build_model()
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(cost)
        acc = fluid.layers.accuracy(input=predict, label=label)

    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    exe.run(start_prog)

    print("-------------------- Sparsity Pruning --------------------")
    ASPHelper.prune_model(test_program, start_prog, place, with_mask=False)
    test_acc_val_mean, test_avg_loss_val_mean = test(test_program, test_reader, 
                                                     feeder, exe, [acc, avg_cost])
    print("-------------------- Warmup --------------------")
    for _ in range(20):
        test_acc_val_mean, test_avg_loss_val_mean = test(test_program, test_reader, 
                                                         feeder, exe, [acc, avg_cost])
    print("-------------------- Start Pretraining --------------------")
    start_t =time.time()
    for _ in range(20):
        test_acc_val_mean, test_avg_loss_val_mean = test(test_program, test_reader, 
                                                         feeder, exe, [acc, avg_cost])
    print((time.time()-start_t)*1000/20)

if __name__ == "__main__":
    paddle.enable_static()
    main()