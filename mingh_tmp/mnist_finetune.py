import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.sparsity import ASPHelper, check_mask_2d, fc_sparse
import numpy as np

def build_model():
    img = fluid.data(name='img', shape=[None, 1, 28, 28], dtype='float16')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    hidden = fluid.layers.fc(input=img, size=160, act='relu')
    hidden = fc_sparse(input=hidden, size=160, act='relu')
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
    BATCH_SIZE = 64
    EPOCHS = 5
    SAVE_DIR = "mnist_asp_test"

    train_prog = fluid.Program()
    start_prog = fluid.Program()

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    with fluid.program_guard(train_prog, start_prog):
        img, label, predict = build_model()
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(cost)
        acc = fluid.layers.accuracy(input=predict, label=label)

    test_program = train_prog.clone(for_test=True)

    with fluid.program_guard(train_prog, start_prog):
        optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        ASPHelper.minimize(avg_cost, optimizer, train_prog, start_prog)

    train_reader = paddle.batch(paddle.reader.shuffle(
                                paddle.dataset.mnist.train(), buf_size=500),
                                batch_size=BATCH_SIZE)

    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    exe.run(start_prog)

    print("-------------------- Loading model --------------------")
    test_acc_val_mean, test_avg_loss_val_mean = test(test_program, test_reader, 
                                                     feeder, exe, [acc, avg_cost])
    print("Before Loaded Model: Loss {:.3f} - Accuracy: {:.3f}".format(
        test_avg_loss_val_mean, test_acc_val_mean
    ))
    fluid.io.load_vars(exe, SAVE_DIR, train_prog, vars=ASPHelper.get_vars(train_prog))
    print("Loaded model from", SAVE_DIR)

    print("-------------------- Sparsity Pruning --------------------")
    ASPHelper.prune_model(train_prog, start_prog, place, func_name="get_mask_2d_greedy")
    test_acc_val_mean, test_avg_loss_val_mean = test(test_program, test_reader, 
                                                     feeder, exe, [acc, avg_cost])
    print("Sparse Model: Loss {:.3f} - Accuracy: {:.3f}".format(
        test_avg_loss_val_mean, test_acc_val_mean
    ))

    print("-------------------- Sparsity Finetuning --------------------")
    for epoch_id in range(EPOCHS):
        for batch_id, data in enumerate(train_reader()):
            metrics = exe.run(train_prog,
                          feed=feeder.feed(data),
                          fetch_list=[avg_cost, acc])

            if batch_id % 100 == 0:
                print("Epoch {:3d} - Batch {:3}:\tTraining Loss {:.3f} - Traing Acc {:.3f}".format(
                        epoch_id, batch_id, metrics[0].mean(), metrics[1].mean()))

        test_acc_val_mean, test_avg_loss_val_mean = test(test_program, test_reader, 
                                                         feeder, exe, [acc, avg_cost])

        print("Epoch {:3d}:\tTraining Loss {:.3f} - Traing Acc {:.3f}".format(
                epoch_id, metrics[0].mean(), metrics[1].mean()))
        print("            \tTesting Loss {:.3f} - Testing Acc {:.3f}".format(
                test_avg_loss_val_mean, test_acc_val_mean))

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
    paddle.enable_static()
    main()