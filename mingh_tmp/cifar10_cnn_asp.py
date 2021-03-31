import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.sparsity import ASPHelper, check_sparsity
import numpy as np

def build_model():
    img = fluid.data(name='img', shape=[None, 3, 32, 32], dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    hidden = fluid.layers.conv2d(input=img, num_filters=32, filter_size=3, padding=2, act="relu")
    hidden = fluid.layers.conv2d(input=hidden, num_filters=32, filter_size=3, padding=0, act="relu")
    hidden = fluid.layers.pool2d(input = hidden,
                                    pool_size = 2,
                                    pool_type = "max",
                                    pool_stride = 1,
                                    global_pooling=False)
    hidden = fluid.layers.dropout(hidden, dropout_prob=0.25)

    hidden = fluid.layers.conv2d(input=hidden, num_filters=64, filter_size=3, padding=2, act="relu")
    hidden = fluid.layers.conv2d(input=hidden, num_filters=64, filter_size=3, padding=0, act="relu")
    hidden = fluid.layers.pool2d(input = hidden,
                                    pool_size = 2,
                                    pool_type = "max",
                                    pool_stride = 1,
                                    global_pooling=False)
    hidden = fluid.layers.dropout(hidden, dropout_prob=0.25)

    hidden = fluid.layers.fc(input=hidden, size=512, act='relu')
    hidden = fluid.layers.dropout(hidden, dropout_prob=0.5)
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
    BATCH_SIZE = 128
    EPOCHS = 60

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
        optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        ASPHelper.minimize(avg_cost, optimizer, train_prog, start_prog)
        # optimizer.minimize(avg_cost)

    train_reader = paddle.batch(paddle.reader.shuffle(
                                paddle.dataset.cifar.train10(), buf_size=600),
                                batch_size=BATCH_SIZE)

    test_reader = paddle.batch(paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    exe.run(start_prog)

    print("-------------------- Start training --------------------")
    for epoch_id in range(EPOCHS):
        for batch_id, data in enumerate(train_reader()):
            metrics = exe.run(train_prog,
                          feed=feeder.feed(data),
                          fetch_list=[avg_cost, acc])

        test_acc_val_mean, test_avg_loss_val_mean = test(test_program, test_reader, 
                                                         feeder, exe, [acc, avg_cost])

        print("Epoch {:3d}:\tTraining Loss {:.3f} - Traing Acc {:.3f}".format(
                epoch_id, metrics[0].mean(), metrics[1].mean()))
        print("           :\tTesting Loss {:.3f} - Testing Acc {:.3f}".format(
                test_avg_loss_val_mean, test_acc_val_mean))

    # SAVE_DIR = "cifar10_cnn_asp_test"
    # print("-------------------- Saving model --------------------")
    # fluid.io.save_inference_model(SAVE_DIR,
    #                                 ['img'], [predict], exe,
    #                                 train_prog,
    #                                 model_filename=None,
    #                                 params_filename=None)
    # print("Saved model to", SAVE_DIR)

    print("-------------------- Sparsity Pruning --------------------")
    ASPHelper.prune_model(train_prog, start_prog, place)
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
            valid = check_sparsity(mat.T, m=4, n=2)
            if valid:
                print(param.name, "Sparsity Validation:", valid)
            else:
                print("!!!!!!!!!!", param.name, "Sparsity Validation:", valid)

if __name__ == "__main__":
    paddle.enable_static()
    main()