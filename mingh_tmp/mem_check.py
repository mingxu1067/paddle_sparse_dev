import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.sparsity import ASPHelper, check_mask_2d
import numpy as np

def build_model():
    img = fluid.data(name='img', shape=[None, 1, 28, 28], dtype='float16')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    hidden = fluid.layers.fc(input=img, size=160, act='relu')
    hidden = fluid.layers.fc_sparse(input=hidden, size=160, act='relu')
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
        break

    return np.array(test_acc_set).mean(), np.array(test_avg_loss_set).mean()

def checking_var(src, dec):
    for name in src:
        src_v = src[name]
        dec_v = dec[name]
        if src_v.shape != dec_v.shape:
            print("Shape not match", src_v.shape, dec_v.shape)
        src_v.flatten()
        dec_v.flatten()
        for i in range(len(src_v)):
            if src_v[i] != dec_v[i]:
                print("{} Not match at {}.".format(name, i), src_v[i], dec_v[i])

def main():
    BATCH_SIZE = 64
    EPOCHS = 1

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
        ASPHelper.minimize(avg_cost, optimizer, place, train_prog, start_prog)

    train_reader = paddle.batch(paddle.reader.shuffle(
                                paddle.dataset.mnist.train(), buf_size=500),
                                batch_size=BATCH_SIZE)

    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    exe.run(start_prog)

    # temp_param_map = {}
    # for param in train_prog.global_block().all_parameters():
    #     val = np.array(fluid.global_scope().find_var(param.name).get_tensor())
    #     temp_param_map[param.name] = val

    print("-------------------- Sparsity Pruning --------------------")
    mask_map = ASPHelper.prune_model(train_prog, start_prog, place)
    test_acc_val_mean, test_avg_loss_val_mean = test(test_program, test_reader,
                                                     feeder, exe, [acc, avg_cost])
    print("Sparse Model: Loss {:.3f} - Accuracy: {:.3f}".format(
        test_avg_loss_val_mean, test_acc_val_mean
    ))

    # after_param_map = {}
    # for param in train_prog.global_block().all_parameters():
    #     val = np.array(fluid.global_scope().find_var(param.name).get_tensor())
    #     after_param_map[param.name] = val
    # checking_var(temp_param_map, after_param_map)

    print("Mask Sparsity Check:", check_mask_2d(mask_map['fc_1.w_0'], 4, 2))
    # print(np.array(fluid.global_scope().find_var('fc_1.w_0').get_tensor()))
    print("Param Sparsity Check After one test:", check_mask_2d(np.array(fluid.global_scope().find_var('fc_1.w_0').get_tensor()), 4, 2))
    maske_param = np.array(fluid.global_scope().find_var('fc_1.w_0_asp_mask').get_tensor())
    print("Mask Sparsity Check After one test:", check_mask_2d(maske_param, 4, 2))

if __name__ == "__main__":
    paddle.enable_static()
    main()