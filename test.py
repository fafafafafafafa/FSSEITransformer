import os
import time
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import initializion
import MyDataset
from model import transformer
import torchinfo
import sklearn.linear_model
import csv


def test_one_epoch(support, query, support_labels, query_labels, encoder, loss_fn):
    """
    单次epoch训练
    参数说明:
        输入:
            my_model: 训练的网络模型
            loader_train: 数据加载器
            criterion: 损失函数
            optimizer: 优化器
        输出:
            train_loss: 单次训练的误差
    """
    encoder.eval()
    with torch.no_grad():
        N_way = query.shape[0]
        N_query = query.shape[1]
        K_shot = support.shape[1]
        seq_len = support.shape[2]
        val_len = support.shape[3]
        # 预测的正确数
        running_correct = 0
        # signals = torch.unsqueeze(signals, 1)   # ([400, 1, 2, 128])

        # signals = torch.transpose(signals, 1, 2)  # ([400, 128, 2])
        # labels = labels.long()
        support = torch.from_numpy(support)
        support_labels = torch.from_numpy(support_labels)
        query = torch.from_numpy(query)
        query_labels = torch.from_numpy(query_labels)
        if torch.cuda.is_available():
            # 获取输入数据X和标签Y并拷贝到GPU上
            # 注意有许多教程再这里使用Variable类来包裹数据以达到自动求梯度的目的，如下
            # Variable(imgs)
            # 但是再pytorch4.0之后已经不推荐使用Variable类，Variable和tensor融合到了一起
            # 因此我们这里不需要用Variable
            # 若我们的某个tensor变量需要求梯度，可以用将其属性requires_grad=True,默认值为False
            # 如，若X和y需要求梯度可设置X.requires_grad=True，y.requires_grad=True
            # 但这里我们的X和y不需要进行更新，因此也不用求梯度
            # X, y = signals.cuda(), labels.cuda()
            support = support.cuda()
            support_labels = support_labels.cuda()
            query = query.cuda()
            query_labels = query_labels.cuda()

        # 将输入X送入模型进行训练
        # feature_s.Size([N_way, feature_len])
        # feature_q.Size([N_query, feature_len])
        # print(support.size())

        support = torch.reshape(support, (-1, seq_len, val_len))
        query = torch.reshape(query, (-1, seq_len, val_len))
        support_labels = torch.reshape(support_labels, (-1, 1)).squeeze(1)
        query_labels = torch.reshape(query_labels, (-1, 1)).squeeze(1)

        feature_s = encoder(support).cpu().numpy()
        feature_q = encoder(query).cpu().numpy()
        support_labels = support_labels.cpu()
        query_labels = query_labels.cpu()
        # linear regression

        classifier = sklearn.linear_model.LogisticRegression(penalty='l2', random_state=0, C=1.0,
                                                             solver='lbfgs', max_iter=1000,
                                                             multi_class='multinomial')
        classifier.fit(feature_s, support_labels)
        pred_query = classifier.predict(feature_q)

        # torch.max()返回两个字，其一是最大值，其二是最大值对应的索引值

        # 计算一个批次的预测正确数
        # temp, y = torch.max(y.detach(), dim=1)
        running_correct = torch.sum(query_labels.eq(torch.tensor(pred_query)))
        acc = sklearn.metrics.accuracy_score(query_labels, pred_query)
        # 打印训练结果

        # acc = 100 * running_correct.item() / (N_way*N_query)
        print(
            'Test Loss:{loss:.4f},Test Acc:{correct}/{lens}={acc:.4f}'.format(
                loss=0,
                correct=running_correct,
                lens=N_way*N_query,
                acc=acc
            ))
    return 0, acc


def normalize_data(X):

    min_value = -0.0014277228
    max_value = 0.00142765778
    min_value = X.min()
    max_value = X.max()

    X = (X - min_value) / (max_value - min_value)
    X = np.float32(X)
    return X


def test():
    args = initializion.init()
    # 获取测试集
    # test_dataset = MyDataset.get_test_dataset(10, normalize_data)
    # test_dataset = MyDataset.get_train_dataset(90, normalize_data)
    # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if args.model_name == 'conformer':
        encoder = transformer.Conformer(args.emb_size, args.heads, args.train_val_dataset_num, mode=False)
    else:
        raise ValueError('model is None!')
    torchinfo.summary(encoder)
    print(torch.cuda.is_available())

    if args.loss_fn_name == 'cross_entropy_loss':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError('loss_fn is None!')

    # my_model.load_state_dict(torch.load(args.early_stopping_dir_path+args.model_name))
    encoder.load_state_dict(torch.load(args.early_stopping_dir_path
                                        +args.model_name+'/'+'checkpoint.pt')["model_state_dict"])

    if torch.cuda.is_available():
        encoder = encoder.cuda()

    encoder.eval()

    N_ways = [20]
    K_shots = [1, 5]
    N_query = 5

    epochs = 300

    for N_way in N_ways:
        test_data_classes = N_way
        X_train, Y_train = MyDataset.get_test_support_dataset(N_way, normalize_data)
        X_test, Y_test = MyDataset.get_test_query_dataset(N_way, normalize_data)
        # print(X_train.shape)
        seq_len = X_train.shape[1]
        val_len = X_train.shape[2]
        for K_shot in K_shots:
            acc_test = []
            acc_test = np.zeros([epochs, 1], dtype=np.float32)
            for epoch in range(epochs):

                print('Epoch {}/{}'.format(epoch, epochs))
                # 随机抽取 N 类
                ep_classes = np.random.permutation(test_data_classes)[:N_way]
                support = np.zeros([N_way, K_shot, seq_len, val_len], dtype=np.float32)
                query = np.zeros([N_way, N_query, seq_len, val_len], dtype=np.float32)
                for i in range(N_way):
                    index_class = np.where(Y_train == ep_classes[i])[0]
                    index_support = np.random.permutation(index_class)[:K_shot]
                    # index_query_all = np.array(list(set(index_class)-set(index_support)))
                    # index_query = np.random.permutation(index_query_all)[:N_query]
                    index_class = np.where(Y_test == ep_classes[i])[0]
                    index_query = np.random.permutation(index_class)[:N_query]

                    support[i, :, :, :] = X_train[index_support, :, :]
                    query[i, :, :, :] = X_test[index_query, :, :]
                support_labels = np.tile(np.arange(N_way)[:, np.newaxis], (1, K_shot)).astype(np.uint8)
                query_labels = np.tile(np.arange(N_way)[:, np.newaxis], (1, N_query)).astype(np.uint8)

                start_time = time.time()
                test_loss, acc = test_one_epoch(support, query, support_labels, query_labels, encoder, loss_fn)
                t = time.time() - start_time
                print('test_time :', t)
                acc_test[epoch, :] = acc
                '''data_loss.append([epoch, test_loss])
                acc_test[epoch, :] = acc
                time_test[epoch, :] = t'''

            test_data_dir_path = args.data_dir_path + args.model_name + '/' + \
                                 '{}way_{}shot_{}query'.format(N_way, K_shot, N_query)
            if not os.path.exists(test_data_dir_path):
                os.makedirs(test_data_dir_path)

            csvfile = open(test_data_dir_path + '/' + 'acc.csv', "w", newline="")  # w覆盖， a追加
            writer = csv.writer(csvfile)
            writer.writerow(['aver_acc', 'min_acc', 'max_acc', 'std_acc', 'aver_time'])
            writer.writerows([[np.mean(acc_test), np.min(acc_test), np.max(acc_test), np.std(acc_test)]])
            csvfile.close()


if __name__ == '__main__':
    test()



