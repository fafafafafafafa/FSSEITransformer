import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
import torchinfo
import MyDataset
from model import transformer
import pytorchtools

import os
import time
import csv
import initializion


def train_one_epoch(args, train_data_loader, my_model, loss_fn, optimizer):
    my_model.train()

    running_loss = 0.0
    running_correct = 0
    train_loss = 0

    for batch, (X, y) in enumerate(train_data_loader, 1):
        if torch.cuda.is_available():
            # 获取输入数据X和标签Y并拷贝到GPU上
            # 注意有许多教程再这里使用Variable类来包裹数据以达到自动求梯度的目的，如下
            # Variable(imgs)
            # 但是再pytorch4.0之后已经不推荐使用Variable类，Variable和tensor融合到了一起
            # 因此我们这里不需要用Variable
            # 若我们的某个tensor变量需要求梯度，可以用将其属性requires_grad=True,默认值为False
            # 如，若X和y需要求梯度可设置X.requires_grad=True，y.requires_grad=True
            # 但这里我们的X和y不需要进行更新，因此也不用求梯度

            signals, labels = X.cuda(), y.cuda()

        else:
            signals, labels = X, y

        pred = my_model(signals)

        loss = loss_fn(pred, labels.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算一个批次的损失值和
        running_loss += loss.detach().item()
        # 计算一个批次的预测正确数
        _, labels_pred = torch.max(pred.detach(), dim=1)
        # running_correct += torch.sum(labels_pred == labels)
        running_correct += torch.sum(labels_pred.eq(labels))

        # 打印训练结果

        if batch == len(train_data_loader):
            train_loss = running_loss / batch
            acc = 100 * running_correct / (args.batch_size * batch)
            print(
                'Batch {batch}/{iter_times},Train Loss:{loss:.4f},Train Acc:{correct}/{lens}={acc:.4f}%'.format(
                    batch=batch,
                    iter_times=len(train_data_loader),
                    loss=running_loss / batch,
                    correct=running_correct,
                    lens=args.batch_size * batch,
                    acc=acc
                ))

    return train_loss


def val_one_epoch(args, val_data_loader, my_model, loss_fn):
    my_model.eval()

    running_loss = 0.0
    running_correct = 0
    val_loss = 0
    acc = 0

    for batch, (X, y) in enumerate(val_data_loader, 1):
        if torch.cuda.is_available():
            # 获取输入数据X和标签Y并拷贝到GPU上
            # 若我们的某个tensor变量需要求梯度，可以用将其属性requires_grad=True,默认值为False
            # 如，若X和y需要求梯度可设置X.requires_grad=True，y.requires_grad=True
            # 但这里我们的X和y不需要进行更新，因此也不用求梯度

            signals, labels = X.cuda(), y.cuda()

        else:
            signals, labels = X, y
        with torch.no_grad():
            pred = my_model(signals)
            loss = loss_fn(pred, labels.long())

        # 计算一个批次的损失值和
        running_loss += loss.detach().item()
        # 计算一个批次的预测正确数
        _, labels_pred = torch.max(pred.detach(), dim=1)
        running_correct += torch.sum(labels_pred.eq(labels))

        # 打印训练结果
        if batch == len(val_data_loader):
            val_loss = running_loss / batch
            acc = 100 * running_correct / (args.batch_size * batch)
            print(
                'Batch {batch}/{iter_times},Val Loss:{loss:.4f},Va; Acc:{correct}/{lens}={acc:.4f}%'.format(
                    batch=batch,
                    iter_times=len(val_data_loader),
                    loss=running_loss / batch,
                    correct=running_correct,
                    lens=args.batch_size * batch,
                    acc=acc
                ))
    return val_loss, acc.detach().item()


def normalize_data(X):
    min_value = X.min()
    max_value = X.max()
    X = (X - min_value) / (max_value - min_value)
    X = np.float32(X)
    return X


def train():

    # 初始化
    args = initializion.init()
    print(args.train_val_dataset_num)
    if not os.path.exists(args.early_stopping_dir_path+args.model_name+'/'):
        os.makedirs(args.early_stopping_dir_path+args.model_name+'/')
    # 获取数据集
    train_dataset = MyDataset.get_train_dataset(args.train_val_dataset_num, normalize_data)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = MyDataset.get_val_dataset(args.train_val_dataset_num, normalize_data)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # model
    # my_model = None
    if args.model_name == 'conformer':
        my_model = transformer.Conformer(args.emb_size, args.heads, args.train_val_dataset_num, mode=True)
    else:
        raise ValueError('model is None!')
    torchinfo.summary(my_model)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        my_model = my_model.cuda()
    # optimizer
    # optimizer = None
    if args.optimizer_name == 'adam':
        optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    else:
        raise ValueError('optimizer is None!')
    if args.loss_fn_name == 'cross_entropy_loss':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError('loss_fn is None!')

    # 数据处理
    # early stopping
    early_stopping = pytorchtools.EarlyStopping(patience=args.early_stopping_patience, verbose=True,
                                                delta=args.early_stopping_delta,
                                                dir_path=args.early_stopping_dir_path+args.model_name+'/')
    # adjust_lr
    adjust_lr = pytorchtools.AdjustLearningRate(patience=args.adjust_lr_patience, verbose=True,
                                                delta=args.adjust_lr_delta,
                                                start_lr=args.adjust_lr_start_lr,
                                                min_lr=args.adjust_lr_min_lr,
                                                gamma=args.adjust_lr_gamma)
    # 训练
    data_loss = []  # 记录训练data_loss[epoch, train_loss, val_loss]
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs))
        start_time = time.time()
        train_loss = train_one_epoch(args, train_data_loader, my_model, loss_fn, optimizer)
        print('train_time :', time.time() - start_time)
        start_time = time.time()
        val_loss, val_acc = val_one_epoch(args, val_data_loader, my_model, loss_fn)
        print('val_time :', time.time() - start_time)
        data_loss.append([epoch, train_loss, val_loss, val_acc])
        adjust_lr(val_loss, optimizer)
        early_stopping(val_loss, my_model, epoch)
        if early_stopping.early_stop:
            print("early_stop")
            break

    # 保存数据
    data_whole_dir_path = args.data_dir_path+args.model_name+'/'
    # 检查路径是否存在
    if not os.path.exists(data_whole_dir_path):
        os.makedirs(data_whole_dir_path)
    # 保存 loss
    filename = 'train_val_loss.csv'
    csvfile = open(data_whole_dir_path+filename, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])
    writer.writerows(data_loss)
    csvfile.close()

    # 保存 hyper parameters
    filename = 'train_hyper_parameters.csv'
    csvfile = open(data_whole_dir_path + filename, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['batch_size', 'lr', 'epochs'])
    writer.writerow([args.batch_size, args.lr, args.epochs])

    writer.writerow(['model_name', 'optimizer_name', 'loss_fn_name'])
    writer.writerow([args.model_name, args.optimizer_name, args.loss_fn_name])

    writer.writerow(['early_stopping_patience', 'early_stopping_delta'])
    writer.writerow([args.early_stopping_patience, args.early_stopping_delta])

    writer.writerow(['adjust_lr_patience', 'adjust_lr_delta', 'adjust_lr_gamma'])
    writer.writerow([args.adjust_lr_patience, args.adjust_lr_delta, args.adjust_lr_gamma])

    csvfile.close()


if __name__ == '__main__':
    train()

