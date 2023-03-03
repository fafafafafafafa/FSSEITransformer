import numpy as np
import torch
"""
放置模型网络
版本说明:
V1
时间: 2022.10.21
作者: fff
说明: EarlyStopping

V3 
时间: 2022.11.3
作者: fff
说明: AdjustLearningRate
"""


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, dir_path='./', filename="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            dir_path (string): 最优模型的保存地址
                            Default: './'
            filename (string): 最优模型的名称
                            Default: ’checkpoint.pt‘
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dir_path = dir_path
        self.filename = filename

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        # Saves model when validation loss decrease.
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # torch.save(model.state_dict(), self.dir_path+self.filename)  # 这里会存储迄今最优模型的参数
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': val_loss,
        }, self.dir_path+self.filename)
        self.val_loss_min = val_loss


class AdjustLearningRate:
    def __init__(self, patience=7, start_lr=0.001, min_lr=1e-6, gamma=0.5, verbose=False, delta=0):
        """
        根据loss 改变学习率
        参数说明:
            输入:
                patience (int): 等待次数
                                Default: 7
                start_lr (float): 起始学习率
                                Default: 0.001
                min_lr (float): 最小学习率
                                Default: 0.000001
                gamma (float): 衰减率
                                Default: 0.5
                verbose (bool): True 时, 每当学习率变化时打印信息
                                Default: False
                delta (float): 衡量loss是否变小的阈值
                                Default: 0

        """
        self.patience = patience
        self.cur_lr = start_lr
        self.min_lr = min_lr
        self.gamma = gamma
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.counter = 0

    def __call__(self, val_loss, optimizer):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter = self.counter + 1
            print(f'AdjustLr counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.counter = 0
                self.cur_lr = self.cur_lr * self.gamma
                if self.cur_lr < self.min_lr:
                    self.cur_lr = self.min_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.cur_lr
                print("Lr has changed to :{:.2E}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        else:
            self.best_score = score
            self.counter = 0
