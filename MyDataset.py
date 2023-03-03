import numpy as np
import torch.utils.data


class MyFsSeiDataset(torch.utils.data.Dataset):

    def __init__(self, filepath_x, filepath_y, transform_x=None, transform_y=None):
        super(MyFsSeiDataset, self).__init__()
        self.X = np.load(filepath_x)
        self.Y = np.uint8(np.load(filepath_y))
        self.transform_x = transform_x
        self.transform_y = transform_y
        if self.transform_x:
            self.X = self.transform_x(self.X)
        if self.transform_y:
            self.Y = self.transform_y(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.Y)


def get_train_dataset(num, transform_x=None, transform_y=None):
    filepath_x = '../SplitFSSEIDataset/X_split_train_{}Classes.npy'.format(num)
    filepath_y = '../SplitFSSEIDataset/Y_split_train_{}Classes.npy'.format(num)
    return MyFsSeiDataset(filepath_x, filepath_y, transform_x, transform_y)


def get_val_dataset(num, transform_x=None, transform_y=None):
    filepath_x = '../SplitFSSEIDataset/X_split_val_{}Classes.npy'.format(num)
    filepath_y = '../SplitFSSEIDataset/Y_split_val_{}Classes.npy'.format(num)
    return MyFsSeiDataset(filepath_x, filepath_y, transform_x, transform_y)


def get_test_dataset(num, transform_x=None, transform_y=None):
    filepath_x = 'H:/demo/FS-SEI_4800/Dataset/X_test_{}Class.npy'.format(num)
    filepath_y = 'H:/demo/FS-SEI_4800/Dataset/Y_test_{}Class.npy'.format(num)

    return MyFsSeiDataset(filepath_x, filepath_y, transform_x, transform_y)


def get_test_support_dataset(num, transform_x=None, transform_y=None):
    x = np.load('H:/demo/FS-SEI_4800/Dataset/X_train_{}Class.npy'.format(num))
    y = np.load('H:/demo/FS-SEI_4800/Dataset/Y_train_{}Class.npy'.format(num))
    if transform_x:
        x = transform_x(x)
    if transform_y:
        y = transform_y(y)

    y = y.astype(np.uint8)

    return x, y


def get_test_query_dataset(num, transform_x=None, transform_y=None):
    x = np.load('H:/demo/FS-SEI_4800/Dataset/X_test_{}Class.npy'.format(num))
    y = np.load('H:/demo/FS-SEI_4800/Dataset/Y_test_{}Class.npy'.format(num))
    if transform_x:
        x = transform_x(x)
    if transform_y:
        y = transform_y(y)

    y = y.astype(np.uint8)

    return x, y


if __name__ == '__main__':
    train_dataset = get_train_dataset(90)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    x, y = next(iter(train_dataloader))
    print("x.size: ", x.size())
    print("y.size: ", y.size())
