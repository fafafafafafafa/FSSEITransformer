import argparse


def init():
    parser = argparse.ArgumentParser('parameters for train')

    # dataset
    parser.add_argument('--train_val_dataset_num', type=int, default=90, help='num of train or val class')
    parser.add_argument('--test_dataset_num', type=list, default=[10, 20, 30], help='num of train class')

    # hyper_parameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size for model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for model')
    parser.add_argument('--epochs', type=int, default=100, help='max epochs for train')

    # model
    parser.add_argument('--model_name', type=str, default='conformer', help='model for train')
    parser.add_argument('--emb_size', type=int, default=32, help='emb_size for attention')
    parser.add_argument('--heads', type=int, default=8, help='nums of parallel attention block')

    # optimizer
    parser.add_argument('--optimizer_name', type=str, default='adam', help='optimizer for train')

    # loss_fn
    parser.add_argument('--loss_fn_name', type=str, default='cross_entropy_loss', help='optimizer for train')

    # early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=25,
                        help='the patience for optimizer stop training')
    parser.add_argument('--early_stopping_delta', type=float, default=0.01,
                        help='the gate determines if loss change or not')
    parser.add_argument('--early_stopping_dir_path', type=str, default='./result/checkpoint/',
                        help='path for model saving(followed by model_name to get specific path. '
                             'thus, early_stopping_dir_path+model_name+/ is the true path for model saving)')
    # adjust learning rate
    parser.add_argument('--adjust_lr_patience', type=int, default=7, help='the patience for optimizer change its lr')
    parser.add_argument('--adjust_lr_delta', type=float, default=0.01,
                        help='the gate determines if loss change or not')
    parser.add_argument('--adjust_lr_start_lr', type=float, default=1e-3,
                        help='remember to be the same as lr')
    parser.add_argument('--adjust_lr_min_lr', type=float, default=1e-5,
                        help='min lr')
    parser.add_argument('--adjust_lr_gamma', type=float, default=0.5,
                        help='decay of lr ')
    # data path
    parser.add_argument('--data_dir_path', type=str, default='./result/data/',
                        help='path for data saving(followed by model_name to get specific path. '
                             'thus, data_dir_path+model_name+/ is the true path for data saving)')

    args = parser.parse_args()
    return args
