import sys
import time
from datetime import datetime
from functools import reduce
import torch
import numpy as np
import argparse



start = time.time()
lasttime = time.time()


def prtime(*args, **kwargs):
    global lasttime
    print(" ".join(map(str, args)), '|time:', str(datetime.now()), '|', time.time() - start, 'secs from start',
          time.time() - lasttime, 'secs from last', **kwargs)
    lasttime = time.time()
    sys.stdout.flush()

def factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('train_sd', default='2018-10-22', type=str,
                        help="start date for the train dataset"+
                        "the input format is \'YYYY-MM-DD\', a string")
    parser.add_argument('train_ed', default='2018-11-22', type=str,
                        help="end date for the train dataset"+
                        "the input format is \'YYYY-MM-DD\', a string")
    parser.add_argument('test_sd', default='2018-11-23', type=str,
                        help="start date for the test dataset"+
                        "the input format is \'YYYY-MM-DD\', a string")
    parser.add_argument('test_ed', default='2018-11-28', type=str,
                        help="end date for the test dataset"+
                        "the input format is \'YYYY-MM-DD\', a string")
    parser.add_argument('transform', default='minmaxscale', type=str,
                        help="normalize or standardize or minmaxscale data, defaults to minmaxscale")
    parser.add_argument('--train',
                        help='True or False, To train the model or not. If False, the test will be run on the existing model')
    parser.add_argument('--num-epochs', default=1000, type=int,
                        help="Number of training, testing epochs")
    parser.add_argument('--resume',
                        help='True or False, To resume from the previous model. If False, a new model will be instantiated and trained')
    parser.add_argument('--pp',
                        help='True or False, To fetch data from API and pre-process. saved csvf files will be used for training and testing')
    parser.add_argument('--arch-type',
                        help='pick architecture type from \'FFNN\', \'RNN\',\'LSTM\',\'GRU\'')


    args, _ = parser.parse_known_args()

    # Sanity check the arguments
    dates = {"train_start_date": args.train_sd,
             "train_end_date": args.train_ed,
             "test_start_date":args.test_sd,
             "test_end_date":args.test_ed}

    for key, value in dates.items():
        if not isinstance(value, str):
            print("Enter a valid {}. Exiting...".format(key))
            parser.print_help()
            exit()
        if not datetime.strptime(value, "%Y-%m-%d"):
            print("Enter a valid {}.Exiting...".format(key))
            parser.print_help()
            exit()

    # args.transform
    if args.transform.lower() in ["standardize", "normalize", "minmaxscale"]:
        transformation_method = args.transform.lower()
    else:
        transformation_method = 'minmaxscale'

    # args.train
    if args.train in ["True", "true"]:
        run_train = True
    elif args.train in ["False", "false"]:
        run_train = False
    else:
        print("Train flag is invalid. It should be True or false. Exiting...")
        parser.print_help()
        exit()

    # args.num_epochs
    num_epochs = args.num_epochs #--num-epochs

    # args.resume
    if args.resume in ["True", "true"]:
        run_resume = True
    elif args.resume in ["False", "false"]:
        run_resume = False
    else:
        print("Resume flag is invalid. It should be True or false. Exiting...")
        parser.print_help()
        exit()

    # args.pp
    if args.pp in ["True", "true"]:
        preprocess = True
    elif args.pp in ["False", "false"]:
        preprocess = False
    else:
        print("Preprocessing flag is invalid. It should be True or false. Exiting...")
        parser.print_help()
        exit()

    # args.arch_type
    if args.arch_type.lower() in ['ffnn', 'rnn','lstm','gru']:
        arch_type = args.arch_type.upper()
    else:
        print("Architecture type is invalid. See help for types of architecture available. Exiting...")
        parser.print_help()
        exit()

    return dates['train_start_date'], dates['train_end_date'], dates['test_start_date'], dates['test_end_date'], transformation_method, run_train, num_epochs, run_resume, preprocess, arch_type