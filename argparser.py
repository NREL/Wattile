import argparse
from datetime import datetime

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
                        help="standardize or minmaxscale data, defaults to minmaxscale")
    parser.add_argument('--train',
                        help='True or False, To train the model or not. If False, the test will be run on the existing model')
    parser.add_argument('--num-epochs', default=1000, type=int,
                        help="Number of training, testing epochs")
    parser.add_argument('--tr-batch-size', default=1000, type=int,
                        help="train batch size")
    parser.add_argument('--te-batch-size', default=200, type=int,
                        help="train batch size")
    parser.add_argument('--resume',
                        help='True or False, To resume from the previous model. If False, a new model will be instantiated and trained')
    parser.add_argument('--pp',
                        help='True or False, To read the jsons and pre-process. saved json files will be used')
    parser.add_argument('--fetch-n-parse',
                        help='True or False, To fetch_n_parse data from API, parse and save as json files, which will be used for preprocessing')
    parser.add_argument('--arch-type',
                        help='pick architecture type from \'FFNN\', \'RNN\',\'LSTM\',\'GRU\'')
    parser.add_argument('--tr-exp-num',
                        help='train experiment number for the given architecture type')
    parser.add_argument('--te-exp-num',
                        help='test experiment number for the given architecture type')
    parser.add_argument('--hidden-nodes',
                        help='number of hidden nodes in each hidden layer')
    parser.add_argument('--weight-decay',
                        help='lambda for regularization')

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

    # args.train_batch_size
    train_batch_size = args.train_batch_size  # --tr-batch-size

    # args.val_batch_size
    val_batch_size = args.val_batch_size  # --te-batch-size

    # args.hidden_nodes
    hidden_nodes = args.hidden_nodes

    # args.weight_decay
    weight_decay = args.weight_decay

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

    # args.fetch_n_parse
    if args.fetch_n_parse in ["True", "true"]:
        fetch_n_parse = True
    elif args.fetch_n_parse in ["False", "false"]:
        fetch_n_parse = False
    else:
        print("Fetch_and_Parse flag is invalid. It should be True or false. Exiting...")
        parser.print_help()
        exit()

    # args.arch_type
    if args.arch_type.lower() in ['ffnn', 'rnn','lstm','gru']:
        arch_type = args.arch_type.upper()
    else:
        print("Architecture type is invalid. See help for types of architecture available. Exiting...")
        parser.print_help()
        exit()

    # args.tr_exp_id and args.te_exp_id
    train_exp_id = args.tr_exp_id
    exp_id = args.te_exp_id

    configs = {"train_start_date": dates['train_start_date'],
               "train_end_date": dates['train_end_date'],
               "test_start_date": dates['test_start_date'],
               "test_end_date": dates['test_end_date'],
               "transformation_method": transformation_method, "run_train": run_train,
               "num_epochs": num_epochs, "train_batch_size":train_batch_size,"val_batch_size":val_batch_size,"run_resume": run_resume,
               "preprocess": preprocess, "arch_type": arch_type,
               "train_exp_id": train_exp_id, "exp_id": exp_id,
               "hidden_nodes": hidden_nodes, 'weight_decay': weight_decay,
               "fetch_n_parse": fetch_n_parse
               }

    return configs