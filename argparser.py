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
    parser.add_argument('transform', default='norm', type=str,
                        help="normalize or standardize or minmaxscale data")
    parser.add_argument('t',
                        help='True or False, To train the model or not. If False, the test will be run on the existing model')
    parser.add_argument('--num-epochs', default=1000, type=int,
                        help="Number of training, testing epochs")
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

    # args.t
    if args.t in ["True", "true"]:
        run_train = True
    elif args.t in ["False", "false"]:
        run_train = False
    else:
        print("Train flag is invalid. It should be True or false. Exiting...")
        parser.print_help()
        exit()

    # args.num_epochs
    num_epochs = args.num_epochs

    return dates['train_start_date'], dates['train_end_date'], dates['test_start_date'], dates['test_end_date'], transformation_method, run_train, num_epochs



