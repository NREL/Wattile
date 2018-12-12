import argparse

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
    parser.add_argument('t',
                        help='True or False, To train the model or not. If False, the test will be run on the existing model')
    parser.add_argument('--num-epochs', default=1000, type=int,
                        help="Number of training, testing epochs")
    args, _ = parser.parse_known_args()

    # Sanity check the arguments
    train_start_date = args.train_sd
    if not isinstance(train_start_date, str):
        print("Enter a valid train start date, the default is 2018-10-22")
        parser.print_help()

    train_end_date = args.train_ed
    if not isinstance(train_end_date, str):
        print("Enter a valid train end date, the default is 2018-11-22")
        parser.print_help()

    test_start_date = args.test_sd
    if not isinstance(test_start_date, str):
        print("Enter a valid test start date, the default is 2018-11-23")
        parser.print_help()

    test_end_date = args.test_ed
    if not isinstance(test_end_date, str):
        print("Enter a valid test end date, the default is 2018-11-28")
        parser.print_help()

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

    return train_start_date, train_end_date, test_start_date, test_end_date, run_train, num_epochs



