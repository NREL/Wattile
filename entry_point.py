import sys
import pandas as pd
import importlib
import data_preprocessing
import json


def main(configs):
    """
    Main function for processing and structuring data.
    Feeds training and testing data to the requested model by calling the script where the model architecture is defined
    :param configs: Dictionary
    :return: None
    """
    # Preprocess if needed
    if configs['preprocess']:
        train_df, test_df, configs = data_preprocessing.main(configs)

        # save the data fetch_n_parseed from API and piped through data_preprocessing (i.e. train_df and test_df)
        train_df.to_csv('./data/STM_Train_Data.csv')
        test_df.to_csv('./data/STM_Test_Data.csv')
    else:
        # preprocessing module defines target_feat_name list and sends it back.
        configs['target_feat_name'] = [configs['target_var']]

    # Choose what ML architecture to use and execute the corresponding script
    if configs['arch_type'] == 'RNN':
        # What RNN version you are implementing?
        # Specified in configs
        rnn_mod = importlib.import_module("algo_main_rnn_v{}".format(configs["arch_version"]))
        if configs["arch_version"] == 1:
            # read the preprocessed data from csvs
            train_df = pd.read_csv('./data/STM_Train_Data_processed.csv')
            test_df = pd.read_csv('./data/STM_Test_Data_processed.csv')
            print("read data from locally stored csvs")
            rnn_mod.main(train_df, test_df, configs)
        elif configs["arch_version"] == 5:
            # Import buildings module to preprocess data
            sys.path.append(configs["shared_dir"])
            bp = importlib.import_module("buildings_processing")

            # Prepare data for the RNN model type
            train_df, test_df = bp.prep_for_seq2seq(configs)
            rnn_mod.main(train_df, test_df, configs)

        else:
            # Import buildings module to preprocess data
            sys.path.append(configs["shared_dir"])
            bp = importlib.import_module("buildings_processing")

            # Prepare data for the RNN model type
            train_df, test_df = bp.prep_for_rnn(configs)
            rnn_mod.main(train_df, test_df, configs)

    print('Run with arch: {}, train_num= {}, test_num= {} and target= {} is done!'.format(configs['arch_type'],
                                                                                          configs['train_exp_num'],
                                                                                          configs['test_exp_num'],
                                                                                          configs['target_var']))

# If the model is being run locally (i.e. a single model is being trained), read in configs.json and pass to main()
if __name__ == "__main__":
    with open("configs.json", "r") as read_file:
        config = json.load(read_file)
    main(config)
