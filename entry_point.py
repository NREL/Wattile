import sys
import pandas as pd
import importlib
import data_preprocessing
import json
import buildings_processing as bp


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

    # Get the full data
    if configs["run_train"]:
        data_full = bp.get_full_data(configs)
    else:
        data_full = bp.get_test_data(configs["external_test"]["building"], configs["external_test"]["year"],
                                  configs["external_test"]["month"], configs["data_dir"])
    # Remove all data columns we dont care about
    important_vars = configs['weather_include'] + [configs['target_var']]
    data = data_full[important_vars]
    # Resample
    resample_bin_size = "{}T".format(configs['resample_bin_min'])
    data = data.resample(resample_bin_size).mean()
    # Clean
    data = bp.clean_data(data, configs)
    # Add calculated features

    # Add time-based dummy variables
    data = bp.time_dummies(data, configs)

    # As of this point, "data" dataframe is assumed to have:
    # only the weather features we want to train on, already resampled, cleaned, have time-based features added, and have all calculated features, if any.
    # The data has not been padded yet, or been split into a test/train split.
    # For feature selection, "data" can be passed into prep_for function. It needs to have gone through the equivilent steps as above.

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

        # Sequence to sequence model
        elif configs["arch_version"] == 5:
            # Prepare data for the RNN model type
            train_df, test_df = bp.prep_for_seq2seq(configs, data)
            rnn_mod.main(train_df, test_df, configs)

        # All other models (2-4)
        else:
            # Prepare data for the RNN model type
            train_df, test_df = bp.prep_for_rnn(configs, data)
            rnn_mod.main(train_df, test_df, configs)

    print('Run with arch: {}, train_num= {}, test_num= {} and target= {} is done!'.format(configs['arch_type'],
                                                                                          configs['building'],
                                                                                          configs['test_exp_num'],
                                                                                          configs['target_var']))

# If the model is being run locally (i.e. a single model is being trained), read in configs.json and pass to main()
if __name__ == "__main__":
    with open("configs.json", "r") as read_file:
        config = json.load(read_file)
    main(config)
