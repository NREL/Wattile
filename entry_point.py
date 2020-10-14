import sys
import pandas as pd
import importlib
import data_preprocessing
import json
import buildings_processing as bp
import logging
import os
import pathlib


class ConfigsError(Exception):
    """Base class for exceptions in this module."""
    pass


def main(configs):
    """
    Main function for processing and structuring data.
    Feeds training and valing data to the requested model by calling the script where the model architecture is defined
    :param configs: Dictionary
    :return: None
    """

    # Initialize logging
    local_results_dir = os.path.join(configs["results_dir"], configs["arch_type"] + '_M' + str(configs["target_var"].replace(" ", "")) + '_T' + str(configs["exp_id"]))
    pathlib.Path(local_results_dir).mkdir(parents=True, exist_ok=True)
    logging_path = os.path.join(local_results_dir, "output.out")
    print("Logging to: {}".format(logging_path))
    logging.basicConfig(filename=logging_path, format='%(asctime)s - %(levelname)-8s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S', level=logging.INFO)

    # Preprocess if needed
    if configs['preprocess']:
        train_df, val_df, configs = data_preprocessing.main(configs)

        # save the data fetch_n_parseed from API and piped through data_preprocessing (i.e. train_df and val_df)
        train_df.to_csv('./data/STM_Train_Data.csv')
        val_df.to_csv('./data/STM_Test_Data.csv')
    else:
        # preprocessing module defines target_feat_name list and sends it back.
        configs['target_feat_name'] = [configs['target_var']]

    # Get the dataset
    if configs["run_train"]:
        data_full = bp.get_full_data(configs)
    else:
        if configs["test_method"] == "external":
            data_full = bp.get_test_data(configs["building"], configs["external_test"]["year"],
                                      configs["external_test"]["month"], configs["data_dir"])
        elif configs["test_method"] == "internal":
            data_full = pd.read_hdf(os.path.join(local_results_dir, "internal_test_{}.h5".format(str(configs["target_var"].replace(" ", "")))))
        else:
            raise ConfigsError("run_train is FALSE but test_method designated in configs.json is not understood")

    # Do some preprocessing, but only if the dataset needs it (i.e. it is not an
    if configs["run_train"] or (not configs["run_train"] and configs["test_method"] == "external"):
        # Remove all data columns we dont care about
        important_vars = configs['weather_include'] + [configs['target_var']]
        data = data_full[important_vars]
        # Resample
        resample_bin_size = "{}T".format(configs['resample_freq'])
        data = data.resample(resample_bin_size).mean()
        # Clean
        data = bp.clean_data(data, configs)
        # Add calculated features (if applicable)

        # Convert data to rolling average (except output) and create min, mean, and max columns
        if configs["rolling_window"]["active"]:
            data = bp.rolling_stats(data, configs)

        # Add time-based dummy variables
        data = bp.time_dummies(data, configs)
    else:
        data = data_full

    # Choose what ML architecture to use and execute the corresponding script
    if configs['arch_type'] == 'RNN':
        # What RNN version you are implementing? Specified in configs.
        rnn_mod = importlib.import_module("algo_main_rnn_v{}".format(configs["arch_version"]))

        if configs["arch_version"] == 1:
            # read the preprocessed data from csvs
            train_df = pd.read_csv('./data/STM_Train_Data_processed.csv')
            val_df = pd.read_csv('./data/STM_Test_Data_processed.csv')
            print("read data from locally stored csvs")
            rnn_mod.main(train_df, val_df, configs)

        # Sequence to sequence model
        elif configs["arch_version"] == 5:
            # Prepare data for the RNN model type
            train_df, val_df = bp.prep_for_seq2seq(configs, data)
            rnn_mod.main(train_df, val_df, configs)

        # All other models (2-4)
        else:
            # Prepare data for the RNN model type
            train_df, val_df = bp.prep_for_rnn(configs, data)
            rnn_mod.main(train_df, val_df, configs)

    logging.info('Run with arch {}({}), on {}, with session ID {}, is done!'.format(configs['arch_type'],
                                                                                                     configs["arch_type_variant"],
                                                                                          configs["target_var"],
                                                                                          configs["exp_id"]))

# If the model is being run locally (i.e. a single model is being trained), read in configs.json and pass to main()
if __name__ == "__main__":
    with open("configs.json", "r") as read_file:
        config = json.load(read_file)
    main(config)
