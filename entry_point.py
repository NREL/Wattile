import sys
import pandas as pd
import importlib
import data_preprocessing
import json
import buildings_processing as bp
import logging
import os
import pathlib
import util


logger = logging.getLogger(str(os.getpid()))
class ConfigsError(Exception):
    """Base class for exceptions in this module."""
    pass


def init_logging(local_results_dir):
    """init logger

    :param local_results_dir: results dir
    :type local_results_dir: Path
    """
    PID = os.getpid()
    pathlib.Path(local_results_dir).mkdir(parents=True, exist_ok=True)
    logging_path = os.path.join(local_results_dir, "output.out")
    print("Logging to: {}, PID: {}".format(logging_path, PID))
    logger = logging.getLogger(str(PID))
    hdlr = logging.FileHandler(logging_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s', '%m/%d/%Y %I:%M:%S')
    hdlr.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info("PID: {}".format(PID))


def create_input_dataframe(configs):
    """construct dataframe for model input

    :param configs: config dict
    :type configs: dict
    :raises ConfigsError: if config is invalid
    :return: data
    :rtype: DataFrame
    """
    local_results_dir = util.get_exp_dir(configs)

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
    if configs["run_train"] or configs["test_method"] == "external":
        data = bp.get_full_data(configs)
    
    elif configs["test_method"] == "internal":
        # temporarily assigning synthetic data for prediction testing
        data = pd.read_hdf(os.path.join(local_results_dir, "internal_test.h5"))

    else:
         raise ConfigsError("run_train is FALSE but test_method designated in configs.json is not understood")

    # Do some preprocessing, but only if the dataset needs it (i.e. it is not an
    if configs["run_train"]:

        # Clean
        data = bp.clean_data(data, configs)

        # Add time-based features 
        data = bp.time_dummies(data, configs)

        # Add statistics features 
        if configs["rolling_window"]["active"]:
            data = bp.rolling_stats(data, configs)

        # Add lag features
        configs['input_dim'] = data.shape[1] - 1
        logger.info("Number of features: {}".format(configs['input_dim']))
        logger.debug("Features: {}".format(data.columns.values))
        if configs["arch_version"] == 4:
            data, target = bp.pad_full_data(data, configs)
        elif configs["arch_version"] == 5:
            data, target = bp.pad_full_data_s2s(data, configs)

        # removing columns with zero
        data = data.loc[:, (data != 0).any(axis=0)]

    elif not configs['run_train']:

        logger.info("performing data transformation for prediction")

        # add time-based features (based on configs file from previous model training)
        logger.info("adding time-based features")
        data = bp.time_dummies(data, configs)

        # add statistics features (based on configs file from previous model training)
        if configs["rolling_window"]["active"]:
            logger.info("adding statistic features")
            data = bp.rolling_stats(data, configs)

        # add lag features (based on configs file from previous model training)
        configs['input_dim'] = data.shape[1] - 1
        logger.info("Number of features: {}".format(configs['input_dim']))
        logger.debug("Features: {}".format(data.columns.values))
        if configs["arch_version"] == 4:
            data, target = bp.pad_full_data(data, configs)
        elif configs["arch_version"] == 5:
            data, target = bp.pad_full_data_s2s(data, configs)

        # filtering features based on down-selected features resulted from feature selection
        # place holder  

    return data


def run_model(configs, data):
    """train, validate, or predict using a model

    :param configs: dict of configs
    :type configs: dcit
    :param data: input data
    :type data: DataFrame

    - reads raw data for prediction
    - adds same features (time-based, statistics, time lag) that were added in the previous model training
    - filters features based on down-selected features list from the previous model training
    
    """
    local_results_dir = util.get_exp_dir(configs)

    if configs["run_train"]:
        # Check the model training process
        torch_file = os.path.join(local_results_dir, 'torch_model')
        if os.path.exists(torch_file):
            check = bp.check_complete(torch_file, configs["num_epochs"])
            # If we already have the desired number of epochs, don't do anything else
            if check:
                print("{} already completed. Moving on...".format(configs["target_var"]))
                return
        # If the torch file doesnt exist yet, and run_resume=True, then reset it to false so it can start from scratch
        else:
            if configs["run_resume"]:
                configs["run_resume"] = False
                print("Model for {} doesnt exist yet. Resetting run_resume to False".format(configs["target_var"]))

    # Choose what ML architecture to use and execute the corresponding script
    if configs['arch_type'] == 'RNN':
        # What RNN version you are implementing? Specified in configs.
        rnn_mod = importlib.import_module("algo_main_rnn_v{}".format(configs["arch_version"]))
        logger.info("training with arch version {}".format(configs["arch_version"]))

        if configs["arch_version"] == 1:
            # read the preprocessed data from csvs
            train_df = pd.read_csv('./data/STM_Train_Data_processed.csv')
            val_df = pd.read_csv('./data/STM_Test_Data_processed.csv')
            print("read data from locally stored csvs")
            rnn_mod.main(train_df, val_df, configs)

        # Sequence to sequence model
        elif (configs["arch_version"] == 5) or (configs["arch_version"] == 6):
            # Prepare data for the RNN model type
            train_df, val_df = bp.prep_for_seq2seq(configs, data)
            rnn_mod.main(train_df, val_df, configs)

        # All other models (2-4)
        else:
            # Prepare data for the RNN model type
            train_df, val_df = bp.prep_for_rnn(configs, data)
            rnn_mod.main(train_df, val_df, configs)

    logger.info('Run with arch {}({}), on {}, with session ID {}, is done!'.format(configs['arch_type'],
                                                                                                     configs["arch_type_variant"],
                                                                                          configs["target_var"],
                                                                                          configs["exp_id"]))

def main(configs):
    """
    Main function for processing and structuring data.
    Feeds training and valing data to the requested model by calling the script where the model architecture is defined
    :param configs: Dictionary
    :return: None
    """

    if not configs["run_train"]:
        with open(configs["trained_model_path"] + "/" + "configs.json", "r") as read_file:
            print("reading statistics data (for normalization) from previously trained model results: {}".format(configs["trained_model_path"]))
            configs = json.load(read_file)
            configs['run_train'] = False

    init_logging(local_results_dir=util.get_exp_dir(configs))
    data = create_input_dataframe(configs)
    run_model(configs, data)


# If the model is being run locally (i.e. a single model is being trained), read in configs.json and pass to main()
if __name__ == "__main__":
    with open("configs.json", "r") as read_file:
        configs = json.load(read_file)
    main(configs)
