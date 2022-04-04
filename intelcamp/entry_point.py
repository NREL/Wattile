import sys
import pandas as pd
import importlib
from  intelcamp import data_preprocessing
import json
import intelcamp.buildings_processing as bp
import logging
import os
import pathlib
from  intelcamp import util

PACKAGE_PATH = pathlib.Path(__file__).parent
CONFIGS_PATH = PACKAGE_PATH / "configs.json"

logger = logging.getLogger(str(os.getpid()))

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
    local_results_dir = util.Path(configs["exp_dir"])

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
    if configs["use_case"] == "validation" and configs["test_method"] == "internal":
        data = pd.read_hdf(os.path.join(local_results_dir, "internal_test.h5"))
    else:
        data = bp.get_full_data(configs)

    # if certain predictor variables are pre-defined, then include only those.
    if configs['weather_include']:
        keep_cols = configs['weather_include'] + [configs['target_var']]
        data = data[keep_cols]
        logger.info("columns specified in the configs.json are only included")
    else:
        logger.info("all available predictor variables and target variable ({}) are included".format(configs['target_var']))

    # Do some preprocessing, but only if the dataset needs it
    if configs["use_case"] == "training":
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
        data = bp.pad_full_data(data, configs)
    elif configs["arch_version"] == 5:
        data = bp.pad_full_data_s2s(data, configs)

    if configs["use_case"] == "training":
        data = data.loc[:, (data != 0).any(axis=0)]

    train_df, val_df = bp.prep_for_rnn(configs, data)

    return train_df, val_df

def run_model(configs, train_df, val_df):
    """train, validate, or predict using a model

    :param configs: dict of configs
    :type configs: dcit
    :param train_df: input data for training
    :type train_df: DataFrame
    :param val_df: input data for validation
    :type val_df: DataFrame
    """
    local_results_dir = util.Path(configs["exp_dir"])

    if configs["use_case"] == "train":
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
        rnn_mod = importlib.import_module("intelcamp.algo_main_rnn_v{}".format(configs["arch_version"]))
        logger.info("training with arch version {}".format(configs["arch_version"]))

        # Prepare data for the RNN model type
        results = rnn_mod.main(train_df, val_df, configs)

    logger.info('Run with arch {}({}), on {}, with session ID {}, is done!'.format(configs['arch_type'],
                                                                                                     configs["arch_type_variant"],
                                                                                          configs["target_var"],
                                                                                          configs["exp_id"]))
    return results


def main(configs):
    """
    Main function for processing and structuring data.
    Feeds training and valing data to the requested model by calling the script where the model architecture is defined
    :param configs: Dictionary
    :return: None
    """
    init_logging(local_results_dir=util.Path(configs["exp_dir"]))
    train_df, val_df = create_input_dataframe(configs)
    
    return run_model(configs, train_df, val_df)

# If the model is being run locally (i.e. a single model is being trained), read in configs.json and pass to main()
if __name__ == "__main__":
    with open(CONFIGS_PATH, "r") as read_file:
        configs = json.load(read_file)
    main(configs)