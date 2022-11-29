import json
import logging
import os
import pathlib

import pandas as pd

import wattile.buildings_processing as bp
from wattile.data_reading import read_dataset_from_file
from wattile.models import ModelFactory

PACKAGE_PATH = pathlib.Path(__file__).parent
CONFIGS_PATH = PACKAGE_PATH / "configs" / "configs.json"

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
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(message)s", "%m/%d/%Y %I:%M:%S"
    )
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
    local_results_dir = pathlib.Path(configs["data_output"]["exp_dir"])

    configs["target_feat_name"] = [configs["data_input"]["target_var"]]

    # Get the dataset
    if (
        configs["learning_algorithm"]["use_case"] == "validation"
        and configs["learning_algorithm"]["test_method"] == "internal"
    ):
        data = pd.read_hdf(os.path.join(local_results_dir, "internal_test.h5"))
    else:
        data = read_dataset_from_file(configs)

    train_df, val_df = bp.prep_for_rnn(configs, data)

    return train_df, val_df, configs


def run_model(configs, train_df, val_df):
    """train, validate, or predict using a model

    :param configs: dict of configs
    :type configs: dcit
    :param train_df: input data for training
    :type train_df: DataFrame
    :param val_df: input data for validation
    :type val_df: DataFrame
    """
    local_results_dir = pathlib.Path(configs["data_output"]["exp_dir"])

    if configs["learning_algorithm"]["use_case"] == "train":
        # Check the model training process
        torch_file = os.path.join(local_results_dir, "torch_model")
        if os.path.exists(torch_file):
            check = bp.check_complete(
                torch_file, configs["learning_algorithm"]["num_epochs"]
            )
            # If we already have the desired number of epochs, don't do anything else
            if check:
                print(
                    "{} already completed. Moving on...".format(
                        configs["data_input"]["target_var"]
                    )
                )
                return
        # If the torch file doesnt exist yet, and run_resume=True, then reset it to false so it can
        # start from scratch
        else:
            if configs["learning_algorithm"]["run_resume"]:
                configs["learning_algorithm"]["run_resume"] = False
                print(
                    "Model for {} doesnt exist yet. Resetting run_resume to False".format(
                        configs["data_input"]["target_var"]
                    )
                )

    # Choose what ML architecture to use and execute the corresponding script
    if configs["learning_algorithm"]["arch_type"] == "RNN":
        model = ModelFactory.create_model(configs)

        logger.info(
            "training with arch version {}".format(
                configs["learning_algorithm"]["arch_version"]
            )
        )

        # Prepare data for the RNN model type
        results = model.main(train_df, val_df)

    logger.info(
        "Run with arch {}({}), on {} is done!".format(
            configs["learning_algorithm"]["arch_type"],
            configs["learning_algorithm"]["arch_type_variant"],
            configs["data_input"]["target_var"],
        )
    )
    return results


def main(configs):
    """
    Main function for processing and structuring data.
    Feeds training and valing data to the requested model by calling the script where the model
     architecture is defined
    :param configs: Dictionary
    :return: None
    """
    init_logging(local_results_dir=pathlib.Path(configs["data_output"]["exp_dir"]))
    train_df, val_df, configs = create_input_dataframe(configs)

    return run_model(configs, train_df, val_df)


# If the model is being run locally (i.e. a single model is being trained), read in configs.json and
# pass to main()
if __name__ == "__main__":
    with open(CONFIGS_PATH, "r") as read_file:
        configs = json.load(read_file)
    main(configs)
