import datetime as dt
import json
import logging
import os
import pathlib
from pathlib import Path

import pandas as pd

from wattile.error import ConfigsError

PROJECT_DIRECTORY = pathlib.Path(__file__).resolve().parent

logger = logging.getLogger(str(os.getpid()))


def save_data_config_to_exp_dir(configs):
    """save data config to exp dir

    :param configs: config
    :type configs: dict
    """
    dataset_dir = Path(configs["data_dir"]) / configs["building"]
    dataset_configs_file = dataset_dir / f"{configs['building']} Config.json"

    with open(dataset_configs_file, "r") as f:
        dataset_configs = json.load(f)

    data_configs = {
        "predictors": [
            p
            for p in dataset_configs["predictors"]
            if p["column"] in configs["predictor_columns"]
        ],
        "target": next(
            t
            for t in dataset_configs["targets"]
            if t["column"] == configs["target_var"]
        ),
        "window": {
            "interval": configs["rolling_window"]["minutes"],
            "lags": configs["window"],
            "duration": (configs["window"] + 1) * configs["sequence_freq_min"],
        },
    }

    train_data_config_path = Path(configs["exp_dir"]) / "data_config.json"
    with open(train_data_config_path, "w+") as f:
        json.dump(data_configs, f, ensure_ascii=False)


def _concat_data_from_files(filepaths, needed_columns):
    """Concat the data in the files

    Only get the needed columns.
    Data must include column "Timestamp".

    :param filepaths: list of filepaths
    :type filepaths: list[Path]
    :param needed_columns: list of column names to keep
    :type needed_columns: list[str]
    :return: full data
    :rtype: pd.DataFrame
    """
    full_data = pd.DataFrame()

    for filepaths in filepaths:
        try:
            data = pd.read_csv(Path(filepaths))[["Timestamp"] + needed_columns]
            full_data = pd.concat([full_data, data])

        except Exception:
            logger.warning(f"Could not read {filepaths}. skipping...")
        else:
            logger.info(f"Read {filepaths} and added to data ...")

    if not full_data.empty:
        full_data["Timestamp"] = pd.to_datetime(
            full_data["Timestamp"], format="%Y-%m-%dT%H:%M:%S%z", exact=False, utc=True
        )

        full_data = full_data.set_index("Timestamp")

    return full_data


def _get_dataset_config(configs):
    """Get dataset config as dataframe

    :param configs: configs
    :type configs: dict
    :return: dataset config
    :rtype: pd.DataFrame
    """
    dataset_dir = Path(configs["data_dir"]) / configs["building"]
    configs_file_inputdata = dataset_dir / f"{configs['building']} Config.json"

    logger.info(
        "Pre-process: reading input data summary json file from {}".format(
            configs_file_inputdata
        )
    )

    with open(configs_file_inputdata, "r") as read_file:
        configs_input = json.load(read_file)
        df_inputdata = pd.DataFrame(configs_input["files"])

    # converting date time column into pandas datetime (raw format based on ISO 8601)
    df_inputdata["start"] = pd.to_datetime(
        df_inputdata.start, format="t:%Y-%m-%dT%H:%M:%S%z", exact=False, utc=True
    )
    df_inputdata["end"] = pd.to_datetime(
        df_inputdata.end, format="t:%Y-%m-%dT%H:%M:%S%z", exact=False, utc=True
    )

    df_inputdata["path"] = str(dataset_dir) + "/" + df_inputdata["filename"]

    return df_inputdata


def read_dataset_from_file(configs):
    """
    Fetches all data for a requested building based on the information reflected in the input data
     summary json file.

    :param configs: (Dictionary)
    :return: (DataFrame)
    """
    df_inputdata = _get_dataset_config(configs)

    # only read from files that's timespan intersects with the configs
    # the extra will be removed in `prep_for_rnn`
    timestamp_start = dt.datetime.fromisoformat(configs["start_time"])
    timestamp_end = dt.datetime.fromisoformat(configs["end_time"])
    df_inputdata = df_inputdata.loc[
        (df_inputdata.start <= timestamp_end) & (df_inputdata.end >= timestamp_start), :
    ]

    if df_inputdata.empty:
        logger.info(
            "Pre-process: measurements during the specified time period "
            f"({timestamp_start} to {timestamp_end}) are empty."
        )

        raise ConfigsError("No datapoints found in dataset for specified timeframe.")

    # read in predictor data
    predictor_data_info = df_inputdata[df_inputdata.contentType == "predictors"]
    data_full_p = _concat_data_from_files(
        predictor_data_info.path,
        needed_columns=configs["predictor_columns"],
    )

    # read in target data
    target_data_info = df_inputdata[df_inputdata.contentType == "targets"]
    data_full_t = _concat_data_from_files(
        target_data_info.path, needed_columns=[configs["target_var"]]
    )

    if data_full_p.empty:
        message = "No predictor data found in dataset for specified timeframe."
        logger.info(f"{message} Exiting process...")

        raise ConfigsError(message)

    elif data_full_t.empty and configs["use_case"] != "prediction":
        message = "No target data found in dataset for specified timeframe."
        logger.info(f"{message} Exiting process...")

        raise ConfigsError(message)

    # the rest of the code expects a shape with a predictor column.
    # TODO: remove if
    if configs["use_case"] == "prediction":
        data_full = data_full_p
        data_full[configs["target_var"]] = -999
    else:
        data_full = pd.merge(data_full_p, data_full_t, how="outer", on="Timestamp")

    return data_full
