import datetime as dt
import json
import logging
import os
import pathlib
from pathlib import Path

import pandas as pd

from wattile.error import ConfigsError

PROJECT_DIRECTORY = pathlib.Path(__file__).resolve().parent

print("PROJECT_DIRECTORY = {}".format(PROJECT_DIRECTORY))

logger = logging.getLogger(str(os.getpid()))


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
            if len(needed_columns) == 0:
                data = pd.read_csv(Path(filepaths))
            else:
                data = pd.read_csv(Path(filepaths))[["Timestamp"] + needed_columns]
            full_data = pd.concat([full_data, data])

        except Exception:
            logger.warning(f"Could not read {filepaths}. skipping...")
        else:
            logger.info(f"Read {filepaths} and added to data ...")

    if not full_data.empty:
        full_data["Timestamp"] = full_data["Timestamp"].str.split(" ", 1).str[0]
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
    :rtype: Tuple[pd.DataFrame, List[Dict]]
    """
    dataset_dir = Path(configs["data_input"]["data_dir"])
    configs_file_inputdata = dataset_dir / configs["data_input"]["data_config"]

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

    return df_inputdata, configs_input


def read_dataset_from_file(configs):
    """
    Fetches all data for a requested building based on the information reflected in the input data
     summary json file.

    :param configs: (Dictionary)
    :return: (DataFrame)
    """
    df_inputdata, configs_input = _get_dataset_config(configs)

    # only read from files that's timespan intersects with the configs
    # the extra will be removed in `prep_for_rnn`
    timestamp_start = dt.datetime.fromisoformat(configs["data_input"]["start_time"])
    timestamp_end = dt.datetime.fromisoformat(configs["data_input"]["end_time"])

    if configs["data_processing"]["resample"]["bin_closed"] == "left":
        mask = (df_inputdata.start >= timestamp_start) & (
            df_inputdata.end < timestamp_end
        )

    elif configs["data_processing"]["resample"]["bin_closed"] == "right":
        mask = (df_inputdata.start > timestamp_start) & (
            df_inputdata.end <= timestamp_end
        )

    else:
        raise ConfigsError(
            'configs["data_processing"]["resample"]["bin_closed"] must be "left" or "right"'
        )

    df_inputdata = df_inputdata.loc[mask]

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
        needed_columns=configs["data_input"]["predictor_columns"],
    )

    # save final input data based on data config format
    predictor_path = (
        Path(configs["data_output"]["exp_dir"]) / "predictors_target_config.json"
    )
    final_predictors_data = {}
    final_predictors_data["predictors"] = []
    with open(predictor_path, "w") as fp:
        final_predictors_data["predictors"] = [
            p
            for p in configs_input["predictors"]
            if p["column"] in list(data_full_p.columns)
        ]
        for t in configs_input["targets"]:
            if t["column"] == configs["data_input"]["target_var"]:
                final_predictors_data["target"] = t
        json.dump(final_predictors_data, fp, ensure_ascii=False)

    # read in target data
    target_data_info = df_inputdata[df_inputdata.contentType == "targets"]
    data_full_t = _concat_data_from_files(
        target_data_info.path, needed_columns=[configs["data_input"]["target_var"]]
    )

    if data_full_p.empty:
        message = "No predictor data found in dataset for specified timeframe."
        logger.info(f"{message} Exiting process...")

        raise ConfigsError(message)

    elif (
        data_full_t.empty and configs["learning_algorithm"]["use_case"] != "prediction"
    ):
        message = "No target data found in dataset for specified timeframe."
        logger.info(f"{message} Exiting process...")

        raise ConfigsError(message)

    # the rest of the code expects a shape with a predictor column.
    # TODO: remove if
    if configs["learning_algorithm"]["use_case"] == "prediction":
        data_full = data_full_p
        data_full[configs["data_input"]["target_var"]] = -999
    else:
        data_full = pd.merge(data_full_p, data_full_t, how="outer", on="Timestamp")

    return data_full
