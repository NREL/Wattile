import datetime as dt
import json
import logging
import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch

# import tables
from intelcamp.error import ConfigsError

PROJECT_DIRECTORY = pathlib.Path(__file__).resolve().parent

logger = logging.getLogger(str(os.getpid()))


def check_complete(torch_file, des_epochs):
    """
    Checks if an existing training session is complete
    :param results_dir:
    :param epochs:
    :return:
    """

    torch_model = torch.load(torch_file)
    check = des_epochs == torch_model["epoch_num"] + 1
    return check


def time_dummies(data, configs):  # noqa: C901 TODO: remove noqa
    """
    Adds time-based indicator variables. Elements in configs describe what method to use.
    regDummy: Binary indicator variables, one column for each entry.
    fuzzy: Same as regDummy, but binary edges are smoothed.
    sincos: Cyclic time variables are used (one sin column and one cos column)

    :param data: (DataFrame)
    :param configs: (Dictionary)
    :return: (Dictionary)
    """

    # HOD
    if "sincos" in configs["HOD"]:
        data["sin_HOD"] = np.sin(
            2
            * np.pi
            * (
                data.index.hour * 3600 + data.index.minute * 60 + data.index.second
            ).values
            / (24 * 60 * 60)
        )
        data["cos_HOD"] = np.cos(
            2
            * np.pi
            * (
                data.index.hour * 3600 + data.index.minute * 60 + data.index.second
            ).values
            / (24 * 60 * 60)
        )
    if "binary_reg" in configs["HOD"]:
        for i in range(0, 24):
            data["HOD_binary_reg_{}".format(i)] = (data.index.hour == i).astype(int)

        # data = data.join(
        #     pd.get_dummies(
        #         data.index.hour, prefix="HOD_binary_reg", drop_first=True
        #     ).set_index(data.index)
        # )

    if "binary_fuzzy" in configs["HOD"]:
        for HOD in range(0, 24):
            data["HOD_binary_fuzzy_{}".format(HOD)] = np.maximum(
                1 - abs((data.index.hour + data.index.minute / 60) - HOD) / 1, 0
            )

    # DOW
    if "binary_reg" in configs["DOW"]:
        for i in range(0, 7):
            data["DOW_binary_reg_{}".format(i)] = (data.index.weekday == i).astype(int)
        # data = data.join(
        #     pd.get_dummies(
        #         data.index.weekday, prefix="DOW_binary_reg", drop_first=True
        #     ).set_index(data.index)
        # )
    if "binary_fuzzy" in configs["DOW"]:
        for i in range(0, 7):
            data["DOW_binary_fuzzy_{}".format(i)] = (data.index.weekday == i).astype(
                int
            )
        # data = data.join(
        #     pd.get_dummies(
        #         data.index.weekday, prefix="DOW_binary_fuzzy", drop_first=True
        #     ).set_index(data.index)
        # )
        for DOW in range(0, 7):
            data["DOW_binary_fuzzy_{}".format(DOW)] = np.maximum(
                1 - abs((data.index.weekday + data.index.hour / 24) - DOW) / 1, 0
            )

    # MOY
    if "sincos" in configs["MOY"]:
        data["sin_MOY"] = np.sin(2 * np.pi * (data.index.dayofyear).values / (365))
        data["cos_MOY"] = np.cos(2 * np.pi * (data.index.dayofyear).values / (365))

    if "Holidays" in configs and configs["Holidays"]:
        # -----Automatic (fetches federal holidays based on dates in imported data
        # cal = USFederalHolidayCalendar()
        # holidays = cal.holidays(
        #     start=data.index[0].strftime("%Y-%m-%d"),
        #     end=data.index[-1].strftime("%Y-%m-%d"),
        # )
        # data['Holiday'] = pd.to_datetime(data.index.date).isin(holidays).astype(int)

        # -----Read from JSON file
        with open("holidays.json", "r") as read_file:
            holidays = json.load(read_file)
        data["Holiday"] = pd.to_datetime(data.index.date).isin(holidays).astype(int)
        data["Holiday_forward"] = (
            pd.to_datetime(data.index.date + dt.timedelta(days=1))
            .isin(holidays)
            .astype(int)
        )

    return data


def input_data_split(data, configs):
    """
    Split a data set into a training set and a validation (val) set.
    Methods: "Random" or "Sequential", specified in configs

    :param data: (DataFrame)
    :param configs: (Dict)
    :return:
    """
    train_ratio = int(configs["data_split"].split(":")[0]) / 100
    val_ratio = int(configs["data_split"].split(":")[1]) / 100
    test_ratio = int(configs["data_split"].split(":")[2]) / 100

    file_prefix = Path(configs["exp_dir"])

    if configs["train_val_split"] == "Random":
        pathlib.Path(configs["data_dir"]).mkdir(parents=True, exist_ok=True)
        mask_file = os.path.join(file_prefix, "mask.h5")
        logger.info("Creating random training mask and writing to file")

        # If you want to group datasets together into sequential chunks
        if configs["splicer"]["active"]:
            # Set indices for training set
            np.random.seed(seed=configs["random_seed"])
            splicer = (
                (data.index - data.index[0]) // pd.Timedelta(configs["splicer"]["time"])
            ).values
            num_chunks = splicer[-1]
            num_train_chunks = (train_ratio * num_chunks) - (
                (train_ratio * num_chunks) % configs["train_size_factor"]
            )
            if num_train_chunks == 0:
                raise Exception(
                    "Total number of data chunks is zero. train_size_factor value might be too "
                    "large compared to the data size. Exiting.."
                )

            msk = np.zeros(data.shape[0]) + 2
            train_chunks = np.random.choice(
                np.arange(num_chunks), replace=False, size=int(num_train_chunks)
            )
            for chunk in train_chunks:
                indices = np.where(splicer == chunk)
                msk[indices] = 0

            # Set indices for validation and test set
            remaining_chunks = np.setdiff1d(np.arange(num_chunks), train_chunks)
            if test_ratio == 0:
                msk[msk != 0] = 1
            else:
                num_val_chunks = int(
                    (val_ratio / (1 - train_ratio)) * remaining_chunks.shape[0]
                )
                val_chunks = np.random.choice(
                    remaining_chunks, replace=False, size=num_val_chunks
                )
                for chunk in val_chunks:
                    indices = np.where(splicer == chunk)
                    msk[indices] = 1

        # If you DONT want to group data into sequential chunks
        else:
            # Set indices for training set
            np.random.seed(seed=configs["random_seed"])
            data_size = data.shape[0]
            num_ones = (train_ratio * data_size) - (
                (train_ratio * data_size) % configs["train_size_factor"]
            )
            msk = np.zeros(data_size) + 2
            indices = np.random.choice(
                np.arange(data_size), replace=False, size=int(num_ones)
            )
            msk[indices] = 0

            # Set indices for validation and test set
            remaining_indices = np.where(msk != 0)[0]
            if test_ratio == 0:
                msk[remaining_indices] = 1
            else:
                num_val = int(
                    (val_ratio / (1 - train_ratio)) * remaining_indices.shape[0]
                )
                val_indices = np.random.choice(
                    remaining_indices, replace=False, size=num_val
                )
                msk[val_indices] = 1

        logger.info(
            "Train: {}, validation: {}, test: {}".format(
                (msk == 0).sum() / msk.shape[0],
                (msk == 1).sum() / msk.shape[0],
                (msk == 2).sum() / msk.shape[0],
            )
        )
        # Assign dataframes
        train_df = data[msk == 0]
        val_df = data[msk == 1]
        test_df = data[msk == 2]

        # Save test_df to file for later use
        test_df.to_hdf(
            os.path.join(file_prefix, "internal_test.h5"), key="df", mode="w"
        )

        # Still save dataframe to file to preserve timeseries index
        mask = pd.DataFrame()
        mask["msk"] = msk
        mask.index = data.index
        mask.to_hdf(mask_file, key="df", mode="w")

        # Get rid of datetime index
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

    else:
        raise ConfigsError(
            "{} is not a supported form of data splitting".format(
                configs["train_val_split"]
            )
        )

    return train_df, val_df


def pad_full_data(data, configs):
    """
    Create lagged versions of exogenous variables in a DataFrame.
    Used specifically for RNN and LSTM deep learning methods.

    :param data: (DataFrame)
    :param configs: (Dict)
    :return: (DataFrame)
    """
    target = data[configs["target_var"]]
    data = data.drop(configs["target_var"], axis=1)
    data_orig = data

    # Pad the exogenous variables
    temp_holder = list()
    temp_holder.append(data_orig)
    for i in range(1, configs["window"] + 1):
        shifted = (
            data_orig.shift(i * int(configs["sequence_freq_min"]), freq="min")
            .astype("float32")
            .add_suffix("_lag{}".format(i))
        )
        temp_holder.append(shifted)
    temp_holder.reverse()
    data = pd.concat(temp_holder, axis=1)

    # If this is a linear quantile regression model (iterative)
    if configs["arch_type"] == "quantile" and configs["iterative"]:
        for i in range(0, configs["EC_future_gap_min"]):
            if i == 0:
                data[configs["target_var"]] = target
            else:
                data["{}_lag_{}".format(configs["target_var"], i)] = target.shift(-i)

        # Drop all nans
        data = data.dropna(how="any")

    # If this is a linear quantile regression model (point)
    elif configs["arch_type"] == "quantile" and not configs["iterative"]:
        # Re-append the shifted target column to the dataframe
        data[configs["target_var"]] = target.shift(-configs["EC_future_gap_min"])

        # Drop all nans
        data = data.dropna(how="any")

        # Adjust time index to match the EC values
        data.index = data.index + pd.DateOffset(minutes=(configs["EC_future_gap_min"]))

    # If this is an RNN model
    elif configs["arch_type"] == "RNN":
        # Re-append the shifted target column to the dataframe
        data[configs["target_var"]] = target.shift(-configs["EC_future_gap_min"])

        # Drop all nans
        data = data.dropna(how="any")

        # Adjust time index to match the EC values
        data.index = data.index + pd.DateOffset(minutes=(configs["EC_future_gap_min"]))

    return data


def pad_full_data_s2s(data, configs):
    """
    Create lagged versions of exogenous variables in a DataFrame.
    Used specifically for Sequence to Sequence (S2S) deep learning methods.

    :param data: (DataFrame)
    :param configs: (Dict)
    :return: (DataFrame)
    """

    target = data[configs["target_var"]]
    data = data.drop(configs["target_var"], axis=1)
    data_orig = data
    # Pad the exogenous variables
    temp_holder = list()
    temp_holder.append(data_orig)
    for i in range(1, configs["window"] + 1):
        shifted = (
            data_orig.shift(i * int(configs["sequence_freq_min"]), freq="min")
            .astype("float32")
            .add_suffix("_lag{}".format(i))
        )
        temp_holder.append(shifted)
    temp_holder.reverse()
    data = pd.concat(temp_holder, axis=1)

    # Do fine padding for future predictions. Create a new df to preserve memory usage.
    local = pd.DataFrame()
    for i in range(0, configs["S2S_stagger"]["initial_num"]):
        local["{}_lag_{}".format(configs["target_var"], i)] = target.shift(
            -i * int(configs["sequence_freq_min"]), freq="min"
        )

    # Do additional coarse padding for future predictions
    for i in range(1, configs["S2S_stagger"]["secondary_num"] + 1):
        base = configs["S2S_stagger"]["initial_num"]
        new = base + configs["S2S_stagger"]["decay"] * i
        local["{}_lag_{}".format(configs["target_var"], base + i)] = target.shift(
            -new * int(configs["sequence_freq_min"], freq="min")
        )

    data = pd.concat([data, local], axis=1)

    # Drop all nans
    data = data.dropna(how="any")

    return data


def corr_heatmap(data):
    """
    Plot a correlation heatmap to see the (linear) relationships between exogenous variables

    :param data: (DataFrame)
    :return: None
    """
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    sns.heatmap(
        data.corr(), mask=mask, cmap="coolwarm", linewidths=1, linecolor="white"
    )


def correct_predictor_columns(configs, data):
    """assert we have the correct columns and order them

    :param configs: configs
    :type configs: dict
    :param data: data
    :type data: pandas.DataFrame
    :raises ConfigsError:if data doesn't contain needed columns
    :return: data with correct columns
    :rtype: pandas.DataFrame
    """
    keep_cols = configs["predictor_columns"] + [configs["target_var"]]

    # raise error if missing columns
    missing_colums = set(keep_cols).difference(set(data.columns))
    if len(missing_colums) > 0:
        raise ConfigsError(f"data is missing predictor_columns: {missing_colums}")

    # remove extra columns
    extra_colums = set(data.columns).difference(set(keep_cols))
    if len(extra_colums) > 0:
        data = data[keep_cols]
        logger.info(
            f"Removed columns from data that are not specified in \
            configs['predictor_columns']: {extra_colums}"
        )

    # sort columns
    return data.reindex(keep_cols, axis="columns")


def correct_timestamps(configs, data):
    """sort and trim data specified time period

    :param configs: configs
    :type configs: dict
    :param data: data
    :type data: pandas.DataFrame
    :raises ConfigsError: no data within specified time period
    :return: sorted data for time period
    :rtype: pandas.DataFrame
    """
    data = data.sort_index()

    # TODO: think about timezones.
    start_time = dt.datetime.fromisoformat((configs["start_time"]))
    end_time = dt.datetime.fromisoformat(configs["end_time"])
    data = data[start_time:end_time]

    if data.shape[0] == 0:
        raise ConfigsError(
            "data has no data within specified time period:"
            f"{start_time.year}-{start_time.month}-{start_time.day} to "
            f"{end_time.year}-{end_time.month}-{end_time.day}"
        )

    return data


def _preprocess_data(configs, data):
    """Preprocess data as dictated by the configs.

    :param configs: configs
    :type configs: dict
    :param data: data
    :type data: pd.dataframe
    :return: data
    :rtype: pd.dataframe
    """
    # assert we have the correct columns and order them
    data = correct_predictor_columns(configs, data)

    # sort and trim data specified time period
    data = correct_timestamps(configs, data)

    # Add time-based features
    data = time_dummies(data, configs)

    # Add statistics features
    if configs["rolling_window"]["active"]:
        data = rolling_stats(data, configs)

    # Add lag features
    configs["input_dim"] = data.shape[1] - 1
    logger.info("Number of features: {}".format(configs["input_dim"]))
    logger.debug("Features: {}".format(data.columns.values))

    if configs["arch_version"] == 4:
        data = pad_full_data(data, configs)
    elif configs["arch_version"] == 5:
        data = pad_full_data_s2s(data, configs)

    return data


def prep_for_rnn(configs, data):
    """
    Prepare data for input to a RNN model.

    :param configs: (Dict)
    :return: train and val DataFrames
    """
    data = _preprocess_data(configs, data)

    # if validatate with external data, write data to h5 for future testing.
    if configs["use_case"] == "validation" and configs["test_method"] == "external":
        filepath = (
            pathlib.Path(configs["data_dir"])
            / f"{configs['target_var']}_external_test.h5"
        )
        data.to_hdf(filepath, key="df", mode="w")

    if configs["use_case"] == "train":
        train_df, val_df = input_data_split(data, configs)

    else:
        train_df, val_df = pd.DataFrame(), data

    return train_df, val_df


def rolling_stats(data, configs):
    # Convert data to rolling average (except output) and create min, mean, and max columns
    target = data[configs["target_var"]]
    X_data = data.drop(configs["target_var"], axis=1)

    # inferring timestep (frequency) from the dataframe
    dt = configs["data_time_interval_mins"]
    windowsize = int(configs["rolling_window"]["minutes"] / dt) + 1
    logging.debug(
        "Feature extraction: rolling window size = {} rows".format(windowsize)
    )

    if configs["rolling_window"]["type"] == "rolling":
        mins = X_data.rolling(window=windowsize, min_periods=1).min().add_suffix("_min")
        means = (
            X_data.rolling(window=windowsize, min_periods=1).mean().add_suffix("_mean")
        )
        maxs = X_data.rolling(window=windowsize, min_periods=1).max().add_suffix("_max")
        data = pd.concat([mins, means, maxs], axis=1)
        data[configs["target_var"]] = target

    elif configs["rolling_window"]["type"] == "binned":
        mins = (
            X_data.resample(str(configs["rolling_window"]["minutes"]) + "T")
            .min()
            .add_suffix("_min")
        )
        means = (
            X_data.resample(str(configs["rolling_window"]["minutes"]) + "T")
            .mean()
            .add_suffix("_mean")
        )
        maxs = (
            X_data.resample(str(configs["rolling_window"]["minutes"]) + "T")
            .max()
            .add_suffix("_max")
        )
        data = pd.concat([mins, means, maxs], axis=1)
        data[configs["target_var"]] = (
            pd.DataFrame(target)
            .resample(str(configs["rolling_window"]["minutes"]) + "T")
            .mean()
        )

    return data
