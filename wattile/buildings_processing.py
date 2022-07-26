import datetime as dt
import logging
import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch

# import tables
from wattile.error import ConfigsError
from wattile.time_processing import add_processed_time_columns

PROJECT_DIRECTORY = pathlib.Path(__file__).resolve().parent

logger = logging.getLogger(str(os.getpid()))


def check_complete(torch_file, des_epochs):
    """
    Checks if an existing training session is complete
    :param results_dir:
    :param epochs:
    :return:
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = torch.load(torch_file, map_location=device)
    check = des_epochs == torch_model["epoch_num"] + 1
    return check


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
        if configs["sequential_splicer"]["active"]:
            # Set indices for training set
            np.random.seed(seed=configs["random_seed"])
            splicer = (
                (data.index - data.index[0]) // pd.Timedelta(configs["sequential_splicer"]["window_width"])
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
    for i in range(1, configs["feat_timelag"]["lag_count"] + 1):
        shifted = (
            data_orig.shift(configs["feat_timelag"]["lag_interval"])
            .astype("float32")
            .add_suffix("_lag{}".format(i))
        )
        temp_holder.append(shifted)
    temp_holder.reverse()
    data = pd.concat(temp_holder, axis=1)

    # If this is a linear quantile regression model (iterative)
    if configs["arch_type"] == "quantile" and configs["iterative"]:
        for i in range(0, configs["feat_timelag"]["lag_interval_target"]):
            if i == 0:
                data[configs["target_var"]] = target
            else:
                data["{}_lag_{}".format(configs["target_var"], i)] = target.shift(-i)

        # Drop all nans
        data = data.dropna(how="any")

    # If this is a linear quantile regression model (point)
    elif configs["arch_type"] == "quantile" and not configs["iterative"]:
        # Re-append the shifted target column to the dataframe
        data[configs["target_var"]] = target.shift(-configs["feat_timelag"]["lag_interval_target"])

        # Drop all nans
        data = data.dropna(how="any")

        # Adjust time index to match the EC values
        data.index = data.index + pd.DateOffset(minutes=(configs["feat_timelag"]["lag_interval_target"]))

    # If this is an RNN model
    elif configs["arch_type"] == "RNN":
        # Re-append the shifted target column to the dataframe
        data[configs["target_var"]] = target.shift(-configs["feat_timelag"]["lag_interval_target"])

        # Drop all nans
        data = data.dropna(how="any")

        # Adjust time index to match the EC values
        data.index = data.index + pd.DateOffset(minutes=(configs["feat_timelag"]["lag_interval_target"]))

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
    for i in range(1, configs["feat_timelag"]["lag_count"] + 1):
        shifted = (
            data_orig.shift(configs["feat_timelag"]["lag_interval"])
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
            -i * int(configs["feat_timelag"]["lag_interval"]), freq="min"
        )

    # Do additional coarse padding for future predictions
    for i in range(1, configs["S2S_stagger"]["secondary_num"] + 1):
        base = configs["S2S_stagger"]["initial_num"]
        new = base + configs["S2S_stagger"]["decay"] * i
        local["{}_lag_{}".format(configs["target_var"], base + i)] = target.shift(
            -new * int(configs["feat_timelag"]["lag_interval"], freq="min")
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
    data = add_processed_time_columns(data, configs)

    # Add statistics features
    if configs["feat_stats"]["active"]:
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

    # reading configuration parameters
    # window_closing and window_position are hard coded for now
    # default is right-closed and backward-looking window
    window_width = configs["feat_stats"]["window_width"]
    window_increment = configs["feat_stats"]["window_increment"]
    window_closing = "right"  # left, right
    window_position = "backward"  # forward, center, backward

    # seperate predictors and target
    target = data[configs["target_var"]]
    X_data = data.drop(configs["target_var"], axis=1)

    # resampling for each statistics separately
    data_resampler = X_data.resample(
        rule=window_increment, closed=window_closing, label=window_closing
    )
    data_resample_min = data_resampler.min().add_suffix("_min")
    data_resample_max = data_resampler.max().add_suffix("_max")
    data_resample_sum = data_resampler.sum().add_suffix("_sum")
    data_resample_count = data_resampler.count().add_suffix("_count")

    # setting configuration settings depending on window_position and window_closing
    if window_position == "backward":
        arg_center = False
    elif window_position == "center":
        arg_center = True
    elif window_position == "forward":
        arg_center = False
        data_resample_min = data_resample_min[::-1]
        data_resample_max = data_resample_max[::-1]
        data_resample_sum = data_resample_sum[::-1]
        data_resample_count = data_resample_count[::-1]
        if window_closing == "left":
            window_closing = "right"
        elif window_closing == "right":
            window_closing = "left"

    # adding rolling window statistics: minimum
    mins = data_resample_min.rolling(
        window=window_width, min_periods=1, center=arg_center, closed=window_closing
    ).min()

    # adding rolling window statistics: maximum
    maxs = data_resample_max.rolling(
        window=window_width, min_periods=1, center=arg_center, closed=window_closing
    ).max()

    # adding rolling window statistics: sum
    sums = data_resample_sum.rolling(
        window=window_width, min_periods=1, center=arg_center, closed=window_closing
    ).sum()

    # adding rolling window statistics: count
    counts = data_resample_count.rolling(
        window=window_width, min_periods=1, center=arg_center, closed=window_closing
    ).sum()  # this has to be sum for proper count calculation

    # adding rolling window statistics: mean
    means = sums.copy()
    means.columns = means.columns.str.replace("_sum", "_mean")
    np.seterr(invalid="ignore")  # supress/hide the warning
    means.loc[:, :] = sums.values / counts.values

    # combining min and max stats
    data = pd.concat([mins, maxs, means], axis=1)

    # reordering dataframe based on window_position
    if window_position == "forward":
        data = data[::-1]

    # adding target back to the dataframe
    target = target.asfreq(freq=window_increment, method="pad")
    data[configs["target_var"]] = target

    return data
