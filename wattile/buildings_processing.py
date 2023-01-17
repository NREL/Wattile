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


def _create_split_mask(timestamp, data_size, configs):

    # set configuration parameters
    np.random.seed(seed=configs["data_processing"]["random_seed"])
    active_sequential = configs["data_processing"]["sequential_splicer"]["active"]
    train_ratio = int(configs["data_processing"]["data_split"].split(":")[0]) / 100
    val_ratio = int(configs["data_processing"]["data_split"].split(":")[1]) / 100
    test_ratio = int(configs["data_processing"]["data_split"].split(":")[2]) / 100
    window_witdh = configs["data_processing"]["sequential_splicer"]["window_width"]
    train_size_factor = configs["data_processing"]["train_size_factor"]

    # split data based on random sequential chunks
    if active_sequential:
        # set indices for training set
        splicer = ((timestamp - timestamp[0]) // pd.Timedelta(window_witdh)).values
        num_chunks = splicer[-1]
        num_train_chunks = (train_ratio * num_chunks) - (
            (train_ratio * num_chunks) % train_size_factor
        )
        if num_train_chunks == 0:
            raise Exception(
                "Total number of data chunks is zero. train_size_factor value might be too "
                "large compared to the data size. Exiting.."
            )
        msk = np.zeros(timestamp.shape[0]) + 2
        train_chunks = np.random.choice(
            np.arange(num_chunks), replace=False, size=int(num_train_chunks)
        )
        for chunk in train_chunks:
            indices = np.where(splicer == chunk)
            msk[indices] = 0

        # set indices for validation and test set
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

    # split data based on random timestamp sampling
    else:
        # set indices for training set
        num_ones = (train_ratio * data_size) - (
            (train_ratio * data_size) % train_size_factor
        )
        msk = np.zeros(data_size) + 2
        indices = np.random.choice(
            np.arange(data_size), replace=False, size=int(num_ones)
        )
        msk[indices] = 0

        # set indices for validation and test set
        remaining_indices = np.where(msk != 0)[0]
        if test_ratio == 0:
            msk[remaining_indices] = 1
        else:
            num_val = int((val_ratio / (1 - train_ratio)) * remaining_indices.shape[0])
            val_indices = np.random.choice(
                remaining_indices, replace=False, size=num_val
            )
            msk[val_indices] = 1

    return msk


def input_data_split(data, configs):
    """
    Split a data set into a training set and a validation (val) set.
    Methods: "Random" or "Sequential", specified in configs

    :param data: (DataFrame)
    :param configs: (Dict)
    :return:
    """
    # setting configuration parameters
    arch_version = configs["learning_algorithm"]["arch_version"]
    file_prefix = Path(configs["data_output"]["exp_dir"])
    mask_file = os.path.join(file_prefix, "mask.h5")

    # assign timestamp and data size depending on arch_version
    if (arch_version == "alfa") | (arch_version == "bravo"):
        timestamp = data.index
        data_size = data.shape[0]

    elif arch_version == "charlie":
        timestamp = data["timestamp"]
        data_size = data["predictor"].shape[0]

    msk = _create_split_mask(timestamp, data_size, configs)

    # assign train, validation, and test data
    if (arch_version == "alfa") | (arch_version == "bravo"):
        train_df = data[msk == 0]
        val_df = data[msk == 1]
        test_df = data[msk == 2]
        # Get rid of datetime index
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        # save test_df to file for later use
        test_df.to_hdf(
            os.path.join(file_prefix, "internal_test.h5"), key="df", mode="w"
        )

    elif arch_version == "charlie":
        train_df = {}
        val_df = {}
        train_df_predictor = data["predictor"][msk == 0, :, :]
        train_df_target = data["target"][msk == 0, :, :]
        val_df_predictor = data["predictor"][msk == 1, :, :]
        val_df_target = data["target"][msk == 1, :, :]
        test_df_predictor = data["predictor"][msk == 2, :, :]
        test_df_target = data["target"][msk == 2, :, :]
        train_df["predictor"] = train_df_predictor
        train_df["target"] = train_df_target
        val_df["predictor"] = val_df_predictor
        val_df["target"] = val_df_target
        # save test_df to file for later use
        np.save(
            os.path.join(file_prefix, "internal_test_predictor.npy"), test_df_predictor
        )
        np.save(os.path.join(file_prefix, "internal_test_target.npy"), test_df_target)

    # save mask file to preserve timeseries index
    mask = pd.DataFrame()
    mask["msk"] = msk
    mask["index"] = timestamp
    mask = mask.set_index("index")
    mask.to_hdf(mask_file, key="df", mode="w")

    return train_df, val_df


def timelag_predictors(data, configs):
    """
    Create lagged versions of predictor variables in a DataFrame.
    Used specifically for alfa learning methods.
    :param data: (DataFrame)
    :param configs: (Dict)
    :return: (DataFrame)
    """

    # reading configuration parameters
    lag_interval = configs["data_processing"]["feat_timelag"]["lag_interval"]
    lag_count = configs["data_processing"]["feat_timelag"]["lag_count"]
    window_width_futurecast = configs["data_processing"]["input_output_window"][
        "window_width_futurecast"
    ]
    target_var = configs["data_input"]["target_var"]

    # splitting predictors and target
    target = data[target_var]
    data = data.drop(target_var, axis=1)
    data_orig = data

    # padding predictors
    temp_holder = list()
    temp_holder.append(data_orig)
    for i in range(1, lag_count + 1):
        shifted = (
            data_orig.shift(freq=i * lag_interval)
            .astype("float32")
            .add_suffix("_lag{}".format(i))
        )
        temp_holder.append(shifted)
    temp_holder.reverse()
    data = pd.concat(temp_holder, axis=1)

    if configs["learning_algorithm"]["use_case"] != "prediction":
        data[target_var] = target.shift(freq="-" + window_width_futurecast)
    else:
        data[target_var] = 0  # dummy

    data = data.dropna(how="any")

    return data


def timelag_predictors_target(data, configs):
    """
    Create lagged versions of predictor and target variables in a DataFrame.
    Used specifically for bravo learning methods.
    :param data: (DataFrame)
    :param configs: (Dict)
    :return: (DataFrame)
    """

    # reading configuration parameters
    lag_interval = configs["data_processing"]["feat_timelag"]["lag_interval"]
    lag_count = configs["data_processing"]["feat_timelag"]["lag_count"]
    window_width_target = configs["data_processing"]["input_output_window"][
        "window_width_target"
    ]
    window_width_futurecast = configs["data_processing"]["input_output_window"][
        "window_width_futurecast"
    ]
    bin_interval = configs["data_processing"]["resample"]["bin_interval"]
    initial_num = (pd.Timedelta(window_width_target) // pd.Timedelta(bin_interval)) + 1
    target_var = configs["data_input"]["target_var"]
    target_temp = data[target_var].copy()

    # shift target for futurecast
    data[target_var] = target_temp.shift(freq="-" + window_width_futurecast)

    # split predictors and target
    target = data[target_var]
    data = data.drop(target_var, axis=1)
    data_orig = data

    # Pad the exogenous variables
    temp_holder = list()
    temp_holder.append(data_orig)
    for i in range(1, lag_count + 1):
        shifted = (
            data_orig.shift(freq=i * lag_interval)
            .astype("float32")
            .add_suffix("_lag{}".format(i))
        )
        temp_holder.append(shifted)
    temp_holder.reverse()
    data = pd.concat(temp_holder, axis=1)

    # Do fine padding for future predictions. Create a new df to preserve memory usage.
    local = pd.DataFrame()
    for i in range(0, initial_num):
        if i == 0:
            local["{}_lag_{}".format(target_var, i)] = target.shift(i)
        else:
            local["{}_lag_{}".format(target_var, i)] = target.shift(
                freq="-" + (i * bin_interval)
            )

    if configs["learning_algorithm"]["use_case"] != "prediction":
        data = pd.concat([data, local], axis=1)
    else:
        for col in local.columns:
            data[col] = 0  # dummy

    data = data.dropna(how="any")

    return data


def roll_predictors_target(data, configs):
    """
    Create rolling windows of predictor and target variables in a DataFrame.
    Used specifically for charlie learning methods.

    :param data: (DataFrame)
    :param configs: (Dict)
    :return: (Dict)
    """

    # setting configuration parameters
    window_width_source = configs["data_processing"]["input_output_window"][
        "window_width_source"
    ]
    window_width_futurecast = configs["data_processing"]["input_output_window"][
        "window_width_futurecast"
    ]
    window_width_target = configs["data_processing"]["input_output_window"][
        "window_width_target"
    ]
    bin_interval = configs["data_processing"]["resample"]["bin_interval"]
    target_var = configs["data_input"]["target_var"]

    # initialize lists
    data_predictor = []
    data_target = []

    # calculate number of rows based on window size defined by time
    window_source_size_count = pd.Timedelta(window_width_source) // pd.Timedelta(
        bin_interval
    )
    window_target_size_count = pd.Timedelta(window_width_target) // pd.Timedelta(
        bin_interval
    )
    window_futurecast_size_count = pd.Timedelta(
        window_width_futurecast
    ) // pd.Timedelta(bin_interval)

    # set aside timeindex
    timestamp = data.iloc[
        window_source_size_count : -(
            window_target_size_count + window_futurecast_size_count
        ),
        :,
    ].index

    # create 3D predictor data
    data_shifted_predictor = data.iloc[
        : -(window_target_size_count + window_futurecast_size_count), :
    ].loc[:, data.columns != target_var]
    for window in data_shifted_predictor.rolling(
        window=window_width_source, closed="both"
    ):
        if window.shape[0] == window_source_size_count + 1:
            data_predictor.append(
                window.values.reshape(
                    (1, window_source_size_count + 1, data_shifted_predictor.shape[1])
                )
            )
    # reshape data dimension
    data_predictor = np.concatenate(np.array(data_predictor), axis=0)

    # create 3D target data
    data_shifted_target = data.iloc[
        (window_source_size_count + window_futurecast_size_count) :, :
    ][target_var]
    for window in data_shifted_target.rolling(
        window=window_width_target, closed="both"
    ):
        if window.shape[0] == window_target_size_count + 1:
            data_target.append(
                window.values.reshape((1, window_target_size_count + 1, 1))
            )
    # reshape data dimension
    data_target = np.concatenate(np.array(data_target), axis=0)

    # combine 3D predictor and target data into dictionary
    data = {}
    data["predictor"] = data_predictor
    data["target"] = data_target
    data["timestamp"] = timestamp

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
    if configs["data_input"]["predictor_columns"] != []:

        keep_cols = configs["data_input"]["predictor_columns"] + [
            configs["data_input"]["target_var"]
        ]

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

    else:
        # not validating pre-defined predictor list
        keep_cols = list(data.columns)

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
    start_time = dt.datetime.fromisoformat(configs["data_input"]["start_time"])
    end_time = dt.datetime.fromisoformat(configs["data_input"]["end_time"])

    if configs["data_processing"]["resample"]["bin_closed"] == "left":
        mask = (data.index >= start_time) & (data.index < end_time)

    elif configs["data_processing"]["resample"]["bin_closed"] == "right":
        mask = (data.index > start_time) & (data.index <= end_time)

    else:
        raise ConfigsError(
            'configs["data_processing"]["resample"]["bin_closed"] must be "left" or "right"'
        )

    data = data.loc[mask]

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
    if configs["data_processing"]["feat_stats"]["active"]:
        data = roll_data(data, configs)
    else:
        data = resample_data(data, configs)

    # Add lag features
    configs["input_dim"] = data.shape[1] - 1
    logger.info("Number of features: {}".format(configs["input_dim"]))
    logger.debug("Features: {}".format(data.columns.values))

    if configs["learning_algorithm"]["arch_version"] == "alfa":
        data = timelag_predictors(data, configs)
    elif configs["learning_algorithm"]["arch_version"] == "bravo":
        data = timelag_predictors_target(data, configs)
    elif configs["learning_algorithm"]["arch_version"] == "charlie":
        data = roll_predictors_target(data, configs)

    return data


def prep_for_rnn(configs, data):
    """
    Prepare data for input to a RNN model.

    :param configs: (Dict)
    :return: train and val DataFrames
    """
    data = _preprocess_data(configs, data)

    # if validatate with external data, write data to h5 for future testing.
    if (
        configs["learning_algorithm"]["use_case"] == "validation"
        and configs["learning_algorithm"]["test_method"] == "external"
    ):
        filepath = pathlib.Path(
            configs["data_input"]["data_dir"]
        ) / "{}_external_test.h5".format(configs["data_input"]["target_var"])
        data.to_hdf(filepath, key="df", mode="w")

    if configs["learning_algorithm"]["use_case"] == "train":
        train_df, val_df = input_data_split(data, configs)

    else:
        train_df, val_df = pd.DataFrame(), data

    return train_df, val_df


def roll_data(data, configs):
    # reading configuration parameters.
    bin_interval = configs["data_processing"]["resample"]["bin_interval"]
    bin_closed = configs["data_processing"]["resample"]["bin_closed"]
    bin_label = configs["data_processing"]["resample"]["bin_label"]
    window_width = configs["data_processing"]["feat_stats"]["window_width"]

    # seperate predictors and target
    target = data[configs["data_input"]["target_var"]]
    X_data = data.drop(configs["data_input"]["target_var"], axis=1)

    # resampling for each statistics separately
    data_resampler = X_data.resample(
        rule=bin_interval, closed=bin_closed, label=bin_label
    )
    data_resample_min = data_resampler.min().add_suffix("_min")
    data_resample_max = data_resampler.max().add_suffix("_max")
    data_resample_sum = data_resampler.sum().add_suffix("_sum")
    data_resample_count = data_resampler.count().add_suffix("_count")

    # adding rolling window statistics: minimum
    mins = data_resample_min.rolling(window=window_width, min_periods=1).min()

    # adding rolling window statistics: maximum
    maxs = data_resample_max.rolling(window=window_width, min_periods=1).max()

    # adding rolling window statistics: sum
    sums = data_resample_sum.rolling(window=window_width, min_periods=1).sum()

    # adding rolling window statistics: count
    counts = data_resample_count.rolling(
        window=window_width, min_periods=1
    ).sum()  # this has to be sum for proper count calculation

    # adding rolling window statistics: mean
    means = sums.copy()
    means.columns = means.columns.str.replace("_sum", "_mean")
    np.seterr(invalid="ignore")  # supress/hide the warning
    means.loc[:, :] = sums.values / counts.values

    # combining min and max stats
    data = pd.concat([mins, maxs, means], axis=1)

    # adding resampled target back to the dataframe
    target = resample_data(target, configs)
    data[configs["data_input"]["target_var"]] = target

    return data


def resample_data(data, configs):

    # reading configuration parameters.
    bin_interval = configs["data_processing"]["resample"]["bin_interval"]
    bin_closed = configs["data_processing"]["resample"]["bin_closed"]
    bin_label = configs["data_processing"]["resample"]["bin_label"]

    # resample data
    data = data.resample(rule=bin_interval, label=bin_label, closed=bin_closed)

    # take the closest value from the label
    if bin_label == "left":
        data = data.first()
    elif bin_label == "right":
        data = data.last()

    return data
