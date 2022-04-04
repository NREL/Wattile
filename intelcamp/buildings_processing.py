import numpy as np
import pathlib
import glob
import sys
import pandas as pd
import datetime as dt
# import tables
from pandas.tseries.holiday import USFederalHolidayCalendar, get_calendar
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import torch
from pathlib import Path
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
    model = torch_model['torch_model']
    check = des_epochs == torch_model['epoch_num']+1
    return check


def get_full_data(configs):
    """
    Fetches all data for a requested building based on the information reflected in the input data summary json file.

    :param configs: (Dictionary)
    :return: (DataFrame)
    """

    # assuming there is only one json file in the folder summerizing input data
    # read json file
    configs_file_inputdata = Path(configs['data_dir']) / configs['building'] / f"{configs['building']} Config.json"
    logger.info("Pre-process: reading input data summary json file from {}".format(configs_file_inputdata))
    with open(configs_file_inputdata, "r") as read_file:
        configs_input = json.load(read_file)

    # converting json into dataframe 
    df_inputdata = pd.DataFrame(configs_input['files'])

    # converting date time column into pandas datetime (raw format based on ISO 8601)
    df_inputdata['start'] = pd.to_datetime(df_inputdata.start, format="t:%Y-%m-%dT%H:%M:%S%z", exact=False, utc=True)
    df_inputdata['end'] = pd.to_datetime(df_inputdata.end, format="t:%Y-%m-%dT%H:%M:%S%z", exact=False, utc=True)

    # creating thresholds dates from configs json file
    timestamp_start = pd.Timestamp(configs['start_year'], configs['start_month'], configs['start_day'], 0)
    if (configs['end_month']==12) & (configs['end_day']==31):
        timestamp_end = pd.Timestamp(configs['end_year']+1, 1, 1, 0)
    else:
        timestamp_end = pd.Timestamp(configs['end_year'], configs['end_month']+1, configs['end_day'], 0)

    # filtering input data based on user specified date period
    df_inputdata = df_inputdata.loc[ (df_inputdata.start.dt.date>=timestamp_start) & (df_inputdata.end.dt.date<=timestamp_end) , :]
    df_inputdata['path'] = configs['data_dir'] + "/" + configs['building'] + "/" + df_inputdata['filename']
    
    if df_inputdata.empty:
        logger.info("Pre-process: measurements during the specified time period ({} to {}) are empty.".format(timestamp_start, timestamp_end))
        
        raise ConfigsError("No datapoints found in dataset for specified timeframe.")

    else:
        data_full_p = pd.DataFrame()
        data_full_t = pd.DataFrame()
        for datatype in df_inputdata.contentType.unique():
            
            df_list_datatype = df_inputdata.loc[df_inputdata.contentType==datatype,:]
            
            for filepath in df_list_datatype.path:
                
                if datatype=="predictors":
                    logger.info("Pre-process: reading predictor file = {}".format(filepath.split(configs['data_dir'])[1]))
                    try:
                        data_full_p = pd.concat([data_full_p, pd.read_csv(filepath)])
                    except:
                        logger.info("Pre-process: error in read_csv with predictor file {}. not reading..".format(filepath.split(configs['data_dir'])[1]))
                        continue
                elif datatype=="targets":
                    logger.info("Pre-process: reading target file = {}".format(filepath.split(configs['data_dir'])[1]))
                    try:
                        data_full_t = pd.concat([data_full_t, pd.read_csv(filepath)[['Timestamp', configs["target_var"]]]])
                    except:
                        logger.info("Pre-process: error in read_csv with target file {}. not reading..".format(filepath.split(configs['data_dir'])[1]))
                        continue
                else:
                    logger.info("Pre-process: input file not properly differentiated between Predictors and Targets")

        if data_full_p.empty:
            logger.info("Pre-process: predictor dataframe is empty. Exiting process...")

            raise ConfigsError("No datapoints found in dataset for specified timeframe.")

        elif data_full_t.empty and configs["use_case"] != "prediction":
            logger.info("Pre-process: target dataframe is empty. Exiting process...")

            raise ConfigsError("No datapoints found in dataset for specified timeframe.")          

        if configs["use_case"] == "prediction":
            data_full = data_full_p
            data_full[configs["target_var"]] = -999
            
        else:
            data_full = pd.merge(data_full_p, data_full_t, how='outer', on='Timestamp')
        
    data_full['Timestamp'] = pd.to_datetime(data_full['Timestamp'], format="%Y-%m-%dT%H:%M:%S%z", exact=False, utc=True)
    data_full = data_full.set_index('Timestamp')

    return data_full


def clean_data(data, configs):
    """
    Clean data that is passed in a DataFrame. Certain columns will be cleaned with pre-defined criteria.

    :param data: (DataFrame)
    :param configs: (Dictionary)
    :return: (DataFrame)
    """

    # Clean data: Set negative GHI values to 0
    var_ref = 'SRRL BMS Global Horizontal Irradiance (W/m²_irr)'
    if var_ref in data.columns:
        data[var_ref][data[var_ref] < 0] = 0

    # Clean data: Total cloud cover: Set -1's to 0 and interpolate negative false values
    var_ref = 'SRRL BMS Total Cloud Cover (%)'
    if var_ref in data.columns:
        data[var_ref][data[var_ref] == -1] = 0
        data[var_ref][data[var_ref] < 0] = float("NaN")
        data[var_ref].interpolate(inplace=True)

    # Clean data: Opaque cloud cover: Set -1's to 0 and interpolate negative false values
    var_ref = 'SRRL BMS Opaque Cloud Cover (%)'
    if var_ref in data.columns:
        data[var_ref][data[var_ref] == -1] = 0
        data[var_ref][data[var_ref] < 0] = float("NaN")
        data[var_ref].interpolate(inplace=True)

    # Clean data: Snow Depth: Set small and negative values to 0 to remove noise, and convolve to remove sharp/incorrect gradients
    var_ref = "SRRL BMS Snow Depth (in)"
    if var_ref in data.columns:
        data[var_ref][data[var_ref] < 0.3] = 0
        data[var_ref].interpolate(inplace=True)
        box_pts = 20
        box = np.ones(box_pts) / box_pts
        data[var_ref] = np.convolve(data[var_ref], box, mode="same")

    return data


def time_dummies(data, configs):
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
        data['sin_HOD'] = np.sin(
            2 * np.pi * (data.index.hour * 3600 + data.index.minute * 60 + data.index.second).values / (
                    24 * 60 * 60))
        data['cos_HOD'] = np.cos(
            2 * np.pi * (data.index.hour * 3600 + data.index.minute * 60 + data.index.second).values / (
                    24 * 60 * 60))
    if "binary_reg" in configs["HOD"]:
        for i in range(0, 24):
            data["HOD_binary_reg_{}".format(i)] = (data.index.hour == i).astype(int)

        #data = data.join(pd.get_dummies(data.index.hour, prefix='HOD_binary_reg', drop_first=True).set_index(data.index))

    if "binary_fuzzy" in configs["HOD"]:
        for HOD in range(0, 24):
            data["HOD_binary_fuzzy_{}".format(HOD)] = np.maximum(1 - abs((data.index.hour + data.index.minute / 60) - HOD) / 1, 0)

    # DOW
    if "binary_reg" in configs["DOW"]:
        for i in range(0, 7):
            data["DOW_binary_reg_{}".format(i)] = (data.index.weekday == i).astype(int)
        #data = data.join(pd.get_dummies(data.index.weekday, prefix='DOW_binary_reg', drop_first=True).set_index(data.index))
    if "binary_fuzzy" in configs["DOW"]:
        for i in range(0, 7):
            data["DOW_binary_fuzzy_{}".format(i)] = (data.index.weekday == i).astype(int)
        #data = data.join(pd.get_dummies(data.index.weekday, prefix='DOW_binary_fuzzy', drop_first=True).set_index(data.index))
        for DOW in range(0, 7):
            data["DOW_binary_fuzzy_{}".format(DOW)] = np.maximum(1 - abs((data.index.weekday + data.index.hour / 24) - DOW) / 1, 0)

    # MOY
    if "sincos" in configs['MOY']:
        data['sin_MOY'] = np.sin(2 * np.pi * (data.index.dayofyear).values / (365))
        data['cos_MOY'] = np.cos(2 * np.pi * (data.index.dayofyear).values / (365))

    if configs['Holidays']:
        # -----Automatic (fetches federal holidays based on dates in imported data
        # cal = USFederalHolidayCalendar()
        # holidays = cal.holidays(start=data.index[0].strftime("%Y-%m-%d"), end=data.index[-1].strftime("%Y-%m-%d"))
        # data['Holiday'] = pd.to_datetime(data.index.date).isin(holidays).astype(int)

        # -----Read from JSON file
        with open("holidays.json", "r") as read_file:
            holidays = json.load(read_file)
        data['Holiday'] = pd.to_datetime(data.index.date).isin(holidays).astype(int)
        data['Holiday_forward'] = pd.to_datetime(data.index.date + dt.timedelta(days=1)).isin(holidays).astype(int)

    return data


def input_data_split(data, configs):
    """
    Split a data set into a training set and a validation (val) set.
    Methods: "Random" or "Sequential", specified in configs

    :param data: (DataFrame)
    :param configs: (Dict)
    :return:
    """
    train_ratio = int(configs["data_split"].split(":")[0])/100
    val_ratio = int(configs["data_split"].split(":")[1])/100
    test_ratio = int(configs["data_split"].split(":")[2])/100

    file_prefix = Path(configs["exp_dir"])

    if configs['train_val_split'] == 'Random':
        pathlib.Path(configs["data_dir"]).mkdir(parents=True, exist_ok=True)
        mask_file = os.path.join(file_prefix, "mask.h5")
        logger.info("Creating random training mask and writing to file")

        # If you want to group datasets together into sequential chunks
        if configs["splicer"]["active"]:
            # Set indices for training set
            np.random.seed(seed=configs["random_seed"])
            splicer = ((data.index - data.index[0]) // pd.Timedelta(configs["splicer"]["time"])).values
            num_chunks = splicer[-1]
            num_train_chunks = (train_ratio * num_chunks) - ((train_ratio * num_chunks) % configs["train_size_factor"])
            msk = np.zeros(data.shape[0]) + 2
            train_chunks = np.random.choice(np.arange(num_chunks), replace=False, size=int(num_train_chunks))
            for chunk in train_chunks:
                indices = np.where(splicer == chunk)
                msk[indices] = 0

            # Set indices for validation and test set
            remaining_chunks = np.setdiff1d(np.arange(num_chunks), train_chunks)
            if test_ratio == 0:
                msk[msk != 0] = 1
            else:
                num_val_chunks = int((val_ratio / (1-train_ratio)) * remaining_chunks.shape[0])
                val_chunks = np.random.choice(remaining_chunks, replace=False, size=num_val_chunks)
                for chunk in val_chunks:
                    indices = np.where(splicer == chunk)
                    msk[indices] = 1

        # If you DONT want to group data into sequential chunks
        else:
            # Set indices for training set
            np.random.seed(seed=configs["random_seed"])
            data_size = data.shape[0]
            num_ones = (train_ratio * data_size) - ((train_ratio * data_size) % configs["train_size_factor"])
            msk = np.zeros(data_size) + 2
            indices = np.random.choice(np.arange(data_size), replace=False, size=int(num_ones))
            msk[indices] = 0

            # Set indices for validation and test set
            remaining_indices = np.where(msk != 0)[0]
            if test_ratio == 0:
                msk[remaining_indices] = 1
            else:
                num_val = int((val_ratio / (1-train_ratio)) * remaining_indices.shape[0])
                val_indices = np.random.choice(remaining_indices, replace=False, size=num_val)
                msk[val_indices] = 1


        logger.info("Train: {}, validation: {}, test: {}".format((msk == 0).sum()/msk.shape[0], (msk == 1).sum()/msk.shape[0], (msk == 2).sum()/msk.shape[0]))
        # Assign dataframes
        train_df = data[msk == 0]
        val_df = data[msk == 1]
        test_df = data[msk == 2]

        # Save test_df to file for later use
        test_df.to_hdf(os.path.join(file_prefix, "internal_test.h5"), key='df', mode='w')

        # Still save dataframe to file to preserve timeseries index
        mask = pd.DataFrame()
        mask['msk'] = msk
        mask.index = data.index
        mask.to_hdf(mask_file, key='df', mode='w')

        # Get rid of datetime index
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

    else:
        raise ConfigsError("{} is not a supported form of data splitting".format(configs['train_val_split']))

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
    data = data.drop(configs['target_var'], axis=1)
    data_orig = data

    # Pad the exogenous variables
    temp_holder = list()
    temp_holder.append(data_orig)
    for i in range(1, configs['window']+1):
        shifted = data_orig.shift(i * int(configs["sequence_freq_min"]), freq='min').astype("float32").add_suffix("_lag{}".format(i))
        temp_holder.append(shifted)
    temp_holder.reverse()
    data = pd.concat(temp_holder, axis=1)

    # If this is a linear quantile regression model (iterative)
    if configs["arch_type"] == "quantile" and configs["iterative"] == True:
        for i in range(0, configs["EC_future_gap_min"]):
            if i == 0:
                data[configs["target_var"]] = target
            else:
                data["{}_lag_{}".format(configs["target_var"], i)] = target.shift(-i)

        # Drop all nans
        data = data.dropna(how='any')

    # If this is a linear quantile regression model (point)
    elif configs["arch_type"] == "quantile" and configs["iterative"] == False:
        # Re-append the shifted target column to the dataframe
        data[configs["target_var"]] = target.shift(-configs['EC_future_gap_min'])

        # Drop all nans
        data = data.dropna(how='any')

        # Adjust time index to match the EC values
        data.index = data.index + pd.DateOffset(minutes=(configs["EC_future_gap_min"]))

    # If this is an RNN model
    elif configs["arch_type"] == "RNN":
        # Re-append the shifted target column to the dataframe
        data[configs["target_var"]] = target.shift(-configs['EC_future_gap_min'])

        # Drop all nans
        data = data.dropna(how='any')

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
    data = data.drop(configs['target_var'], axis=1)
    data_orig = data
    # Pad the exogenous variables
    temp_holder = list()
    temp_holder.append(data_orig)
    for i in range(1, configs['window']+1):
        shifted = data_orig.shift(i * int(configs["sequence_freq_min"]), freq='min').astype(
            "float32").add_suffix("_lag{}".format(i))
        temp_holder.append(shifted)
    temp_holder.reverse()
    data = pd.concat(temp_holder, axis=1)

    # Do fine padding for future predictions. Create a new df to preserve memory usage.
    local = pd.DataFrame()
    for i in range(0, configs["S2S_stagger"]["initial_num"]):
        local["{}_lag_{}".format(configs["target_var"], i)] = target.shift(-i * int(configs["sequence_freq_min"]), freq='min')

    # Do additional coarse padding for future predictions
    for i in range(1, configs["S2S_stagger"]["secondary_num"] + 1):
        base = configs["S2S_stagger"]["initial_num"]
        new = base + configs["S2S_stagger"]["decay"] * i
        local["{}_lag_{}".format(configs["target_var"], base+i)] = target.shift(-new * int(configs["sequence_freq_min"], freq='min'))

    data = pd.concat([data, local], axis=1)

    # Drop all nans
    data = data.dropna(how='any')

    return data


def corr_heatmap(data):
    """
    Plot a correlation heatmap to see the (linear) relationships between exogenous variables

    :param data: (DataFrame)
    :return: None
    """
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    sns.heatmap(data.corr(), mask=mask, cmap='coolwarm', linewidths=1, linecolor='white')


def prep_for_rnn(configs, data):
    """
    Prepare data for input to a RNN model.

    :param configs: (Dict)
    :return: train and val DataFrames
    """

    if configs["use_case"] == "train" or configs["use_case"] == "prediction":
        # split data into training/validation/testing sets
        train_df, val_df = input_data_split(data, configs)

    elif configs["use_case"] == "validation" and configs["test_method"] == "external":

        # split data into training/validation/testing sets
        val_df = data
        train_df = pd.DataFrame()
        filepath = filepath = pathlib.Path(configs["data_dir"]) / f"{configs['target_var']}_external_test.h5"
        val_df.to_hdf(filepath, key='df', mode='w')

    elif configs["use_case"] == "validation" and configs["test_method"] == "internal":
        local_results_dir = Path(configs["exp_dir"])
        temp_config_file = os.path.join(local_results_dir, "configs.json")
        with open(temp_config_file, 'r') as f:
            temp_configs = json.load(f)
        configs["input_dim"] = temp_configs["input_dim"]

        train_df = pd.DataFrame()
        val_df = data

    else:
        raise ConfigsError("run_train and/or test_method not valid.")

    return train_df, val_df


def get_test_data(building, year, months, dir):
    data_e = pd.DataFrame()
    data_w = pd.DataFrame()

    test_set_dir = dir
    pathlib.Path(test_set_dir).mkdir(parents=True, exist_ok=True)

    dataset = pd.DataFrame()
    for month in months:
        file = os.path.join(test_set_dir, "{}-{}-{}.h5".format(building, "{:02d}".format(month), year))
        if pathlib.Path(file).exists():
            sub_dataset = pd.read_hdf(file, key='df')

        else:
            # Get energy data
            network_path = "Z:\\Data"
            sub_dir = "Building Load Data"
            suffix = " Meter Trends"
            energy_data_dir = os.path.join(network_path, sub_dir)
            energy_file = "{} {}-{}{}.csv".format(building, year, "{:02d}".format(month), suffix)
            dateparse = lambda date: dt.datetime.strptime(date[:-13], '%Y-%m-%dT%H:%M:%S')
            csv_path = os.path.join(energy_data_dir, energy_file)
            df_e = pd.read_csv(csv_path,
                               parse_dates=['Timestamp'],
                               date_parser=dateparse,
                               index_col='Timestamp')
            data_e = pd.concat([data_e, df_e])

            site = 'STM'
            weather_data_dir = os.path.join(network_path, "Weather")
            weather_file = '{} Site Weather {}-{}.csv'.format(site, year, "{:02d}".format(month))
            csv_path = os.path.join(weather_data_dir, weather_file)
            df_w = pd.read_csv(csv_path,
                               parse_dates=['Timestamp'],
                               date_parser=dateparse,
                               index_col='Timestamp')
            data_w = pd.concat([data_w, df_w])
            dataset = pd.concat([data_e, data_w], axis=1)
            sub_dataset.to_hdf(file, key='df', mode='w')

        dataset = pd.concat([dataset, sub_dataset])

    return dataset


def rolling_stats(data, configs):
    # Convert data to rolling average (except output) and create min, mean, and max columns
    target = data[configs["target_var"]]
    X_data = data.drop(configs["target_var"], axis=1)

    # inferring timestep (frequency) from the dataframe
    dt = configs["data_time_interval_mins"]
    windowsize = int(configs["rolling_window"]["minutes"] / dt) + 1
    logging.debug("Feature extraction: rolling window size = {} rows".format(windowsize))

    if configs["rolling_window"]["type"] == "rolling":
        mins = X_data.rolling(window=windowsize, min_periods=1).min().add_suffix("_min")
        means = X_data.rolling(window=windowsize, min_periods=1).mean().add_suffix("_mean")
        maxs = X_data.rolling(window=windowsize, min_periods=1).max().add_suffix("_max")
        data = pd.concat([mins, means, maxs], axis=1)
        data[configs["target_var"]] = target

    elif configs["rolling_window"]["type"] == "binned":
        mins = X_data.resample(str(configs["rolling_window"]["minutes"]) + "T").min().add_suffix("_min")
        means = X_data.resample(str(configs["rolling_window"]["minutes"]) + "T").mean().add_suffix("_mean")
        maxs = X_data.resample(str(configs["rolling_window"]["minutes"]) + "T").max().add_suffix("_max")
        data = pd.concat([mins, means, maxs], axis=1)
        data[configs["target_var"]] = pd.DataFrame(target).resample(str(configs["rolling_window"]["minutes"]) + "T").mean()

    return data