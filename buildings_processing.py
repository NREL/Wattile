import numpy as np
import pathlib
import pandas as pd
import datetime as dt
import tables
from pandas.tseries.holiday import USFederalHolidayCalendar, get_calendar
import json


def import_from_lan(configs, year):
    # Imports EC data and weather data one year at a time
    data_e = pd.DataFrame()
    data_w = pd.DataFrame()

    # Read in energy consumption data from LAN file (one month at a time)
    for month in range(1, 13):
        energy_data_dir = configs['LAN_path'] + "\\Building Load data\\"
        energy_file = "{} {}-{} Meter Trends.csv".format(configs['building'], year, "{:02d}".format(month))
        dateparse = lambda date: dt.datetime.strptime(date[:-13], '%Y-%m-%dT%H:%M:%S')
        df_e = pd.read_csv(energy_data_dir + energy_file,
                           parse_dates=['Timestamp'],
                           date_parser=dateparse,
                           index_col='Timestamp')
        data_e = pd.concat([data_e, df_e])
        print('Read energy month {}/12 in {} for {}'.format(month, year, configs['building']))
    print('Done reading in energy data')

    # Read in weather data (one month at a time)
    file_extension = "data/Weather_{}.h5".format(year)
    if pathlib.Path(file_extension).exists():
        data_w = pd.read_hdf(file_extension, key='df')
    else:
        site = 'STM'
        weather_data_dir = configs['LAN_path'] + "\Weather\\"
        for month in range(1, 13):
            weather_file = '{} Site Weather {}-{}.csv'.format(site, year, "{:02d}".format(month))
            df_w = pd.read_csv(weather_data_dir + weather_file,
                               parse_dates=['Timestamp'],
                               date_parser=dateparse,
                               index_col='Timestamp')
            data_w = pd.concat([data_w, df_w])
            print('Read weather month {}/12 in {}'.format(month, year))
        data_w.to_hdf(file_extension, key='df', mode='w')
    print('Done reading in weather data')

    dataset = pd.concat([data_e, data_w], axis=1)

    pathlib.Path('data').mkdir(parents=True, exist_ok=True)
    output_string = 'data/Data_{}_{}.h5'.format(configs['building'], year)
    dataset.to_hdf(output_string, key='df', mode='w')

    return output_string


def get_full_data(configs):
    # For quantile regression model, "year" input will be a single year (str)
    if type(configs['year']) == str:
        iterable = [configs['year']]
    else:
        iterable = configs['year']

    # Collect data from the requested year(s) and put it in a single dataframe
    dataset = dict()
    for year in iterable:
        # Read in preprocessed data from HDFs
        file_extension = "data/Data_{}_{}.h5".format(configs['building'], year)
        if pathlib.Path(file_extension).exists():
            # print('data already processed. Reading from hdf file')
            data_full = pd.read_hdf(file_extension, key='df')
        else:
            # print('Processing data to hdf file')
            file_path = import_from_lan(configs, year)
            data_full = pd.read_hdf(file_path, key='df')
        dataset[year] = data_full

    data_full = pd.DataFrame()
    for year in dataset:
        data_full = pd.concat([data_full, dataset[year]])

    return data_full


def clean_data(data, configs):
    # Clean data: Set negative GHI values to 0
    var_ref = 'SRRL BMS Global Horizontal Irradiance (W/m²_irr)'
    if var_ref in configs['weather_include']:
        data[var_ref][data[var_ref] < 0] = 0

    # Clean data: Total cloud cover
    var_ref = 'SRRL BMS Total Cloud Cover (%)'
    if var_ref in configs['weather_include']:
        data[var_ref][data[var_ref] < 0] = 0
    return data


def add_weather_lags(data, configs):
    # Add main weather lag
    if configs['main_lag'] > 0:
        # print("Adding main weather variable lag")
        for variable in configs['weather_include']:
            data[variable] = data[variable].shift(int(configs['main_lag'] / (configs['resample_bin_min'] / 60)))

    # Add extra weather lags
    if configs['num_weather_lags'] > 0:
        # print('Adding extra weather variable lags')
        for variable in configs['weather_include']:
            for lag_num in range(1, configs['num_weather_lags'] + 1):
                data[variable + '_lag_{}'.format(lag_num)] = data[variable].shift(periods=lag_num)
    return data


def time_dummies(data, configs):
    # print("Adding dummy variables")
    if configs['HOD']:
        if configs['HOD_indicator'] == 'regDummy':
            data = data.join(pd.get_dummies(data.index.hour, prefix='HOD', drop_first=True).set_index(data.index))
        elif configs['HOD_indicator'] == 'sincos':
            data['sin_HOD'] = np.sin(
                2 * np.pi * (data.index.hour * 3600 + data.index.minute * 60 + data.index.minute).values / (
                        24 * 60 * 60))
            data['cos_HOD'] = np.cos(
                2 * np.pi * (data.index.hour * 3600 + data.index.minute * 60 + data.index.minute).values / (
                        24 * 60 * 60))
        elif configs['HOD_indicator'] == 'fuzzy':
            data = data.join(pd.get_dummies(data.index.hour, prefix='HOD', drop_first=True).set_index(data.index))
            for HOD in range(1, 24):
                data["HOD_{}".format(HOD)] = np.maximum(1 - abs((data.index.hour + data.index.minute / 60) - HOD) / 1,
                                                        0)
        else:
            print('Time-indicator type not recognized')
    if configs['DOW']:
        data = data.join(pd.get_dummies(data.index.weekday, prefix='DOW', drop_first=True).set_index(data.index))
    if configs['MOY']:
        data['sin_MOY'] = np.sin(2 * np.pi * (data.index.dayofyear).values / (365))
        data['cos_MOY'] = np.cos(2 * np.pi * (data.index.dayofyear).values / (365))
        #data = data.join(pd.get_dummies(data.index.month, prefix='MOY', drop_first=True).set_index(data.index))
    if configs['Holidays']:
        # -----Automatic (fetches federal holidays based on dates in imported data
        # cal = USFederalHolidayCalendar()
        # holidays = cal.holidays(start=data.index[0].strftime("%Y-%m-%d"), end=data.index[-1].strftime("%Y-%m-%d"))
        # data['Holiday'] = pd.to_datetime(data.index.date).isin(holidays).astype(int)

        # -----Manual in-script
        # holidays = ['2019-01-01', '2019-02-18', '2019-05-27',
        #             '2019-07-04', '2019-09-02', '2019-11-28',
        #             '2019-12-24', '2019-12-25', '2019-12-31']
        # data['Holiday'] = pd.to_datetime(data.index.date).isin(holidays).astype(int)

        # -----Read from JSON file
        with open("holidays.json", "r") as read_file:
            holidays = json.load(read_file)
        data['Holiday'] = pd.to_datetime(data.index.date).isin(holidays).astype(int)

    return data


def train_test_split(data, configs):
    if configs['TrainTestSplit'] == 'Random':
        pathlib.Path('data').mkdir(parents=True, exist_ok=True)
        mask_file = "data/mask_{}_{}.json".format(configs['building'], "-".join(configs['year']))
        # Check if a mask for the building/year combination already exists
        if pathlib.Path(mask_file).exists():
            # Open the mask file
            with open(mask_file, "r") as read_file:
                msk = json.load(read_file)
            # Check if the saved mask is the same size as the data file
            if len(msk) == data.shape[0]:
                msk = np.array(msk)
                train_df = data[msk]
                test_df = data[~msk]
                print("Using an existing training mask: {}".format(mask_file))
            # If not, a recent architectural change must have changed the length of data. Make a new one.
            else:
                print("There was a length mismatch between the existing mask and the data. Making a new mask and writing to file.")
                data_size = data.shape[0]
                num_ones = (0.9 * data_size) - ((0.9 * data_size) % 32)
                msk = np.zeros(data_size)
                indices = np.random.choice(np.arange(data_size), replace=False, size=int(num_ones))
                msk[indices] = 1
                msk = msk.astype(bool)
                train_df = data[msk]
                test_df = data[~msk]
                with open(mask_file, "w") as write_file:
                    json.dump(msk.tolist(), write_file)

        # If no previously-saved mask exists, make one
        else:
            print("Creating random training mask and writing to file")
            data_size = data.shape[0]
            num_ones = (0.9 * data_size) - ((0.9 * data_size) % 32)
            msk = np.zeros(data_size)
            indices = np.random.choice(np.arange(data_size), replace=False, size=int(num_ones))
            msk[indices] = 1
            msk = msk.astype(bool)
            train_df = data[msk]
            test_df = data[~msk]
            with open(mask_file, "w") as write_file:
                json.dump(msk.tolist(), write_file)

    if configs['TrainTestSplit'] == 'Sequential':
        train_df = data[~data.index.isin(data[configs['test_start']:configs['test_end']].index)]
        test_df = data[data.index.isin(data[configs['test_start']:configs['test_end']].index)]

    return train_df, test_df

def pad_full_data(data, configs):
    target = data[configs['target_var']]
    a = data.drop(configs['target_var'], axis=1).values.astype(dtype='float32')
    rows = a.shape[0]
    cols = a.shape[1]
    b = np.zeros((rows, configs['window'] * cols))

    # Make new columns for the time-lagged values. Lagged spaces filled with zeros.
    for i in range(configs['window']):
        # The first window isnt lagged and is just a copy of "a"
        if i == 0:
            b[:, 0:cols] = a
        # For all remaining windows, just paste a slightly cropped version (so it fits) of "a" into "b"
        else:
            b[i:, i * cols:(i + 1) * cols] = a[:-i, :]

    # The zeros are replaced with a copy of the first n rows
    for i in list(np.arange(configs['window'] - 1)):
        j = (i * cols) + cols
        b[i, j:] = np.tile(b[i, 0:cols], configs['window'] - i - 1)

    # Shift the output column
    target_shifted = target.shift(-configs['EC_future_gap']).fillna(method='ffill')
    data = pd.DataFrame(b)
    # Recombine the input matrix to the output column
    data[configs['target_var']] = target_shifted.values

    return data


def prep_for_rnn(configs):
    # Get full dataset from LAN for specific building
    data_full = get_full_data(configs)

    # Remove all data columns we dont care about
    important_vars = configs['weather_include'] + [configs['target_var']]
    data = data_full[important_vars]

    # Resample
    resample_bin_size = "{}T".format(configs['resample_bin_min'])
    data = data.resample(resample_bin_size).mean()

    # Clean
    data = clean_data(data, configs)

    # Deal with NANs
    #data = data.dropna(how='any')
    data = data.interpolate()

    # Add time-based dummy variables
    data = time_dummies(data, configs)

    # Determine input dimension
    configs['input_dim'] = data.shape[1] - 1

    # Do sequential padding now if we are doing random train/test splitting
    if configs["TrainTestSplit"] == 'Random':
        # Do padding
        data = pad_full_data(data, configs)
        data.columns = data.columns.values.astype(str)

    # Split into training and test dataframes
    train_df, test_df = train_test_split(data, configs)

    return train_df, test_df



