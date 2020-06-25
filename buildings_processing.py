import numpy as np
import pathlib
import pandas as pd
import datetime as dt
import tables
from pandas.tseries.holiday import USFederalHolidayCalendar, get_calendar


def import_from_lan(configs):
    # Does one year at a time
    data = pd.DataFrame()

    for month in range(1, 13):
        # Read in energy consumption data from LAN file
        energy_data_dir = configs['LAN_path'] + "\\Building Load data\\"
        energy_file = "{} {}-{} Meter Trends.csv".format(configs['building'], configs['year'], "{:02d}".format(month))
        dateparse = lambda date: dt.datetime.strptime(date[:-13], '%Y-%m-%dT%H:%M:%S')
        df_e = pd.read_csv(energy_data_dir + energy_file,
                           parse_dates=['Timestamp'],
                           date_parser=dateparse,
                           index_col='Timestamp')
        data = pd.concat([data, df_e])
        print('Read month {} in {}'.format(month, configs['year']))
    print('Done reading in energy data')

    # Read in weather data
    file_extension = "data/Weather_{}.h5".format(configs['year'])
    if pathlib.Path(file_extension).exists():
        df_w = pd.read_hdf(file_extension, key='df')
    else:
        site = 'STM'
        weather_data_dir = configs['LAN_path'] + "\Weather\\"
        weather_file = '{} Site Weather {}.csv'.format(site, configs['year'])
        df_w = pd.read_csv(weather_data_dir + weather_file,
                           parse_dates=['Timestamp'],
                           date_parser=dateparse,
                           index_col='Timestamp')
        df_w.to_hdf(file_extension, key='df', mode='w')
    print('Done reading in weather data')

    dataset = pd.concat([data, df_w], axis=1)

    pathlib.Path('data').mkdir(parents=True, exist_ok=True)
    output_string = 'data/Data_{}_{}.h5'.format(configs['building'], configs['year'])
    dataset.to_hdf(output_string, key='df', mode='w')

    return output_string


def get_full_data(configs):
    # For quantile regression model, "year" input will be a single year (str)
    if type(configs['year']) == str:
        iterable = [configs['year']]
    else:
        iterable = configs['year']

    dataset = dict()
    for year in iterable:
        # Read in preprocessed data from HDFs
        file_extension = "data/Data_{}_{}.h5".format(configs['building'], year)
        if pathlib.Path(file_extension).exists():
            # print('data already processed. Reading from hdf file')
            data_full = pd.read_hdf(file_extension, key='df')
        else:
            # print('Processing data to hdf file')
            file_path = import_from_lan(configs)
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
            data['sin_time'] = np.sin(
                2 * np.pi * (data.index.hour * 3600 + data.index.minute * 60 + data.index.minute).values / (
                        24 * 60 * 60))
            data['cos_time'] = np.cos(
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
        data = data.join(pd.get_dummies(data.index.month, prefix='MOY', drop_first=True).set_index(data.index))
    if configs['Holidays']:
        # Automatic
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=data.index[0].strftime("%Y-%m-%d"), end=data.index[-1].strftime("%Y-%m-%d"))
        data['Holiday'] = pd.to_datetime(data.index.date).isin(holidays).astype(int)
        # Manual
        # holidays = ['2019-01-01', '2019-02-18', '2019-05-27',
        #             '2019-07-04', '2019-09-02', '2019-11-28',
        #             '2019-12-24', '2019-12-25', '2019-12-31']
        # data['Holiday'] = pd.to_datetime(data.index.date).isin(holidays).astype(int)
    return data


def train_test_split(data, configs):
    if configs['TrainTestSplit'] == 'Random':
        msk = np.random.rand(len(data)) < 0.9
        train_df = data[msk]
        test_df = data[~msk]
    if configs['TrainTestSplit'] == 'Sequential':
        train_df = data[~data.index.isin(data[configs['test_start']:configs['test_end']].index)]
        test_df = data[data.index.isin(data[configs['test_start']:configs['test_end']].index)]

    return train_df, test_df
