import sys
import pathlib
import numpy as np
import pandas as pd
import requests
import re
from dateutil.parser import parse
from functools import reduce
from util import prtime


# train_start_date = '2018-10-22'
# train_end_date = '2018-11-22'
# test_start_date = '2018-11-23'
# test_end_date = '2018-11-28'

train_exp_num = 1  # increment this number everytime a new model is trained
test_exp_num = 1   # increment this number when the tests are run on an existing model (run_train = False)


# Define the Directories to save the trained model and results.
# Create the dir if it does not exist using pathlib
MODEL_DIR = 'LoadForecasting_Results/Model_' + str(train_exp_num)
RESULTS_DIR = 'LoadForecasting_Results/Model_' + str(train_exp_num) + '/Test_num_' +  str(test_exp_num)
pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
log_file = RESULTS_DIR + '/' + 'console.log'
#print("Writing print statements to ", log_file)
#sys.stdout = open(log_file, 'w')  # Redirect print statement's outputs to file
#print("Stdout:")

start_time = '00:01:00'
end_time = '23:59:00'
EC_start_time = '00:00:00'
EC_end_time = '23:45:00'

def fetch_data(root_url, reference_id, train_date_range, test_date_range, feat_name, run_train):
    train_response_dict = {}
    test_response_dict = {}
    if run_train:
        for i in range(len(reference_id)):
            train_response_dict['resp_'+feat_name[i]] = requests.get(root_url+reference_id[i]+train_date_range)
            if train_response_dict['resp_'+feat_name[i]].status_code == 200:
                pass
            else:
                print("response from {} is not getting fetched from API for training data".format(feat_name[i]))

    for i in range(len(reference_id)):
        test_response_dict['resp_' + feat_name[i]] = requests.get(root_url + reference_id[i] + test_date_range)
        if test_response_dict['resp_' + feat_name[i]].status_code == 200:
            pass
        else:
            print("response from {} is not getting fetched from API for testing data".format(feat_name[i]))

    return train_response_dict, test_response_dict

def str_split(row):
    time_val = row.split(",")[0].strip(" Denver")
    energy_val = row.split(",")[1]
    energy_val = re.sub('[kwh%RHmbar°FW/m²_irrp]','', energy_val)
    return (time_val, float(energy_val))

def date_parser(row):
    parsed = parse(row)
    datetime_var = parsed.strftime(format='%m-%d-%y %H:%M:%S')
    date = parsed.date()
    time = parsed.time()
    return (datetime_var ,date, time)


def parse_data(train_response_dict, test_response_dict, feat_name, run_train):
    train_parsed_dict = {}
    test_parsed_dict = {}
    if run_train:

        for i in range(len(feat_name)):
            train_parsed_dict[feat_name[i]] = train_response_dict['resp_' + feat_name[i]].content.decode('utf-8').split("\n")
            train_parsed_dict[feat_name[i]] = train_parsed_dict[feat_name[i]][2:]
            train_parsed_dict[feat_name[i]] = filter(None, train_parsed_dict[feat_name[i]])
            train_parsed_dict[feat_name[i]] = list(map(str_split, train_parsed_dict[feat_name[i]]))


            # the following line gives list (len 2) of lists (i.e. EC_dt and EC_value)
            # i.e. EC_dt, EC_value = EC[0], EC[1]
            train_parsed_dict[feat_name[i]] = list(zip(*train_parsed_dict[feat_name[i]]))

            # parsing the datetimeinfo obtained in above list into datetime string, date and time
            # the lists can be unpacked as:
            # EC_datetime, EC_date, EC_time = EC_dt_parsed[0], EC_dt_parsed[1], EC_dt_parsed[2]
            train_parsed_dict[feat_name[i] + '_dt_parsed'] = list(map(date_parser, train_parsed_dict[feat_name[i]][0]))
            train_parsed_dict[feat_name[i] + '_dt_parsed'] = list(zip(*train_parsed_dict[feat_name[i] + '_dt_parsed']))

    for i in range(len(feat_name)):
        test_parsed_dict[feat_name[i]] = test_response_dict['resp_' + feat_name[i]].content.decode('utf-8').split(
            "\n")
        test_parsed_dict[feat_name[i]] = test_parsed_dict[feat_name[i]][2:]
        test_parsed_dict[feat_name[i]] = filter(None, test_parsed_dict[feat_name[i]])
        test_parsed_dict[feat_name[i]] = list(map(str_split, test_parsed_dict[feat_name[i]]))

        test_parsed_dict[feat_name[i]] = list(zip(*test_parsed_dict[feat_name[i]]))

        # parsing the datetimeinfo obtained in above list into datetime string, date and time
        test_parsed_dict[feat_name[i] + '_dt_parsed'] = list(map(date_parser, test_parsed_dict[feat_name[i]][0]))
        test_parsed_dict[feat_name[i] + '_dt_parsed'] = list(zip(*test_parsed_dict[feat_name[i] + '_dt_parsed']))

    return train_parsed_dict, test_parsed_dict


def input_feat_dfs(train_parsed_dict, test_parsed_dict, input_feat_name, run_train, train_start_date, train_end_date, test_start_date, test_end_date):

    # creating a dictionary of input features dataframes with the parsed lists generated in the parsing module
    train_df_dict = {}
    test_df_dict = {}
    if run_train:

        for i in range(len(input_feat_name)):
            train_df_dict["df_" + input_feat_name[i]] = pd.DataFrame({'datetime_str': train_parsed_dict[input_feat_name[i] + '_dt_parsed'][0],
                 input_feat_name[i]: train_parsed_dict[input_feat_name[i]][1]},columns=['datetime_str', input_feat_name[i]])

            df_temp = train_df_dict["df_" + input_feat_name[i]]
            df_temp.name = "df_" + input_feat_name[i]
            prtime("raw_train dataframe = {}, shape = {}".format(df_temp.name, df_temp.shape))
            df_temp['datetime_str'] = pd.to_datetime(df_temp['datetime_str'])

            if not (df_temp.loc[0, 'datetime_str'] == pd.to_datetime(train_start_date + ' ' + start_time)):
                df_temp.loc[0, 'datetime_str'] = pd.to_datetime(train_start_date + ' ' + start_time)
            if not (df_temp.loc[df_temp.index[-1], 'datetime_str'] == pd.to_datetime(train_end_date + ' ' + end_time)):
                df_temp.loc[df_temp.index[-1], 'datetime_str'] = pd.to_datetime(train_end_date + ' ' + end_time)

            df_temp = df_temp.set_index('datetime_str').resample("1min").first().reset_index().reindex(columns=df_temp.columns)
            cols = df_temp.columns.difference([input_feat_name[i]])
            df_temp[cols] = df_temp[cols].ffill()
            df_temp[input_feat_name[i]] = df_temp[input_feat_name[i]].fillna(
                ((df_temp[input_feat_name[i]].shift() + df_temp[input_feat_name[i]].shift(-1)) / 2))
            prtime("shape of processed train dataframe: {}".format(df_temp.shape))

            train_df_dict["df_" + input_feat_name[i]] = df_temp
            del df_temp

    for i in range(len(input_feat_name)):
        test_df_dict["df_" + input_feat_name[i]] = pd.DataFrame(
            {'datetime_str': test_parsed_dict[input_feat_name[i] + '_dt_parsed'][0],
             input_feat_name[i]: test_parsed_dict[input_feat_name[i]][1]},
            columns=['datetime_str', input_feat_name[i]])

        df_temp = test_df_dict["df_" + input_feat_name[i]]
        df_temp.name = "df_" + input_feat_name[i]
        prtime("raw_test dataframe = {}, shape = {}".format(df_temp.name, df_temp.shape))
        df_temp['datetime_str'] = pd.to_datetime(df_temp['datetime_str'])

        if not (df_temp.loc[0, 'datetime_str'] == pd.to_datetime(test_start_date + ' ' + start_time)):
            df_temp.loc[0, 'datetime_str'] = pd.to_datetime(test_start_date + ' ' + start_time)
        if not (df_temp.loc[df_temp.index[-1], 'datetime_str'] == pd.to_datetime(test_end_date + ' ' + end_time)):
            df_temp.loc[df_temp.index[-1], 'datetime_str'] = pd.to_datetime(test_end_date + ' ' + end_time)

        df_temp = df_temp.set_index('datetime_str').resample("1min").first().reset_index().reindex(
            columns=df_temp.columns)
        cols = df_temp.columns.difference([input_feat_name[i]])
        df_temp[cols] = df_temp[cols].ffill()
        df_temp[input_feat_name[i]] = df_temp[input_feat_name[i]].fillna(
            ((df_temp[input_feat_name[i]].shift() + df_temp[input_feat_name[i]].shift(-1)) / 2))
        prtime("shape of processed test dataframe: {}".format(df_temp.shape))

        test_df_dict["df_" + input_feat_name[i]] = df_temp
        del df_temp

    return train_df_dict, test_df_dict


def target_df(train_parsed_dict, test_parsed_dict, run_train, train_start_date, train_end_date, test_start_date, test_end_date):
    # processing Energy Consumption values separately
    if run_train:
        train_df_EC = pd.DataFrame({'datetime_str': train_parsed_dict['EC_dt_parsed'][0], 'EC': train_parsed_dict['EC'][1]},
                         columns=['datetime_str', 'EC'])
        prtime("shape of raw EC dataframe: {}".format(train_df_EC.shape))

        train_df_EC['datetime_str'] = pd.to_datetime(train_df_EC['datetime_str'])
        if not (train_df_EC.loc[0, 'datetime_str'] == pd.to_datetime(train_start_date + ' ' + EC_start_time)):
            train_df_EC.loc[0, 'datetime_str'] = pd.to_datetime(train_start_date + ' ' + EC_start_time)
        if not (train_df_EC.loc[train_df_EC.index[-1], 'datetime_str'] == pd.to_datetime(train_end_date + ' ' + EC_end_time)):
            train_df_EC.loc[train_df_EC.index[-1], 'datetime_str'] = pd.to_datetime(train_end_date + ' ' + EC_end_time)

        train_df_EC = train_df_EC.set_index('datetime_str').resample("15min").first().reset_index().reindex(columns=train_df_EC.columns)
        cols = train_df_EC.columns.difference(['EC'])
        train_df_EC[cols] = train_df_EC[cols].ffill()
        train_df_EC['EC'] = train_df_EC['EC'].fillna(((train_df_EC['EC'].shift() + train_df_EC['EC'].shift(-1)) / 2))
        prtime("shape of processed train EC dataframe: {}".format(train_df_EC.shape))


    test_df_EC = pd.DataFrame({'datetime_str': test_parsed_dict['EC_dt_parsed'][0], 'EC': test_parsed_dict['EC'][1]},
                              columns=['datetime_str', 'EC'])
    prtime("shape of raw EC dataframe: {}".format(test_df_EC.shape))

    test_df_EC['datetime_str'] = pd.to_datetime(test_df_EC['datetime_str'])
    if not (test_df_EC.loc[0, 'datetime_str'] == pd.to_datetime(test_start_date + ' ' + EC_start_time)):
        test_df_EC.loc[0, 'datetime_str'] = pd.to_datetime(test_start_date + ' ' + EC_start_time)
    if not (test_df_EC.loc[test_df_EC.index[-1], 'datetime_str'] == pd.to_datetime(
            test_end_date + ' ' + EC_end_time)):
        test_df_EC.loc[test_df_EC.index[-1], 'datetime_str'] = pd.to_datetime(test_end_date + ' ' + EC_end_time)

    test_df_EC = test_df_EC.set_index('datetime_str').resample("15min").first().reset_index().reindex(
        columns=test_df_EC.columns)
    cols = test_df_EC.columns.difference(['EC'])
    test_df_EC[cols] = test_df_EC[cols].ffill()
    test_df_EC['EC'] = test_df_EC['EC'].fillna(((test_df_EC['EC'].shift() + test_df_EC['EC'].shift(-1)) / 2))
    prtime("shape of processed test EC dataframe: {}".format(test_df_EC.shape))

    return train_df_EC, test_df_EC

def merge_n_resample(train_df_dict, test_df_dict, train_df_EC, test_df_EC, run_train):
    # merging the multiple dataframes (in the dictionary) containing input features
    if run_train:
        train_df_list = []
        for key, value in train_df_dict.items():
            train_df_list.append(train_df_dict[key])

        train_input_df = reduce(lambda left, right: pd.merge(left, right, on=['datetime_str'], how='outer'), train_df_list)

        # re-sampling (downsampling 1-min data to 15-min, since target values are for 15-min)
        train_input_df =train_input_df.set_index('datetime_str').resample("15min").mean().reset_index().reindex(columns=train_input_df.columns)

        # merging input_df (dataframe with input features) with df_EC (target dataframe)
        train_df = train_input_df.merge(train_df_EC, how='outer', on='datetime_str')


    test_df_list = []
    for key, value in test_df_dict.items():
        test_df_list.append(test_df_dict[key])

    test_input_df = reduce(lambda left, right: pd.merge(left, right, on=['datetime_str'], how='outer'), test_df_list)

    # re-sampling (downsampling 1-min data to 15-min, since target values are for 15-min)
    test_input_df = test_input_df.set_index('datetime_str').resample("15min").mean().reset_index().reindex(
        columns=test_input_df.columns)

    # merging input_df (dataframe with input features) with df_EC (target dataframe)
    test_df = test_input_df.merge(test_df_EC, how='outer', on='datetime_str')

    return train_df, test_df


def get_static_features(train_df, test_df, run_train):

    if run_train:
        # inserting new columns at index 7 and onward
        idx = 7
        new_col = train_df.datetime_str.dt.dayofyear.astype(np.float32)
        train_df.insert(loc=idx, column='doy', value=new_col)

        idx = idx + 1
        new_col = pd.to_timedelta(train_df.datetime_str.dt.strftime('%H:%M:%S')).dt.total_seconds().astype(int)
        train_df.insert(loc=idx, column='timeinSec', value=new_col)

        # conversion to cyclic coordinates
        seconds_in_day = 24 * 60 * 60

        idx = idx + 1
        new_col = np.sin(2 * np.pi * train_df.timeinSec / seconds_in_day)
        train_df.insert(loc=idx, column='sin_time', value=new_col)

        idx = idx + 1
        new_col = np.cos(2 * np.pi * train_df.timeinSec / seconds_in_day)
        train_df.insert(loc=idx, column='cos_time', value=new_col)

        # adding the lagged EC features
        idx = idx + 1
        new_col = train_df['EC'].shift(4)
        train_df.insert(loc=idx, column='EC_t-4', value=new_col)

        idx = idx + 1
        new_col = train_df['EC'].shift(3)
        train_df.insert(loc=idx, column='EC_t-3', value=new_col)

        idx = idx + 1
        new_col = train_df['EC'].shift(2)
        train_df.insert(loc=idx, column='EC_t-2', value=new_col)

        idx = idx + 1
        new_col = train_df['EC'].shift(1)
        train_df.insert(loc=idx, column='EC_t-1', value=new_col)

    idx = 7
    new_col = test_df.datetime_str.dt.dayofyear.astype(np.float32)
    test_df.insert(loc=idx, column='doy', value=new_col)

    idx = idx + 1
    new_col = pd.to_timedelta(test_df.datetime_str.dt.strftime('%H:%M:%S')).dt.total_seconds().astype(int)
    test_df.insert(loc=idx, column='timeinSec', value=new_col)

    # conversion to cyclic coordinates
    seconds_in_day = 24 * 60 * 60

    idx = idx + 1
    new_col = np.sin(2 * np.pi * test_df.timeinSec / seconds_in_day)
    test_df.insert(loc=idx, column='sin_time', value=new_col)

    idx = idx + 1
    new_col = np.cos(2 * np.pi * test_df.timeinSec / seconds_in_day)
    test_df.insert(loc=idx, column='cos_time', value=new_col)

    # adding the lagged EC features
    idx = idx + 1
    new_col = test_df['EC'].shift(4)
    test_df.insert(loc=idx, column='EC_t-4', value=new_col)

    idx = idx + 1
    new_col = test_df['EC'].shift(3)
    test_df.insert(loc=idx, column='EC_t-3', value=new_col)

    idx = idx + 1
    new_col = test_df['EC'].shift(2)
    test_df.insert(loc=idx, column='EC_t-2', value=new_col)

    idx = idx + 1
    new_col = test_df['EC'].shift(1)
    test_df.insert(loc=idx, column='EC_t-1', value=new_col)

    return train_df, test_df


def fill_nan(train_df, test_df, run_train):

    # carefully filling in the nan values created during the shift operation
    # doing so by taking the mean of the row of the associated columns which have the same timestamp as nan-valued cell

    if run_train:

        train_df.loc[0, 'EC_t-4'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[0].time()]['EC_t-4'].mean(
            axis=0)
        train_df.loc[1, 'EC_t-4'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[1].time()]['EC_t-4'].mean(
            axis=0)
        train_df.loc[2, 'EC_t-4'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[2].time()]['EC_t-4'].mean(
            axis=0)
        train_df.loc[3, 'EC_t-4'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[3].time()]['EC_t-4'].mean(
            axis=0)

        train_df.loc[0, 'EC_t-3'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[0].time()]['EC_t-3'].mean(
            axis=0)
        train_df.loc[1, 'EC_t-3'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[1].time()]['EC_t-3'].mean(
            axis=0)
        train_df.loc[2, 'EC_t-3'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[2].time()]['EC_t-3'].mean(
            axis=0)

        train_df.loc[0, 'EC_t-2'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[0].time()]['EC_t-2'].mean(
            axis=0)
        train_df.loc[1, 'EC_t-2'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[1].time()]['EC_t-2'].mean(
            axis=0)

        train_df.loc[0, 'EC_t-1'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[0].time()]['EC_t-1'].mean(
            axis=0)

    test_df.loc[0, 'EC_t-4'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[0].time()]['EC_t-4'].mean(
        axis=0)
    test_df.loc[1, 'EC_t-4'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[1].time()]['EC_t-4'].mean(
        axis=0)
    test_df.loc[2, 'EC_t-4'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[2].time()]['EC_t-4'].mean(
        axis=0)
    test_df.loc[3, 'EC_t-4'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[3].time()]['EC_t-4'].mean(
        axis=0)

    test_df.loc[0, 'EC_t-3'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[0].time()]['EC_t-3'].mean(
        axis=0)
    test_df.loc[1, 'EC_t-3'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[1].time()]['EC_t-3'].mean(
        axis=0)
    test_df.loc[2, 'EC_t-3'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[2].time()]['EC_t-3'].mean(
        axis=0)

    test_df.loc[0, 'EC_t-2'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[0].time()]['EC_t-2'].mean(
        axis=0)
    test_df.loc[1, 'EC_t-2'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[1].time()]['EC_t-2'].mean(
        axis=0)

    test_df.loc[0, 'EC_t-1'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[0].time()]['EC_t-1'].mean(
        axis=0)

    return train_df, test_df


def main(train_start_date, train_end_date, test_start_date, test_end_date, run_train):
    root_url = 'https://internal-apis.nrel.gov/intelligentcampus/hisRead?id='
    reference_id = ['@p:nrel:r:225918db-bfbda16a', '@p:nrel:r:20ed5e0a-275dbdc2', '@p:nrel:r:20ed5e0a-53e174aa',
                    '@p:nrel:r:20ed5e0a-fe755c80', '@p:nrel:r:20ed5df2-2c0e126b', '@p:nrel:r:20ed5e0a-acc8beff',
                    '@p:nrel:r:20ed5df2-fd2eecc5']
    train_date_range = '&range=\"' + train_start_date + '%2c' + train_end_date + '\"'
    test_date_range = '&range=\"' + test_start_date + '%2c' + test_end_date + '\"'
    feat_name = ['EC', 'RH', 'BP', 'DBT', 'GHI', 'TCC', 'WS']
    input_feat_name = ['RH', 'BP', 'DBT', 'GHI', 'TCC', 'WS']

    train_response_dict, test_response_dict = fetch_data(root_url, reference_id, train_date_range, test_date_range, feat_name, run_train)
    prtime("data fetched from the API successfully, now parsing...")

    train_parsed_dict, test_parsed_dict = parse_data(train_response_dict, test_response_dict, feat_name, run_train)
    prtime("data parsed in a dictionary successfully, constructing input feature dataframes")

    train_df_dict, test_df_dict = input_feat_dfs(train_parsed_dict, test_parsed_dict, input_feat_name, run_train, train_start_date, train_end_date, test_start_date, test_end_date)

    train_df_EC, test_df_EC = target_df(train_parsed_dict, test_parsed_dict, run_train, train_start_date, train_end_date, test_start_date, test_end_date)
    prtime("feature and target data strucuted successfully into dataframe, further processing...")

    train_df, test_df = merge_n_resample(train_df_dict, test_df_dict, train_df_EC, test_df_EC, run_train)
    train_df, test_df = get_static_features(train_df, test_df, run_train)
    train_df, test_df = fill_nan(train_df, test_df, run_train)
    prtime("train and test dataframes merged and resampled. Exiting data_preprocessing module")

    return train_df, test_df





