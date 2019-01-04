import sys
import pathlib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import re
from scipy import stats
from dateutil.parser import parse
from functools import reduce
from util import prtime
import time
import json


start_time = '00:01:00'
end_time = '23:59:00'
tar_start_time = '00:00:00'
tar_end_time = '23:45:00'


def requests_retry_session(retries=5,
                           backoff_factor=0.3,
                           status_forcelist=(500, 502, 504, 404),
                           session=None,):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    return session

def fetch_data(root_url, reference_id, train_date_range, test_date_range, feat_name, run_train):
    train_response_dict = {}
    test_response_dict = {}
    if run_train:
        for i in range(len(reference_id)):

            t0 = time.time()
            try:
                train_response_dict['resp_' + feat_name[i]] = requests_retry_session().get(
                    root_url+reference_id[i]+train_date_range,
                )
            except Exception as x:
                print('It failed :(', x.__class__.__name__)
            else:
                print('API has sent data for training', train_response_dict['resp_' + feat_name[i]].status_code)
            finally:
                t1 = time.time()
                time_took = t1-t0
                print('Train data fetch_n_parse took {} seconds for {}'.format(time_took, feat_name[i]))

    for i in range(len(reference_id)):

        t0 = time.time()
        try:
            test_response_dict['resp_' + feat_name[i]] = requests_retry_session().get(
                root_url + reference_id[i] + test_date_range,
            )
        except Exception as x:
            print('It failed :(', x.__class__.__name__)
        else:
            print('API has sent data for testing', test_response_dict['resp_' + feat_name[i]].status_code)
        finally:
            t1 = time.time()
            time_took = t1 - t0
            print('Test data fetch_n_parse took {} seconds for {}'.format(time_took, feat_name[i]))

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


            # the following line gives list (len 2) of lists (i.e. GHI_dt and GHI_value)
            # i.e. GHI_dt, GHI_value = GHI[0], GHI[1]
            train_parsed_dict[feat_name[i]] = list(zip(*train_parsed_dict[feat_name[i]]))


    for i in range(len(feat_name)):
        test_parsed_dict[feat_name[i]] = test_response_dict['resp_' + feat_name[i]].content.decode('utf-8').split(
            "\n")
        test_parsed_dict[feat_name[i]] = test_parsed_dict[feat_name[i]][2:]
        test_parsed_dict[feat_name[i]] = filter(None, test_parsed_dict[feat_name[i]])
        test_parsed_dict[feat_name[i]] = list(map(str_split, test_parsed_dict[feat_name[i]]))

        test_parsed_dict[feat_name[i]] = list(zip(*test_parsed_dict[feat_name[i]]))

    return train_parsed_dict, test_parsed_dict


def input_feat_dfs(train_parsed_dict, test_parsed_dict, input_feat_name, run_train, train_start_date, train_end_date, test_start_date, test_end_date):

    # creating a dictionary of input features dataframes with the parsed lists generated in the parsing module
    train_df_dict = {}
    test_df_dict = {}
    if run_train:

        for i in range(len(input_feat_name)):
            # parsing the datetimeinfo str into datetime string, date and time
            train_parsed_dict[input_feat_name[i] + '_dt_parsed'] = list(map(date_parser, train_parsed_dict[input_feat_name[i]][0]))
            train_parsed_dict[input_feat_name[i] + '_dt_parsed'] = list(zip(*train_parsed_dict[input_feat_name[i] + '_dt_parsed']))

            train_df_dict["df_" + input_feat_name[i]] = pd.DataFrame({'datetime_str': train_parsed_dict[input_feat_name[i] + '_dt_parsed'][0], input_feat_name[i]: train_parsed_dict[input_feat_name[i]][1]},columns=['datetime_str', input_feat_name[i]])

            df_temp = train_df_dict["df_" + input_feat_name[i]]
            df_temp.name = "df_" + input_feat_name[i]
            prtime("raw_train input df = {}, shape = {}".format(df_temp.name, df_temp.shape))

            z_temp = np.abs(stats.zscore(df_temp[input_feat_name[i]]))
            threshold = 3
            mask = np.where(~(z_temp > threshold))
            # print(len(mask))
            df_temp = df_temp.iloc[mask]
            df_temp.name = "df_" + input_feat_name[i]
            prtime("outlier-removed train input df = {}, shape = {}".format(df_temp.name, df_temp.shape))


            df_temp['datetime_str'] = pd.to_datetime(df_temp['datetime_str'])
            if not (df_temp.loc[0, 'datetime_str'] == pd.to_datetime(train_start_date + ' ' + start_time)):
                df_temp.loc[0, 'datetime_str'] = pd.to_datetime(train_start_date + ' ' + start_time)
            if not (df_temp.loc[df_temp.index[-1], 'datetime_str'] == pd.to_datetime(train_end_date + ' ' + end_time)):
                df_temp.loc[df_temp.index[-1], 'datetime_str'] = pd.to_datetime(train_end_date + ' ' + end_time)


            #df_temp = df_temp.set_index('datetime_str').resample("1min").first().reset_index().reindex(columns=df_temp.columns)
            # df_temp[input_feat_name[i] + '_roll_mean'] = df_temp[input_feat_name[i]].rolling(12, center=True,
            #                                                                                  min_periods=1).mean()
            # df_temp[input_feat_name[i] + '_roll_mean'].update(df_temp[input_feat_name[i]])
            # df_temp.drop([input_feat_name[i]], inplace=True, axis=1)
            # df_temp = df_temp.rename(columns={input_feat_name[i] + '_roll_mean': input_feat_name[i]})

            cols = df_temp.columns
            df_temp = df_temp.set_index('datetime_str').resample("1min").first()
            df_temp.interpolate(inplace=True)
            df_temp = df_temp.reset_index().reindex(columns=cols)
            df_temp.name = "df_" + input_feat_name[i]
            prtime("final-processed train input df = {}, shape = {}".format(df_temp.name, df_temp.shape))

            train_df_dict["df_" + input_feat_name[i]] = df_temp
            del df_temp

    for i in range(len(input_feat_name)):

        # parsing the datetimeinfo obtained in above list into datetime string, date and time
        test_parsed_dict[input_feat_name[i] + '_dt_parsed'] = list(map(date_parser, test_parsed_dict[input_feat_name[i]][0]))
        test_parsed_dict[input_feat_name[i] + '_dt_parsed'] = list(zip(*test_parsed_dict[input_feat_name[i] + '_dt_parsed']))
        test_df_dict["df_" + input_feat_name[i]] = pd.DataFrame(
            {'datetime_str': test_parsed_dict[input_feat_name[i] + '_dt_parsed'][0],
             input_feat_name[i]: test_parsed_dict[input_feat_name[i]][1]},
            columns=['datetime_str', input_feat_name[i]])

        df_temp = test_df_dict["df_" + input_feat_name[i]]
        df_temp.name = "df_" + input_feat_name[i]
        prtime("raw_test input df = {}, shape = {}".format(df_temp.name, df_temp.shape))

        z_temp = np.abs(stats.zscore(df_temp[input_feat_name[i]]))
        threshold = 3
        mask = np.where(~(z_temp > threshold))
        # print(len(mask))
        df_temp = df_temp.iloc[mask]
        df_temp.name = "df_" + input_feat_name[i]
        prtime("outlier-removed test input df = {}, shape = {}".format(df_temp.name, df_temp.shape))

        df_temp['datetime_str'] = pd.to_datetime(df_temp['datetime_str'])
        if not (df_temp.loc[0, 'datetime_str'] == pd.to_datetime(test_start_date + ' ' + start_time)):
            df_temp.loc[0, 'datetime_str'] = pd.to_datetime(test_start_date + ' ' + start_time)
        if not (df_temp.loc[df_temp.index[-1], 'datetime_str'] == pd.to_datetime(test_end_date + ' ' + end_time)):
            df_temp.loc[df_temp.index[-1], 'datetime_str'] = pd.to_datetime(test_end_date + ' ' + end_time)

        # df_temp = df_temp.set_index('datetime_str').resample("1min").first().reset_index().reindex(
        #     columns=df_temp.columns)
        # df_temp[input_feat_name[i] + '_roll_mean'] = df_temp[input_feat_name[i]].rolling(12, center=True,
        #                                                                                  min_periods=1).mean()
        # df_temp[input_feat_name[i] + '_roll_mean'].update(df_temp[input_feat_name[i]])
        # df_temp.drop([input_feat_name[i]], inplace=True, axis=1)
        # df_temp = df_temp.rename(columns={input_feat_name[i] + '_roll_mean': input_feat_name[i]})

        cols = df_temp.columns
        df_temp = df_temp.set_index('datetime_str').resample("1min").first()
        df_temp.interpolate(inplace=True)
        df_temp = df_temp.reset_index().reindex(columns=cols)
        df_temp.name = "df_" + input_feat_name[i]
        prtime("final-processed train input df = {}, shape = {}".format(df_temp.name, df_temp.shape))

        test_df_dict["df_" + input_feat_name[i]] = df_temp
        del df_temp

    return train_df_dict, test_df_dict


def target_df(train_parsed_dict, test_parsed_dict, run_train, train_start_date, train_end_date, test_start_date,
              test_end_date, target_feat_name):
    # processing Energy Consumption values separately
    train_df_target = {}
    test_df_target = {}
    if run_train:
        for i in range(len(target_feat_name)):

            # parsing the datetimeinfo obtained in above list into datetime string, date and time
            train_parsed_dict[target_feat_name[i] + '_dt_parsed'] = list(
                map(date_parser, train_parsed_dict[target_feat_name[i]][0]))
            train_parsed_dict[target_feat_name[i] + '_dt_parsed'] = list(
                zip(*train_parsed_dict[target_feat_name[i] + '_dt_parsed']))

            train_df_target["df_"+target_feat_name[i]] = pd.DataFrame({'datetime_str': train_parsed_dict[target_feat_name[i]+'_dt_parsed'][0], target_feat_name[i]: train_parsed_dict[target_feat_name[i]][1]},
                             columns=['datetime_str', target_feat_name[i]])

            df_temp = train_df_target["df_" + target_feat_name[i]]
            df_temp.name = "df_" + target_feat_name[i]
            prtime("raw_train target df = {}, shape = {}".format(df_temp.name, df_temp.shape))

            z_temp = np.abs(stats.zscore(df_temp[target_feat_name[i]]))
            threshold = 3.2
            mask = np.where(~(z_temp > threshold))
            # print(len(mask))
            df_temp = df_temp.iloc[mask]
            df_temp.name = "df_" + target_feat_name[i]
            prtime("outlier-removed train target df = {}, shape = {}".format(df_temp.name, df_temp.shape))

            df_temp['datetime_str'] = pd.to_datetime(df_temp['datetime_str'])
            if not (df_temp.loc[0, 'datetime_str'] == pd.to_datetime(train_start_date + ' ' + tar_start_time)):
                df_temp.loc[0, 'datetime_str'] = pd.to_datetime(train_start_date + ' ' + tar_start_time)
            if not (df_temp.loc[df_temp.index[-1], 'datetime_str'] == pd.to_datetime(train_end_date + ' ' + tar_end_time)):
                df_temp.loc[df_temp.index[-1], 'datetime_str'] = pd.to_datetime(train_end_date + ' ' + tar_end_time)

            # train_df_target = train_df_target.set_index('datetime_str').resample("15min").first().reset_index().reindex(columns=train_df_target.columns)
            # train_df_target['EC_roll_mean'] = train_df_target['EC'].rolling(12, center=True, min_periods=1).mean()
            # train_df_target['EC_roll_mean'].update(train_df_target['EC'])
            # train_df_target.drop(['EC'], inplace=True, axis=1)
            # train_df_target = train_df_target.rename(columns={'EC_roll_mean': 'EC'})

            cols = df_temp.columns
            df_temp = df_temp.set_index('datetime_str').resample("1min").first()
            df_temp.interpolate(inplace=True)
            df_temp = df_temp.reset_index().reindex(columns=cols)

            train_df_target["df_" + target_feat_name[i]] = df_temp
            df_temp.name = "df_" + target_feat_name[i]
            prtime("final-processed train target df = {}, shape = {}".format(df_temp.name, df_temp.shape))
            del df_temp

    else:
        # if run_train = False, then return empty train dataframe
        train_df_target = pd.DataFrame()

    for i in range(len(target_feat_name)):

        # parsing the datetimeinfo obtained in above list into datetime string, date and time
        test_parsed_dict[target_feat_name[i] + '_dt_parsed'] = list(
            map(date_parser, test_parsed_dict[target_feat_name[i]][0]))
        test_parsed_dict[target_feat_name[i] + '_dt_parsed'] = list(
            zip(*test_parsed_dict[target_feat_name[i] + '_dt_parsed']))

        test_df_target["df_"+target_feat_name[i]] = pd.DataFrame({'datetime_str': test_parsed_dict[target_feat_name[i]+'_dt_parsed'][0], target_feat_name[i]: test_parsed_dict[target_feat_name[i]][1]},
                                 columns=['datetime_str', target_feat_name[i]])

        df_temp = test_df_target["df_" + target_feat_name[i]]
        df_temp.name = "df_" + target_feat_name[i]
        prtime("raw_train target dataframe = {}, shape = {}".format(df_temp.name, df_temp.shape))

        z_temp = np.abs(stats.zscore(df_temp[target_feat_name[i]]))
        threshold = 3.2
        mask = np.where(~(z_temp > threshold))
        # print(len(mask))
        df_temp = df_temp.iloc[mask]
        df_temp.name = "df_" + target_feat_name[i]
        prtime("outlier-removed test target df = {}, shape = {}".format(df_temp.name, df_temp.shape))

        df_temp['datetime_str'] = pd.to_datetime(df_temp['datetime_str'])
        if not (df_temp.loc[0, 'datetime_str'] == pd.to_datetime(test_start_date + ' ' + tar_start_time)):
            df_temp.loc[0, 'datetime_str'] = pd.to_datetime(test_start_date + ' ' + tar_start_time)
        if not (df_temp.loc[df_temp.index[-1], 'datetime_str'] == pd.to_datetime(test_end_date + ' ' + tar_end_time)):
            df_temp.loc[df_temp.index[-1], 'datetime_str'] = pd.to_datetime(test_end_date + ' ' + tar_end_time)

        # test_df_target = test_df_target.set_index('datetime_str').resample("15min").first().reset_index().reindex(
        #     columns=test_df_target.columns)
        # test_df_target['EC_roll_mean'] = test_df_target['EC'].rolling(12, center=True, min_periods=1).mean()
        # test_df_target['EC_roll_mean'].update(test_df_target['EC'])
        # test_df_target.drop(['EC'], inplace=True, axis=1)
        # test_df_target = test_df_target.rename(columns={'EC_roll_mean': 'EC'})
        cols = df_temp.columns
        df_temp = df_temp.set_index('datetime_str').resample("1min").first()
        df_temp.interpolate(inplace=True)
        df_temp = df_temp.reset_index().reindex(columns=cols)

        test_df_target["df_" + target_feat_name[i]] = df_temp
        df_temp.name = "df_" + target_feat_name[i]
        prtime("final-processed test target df = {}, shape = {}".format(df_temp.name, df_temp.shape))
        del df_temp

    return train_df_target, test_df_target

def merge_n_resample(train_df_dict, test_df_dict, train_df_target, test_df_target, run_train):
    # merging the multiple dataframes (in the dictionary) containing input features
    if run_train:
        train_df_list = []
        for key, value in train_df_dict.items():
            train_df_list.append(train_df_dict[key])

        train_input_df = reduce(lambda left, right: pd.merge(left, right, on=['datetime_str'], how='outer'), train_df_list)

        # re-sampling (downsampling 1-min data to 15-min, since target values are for 15-min)
        #train_input_df =train_input_df.set_index('datetime_str').resample("1min").mean().reset_index().reindex(#columns=train_input_df.columns)

        train_df_target_list = []
        for key, value in train_df_target.items():
            train_df_target_list.append(train_df_target[key])

        train_df_target = reduce(lambda left, right: pd.merge(left, right, on=['datetime_str'], how='outer'), train_df_target_list)

        # merging input_df (dataframe with input features) with target dataframe
        train_df = train_input_df.merge(train_df_target, how='outer', on='datetime_str')

    else:
        train_df = pd.DataFrame()

    test_df_list = []
    for key, value in test_df_dict.items():
        test_df_list.append(test_df_dict[key])

    test_input_df = reduce(lambda left, right: pd.merge(left, right, on=['datetime_str'], how='outer'), test_df_list)

    # re-sampling (downsampling 1-min data to 15-min, since target values are for 15-min)
    # test_input_df = test_input_df.set_index('datetime_str').resample("1min").mean().reset_index().reindex(
    #    columns=test_input_df.columns)

    test_df_target_list = []
    for key, value in test_df_target.items():
        test_df_target_list.append(test_df_target[key])

    test_df_target = reduce(lambda left, right: pd.merge(left, right, on=['datetime_str'], how='outer'),
                             test_df_target_list)

    # merging input_df (dataframe with input features) with df_EC (target dataframe)
    test_df = test_input_df.merge(test_df_target, how='outer', on='datetime_str')

    return train_df, test_df


def get_static_features(train_df, test_df, run_train, target_feat_name):

    if run_train:
        # inserting new columns at index 7 and onward
        idx = 7
        new_col = train_df.datetime_str.dt.dayofyear.astype(np.float32)
        train_df.insert(loc=idx, column='doy', value=new_col)

        idx = idx + 1
        new_col = train_df.datetime_str.dt.dayofweek.astype(np.float32)
        train_df.insert(loc=idx, column='dow', value=new_col)

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
        new_col = train_df[target_feat_name[0]].shift(4)
        train_df.insert(loc=idx, column='t-4', value=new_col)

        idx = idx + 1
        new_col = train_df[target_feat_name[0]].shift(3)
        train_df.insert(loc=idx, column='t-3', value=new_col)

        idx = idx + 1
        new_col = train_df[target_feat_name[0]].shift(2)
        train_df.insert(loc=idx, column='t-2', value=new_col)

        idx = idx + 1
        new_col = train_df[target_feat_name[0]].shift(1)
        train_df.insert(loc=idx, column='t-1', value=new_col)

    idx = 7
    new_col = test_df.datetime_str.dt.dayofyear.astype(np.float32)
    test_df.insert(loc=idx, column='doy', value=new_col)

    idx = idx + 1
    new_col = test_df.datetime_str.dt.dayofweek.astype(np.float32)
    test_df.insert(loc=idx, column='dow', value=new_col)

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
    new_col = test_df[target_feat_name[0]].shift(4)
    test_df.insert(loc=idx, column='t-4', value=new_col)

    idx = idx + 1
    new_col = test_df[target_feat_name[0]].shift(3)
    test_df.insert(loc=idx, column='t-3', value=new_col)

    idx = idx + 1
    new_col = test_df[target_feat_name[0]].shift(2)
    test_df.insert(loc=idx, column='t-2', value=new_col)

    idx = idx + 1
    new_col = test_df[target_feat_name[0]].shift(1)
    test_df.insert(loc=idx, column='t-1', value=new_col)

    return train_df, test_df


def fill_nan(train_df, test_df, run_train):

    # carefully filling in the nan values created during the shift operation
    # doing so by taking the mean of the row of the associated columns which have the same timestamp as nan-valued cell

    if run_train:

        train_df.loc[0, 't-4'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[0].time()]['t-4'].mean(
            axis=0)
        train_df.loc[1, 't-4'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[1].time()]['t-4'].mean(
            axis=0)
        train_df.loc[2, 't-4'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[2].time()]['t-4'].mean(
            axis=0)
        train_df.loc[3, 't-4'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[3].time()]['t-4'].mean(
            axis=0)

        train_df.loc[0, 't-3'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[0].time()]['t-3'].mean(
            axis=0)
        train_df.loc[1, 't-3'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[1].time()]['t-3'].mean(
            axis=0)
        train_df.loc[2, 't-3'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[2].time()]['t-3'].mean(
            axis=0)

        train_df.loc[0, 't-2'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[0].time()]['t-2'].mean(
            axis=0)
        train_df.loc[1, 't-2'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[1].time()]['t-2'].mean(
            axis=0)

        train_df.loc[0, 't-1'] = \
        train_df[train_df.datetime_str.apply(lambda x: x.time()) == train_df.datetime_str[0].time()]['t-1'].mean(
            axis=0)

    test_df.loc[0, 't-4'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[0].time()]['t-4'].mean(
        axis=0)
    test_df.loc[1, 't-4'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[1].time()]['t-4'].mean(
        axis=0)
    test_df.loc[2, 't-4'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[2].time()]['t-4'].mean(
        axis=0)
    test_df.loc[3, 't-4'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[3].time()]['t-4'].mean(
        axis=0)

    test_df.loc[0, 't-3'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[0].time()]['t-3'].mean(
        axis=0)
    test_df.loc[1, 't-3'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[1].time()]['t-3'].mean(
        axis=0)
    test_df.loc[2, 't-3'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[2].time()]['t-3'].mean(
        axis=0)

    test_df.loc[0, 't-2'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[0].time()]['t-2'].mean(
        axis=0)
    test_df.loc[1, 't-2'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[1].time()]['t-2'].mean(
        axis=0)

    test_df.loc[0, 't-1'] = \
    test_df[test_df.datetime_str.apply(lambda x: x.time()) == test_df.datetime_str[0].time()]['t-1'].mean(
        axis=0)

    return train_df, test_df


def main(configs):
    train_start_date = configs['train_start_date']
    train_end_date = configs['train_end_date']
    test_start_date = configs['test_start_date']
    test_end_date = configs['test_end_date']
    run_train = configs['run_train']
    fetch_n_parse = configs['fetch_n_parse']

    train_exp_num = configs['train_exp_num']
    test_exp_num = configs['test_exp_num']
    arch_type = configs['arch_type']

    # Define the Directories to save the trained model and results.
    # Create the dir if it does not exist using pathlib
    RESULTS_DIR = 'EnergyForecasting_Results/' + arch_type + '_M' + str(train_exp_num) + '_T' + str(
        test_exp_num)

    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    log_file = RESULTS_DIR + '/' + 'console.log'
    # print("Writing print statements to ", log_file)
    # sys.stdout = open(log_file, 'w')  # Redirect print statement's outputs to file
    # print("Stdout:")

    # excluding Xcel Energy Meter's EnergyConsumption data (since it is not available for the given date range)
    # reference_id for EC = '@p:stm_campus:r:225918db-bfbda16a'
    # 'Garage_Energy_Net : '@p:stm_campus:r:23752630-b115b3c7'
    # 'Garage_Real_Power_Total'= '@p:stm_campus:r:23295bf9-933c18ac'
    # RSF1 Main= 'Real_Power_Total': '@p:stm_campus:r:1f587070-8c045a5e'
    # RSF2 Main= 'Real_Power_Total': '@p:stm_campus:r:1f587071-6a7f739d'



    root_url = 'https://internal-apis.nrel.gov/intelligentcampus/hisRead?id='
    reference_id = ['@p:stm_campus:r:20ed5e0a-275dbdc2', '@p:stm_campus:r:20ed5e0a-53e174aa',
                    '@p:stm_campus:r:20ed5e0a-fe755c80', '@p:stm_campus:r:20ed5df2-2c0e126b', '@p:stm_campus:r:20ed5e0a-acc8beff',
                    '@p:stm_campus:r:20ed5df2-fd2eecc5', '@p:stm_campus:r:1f587070-8c045a5e', '@p:stm_campus:r:1f587071-6a7f739d']
    train_date_range = '&range=\"' + train_start_date + '%2c' + train_end_date + '\"'
    test_date_range = '&range=\"' + test_start_date + '%2c' + test_end_date + '\"'
    feat_name = ['RH', 'BP', 'DBT', 'GHI', 'TCC', 'WS','RSF1_Real_Power_Total', 'RSF2_Real_Power_Total']
    input_feat_name = ['RH', 'BP', 'DBT', 'GHI', 'TCC', 'WS']
    target_feat_name = ['RSF1_Real_Power_Total', 'RSF2_Real_Power_Total']

    if fetch_n_parse:

        train_response_dict, test_response_dict = fetch_data(root_url, reference_id, train_date_range, test_date_range, feat_name, run_train)
        train_parsed_dict, test_parsed_dict = parse_data(train_response_dict, test_response_dict, feat_name, run_train)
        prtime("data fetched and parsed in a dictionary successfully, dumping it to json")

        with open('train_parsed.json', 'w') as fw:
            json.dump(train_parsed_dict, fw)
        with open('test_parsed.json', 'w') as fw:
            json.dump(test_parsed_dict, fw)

    else:
        prtime("data being read from stored jsons and will be fed to dataframe construction function")
        with open('train_parsed.json', 'r') as fr:
            train_parsed_dict = json.load(fr)
        with open('test_parsed.json', 'r') as fr:
            test_parsed_dict = json.load(fr)



    train_df_dict, test_df_dict = input_feat_dfs(train_parsed_dict, test_parsed_dict, input_feat_name, run_train, train_start_date, train_end_date, test_start_date, test_end_date)

    train_df_target, test_df_target = target_df(train_parsed_dict, test_parsed_dict, run_train, train_start_date, train_end_date, test_start_date, test_end_date, target_feat_name)
    prtime("feature and target data structured successfully into dataframe, further processing...")

    train_df, test_df = merge_n_resample(train_df_dict, test_df_dict, train_df_target, test_df_target, run_train)

    # this adjustment is for RSF Real_Power_Total case alone.
    train_df['RSF_Real_Power_Total'] = train_df['RSF1_Real_Power_Total'] + train_df['RSF2_Real_Power_Total']
    train_df.drop(['RSF1_Real_Power_Total', 'RSF2_Real_Power_Total'], axis=1, inplace=True)
    test_df['RSF_Real_Power_Total'] = test_df['RSF1_Real_Power_Total'] + test_df['RSF2_Real_Power_Total']
    test_df.drop(['RSF1_Real_Power_Total', 'RSF2_Real_Power_Total'], axis=1, inplace=True)

    # the target_feat_name list is being updated specifically for RSF case
    target_feat_name = ['RSF_Real_Power_Total']                  # rep stands for repitative feature
    train_df, test_df = get_static_features(train_df, test_df, run_train, target_feat_name)
    train_df, test_df = fill_nan(train_df, test_df, run_train)
    prtime("train and test dataframes merged and resampled. Exiting data_preprocessing module")

    configs['target_feat_name'] = target_feat_name

    # finally dropping (at max 3) a couple of rows with nan values to clean up the dataset.
    train_df.fillna(train_df.mean(), inplace=True)
    test_df.fillna(test_df.mean(), inplace=True)
    return train_df, test_df, configs





