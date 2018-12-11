import numpy as np
import pandas as pd
import requests
import re
from dateutil.parser import parse
from functools import reduce


start_date = '2018-10-22'
end_date = '2018-11-22'
start_time = '00:01:00'
end_time = '23:59:00'
EC_start_time = '00:00:00'
EC_end_time = '23:45:00'

root_url = 'https://internal-apis.nrel.gov/intelligentcampus/hisRead?id='
reference_id = ['@p:nrel:r:225918db-bfbda16a','@p:nrel:r:20ed5e0a-275dbdc2','@p:nrel:r:20ed5e0a-53e174aa',
                '@p:nrel:r:20ed5e0a-fe755c80','@p:nrel:r:20ed5df2-2c0e126b','@p:nrel:r:20ed5e0a-acc8beff',
                '@p:nrel:r:20ed5df2-fd2eecc5']
date_range = '&range=\"'+start_date+'%2c'+end_date+'\"'

feat_name = ['EC','RH','BP','DBT','GHI','TCC','WS']
response_dict = {}
for i in range(len(reference_id)):
    response_dict['resp_'+feat_name[i]] = requests.get(root_url+reference_id[i]+date_range)
    if response_dict['resp_'+feat_name[i]].status_code == 200:
        pass
    else:
        print("response from {} is not getting fetched from API".format(feat_name[i]))

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


parsed_dict = {}
for i in range(len(feat_name)):
    parsed_dict[feat_name[i]] = response_dict['resp_' + feat_name[i]].content.decode('utf-8').split("\n")
    parsed_dict[feat_name[i]] = parsed_dict[feat_name[i]][2:]
    parsed_dict[feat_name[i]] = filter(None, parsed_dict[feat_name[i]])
    parsed_dict[feat_name[i]] = list(map(str_split, parsed_dict[feat_name[i]]))

    # the following line gives list (len 2) of lists (i.e. EC_dt and EC_value)
    # i.e. EC_dt, EC_value = EC[0], EC[1]
    parsed_dict[feat_name[i]] = list(zip(*parsed_dict[feat_name[i]]))

    # parsing the datetimeinfo obtained in above list into datetime string, date and time
    # the lists can be unpacked as:
    # EC_datetime, EC_date, EC_time = EC_dt_parsed[0], EC_dt_parsed[1], EC_dt_parsed[2]
    parsed_dict[feat_name[i] + '_dt_parsed'] = list(map(date_parser, parsed_dict[feat_name[i]][0]))
    parsed_dict[feat_name[i] + '_dt_parsed'] = list(zip(*parsed_dict[feat_name[i] + '_dt_parsed']))


# creating a dictionary of input features dataframes with the parsed lists generated in the parsing module

input_feat_name = ['RH', 'BP', 'DBT', 'GHI', 'TCC', 'WS']
df_dict = {}
for i in range(len(input_feat_name)):
    df_dict["df_" + input_feat_name[i]] = pd.DataFrame(
        {'datetime_str': parsed_dict[input_feat_name[i] + '_dt_parsed'][0],
         input_feat_name[i]: parsed_dict[input_feat_name[i]][1]},
        columns=['datetime_str', input_feat_name[i]])

    df_temp = df_dict["df_" + input_feat_name[i]]
    df_temp.name = "df_" + input_feat_name[i]
    print("raw_dataframe = {}, shape = {}".format(df_temp.name, df_temp.shape))
    df_temp['datetime_str'] = pd.to_datetime(df_temp['datetime_str'])

    if not (df_temp.loc[0, 'datetime_str'] == pd.to_datetime(start_date + ' ' + start_time)):
        df_temp.loc[0, 'datetime_str'] = pd.to_datetime(start_date + ' ' + start_time)
    if not (df_temp.loc[df_temp.index[-1], 'datetime_str'] == pd.to_datetime(end_date + ' ' + end_time)):
        df_temp.loc[df_temp.index[-1], 'datetime_str'] = pd.to_datetime(end_date + ' ' + end_time)

    df_temp = df_temp.set_index('datetime_str').resample("1min").first().reset_index().reindex(columns=df_temp.columns)
    cols = df_temp.columns.difference([input_feat_name[i]])
    df_temp[cols] = df_temp[cols].ffill()
    df_temp[input_feat_name[i]] = df_temp[input_feat_name[i]].fillna(
        ((df_temp[input_feat_name[i]].shift() + df_temp[input_feat_name[i]].shift(-1)) / 2))
    print("shape of processed dataframe: {}".format(df_temp.shape))

    df_dict["df_" + input_feat_name[i]] = df_temp
    del df_temp


# processing Energy Consumption values separately

df_EC = pd.DataFrame({'datetime_str':parsed_dict['EC_dt_parsed'][0],'EC':parsed_dict['EC'][1],},
                 columns=['datetime_str','EC'])

df_EC = pd.DataFrame({'datetime_str': parsed_dict['EC_dt_parsed'][0], 'EC': parsed_dict['EC'][1], },
                     columns=['datetime_str', 'EC'])
print("shape of raw dataframe: {}".format(df_EC.shape))

df_EC['datetime_str'] = pd.to_datetime(df_EC['datetime_str'])
if not (df_EC.loc[0, 'datetime_str'] == pd.to_datetime(start_date + ' ' + EC_start_time)):
    df_EC.loc[0, 'datetime_str'] = pd.to_datetime(start_date + ' ' + EC_start_time)
if not (df_EC.loc[df_EC.index[-1], 'datetime_str'] == pd.to_datetime(end_date + ' ' + EC_end_time)):
    df_EC.loc[df_EC.index[-1], 'datetime_str'] = pd.to_datetime(end_date + ' ' + EC_end_time)

df_EC = df_EC.set_index('datetime_str').resample("15min").first().reset_index().reindex(columns=df_EC.columns)
cols = df_EC.columns.difference(['EC'])
df_EC[cols] = df_EC[cols].ffill()
df_EC['EC'] = df_EC['EC'].fillna(((df_EC['EC'].shift() + df_EC['EC'].shift(-1)) / 2))
print("shape of processed dataframe: {}".format(df_EC.shape))


# merging and re-sampling the dataframes containing input features
df_list = []
for key, value in df_dict.items():
    df_list.append(df_dict[key])

input_df = reduce(lambda left, right: pd.merge(left, right, on=['datetime_str'], how='outer'), df_list)

input_df =input_df.set_index('datetime_str').resample("15min").mean().reset_index().reindex(columns=input_df.columns)

# merging input_df (dataframe with input features) with df_EC (target dataframe)

df = input_df.merge(df_EC, how='outer', on='datetime_str')


def get_static_features(df):
    # inserting new columns at index 7 and onward
    idx = 7
    new_col = df.datetime_str.dt.dayofyear.astype(np.float32)
    df.insert(loc=idx, column='Doy', value=new_col)

    idx = idx + 1
    new_col = pd.to_timedelta(df.datetime_str.dt.strftime('%H:%M:%S')).dt.total_seconds().astype(int)
    df.insert(loc=idx, column='TimeinSec', value=new_col)

    # conversion to cyclic coordinates
    seconds_in_day = 24 * 60 * 60

    idx = idx + 1
    new_col = np.sin(2 * np.pi * df.TimeinSec / seconds_in_day)
    df.insert(loc=idx, column='sin_time', value=new_col)

    idx = idx + 1
    new_col = np.cos(2 * np.pi * df.TimeinSec / seconds_in_day)
    df.insert(loc=idx, column='cos_time', value=new_col)

    # adding the lagged EC features
    idx = idx + 1
    new_col = df['EC'].shift(4)
    df.insert(loc=idx, column='EC_t-4', value=new_col)

    idx = idx + 1
    new_col = df['EC'].shift(3)
    df.insert(loc=idx, column='EC_t-3', value=new_col)

    idx = idx + 1
    new_col = df['EC'].shift(2)
    df.insert(loc=idx, column='EC_t-2', value=new_col)

    idx = idx + 1
    new_col = df['EC'].shift(1)
    df.insert(loc=idx, column='EC_t-1', value=new_col)

    return df


def fill_nan(df):

    # carefully filling in the nan values created during the shift operation
    # doing so by taking the mean of the row of the associated columns which have the same timestamp as nan-valued cell

    df.loc[0, 'EC_t-4'] = df[df.datetime_str.apply(lambda x: x.time()) == df.datetime_str[0].time()]['EC_t-4'].mean(
        axis=0)
    df.loc[1, 'EC_t-4'] = df[df.datetime_str.apply(lambda x: x.time()) == df.datetime_str[1].time()]['EC_t-4'].mean(
        axis=0)
    df.loc[2, 'EC_t-4'] = df[df.datetime_str.apply(lambda x: x.time()) == df.datetime_str[2].time()]['EC_t-4'].mean(
        axis=0)
    df.loc[3, 'EC_t-4'] = df[df.datetime_str.apply(lambda x: x.time()) == df.datetime_str[3].time()]['EC_t-4'].mean(
        axis=0)

    df.loc[0, 'EC_t-3'] = df[df.datetime_str.apply(lambda x: x.time()) == df.datetime_str[0].time()]['EC_t-3'].mean(
        axis=0)
    df.loc[1, 'EC_t-3'] = df[df.datetime_str.apply(lambda x: x.time()) == df.datetime_str[1].time()]['EC_t-3'].mean(
        axis=0)
    df.loc[2, 'EC_t-3'] = df[df.datetime_str.apply(lambda x: x.time()) == df.datetime_str[2].time()]['EC_t-3'].mean(
        axis=0)

    df.loc[0, 'EC_t-2'] = df[df.datetime_str.apply(lambda x: x.time()) == df.datetime_str[0].time()]['EC_t-2'].mean(
        axis=0)
    df.loc[1, 'EC_t-2'] = df[df.datetime_str.apply(lambda x: x.time()) == df.datetime_str[1].time()]['EC_t-2'].mean(
        axis=0)

    df.loc[0, 'EC_t-1'] = df[df.datetime_str.apply(lambda x: x.time()) == df.datetime_str[0].time()]['EC_t-1'].mean(
        axis=0)

    return df





