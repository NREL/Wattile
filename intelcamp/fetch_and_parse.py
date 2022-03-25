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
from intelcamp.util import prtime
import time
import json
import glob

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


def main():
    train_start_date = '2018-08-01'
    train_end_date = '2018-11-01'
    test_start_date = '2018-11-02'
    test_end_date = '2018-11-20'
    run_train = True



    # Xcel Energy Meter's EnergyConsumption data
    # reference_id for STM_campus EC = '@p:stm_campus:r:225918db-bfbda16a'
    # 'Garage_Energy_Net : '@p:stm_campus:r:23752630-b115b3c7'
    # 'Garage_Real_Power_Total'= '@p:stm_campus:r:23295bf9-933c18ac'
    # RSF1 Main= 'Real_Power_Total': '@p:stm_campus:r:1f587070-8c045a5e'
    # RSF2 Main= 'Real_Power_Total': '@p:stm_campus:r:1f587071-6a7f739d'
    # Cafe Main= 'Energy Net' : '@p:stm_campus:r:23752630-93eb705e'

    root_url = 'https://internal-apis.nrel.gov/intelligentcampus/hisRead?id='
    reference_id = ['@p:stm_campus:r:20ed5e0a-275dbdc2', '@p:stm_campus:r:20ed5e0a-53e174aa',
                    '@p:stm_campus:r:20ed5e0a-fe755c80', '@p:stm_campus:r:20ed5df2-2c0e126b', '@p:stm_campus:r:20ed5e0a-acc8beff',
                    '@p:stm_campus:r:20ed5df2-fd2eecc5']
    train_date_range = '&range=\"' + train_start_date + '%2c' + train_end_date + '\"'
    test_date_range = '&range=\"' + test_start_date + '%2c' + test_end_date + '\"'
    feat_name = ['RH', 'BP', 'DBT', 'GHI', 'TCC', 'WS']
    input_feat_name = ['RH', 'BP', 'DBT', 'GHI', 'TCC', 'WS']
    target_feat_name = ['Garage_Real_Power_Total']



    train_response_dict, test_response_dict = fetch_data(root_url, reference_id, train_date_range, test_date_range, feat_name, run_train)
    train_parsed_dict, test_parsed_dict = parse_data(train_response_dict, test_response_dict, feat_name, run_train)
    prtime("data fetched and parsed in a dictionary successfully, dumping it to json")

    with open('train_parsed.json', 'w') as fw:
        json.dump(train_parsed_dict, fw)
    with open('test_parsed.json', 'w') as fw:
        json.dump(test_parsed_dict, fw)



if __name__ == '__main__':

    main()
