import numpy as np
import pandas as pd
import requests
import re
from dateutil.parser import parse


start_date = '2018-10-22'
end_date = '2018-10-26'
start_time = '00:01:00'
end_time = '23:59:00'

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


feat_name = ['EC', 'RH', 'BP', 'DBT', 'GHI', 'TCC', 'WS']
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


print("done")