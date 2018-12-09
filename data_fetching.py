import numpy as np
import pandas as pd
import requests
import json

import dateutil
from dateutil.parser import parse
import datetime

resp1 = requests.get("https://internal-apis.nrel.gov/intelligentcampus/hisRead?id=@p:nrel:r:225918db-bfbda16a&range=\"2018-08-31%2c2018-09-02\"")
if resp1.status_code == 200:
    pass
else:
    print("Energy consumption data not coming from API")

EC = resp1.content.decode('utf-8').split("\n")
EC = EC[2:]
EC = filter(None, EC)

def str_split(row):
    time_val = row.split(",")[0].strip(" Denver")
    energy_val = row.split(",")[1].strip("kWh")
    return (time_val, float(energy_val))

EC = list(map(str_split, EC))
EC = list(zip(*EC))
EC_dt, EC_value = EC[0], EC[1]

def date_parser(row):
    parsed = parse(row)
    datetime_var = parsed.strftime(format='%m-%d-%y %H:%M:%S')
    date = parsed.date()
    time = parsed.time()
    return (datetime_var ,date, time)

EC_dt_parsed = list(map(date_parser, EC_dt))
EC_dt_parsed = list(zip(*EC_dt_parsed))
EC_datetime, EC_date, EC_time = EC_dt_parsed[0], EC_dt_parsed[1], EC_dt_parsed[2]

print("done")