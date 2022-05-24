"""
Script for retrieving historical weather data for a particular location and time

Sources:
--- ISU ---
- Retrieved from https://mesonet.agron.iastate.edu/request/download.phtml?network=CO_ASOS
'tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'p01i', 'alti',
       'mslp', 'vsby', 'gust', 'skyc1', 'skyc2', 'skyc3', 'skyc4', 'skyl1',
       'skyl2', 'skyl3', 'skyl4', 'wxcodes', 'ice_accretion_1hr',
       'ice_accretion_3hr', 'ice_accretion_6hr', 'peak_wind_gust',
       'peak_wind_drct', 'peak_wind_time', 'feel'

--- National Solar Radiation Database (NSRD) ---
ghi,dhi,dni,air_temperature,relative_humidity,total_precipitable_water,surface_albedo

"""
import json

import pandas as pd
from scipy import spatial


def get_nsrdb(year, target_lat, target_lon):
    # ---------------- Get data from NSRDB ----------------
    with open("nsrd_auth.json", "r") as read_file:
        auth = json.load(read_file)
    attributes = [
        "ghi",
        "dhi",
        "dni",
        "air_temperature",
        "relative_humidity",
        "total_precipitable_water",
        "surface_albedo",
    ]
    year = str(year)
    if year == "2020" or year == "2016":
        leap_year = "true"
    else:
        leap_year = "false"
    interval = "30"
    utc = "false"

    # Get metadata
    url = (
        "https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?"
        f"wkt=POINT({target_lon}%20{target_lat})&"
        f"names={year}&"
        f"leap_day={leap_year}&"
        f"interval={interval}&"
        f"utc={utc}&"
        f"full_name={auth['your_name']}&"
        f"email={auth['your_email']}&"
        f"affiliation={auth['your_affiliation']}&"
        f"mailing_list={auth['mailing_list']}&"
        f"reason={auth['reason_for_use']}"
        f"api_key={auth['api_key']}&"
        f"attributes={','.join(attributes)}"
    )

    # Get actual data
    df = pd.read_csv(url, skiprows=2)

    # Set the time index in the pandas dataframe:
    concat = (
        df["Year"].astype(str)
        + "-"
        + df["Month"].astype(str)
        + "-"
        + df["Day"].astype(str)
        + " "
        + df["Hour"].astype(str)
        + ":"
        + df["Minute"].astype(str)
        + ":00"
    )
    df = df.set_index(pd.to_datetime(concat))
    df = df.drop(["Year", "Month", "Day", "Hour", "Minute"], axis=1)
    return df


# Inputs
target_lat = 43.56
target_lon = -116.21
data_start = "02-02-2017 00:00:00"
data_end = "02-02-2018 00:00:00"

# ---------------- Get data from ISU ----------------
# Find the closest weather station to our site
url = "https://mesonet.agron.iastate.edu/sites/networks.php?network=_ALL_&format=csv&nohtml=on"
station_meta = pd.read_csv(url)
tree = spatial.KDTree(station_meta[["lat", "lon"]])
result = tree.query([(target_lat, target_lon)])
nearest_stid = station_meta["stid"][result[1][0]]
distance = result[0][0]  # In degrees

# Do the main query for the selected site
time1 = pd.to_datetime(data_start)
time2 = pd.to_datetime(data_end)
station_id = nearest_stid
direct = "no"
url = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    f"station={station_id}&"
    "data=all&"
    f"year1={time1.year}&"
    f"month1={time1.month}&"
    f"day1={time1.day}&"
    f"year2={time2.year}&"
    f"month2={time2.month}&"
    f"ay2={time2.day}&"
    "tz=Etc%2FUTC&"
    "format=onlycomma&"
    "latlon=no&"
    "missing=M&"
    "trace=T&"
    f"direct={direct}&"
    "report_type=1&"
    "report_type=2"
)
ISU_data = pd.read_csv(url, na_values=["M"], index_col="valid", parse_dates=True)


# Combine ISU and NSRDB data
