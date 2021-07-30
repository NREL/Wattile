"""
Script for retrieving data for: Building Data Genome Project
- Retrieved from https://github.com/buds-lab/building-data-genome-project-2
- Approx. 1500 commercial buildings
- 60 min resolution
- Data for 2016 and 2017

- Weather data available included:
'airTemperature', 'cloudCoverage', 'dewTemperature', 'precipDepth1HR',
       'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed'

"""

import pandas as pd
import numpy as np
import os
from historical_weather import get_nsrdb
import tables
import pathlib
import json

# Inputs
data_dir = "/projects/intelcamp21/data/GP_new"
weather_url = "https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/weather/weather.csv"
meta_url = "https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/metadata/metadata.csv"
loads_url = "https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/cleaned/electricity_cleaned.csv"

# Get data
pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
if pathlib.Path(os.path.join(data_dir, "weather.csv")).exists():
    weather = pd.read_csv(os.path.join(data_dir, "weather.csv"))
else:
    print("Reading weather file from Github...")
    weather = pd.read_csv(weather_url)
    weather.to_csv(os.path.join(data_dir, "weather.csv"))
if pathlib.Path(os.path.join(data_dir, "metadata.csv")).exists():
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
else:
    print("Reading meta file from Github...")
    meta = pd.read_csv(meta_url)
    meta.to_csv(os.path.join(data_dir, "metadata.csv"))
if pathlib.Path(os.path.join(data_dir, "electricity_cleaned.csv")).exists():
    loads = pd.read_csv(os.path.join(data_dir, "electricity_cleaned.csv"))
else:
    print("Reading load file from Github...")
    loads = pd.read_csv(weather_url)
    loads.to_csv(os.path.join(data_dir, "electricity_cleaned.csv"))

# Alter timestamps
weather["timestamp"] = pd.to_datetime(weather["timestamp"])
weather = weather.set_index("timestamp", drop=True)
loads["timestamp"] = pd.to_datetime(loads["timestamp"])
loads = loads.set_index("timestamp", drop=True)

# Find buildings with electricity data
# sites_with_loads = meta[meta["electricity"].notna()]["site_id"].values.tolist()
c1 = meta["electricity"].notna().values
c2 = meta["lat"].notna().values
c3 = meta["lng"].notna().values
c4 = meta["timezone"].str.contains("US").values
criteria = np.logical_and.reduce((c1, c2, c3, c4))
filtered = dict()
filtered["ids"] = meta[criteria]["building_id"].values.tolist()
filtered["usage"] = np.unique(filtered["usage"], return_counts=True)

# Save the fitlered building IDs for later use
# with open(os.path.join(data_dir, "GP_ids.json"), 'w') as fp:
#     json.dump(filtered["ids"], fp, indent=1)

# Create datasets for each building
print("Iterating")
i = 0

# Make temporary holding stucture for site weather data
sites_weather = dict()
sites_weather["2016"] = dict()
sites_weather["2017"] = dict()

for id in filtered["ids"]:
    # Get load data
    id_load = loads[id]

    # Get weather data from the genome project data
    # site = id.split("_")[0]
    # id_weather = weather[weather["site_id"] == site].drop("site_id", axis=1)

    # Put load and weather data together
    # id_data = pd.concat([id_weather, id_load], axis=1)


    site_id = meta[meta["building_id"] == id]["site_id"].values[0]

    # Iterate through years
    for year in [2016, 2017]:
        output_string = os.path.join(data_dir, "Data_{}_{}.h5".format(id, year))
        if pathlib.Path(output_string).exists():
            pass
        else:
            # Check if weather data is already gathered for this site
            if site_id in sites_weather[str(year)]:
                print("already have nsrdb data")
                df = sites_weather[str(year)][site_id]
            else:
                print("Getting nsrdb data for {}".format(year))
                # Get nsedb weather data for this site
                site_lat = meta[meta["building_id"] == id]["lat"].values[0]
                site_lng = meta[meta["building_id"] == id]["lng"].values[0]
                df = get_nsrdb(year, site_lat, site_lng)
                df = df.resample("60T").mean()
                sites_weather[str(year)][site_id] = df

            # Combine site data together
            site_data = pd.concat([id_load[id_load.index.year == year], df[['GHI', 'Temperature', 'Relative Humidity']]], axis=1)

            # Save dataset
            site_data.to_hdf(output_string, key='df', mode='w')

    print("Done with {}: {}/{} sites".format(id, i, len(filtered["ids"])))
    i = i + 1

print("Done")