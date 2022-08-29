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
import json
import os
import pathlib

import numpy as np
import pandas as pd


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
data_dir = "/projects/wattile/data/GP_new"
building_data_genome_project_url = (
    "https://github.com/buds-lab/building-data-genome-project-2/blob/master/data"
)
weather_url = f"{building_data_genome_project_url}/weather/weather.csv"
meta_url = f"{building_data_genome_project_url}/metadata/metadata.csv"
loads_url = f"{building_data_genome_project_url}/meters/cleaned/electricity_cleaned.csv"

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
            site_data = pd.concat(
                [
                    id_load[id_load.index.year == year],
                    df[["GHI", "Temperature", "Relative Humidity"]],
                ],
                axis=1,
            )

            # Save dataset
            site_data.to_hdf(output_string, key="df", mode="w")

    print("Done with {}: {}/{} sites".format(id, i, len(filtered["ids"])))
    i = i + 1

print("Done")