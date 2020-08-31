import json
import sys
import importlib
import buildings_processing as bp

meters = [
    ["Cafe", "Cafe Main Power (kW)"],
    ["ESIF", "ESIF Office Power (kW)"],
    ["RSF2", "RSF2 Mechanical Power (kW)"],
    ["RSF2", "RSF2 Lighting Power (kW)"],
    ["ESEB", "ESEB Main Power (kW)"],
    ["TTF", "TTF Main Power (kW)"],
    ["SERF", "SERF Main Power (kW)"],
    ["FTLB", "FTLB Main Power (kW)"],
    ["SSEB", "SSEB Mechanical Power (kW)"],
    ["RSF", "RSF Main Power (kW)"],
    ["RSF", "RSF Data Center Power (kW)"],
    ["S&TF", "S&TF Main Power (kW)"],
    ["EC", "EC Main Power (kW)"]
]
# To get full training data for all of the meters: "Train"
# To get the 6 months worth of 2020 testing data for all of the meters: "Test"
get_data = "Train"

# For getting test data
months = [1, 2, 3, 4, 5, 6]
year = "2020"

for meter_ID in meters:
    # Read in base configurations from json file
    with open("configs.json", "r") as read_file:
        configs = json.load(read_file)

    configs["building"] = meter_ID[0]
    configs["target_var"] = meter_ID[1]

    # Get the data
    if get_data == "Train":
        data_full = bp.get_full_data(configs)
        print("Done getting full training data for {}".format(configs["building"]))
    elif get_data == "Test":
        data_full = bp.get_test_data(configs["building"], year, months, configs["data_dir"])
        print("Done getting test data for {}, month {}".format(configs["building"], months))
    else:
        print("get_data command not understood")
        exit()






