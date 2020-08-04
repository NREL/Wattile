import json
import sys
import importlib

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

for meter_ID in meters:
    # Read in base configurations from json file
    with open("configs.json", "r") as read_file:
        configs = json.load(read_file)

    configs["building"] = meter_ID[0]
    configs["target_var"] = meter_ID[1]

    # Import buildings module to preprocess data
    sys.path.append(configs["shared_dir"])
    bp = importlib.import_module("buildings_processing")
    data_full = bp.get_full_data(configs)
    print("Done getting data for {}".format(configs["building"]))





