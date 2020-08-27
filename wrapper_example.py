import pandas as pd
import numpy as np
import json
import sys
import importlib
import re

# Import configs file
with open("configs.json", "r") as read_file:
    configs = json.load(read_file)

# Get some libraries
sys.path.append(configs["shared_dir"])
bp = importlib.import_module("buildings_processing")
rnn_mod = importlib.import_module("algo_main_rnn_v{}".format(configs["arch_version"]))

# Get master dataFrame
# This section will be replaced by the output of Kristens code. This is just here to get a df to work with.
# "data
data_full = bp.get_full_data(configs)
important_vars = configs['weather_include'] + [configs['target_var']]
data = data_full[important_vars]
resample_bin_size = "{}T".format(configs['resample_freq'])
data = data.resample(resample_bin_size).mean()
data = bp.clean_data(data, configs)
data = bp.time_dummies(data, configs)

# Do sequential padding and test/validation split, and test/train split
train_df_master, test_df_master = bp.prep_for_rnn(configs, data)

# Get list of features that is availiable in the master dataframe and separate the target output
train_features = train_df_master.columns.tolist()
test_features = test_df_master.columns.tolist()
train_target = train_df_master[configs["target_var"]]
test_target = test_df_master[configs["target_var"]]


# MAIN WRAPPER ITERATIVE LOOP STARTS HERE
# for....

# Input features I want to test for this iteration (sample)
features = ["SRRL BMS Dry Bulb Temperature (\u00b0F)",
            'sin_HOD',
            'cos_HOD',
            'Holiday']

# Make a subset of the master dataframe to test for this iteration. This grabs anything matching "features" with all of their lagged versions.
filtered_features = [str for str in train_features if any(sub in str for sub in features)]
train_df = train_df_master[filtered_features]
train_df[configs["target_var"]] = train_target
filtered_features = [str for str in test_features if any(sub in str for sub in features)]
test_df = test_df_master[filtered_features]
test_df[configs["target_var"]] = train_target

# Train the model on this subset
rnn_mod.main(train_df, test_df, configs)

# Do other wrapper stuff...


# MAIN WRAPPER ITERATIVE LOOP ENDS HERE