"""
Run-alone script for running testing on hyperparmeter values for a ML model.
"""

import json
import entry_point_building as epb
import os


# User inputs: What configurations will be run for this test? Each line is a separate study run in series
runs = {}
# runs["Test_wd"] = {'iterable': "weight_decay", 'iterables': [0.1, 0.01, 0.001, 0.0001], 'iterable_type': str}
# runs["Test_hn"] = {'iterable': "hidden_nodes", 'iterables': [4, 7, 10, 13, 16], 'iterable_type': str}
# runs["Test_lr"] = {'iterable': "learning_rate_base", 'iterables': [1e-1, 1e-2, 1e-3, 1e-4], 'iterable_type': float}
runs["Test_ec_gap"] = {'iterable': "EC_future_gap", 'iterables': [5, 10, 15, 20], 'iterable_type': int}

# Run tests
for test in runs:
    # Read in base configurations from json file
    with open("configs.json", "r") as read_file:
        configs = json.load(read_file)

    # Make a sub-directory in the main results directory specific to this test study
    configs["results_dir"] = os.path.join(configs["results_dir"], "{}_study_{}".format(configs["arch_type"], runs[test]['iterable']))

    # Test the model
    for i in runs[test]['iterables']:
        configs[runs[test]['iterable']] = runs[test]['iterable_type'](i)
        configs["test_exp_num"] = "{}_{}".format(runs[test]['iterable'], i)
        epb.main(configs)