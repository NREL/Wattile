"""
Run-alone script for running testing on hyperparmeter values for a ML model.

Input format example:
runs["Test_hn"] = {'iterable': "hidden_nodes", 'iterables': [16, 18], 'iterable_type': str}

"Test_hn" - Example of a name for a particular test (str). Required, but value decided by user.
iterable (str) - The name of the parameter (as it is named in configs.json) that will be studied
iterables (list) - The specific values of "iterable" that should be tested
iterable_type (type) - The type of variable "iterable" as it is defined in configs.json
"""

import json
import os

import intelcamp.entry_point as epb

runs = {}

# User inputs: What configurations will be run for this test? Each line is a separate study run in
# series
# runs["Test_type"] = {
#     "iterable": "arch_type_variant",
#     "iterables": ["vanilla", "lstm"],
#     "iterable_type": str,
# }
# runs["Test_wd"] = {
#     "iterable": "weight_decay",
#     "iterables": [0.1, 0.01, 0.001, 0.0001],
#     "iterable_type": float,
# }
# runs["Test_lr"] = {
#     "iterable": "learning_rate_base",
#     "iterables": [1e-1, 1e-2, 1e-3, 1e-4],
#     "iterable_type": float,
# }
# runs["Test_ec_gap"] = {
#     "iterable": "EC_future_gap",
#     "iterables": [5, 10, 15, 20],
#     "iterable_type": int,
# }
# runs["Test_HOD_indicator_RSF"] = {
#     "iterable": "HOD_indicator",
#     "iterables": ["sincos", "regDummy"],
#     "iterable_type": str,
# }
# runs["Test_layer_dim"] = {
#     "iterable": "layer_dim",
#     "iterables": [1, 2],
#     "iterable_type": int,
# }
# runs["Test_hidden_dim"] = {
#     "iterable": "hidden_nodes",
#     "iterables": [5, 10, 15, 20, 25, 30, 35],
#     "iterable_type": int,
# }
# runs["Test_alpha"] = {
#     "iterable": "smoothing_alpha",
#     "iterables": [0.1, 0.01, 0.001],
#     "iterable_type": float,
# }

# Run tests
for test in runs:
    # Read in base configurations from json file
    with open("configs.json", "r") as read_file:
        configs = json.load(read_file)

    # Make a sub-directory in the main results directory specific to this test study
    configs["results_dir"] = os.path.join(
        configs["results_dir"], "STUDY_{}".format(runs[test]["iterable"])
    )

    # Test the model
    for i in runs[test]["iterables"]:
        configs[runs[test]["iterable"]] = runs[test]["iterable_type"](i)
        configs["exp_id"] = "{}_{}".format(runs[test]["iterable"], i)
        epb.main(configs)
