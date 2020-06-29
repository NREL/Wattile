import buildings_processing as bp
import json
import algo_main_rnn_v2
import entry_point_building as epb
import os


# User inputs: What configurations will be run for this test?
runs = {}
runs["Test_wd"] = {'iterable': "weight_decay", 'iterables': [0.1, 0.01, 0.001, 0.001], 'iterable_type': str}
runs["Test_hn"] = {'iterable': "hidden_nodes", 'iterables': [4, 7, 10, 13, 16], 'iterable_type': str}
runs["Test_lr"] = {'iterable': "learning_rate_base", 'iterables': [1e-1, 1e-2, 1e-3, 1e-4], 'iterable_type': float}

# Run tests
for test in runs:
    # Read in base configurations from json file
    with open("configs.json", "r") as read_file:
        configs = json.load(read_file)

    # Make a sub-directory in the main results directory specific to this test study
    configs["results_dir"] = os.path.join(configs["results_dir"], "study_{}".format(runs[test]['iterable']))

    # Test the model
    for i in runs[test]['iterables']:
        configs[runs[test]['iterable']] = runs[test]['iterable_type'](i)
        configs["test_exp_num"] = "{}_{}".format(runs[test]['iterable'], i)
        epb.main(configs)