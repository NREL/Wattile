import json
import entry_point as epb
import os
from multiprocessing import Process
from multiprocessing import Pool

if __name__ == '__main__':
    print("starting file")
    # Read in base configurations from json file
    with open("configs.json", "r") as read_file:
        base_configs = json.load(read_file)

    # Get list of meters
    with open(os.path.join(base_configs["data_dir"], "GP_ids.json"), "r") as read_file:
        meters = json.load(read_file)

    test_ID = "initial_par_test"
    stop_num = 10

    state = "Train"  # Train, Test, get_results
    test_dir_path = os.path.join("EnergyForecasting_Results", "GP_training_initial_tests")

    pool = Pool(processes=3)
    inputs = list()
    # processes = list()
    # Run tests
    if state == "Train":
        i = 0
        for meter_ID in meters:

            # Read in base configuration
            configs = base_configs.copy()

            # Make a sub-directory in the main results directory specific to this test study
            configs["results_dir"] = os.path.join(configs["results_dir"], "GP_training_{}".format(test_ID))

            # Test the model
            configs["building"] = meter_ID
            configs["target_var"] = meter_ID

            inputs.append(configs)

            i = i + 1
            print("Just started process for {} ({}/{})".format(meter_ID, i, len(meters)))
            if i == stop_num:
                break

        pool.map(epb.main, inputs)
        print("Done")


    elif state == "Test":
        # Find the models that were trained and are currently in the folder with the name "test_dir_path"
        meter_models = os.listdir(test_dir_path)

        # Iterate through those results directories, run a test set on each
        i = 0
        for meter in meter_models:
            results_dir = os.path.join(test_dir_path, meter)

            # Read in the configs from that training session
            with open(os.path.join(results_dir, "configs.json"), "r") as read_file:
                configs = json.load(read_file)
            configs["run_train"] = False
            configs["test_method"] = "internal"

            # Test the model
            epb.main(configs)

            i = i + 1
            print("Just finished training for target variable {} ({}/{})".format(meter, i, len(meter_models)))
            if i == stop_num:
                break

    elif state == "get_results":
        # Find the models that were trained and are currently in the folder with the name "test_dir_path"
        meter_models = os.listdir(test_dir_path)

        # Iterate through those results directories, run a test set on each
        i = 0
        for meter in meter_models:
            results_dir = os.path.join(test_dir_path, meter)

            # Get the results specs

            # Get Q data

            i = i + 1
            print("Just finished training for target variable {} ({}/{})".format(meter, i, len(meter_models)))
            if i == stop_num:
                break

    else:
        print("Command not understood")
