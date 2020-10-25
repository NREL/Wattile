import json
import entry_point as epb
import os

# Get list of meters
with open(os.path.join("data", "GP", "GP_ids.json"), "r") as read_file:
    meters = json.load(read_file)

test_ID = "initial_tests"
stop_num = 3

state = "Test" # Train, Test, get_results
test_dir_path = os.path.join("EnergyForecasting_Results", "GP_training_session_initial_tests")

# Run tests
if state == "Train":
    i = 0
    for meter_ID in meters:

            # Read in base configurations from json file
            with open("configs.json", "r") as read_file:
                configs = json.load(read_file)

            # Make a sub-directory in the main results directory specific to this test study
            configs["results_dir"] = os.path.join(configs["results_dir"], "GP_training_session_{}".format(test_ID))

            # Test the model
            configs["building"] = meter_ID
            configs["target_var"] = meter_ID
            epb.main(configs)

            i = i+1
            print("Just finished training for target variable {} ({}/{})".format(meter_ID[1], i, len(meters)))
            if i == stop_num:
                break


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

        i = i+1
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


        i = i+1
        print("Just finished training for target variable {} ({}/{})".format(meter, i, len(meter_models)))
        if i == stop_num:
            break

else:
    print("Command not understood")
