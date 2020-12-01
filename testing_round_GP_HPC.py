import json
import entry_point as epb
import os
import pandas as pd
from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing import set_start_method
import matplotlib.pyplot as plt


if __name__ == '__main__':
    set_start_method("spawn")
    print("starting file")
    # Read in base configurations from json file
    with open("configs.json", "r") as read_file:
        base_configs = json.load(read_file)

    # Get list of meters
    with open(os.path.join(base_configs["data_dir"], "GP_ids.json"), "r") as read_file:
        meters = json.load(read_file)

    test_ID = "batch_1"
    low_bound = 700
    stop_num = 1000
    hpc_processes = 1
    states = ["Train"]  # Train, Test, get_results
    resume = True

    testing_round_dir = os.path.join(base_configs["results_dir"], "GP_training_{}".format(test_ID))

    # Run tests
    if "Train" in states:
        i = 0
        if hpc_processes > 1:
            pool = Pool(processes=hpc_processes, maxtasksperchild=1)
            inputs = list()

        for meter_ID in meters:
            if i < low_bound:
                i = i + 1
                continue

            # Read in base configuration
            configs = base_configs.copy()

            # Make a sub-directory in the main results directory specific to this test study
            configs["results_dir"] = testing_round_dir

            # Alter some configs
            configs["building"] = meter_ID
            configs["target_var"] = meter_ID
            configs["run_resume"] = resume


            if hpc_processes > 1:
                inputs.append(configs)
            else:
                try:
                    epb.main(configs)
                except:
                    print("Error in training for {}. Terminating training process".format(configs["target_var"]))

            i = i + 1
            print("Just started process for {} ({}/{})".format(meter_ID, i, len(meters)))
            if i == stop_num:
                break

        if hpc_processes > 1:
            pool.map(epb.main, inputs, chunksize=1)
        print("Done")

    elif "Test" in states:
        # Find the models that were trained and are currently in the folder with the name "test_dir_path"
        meter_models = os.listdir(testing_round_dir)

        # Iterate through those results directories, run a test set on each
        i = 0
        for meter in meter_models:
            results_dir = os.path.join(testing_round_dir, meter)

            # Read in the configs from that training session
            with open(os.path.join(results_dir, "configs.json"), "r") as read_file:
                configs = json.load(read_file)
            configs["run_train"] = False
            configs["test_method"] = "internal"
            configs["results_dir"] = testing_round_dir

            # Test the model
            epb.main(configs)

            i = i + 1
            print("Just finished testing for {} ({}/{})".format(meter, i, len(meter_models)))
            if i == stop_num:
                break

    elif "plot_results" in states:
        # Find the models that were trained and are currently in the folder with the name "test_dir_path"
        meter_models = os.listdir(testing_round_dir)

        # Iterate through those results directories, get the results data
        i = 0
        fig1, ax1 = plt.subplots()
        for meter in meter_models:
            results_dir = os.path.join(testing_round_dir, meter)

            # Q data
            QQ_data = pd.read_hdf(os.path.join(results_dir, "QQ_data_test.h5"), key='df')
            ax1.scatter(QQ_data["q_requested"], QQ_data["q_actual"], s=20)

            i = i + 1
            print("Just finished plotting for target variable {} ({}/{})".format(meter, i, len(meter_models)))
            if i == stop_num:
                break

        ax1.plot([0, 1], [0, 1], c='k', alpha=0.5)
        ax1.set_xlabel('Requested')
        ax1.set_ylabel('Actual')
        ax1.set_xlim(left=0, right=1)
        ax1.set_ylim(bottom=0, top=1)
        plt.show()

    else:
        print("Command not understood")
