import json
import entry_point as epb
import os
import pandas as pd
from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing import set_start_method
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

if __name__ == '__main__':
    set_start_method("spawn")
    print("starting file")
    # Read in base configurations from json file
    with open("configs.json", "r") as read_file:
        base_configs = json.load(read_file)

    # Get list of meters
    with open(os.path.join(base_configs["data_dir"], "GP_ids.json"), "r") as read_file:
        meters = json.load(read_file)

    test_ID = "final"
    low_bound = 700
    stop_num = 2000
    hpc_processes = 1
    states = ["plot_results"]  # Train, Test, plot_results
    resume = True

    testing_round_dir = os.path.join(base_configs["results_dir"], "BGP_{}".format(test_ID))

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
        # rc('text', usetex=True)
        rc('xtick', labelsize=6)
        rc('ytick', labelsize=6)
        plt.rc('font', family='serif')

        # Find the models that were trained and are currently in the folder with the name "test_dir_path"
        meter_models = os.listdir(testing_round_dir)

        # Get building metadata
        meta = pd.read_csv("data\\GP\\metadata.csv")

        # Iterate through those results directories, get the results data
        i = 0
        # fig1, ax1 = plt.subplots()
        fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.2, wspace=0.1)
        usages = dict()
        data = dict()

        for meter in meter_models:
            # Get QQ data
            results_dir = os.path.join(testing_round_dir, meter)
            QQ_data = pd.read_hdf(os.path.join(results_dir, "QQ_data_test.h5"), key='df')
            if max(abs(QQ_data["q_actual"] - QQ_data["q_requested"])) > 0.7:
                print("Dropping {}".format(meter))
                continue

            # Store data
            usage = meta[meta["building_id"] == meter[5:-5]]["primaryspaceusage"].values[0]
            if usage not in data:
                data[usage] = pd.DataFrame()
            data[usage] = pd.concat([data[usage], QQ_data["q_actual"]], axis=1)

            # plot_index = list(meta.primaryspaceusage.unique()).index(usage)
            # usages[usage] = plot_index

        print("done")
        requested = np.array([0.025, 0.05, 0.1, 0.25, 0.5, 0.7, 0.9, 0.95, 0.975])
        plt_ind = 0

        flierprops = dict(marker=".", markersize=3, markerfacecolor="k")
        num_meters = 0
        for usage in data:
            usage_data = np.array(data[usage]).T
            ax[plt_ind % 4, int(plt_ind / 4)].boxplot(usage_data, whis=(5,95), flierprops=flierprops, showfliers=True, positions=requested, widths=0.08, manage_xticks=False)
            ax[plt_ind % 4, int(plt_ind / 4)].text(.5, 1, usage, ha='center', va="bottom",
                                                         transform=ax[plt_ind % 4, int(plt_ind / 4)].transAxes, fontsize=8)
            ax[plt_ind % 4, int(plt_ind / 4)].text(0.02, 0.975, "n={}".format(usage_data.shape[0]), ha='left', va="top",
                                                         transform=ax[plt_ind % 4, int(plt_ind / 4)].transAxes, fontsize=8)
            ax[plt_ind % 4, int(plt_ind / 4)].plot([0, 1], [0, 1], c='k', alpha=0.5, linewidth=1.0)
            ax[plt_ind % 4, int(plt_ind / 4)].set_xlim(left=0, right=1)
            ax[plt_ind % 4, int(plt_ind / 4)].set_ylim(bottom=0, top=1)
            num_meters = num_meters + usage_data.shape[0]
            plt_ind = plt_ind + 1

        print(num_meters)
        fig.text(0.5, 0.04, r'$\tau_{Requested}$', ha='center', va='center', fontsize=10)
        fig.text(0.06, 0.5, r'$\tau_{Evaluated}$', ha='center', va='center', rotation='vertical', fontsize=10)
        plt.show()

        #     ax[plot_index % 4, int(plot_index/ 4)].scatter(QQ_data["q_requested"], QQ_data["q_actual"], s=5, c="black")
        #
        #     i = i + 1
        #     # print("Just finished plotting for target variable {} ({}/{})".format(meter, i, len(meter_models)))
        #     if i == stop_num:
        #         break
        #
        # for usage in usages:
        #     index = usages[usage]
        #     ax[index % 4, int(index / 4)].text(.5, 1, usage, ha='center', va="bottom",
        #                                                  transform=ax[index % 4, int(index / 4)].transAxes, fontsize=8)
        #     ax[index % 4, int(index / 4)].plot([0, 1], [0, 1], c='k', alpha=0.5, linewidth=1.0)
        #     ax[index % 4, int(index / 4)].set_xlim(left=0, right=1)
        #     ax[index % 4, int(index / 4)].set_ylim(bottom=0, top=1)
        #


    else:
        print("Command not understood")
