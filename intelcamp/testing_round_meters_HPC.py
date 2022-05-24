import json
import os
import sys
from multiprocessing import Process

import intelcamp.entry_point as epb

# User inputs
meters = [
    ["Cafe", "Cafe Main Power (kW)"],
    ["ESIF", "ESIF Office Power (kW)"],
    ["RSF2", "RSF2 Mechanical Power (kW)"],
    ["RSF2", "RSF2 Lighting Power (kW)"],
    ["SSEB", "SSEB Mechanical Power (kW)"],
]

#     ["ESEB", "ESEB Main Power (kW)"],
#     ["TTF", "TTF Main Power (kW)"],
#     ["SERF", "SERF Main Power (kW)"],
#     ["FTLB", "FTLB Main Power (kW)"],
#     ["RSF", "RSF Main Power (kW)"],
#     ["RSF", "RSF Data Center Power (kW)"],
#     ["S&TF", "S&TF Main Power (kW)"],
#     ["EC", "EC Main Power (kW)"]
# ]
test_ID = "MP_test"
processes = list()

# Run tests
for meter_ID in meters:
    # Read in base configurations from json file
    with open("configs.json", "r") as read_file:
        configs = json.load(read_file)

    # Make a sub-directory in the main results directory specific to this test study
    configs["results_dir"] = os.path.join(
        configs["results_dir"], "all_meters_T{}".format(test_ID)
    )

    # Modify inputs
    configs["building"] = meter_ID[0]
    configs["target_var"] = meter_ID[1]

    # # Rewrite configs to disk:
    # with open("configs.json", 'w') as fp:
    #     json.dump(configs, fp, indent=1)

    # Execute
    # target = configs["target_var"].replace(" ", "").replace("(", "-").replace(")", "")
    # python_file = "/projects/intelcamp20/repos/intelligentcampus-pred-analytics/entry_point.py"
    # stdout_file = "/projects/intelcamp20/repos/intelcamp20-hpc/results/Outputs/stdout_{}.txt".format(configs["target_var"].replace(" ", ""))
    # os.system("python -u " + python_file + " > " + stdout_file + " &")

    if __name__ == "__main__":
        p = Process(target=epb.main, args=(configs,))
        processes.append(p)
        p.start()

    print("Started {}".format(configs["target_var"]))

print("Joining Processes...")
for p in processes:
    p.join()
print("Done")
