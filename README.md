# README #

This repository contains the source code for forecasting energy consumption using deep learning.

### What is this repository for? ###

* Quick summary: this work uses weather variables and past time-series data for energy consumption to predict future energy consumption (and PV production). The two deep learning neural network architectures which have been implemented are:
  1. feed-forward neural network
  2. Recurrent-neural network
  
* Version: this is the initial version (0.1) of this project


### How do I get set up to run the code? ###

* Conda environment for running the code:
 A conda environment file is provided for convenience. Assuming you have Anaconda python distribution available on your computer, you can create a new conda environment with the necessary packages using the following command:

`conda env create -f rnn-env.yml -n "ic-lf-deploy"`

* Change the configuration variables in configs.json.
* Change the `shared_dir` variable at the top of entry_point.py to map to the location of external files (described below)
* Run entry_point.py

---
### Local Files

#### entry_point.py

* Currently being iterated upon. (Current) data source is LAN directory. 
* Model configurations defined in external configs.json file.
* For whole campus modeling, specify `"building": "Campus Energy"` and change `target_var` in `configs.json`. As of 7/1/2020 it is recommended to only use 2019 data due to inconsistent data formatting in previous years. 
 
#### algo_main_rnn_v1.py

* Works with preprocessed STM whole-campus data only. Uses previous timestep for prediction feedback. No major architecture changes besides bug fixes.

#### algo_main_rnn_v2.py

* Currently being iterated upon. 

#### testing_round.py

* Lets a user run a series of model trainings by varying a single configuration parameter at once. 
* Any parameter in configs.json can be studied.
* All parameter values not being studied in a particular test will default to value already in configs.json
* Multiple studies can be run in series.
* Results are saved to a sub-directory of the main results directory, specific to the case-study (makes for faster results filtering within TensorBoard).

* `iterable`: What config parameter do you want to study? (str) 
* `iterables`: What values of that parameter do you want to test? (list)
* `iterable_type`: What class is the iterable in the configs.json file? (class)

## Externally referenced files

The code in this repo references some files that are held in external repos. The descriptions below describe how to reference these files. 

#### buildings_processing.py

* Contains functions for data manipulation and cleaning. (Same file is used for both Quantile Regression and ML methods)
* File path specified at top of entry_point_building.py

#### holidays.json

* JSON file containing holidays specific to the region being tested. 
* File path specified in configs.json