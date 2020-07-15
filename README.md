# README #

This repository contains the source code for forecasting energy consumption using deep learning.

### What is this repository for? ###

* Quick summary: this work uses weather variables and past time-series data for energy consumption to predict future energy consumption (and PV production). The two deep learning neural network architectures which have been implemented are:
  1. feed-forward neural network
  2. Recurrent-neural network
  
* Version: this is the initial version (0.1) of this project


### How do I get set up?

* Conda environment for running the code:
 A conda environment file is provided for convenience. Assuming you have Anaconda python distribution available on your computer, you can create a new conda environment with the necessary packages using the following command:

For windows users:
`conda env create -f rnn-env-win.yml -n "ic-lf-deploy"`

For linux users:
`conda env create -f rnn-env-lnx.yml -n "ic-lf-deploy"`


* Change the configuration variables in configs.json.
* Change the `shared_dir` variable at the top of entry_point.py to map to the location of external files (described below)
* Run entry_point.py

### How do I train a single model?

* Description coming soon...

### How do I run a hyperparamter study? 

* Description coming soon...

### What are the parameters in configs.json?
Descriptions coming soon...

| Variable | Type | Description | v1 |
| --- | --- | --- | --- |
| foo | str |  | x |
| foo | str |   | |

```
{
    "train_start_date": "2018-01-01",
    "train_end_date": "2018-10-31",
    "test_start_date": "2018-11-01",
    "test_end_date": "2018-12-31",
    "transformation_method": "minmaxscale",
    "run_train": true,
    "num_epochs": 200,
    "tr_batch_size": 3900,
    "te_batch_size": 6949,
    "run_resume": true,
    "preprocess": false,
    "arch_type": "RNN",
    "arch_type_variant": "lstm",
    "arch_version": 6,
    "train_exp_num": "Cafe",
    "test_exp_num": "dev",
    "hidden_nodes": 100,
    "layer_dim": 1,
    "output_dim": 5,
    "weight_decay": 0.001,
    "fetch_n_parse": false,
    "LAN_path": "Z:\\Data",
    "building": "Cafe",
    "year": [
        "2018",
        "2019"
    ],
    "target_var": "Cafe Main Power (kW)",
    "weather_include": [
        "SRRL BMS Global Horizontal Irradiance (W/m\u00b2_irr)",
        "SRRL BMS Wind Speed at 19' (mph)",
        "SRRL BMS Dry Bulb Temperature (\u00b0F)",
        "SRRL BMS Opaque Cloud Cover (%)",
        "SRRL BMS Relative Humidity (%RH)",
        "SRRL BMS Barometric Pressure (mbar)"
    ],
    "TrainTestSplit": "Random",
    "test_start": "12/28/2019 23:45:00",
    "test_end": "12/31/2019 23:45:00",
    "resample": true,
    "resample_bin_min": 15,
    "HOD": true,
    "DOW": true,
    "MOY": true,
    "Holidays": true,
    "HOD_indicator": "sincos",
    "window": 20,
    "EC_future_gap": 16,
    "lr_schedule": true,
    "lr_config": {
        "base": 0.001,
        "factor": 0.1,
        "min": 1e-03,
        "patience": 20
    },
    "qs": [0.025, 0.2, 0.5, 0.8, 0.975],
    "smoothing_alpha": 0.001,
    "S2S_stagger": {
        "initial_num": 4,
        "decay": 4,
        "secondary_num": 17
    },
    "results_dir": "EnergyForecasting_Results",
    "holidays_file": "C:\\dev\\intelligentcampus-2020summer\\loads\\Stats_Models_Loads\\holidays.json",
    "shared_dir": "C:\\dev\\intelligentcampus-2020summer\\loads\\Stats_Models_Loads"
}
```

---
### Local Files

#### entry_point.py

* Currently being iterated upon. (Current) data source is LAN directory. 
* Model configurations defined in external configs.json file.
* For whole campus modeling, specify `"building": "Campus Energy"` and change `target_var` in `configs.json`. As of 7/1/2020 it is recommended to only use 2019 data due to inconsistent data formatting in previous years. 
 
#### algo_main_rnn_v1.py

* Works with preprocessed STM whole-campus data only. Uses previous timestep for prediction feedback. 
* No major architecture changes besides bug fixes.
* Vanilla and LSTM variants available

#### algo_main_rnn_v2.py

* Predicts the conditional mean of the data.
* Single time point prediction
* Works with individual buildings or whole-campus data, as do all future versions. 
* Vanilla variant available

#### algo_main_rnn_v3.py

* Probabilistic forecasting
* In one training session, predict:
    * Number of future times: 1
    * Number of quantiles per time: 1
* Vanilla variant available

#### algo_main_rnn_v4.py

* Probabilistic forecasting 
* In one training session, predict:
    * Number of future times: 1
    * Number of quantiles per time: *Q*
* Vanilla and LSTM variants available

#### algo_main_rnn_v5.py

* Probabilistic forecasting 
* In one training session, predict:
    * Number of future times: *T*
    * Number of quantiles per time: *Q*
* Vanilla and LSTM variants available
* Supports future time predictions with constant spacing or variable spacing 


#### testing_round.py

* Lets a user run a series of model trainings by varying a single configuration parameter at once. 
* Any parameter in configs.json can be studied.
* All parameter values not being studied in a particular test will default to value already in configs.json
* Multiple studies can be run in series.
* Results are saved to a sub-directory of the main results directory, specific to the case-study (makes for faster results filtering within TensorBoard).

* `iterable`: What config parameter do you want to study? (str) 
* `iterables`: What values of that parameter do you want to test? (list)
* `iterable_type`: What class is the iterable in the configs.json file? (class)

---
### Externally referenced files

The code in this repo references some files that are held in external repos. The descriptions below describe how to reference these files. 

#### buildings_processing.py

* Contains functions for data manipulation and cleaning. (Same file is used for both Quantile Regression and ML methods)
* Directory path specified in configs.json

#### holidays.json

* JSON file containing holidays specific to the region being tested. 
* File path specified in configs.json


