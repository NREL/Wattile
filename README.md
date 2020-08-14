# README #

This repository contains the source code for forecasting energy consumption using deep learning.

### What is this repository for? ###

* Quick summary: this work uses weather variables and past time-series data for energy consumption to predict future energy consumption (and PV production). The two deep learning neural network architectures which have been implemented are:
  1. feed-forward neural network
  2. Recurrent-neural network
  
* Version: this is the initial version (0.1) of this project


### How do I...
#### ... get set up?

* Conda environment for running the code:
 A conda environment file is provided for convenience. Assuming you have Anaconda python distribution available on your computer, you can create a new conda environment with the necessary packages using the following command:

For windows users:
`conda env create -f rnn-env-win.yml -n "ic-lf-deploy"`

For linux users:
`conda env create -f rnn-env-lnx.yml -n "ic-lf-deploy"`


* Change the configuration variables in configs.json.
* Run entry_point.py

#### ... train a single model?

##### Starting from scratch:

...
##### With my own dataframe:

* Follow these instructions if you wish to bypass the preprocessing steps, and just want to train a model on your own dataframe.
* This method assumes your data is already resampled, cleaned, and already has all of the features that you want to train on. 
* The dataframe that is passed will be split between training and testing data. 
* Function calls:  
```
train_df_master, test_df_master = bp.prep_for_rnn(configs, data)  
rnn_mod.main(train_df, test_df, configs)
```
* Requirements:
    * `data` must have a datetime index that is continuous, i.e. if there are missing data values they should be kept as `nan` instead of removing the timestamp completely.

#### ... test an already-trained model?

##### Starting from scratch:

...
##### With my own dataframe:

* This method assumes that the data is in the same format as the data that you already trained the model on.
* The entire dataframe that is passed will be used for tested.
* Function calls:
```
train_df_master, test_df_master = bp.prep_for_rnn(configs, data)  
rnn_mod.main(train_df, test_df, configs)
```
* Requirements:
    * `data` must have a datetime index that is continuous, i.e. if there are missing data values they should be kept as `nan` instead of removing the timestamp completely.

#### ... get a prediction from an already trained model?

* This methods is for predicting a future output given historical data leading up to the present.
* Function calls:  
`final_preds = predict(data, file_prefix)`



#### ... run a hyperparameter study? 

### What are the parameters in configs.json?

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
* Update this file to contain the dates of holidays that will affect building occupancy. This will be used for both training and testing.


