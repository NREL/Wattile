# README #

This repository contains the source code for forecasting energy consumption using deep learning.

### What is this repository for? ###

* Quick summary: this work uses weather variables and past time-series data for energy consumption to predict future energy consumption (and PV production). The two deep learning neural network architectures which have been implemented are:
  1. feed-forward neural network
  2. Recurrent-neural network
  
* Version: this is the initial version (0.1) of this project

---
### Getting set up for development

You can either use Conda or Poetry as your local package manager.

#### Poetry set up

1. Ensure that your local python version is 3.6.6
    ```
    $ python -V
    Python 3.6.6
    ```
    * Note: Recommended python verison managers are [pyenv](https://github.com/pyenv/pyenv) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html), but are not required.
2. Install [Poetry](https://python-poetry.org/docs/#installation).

    For osx / linux / bashonwindows:
    ```
    $ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -      
    ```

    For Windows:
    ```
    $ (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
    ```
3. Install dependencies.
    ```
    $ poetry install
    ```

4. Run singluar commands in poetry's virtual environment
    ```
    $ poetry run python entry_point.py
    ```
    or create a new shell within the virtual environment
    ```
    $ poetry shell
    Spawning shell within /poetry_path/pypoetry/virtualenvs/intelligentcampus-pred-analytics-l9jvZEb_-py3.6
    . /poetry_path/pypoetry/virtualenvs/intelligentcampus-pred-analytics-l9jvZEb_-py3.6/bin/activate

    $ python entry_point.py
    ...

    $ exit
    Saving session...completed.
    Deleting expired sessions...
    ```


#### Conda set up
1. Configuring conda environment 

    A conda environment file is provided for convenience. Assuming you have Anaconda python distribution available on your computer, you can create a new conda environment with the necessary packages using the following command:
    
    For windows users:
    `conda env create -f rnn-env-win.yml -n "ic-lf-deploy"`
    
    For linux users:
    `conda env create -f rnn-env-lnx.yml -n "ic-lf-deploy"`

2. Make sure all of your training and testing data is in a dedicated data directory, in the correct format.
   
* Description to follow
   
3. Change the configuration parameters in configs.json
    
* Description to follow

---
### How do I...
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

#### ... open Tensorboard to visualize the training statistics?

1. In terminal: `tensorboard --logdir=<study directory>` 
    * `<study directory>` can be a directory for a single training session, or it can be a directory containing a collection of study subdirectories, which would result from running a hyperparameter study. 
2. Open `http://localhost:6006/` in your browser of choice.

---
### What are the parameters in configs.json?

`rolling window`
* `active` - True or false, specify whether or not to use rolling window statistics
* `type` 
    * `binned` - This method creates min, max, and mean features for each original feature, computed by calculating the statistic over that last N minutes, separated into stationary bins. This has the same effect as downsampling the data to a lower frequency.    
    * `rolling` - This method creates min, max, and mean features for each original feature, computed by calculating the statistic over that last N minutes in a rolling fashion. The time frequency of the original data is preserved.
* `minutes` - Specifies the number of minutes to use for the window. For type `binned`, this is the size of the downsampling. This should be higher than `configs["resample_freq"]`, since the rolling windows are calculated after this step. For type `rolling`, this is the size of the rolling window.  

---
### Local Files

#### entry_point.py

* Currently being iterated upon. (Current) data source is network directory. 
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

#### testing_round_parameters.py

* Lets a user run a series of model trainings by varying a single configuration parameter at once. 
* Any parameter in configs.json can be studied.
* All parameter values not being studied in a particular test will default to value already in configs.json
* Multiple studies can be run in series.
* Results are saved to a sub-directory of the main results directory, specific to the case-study (makes for faster results filtering within TensorBoard).

* `iterable`: What config parameter do you want to study? (str) 
* `iterables`: What values of that parameter do you want to test? (list)
* `iterable_type`: What class is the iterable in the configs.json file? (class)

#### testing_round_meters.py

#### testing_round_meters_HPC.py

#### training_history.csv

#### buildings_processing.py

* Contains functions for data manipulation and cleaning.
* Directory path specified in configs.json

#### holidays.json

* JSON file containing holidays specific to the region being tested. 
* File path specified in configs.json
* Update this file to contain the dates of holidays that will affect building occupancy. This will be used for both training and testing.

--- 
### Relevant directories

#### Data directory

* Location described by `configs["data_dir"]` as an absolute or relative path.
* Contains all data and mask files needed for training and testing.

#### Results directory

* Location described by `configs["results_dir"]` as an absolute or relative path.
* This directory (example: `Results/`) contains subdirectories, one for each single-target training session (example: `Results/target/`).
* Each `Results/target/` contains: 
    * Tensorboard directories:
        * `Loss/`, `CPU_Utilization/`, `Memory_GB/`, `Iteration_time/`
    * Configs file, specific to the training session:
        * `configs.json`
    * Output file:
        * `output.out`
    * PyTorch model file saved to disk:
        * `torch_model`
    * Training statistics needed to denormalize data set
        * `train_stats.json`

#### Network directory

* This can be specified by `configs["network_path"]` if the user wishes to retrieve files from another remote directory, like a mounted network drive.
* Not required for normal operation.






