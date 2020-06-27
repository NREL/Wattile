# README #

This repository contains the source code for forecasting energy consumption using deep learning.

### What is this repository for? ###

* Quick summary: this work uses weather variables and past time-series data for energy consumption to predict future energy consumption (and PV production). The two deep learning neural network architectures which have been implemented are:
  1. feed-forward neural network
  2. Recurrent-neural network
  
* Version: this is the initial version (0.1) of this project


### How do I get set up? ###

* Conda environment for running the code:
 A conda environment file is provided for convenience. Assuming you have Anaconda python distribution available on your computer, you can create a new conda environment with the necessary packages using the following command:

`conda env create -f ml-env.yml -n "ic-lf-deploy"`

#### algo_main_rnn_v1.py

* Works with STM whole-campus data. Uses previous timestep for prediction feedback. No major architecture changes besides bug fixes.

#### algo_main_rnn_v2.py

* Currently being iterated upon. Built for training on building-specific data. 

#### entry_point_campus.py

* Older entry point script for STM modeling

#### entry_point_building.py

* Currently being iterated upon. Built for individual building use. (Current) data source is LAN directory. 

#### buildings_processing.py

* Contains functions for data manipulation and cleaning. (Same file is used for both Quantile Regression and ML methods)