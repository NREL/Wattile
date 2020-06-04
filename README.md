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
