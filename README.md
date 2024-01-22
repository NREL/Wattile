Wattile
=

Probabilistic Deep Learning-based Forecasting of Building Energy ConsumptionProbabilistic Deep Learning-based Forecasting of Building Energy Consumption.

# Set Up

Requirements:
- [Python](https://www.python.org/downloads/) >= 3.7.0
- [Poetry](https://python-poetry.org/docs/#installation)
- [Microsoft Visual C++ 14.0](https://visualstudio.microsoft.com/visual-cpp-build-tools/), if you're on Windows

If using conda for python management, use `environment.yml` to make an environment and configure poetry to not create a vitural enviroment (else, skip these commands).
```
conda env create -f environment.yml
poetry config virtualenvs.create false
poetry config virtualenvs.in-project true
```
Then install dependencies.
```
poetry install
```

Finally, install pre-commit.
```
poetry run pre-commit install
```

Quick Start
----
Please see [our example notebooks](./notebooks/examples) to see how to run Wattile out of the box.

In short, one must prep their data, create the model, and send the data through.
```py
train_df, val_df = prep_for_rnn(configs, data)
model = ModelFactory.create_model(configs)
model.train(train_df, val_df)
```

After running, you may use tensordboard on the results.

```
tensorboard --logdir=<study directory>
```
Wattile is highly configurable.

Docs for the configs [here](./docs/Configs.md).

Docs for the format of the raw data is [here](./docs/Data_configs.md).

Docs for the format of the output is [here](./docs/Output.md).

Available Models
----
### [Alfa](./wattile/models/alfa_model.py)

* Probabilistic forecasting 
* In one training session, predict:
    * Number of future times: 1
    * Number of quantiles per time: *Q*
* Vanilla and LSTM variants available

### [Bravo](./wattile/models/bravo_model.py)

* Probabilistic forecasting 
* In one training session, predict:
    * Number of future times: *T*
    * Number of quantiles per time: *Q*
* Vanilla and LSTM variants available
* Supports future time predictions with constant spacing or variable spacing 


### [Alfa Ensemble](./wattile/models/alfa_ensemble_model.py)

* Probabilistic forecasting 
* In one training session, predict:
    * Number of future times: *T*
    * Number of quantiles per time: *Q*
* Vanilla and LSTM variants available
* Supports future time predictions with constant spacing or variable spacing 

Development
----

### Testing
```
$ poetry run pytest tests
```
To see test coverage, add args `--cov-report html --cov=$PROJECT_DIR` and open `./htmlcov/index.html`

### Styleing
```sh
$ make format
```
