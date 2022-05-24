Intelligent Campus
====

Deep learning algorithms for probabilistic forecasting of time series building performance data.

Set Up
----

Requirements:
- [Python](https://www.python.org/downloads/) >= 3.7.0
- [Poetry](https://python-poetry.org/docs/#installation)
- [Microsoft Visual C++ 14.0](https://visualstudio.microsoft.com/visual-cpp-build-tools/), if you're on Windows

If using conda for python management, use `environment.yml` to make an environment and configure poetry to not create a vitural enviroment.
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

IntelCamp has two main functions:
- `create_input_dataframe`, which creates a dataframe for model input from the configs file and raw data.
- `run_model` which runs either training, validation, or prediction on input dataframes according to the configs.

Docs for the configs [here](./tests/fixtures/README.md).

Docs for the format of the raw data is [here](./tests/data/README.md).

Docs for the format of the output is [here](./tests/fixtures/v5_exp_dir/README.md).

```py
import json 

from intelcamp.entry_point import create_input_dataframe, run_model

with open("intelcamp/configs.json", "r") as f:
    configs = json.load(f)

train_df, val_df = create_input_dataframe(configs)
run_model(configs, train_val, val_df)
```

After running, you may use tensordboard on the results.

```
tensorboard --logdir=<study directory>
```

Available Models
----
algo_main_rnn_v4.py

* Probabilistic forecasting 
* In one training session, predict:
    * Number of future times: 1
    * Number of quantiles per time: *Q*
* Vanilla and LSTM variants available

algo_main_rnn_v5.py

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
