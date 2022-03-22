import pytest
import json
import pathlib
import pandas as pd

import entry_point as epb
from util import get_exp_dir

TESTS_PATH = pathlib.Path(__file__).parents[1]
TESTS_FIXTURES_PATH = TESTS_PATH / "fixtures"

@pytest.fixture
def config_for_tests():
    """
    Get test configs
    """
    with open(TESTS_FIXTURES_PATH / "test_configs.json", "r") as read_file:
        configs = json.load(read_file)

    return configs
    
    
def test_create_input_dataframe(config_for_tests, tmpdir):
    # patch configs and create temporary, unquie output file
    config_for_tests["results_dir"] = tmpdir / "train_results"

    # creat data frame
    exp_dir = get_exp_dir(config_for_tests)
    exp_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df = epb.create_input_dataframe(config_for_tests)

    excepted_data_columns = []
    # add weather columns
    for pred in config_for_tests["weather_include"]:
        excepted_data_columns += [f"{pred}_{m}" for m in ["max", "min", "mean"]]
        excepted_data_columns += [f"{pred}_{m}_lag{l + 1}" for m in ["max", "min", "mean"] for l in range(config_for_tests["window"])]

    # add year and hour columns
    excepted_data_columns += [
        f"{f}_{t}_{m}"
        for t in ["MOY", "HOD"] 
        for f in ["cos", "sin"] 
        for m in ["max", "min", "mean"]
    ]
    excepted_data_columns += [
        f"{f}_{t}_{m}_lag{l + 1}" 
        for t in ["MOY", "HOD"] 
        for f in ["cos", "sin"] 
        for m in ["max", "min", "mean"] 
        for l in range(config_for_tests["window"])
    ]

    # add week columns
    excepted_data_columns += [
        f"DOW_binary_reg_{i}_{m}" 
        for i in range(0,7) 
        for m in ["max", "min", "mean"]
    ]
    excepted_data_columns += [
        f"DOW_binary_reg_{i}_{m}_lag{l + 1}" 
        for i in range(0,7) 
        for m in ["max", "min", "mean"] 
        for l in range(config_for_tests["window"])
    ]

    # add target var
    excepted_data_columns.append(config_for_tests["target_var"])

    assert set(train_df.columns) == set(excepted_data_columns)
    assert train_df.shape == (480, len(excepted_data_columns))

    assert set(val_df.columns) == set(excepted_data_columns)
    assert val_df.shape == (84, len(excepted_data_columns))

    test_df = pd.read_hdf(exp_dir / "internal_test.h5", key='df')
    assert set(test_df.columns) == set(excepted_data_columns)
    assert test_df.shape == (96, len(excepted_data_columns))
