import json
import pathlib

import pandas as pd
import pytest

from wattile.buildings_processing import prep_for_rnn
from wattile.data_reading import read_dataset_from_file

TESTS_PATH = pathlib.Path(__file__).parents[1]
TESTS_FIXTURES_PATH = TESTS_PATH / "fixtures"
TESTS_DATA_PATH = TESTS_PATH / "data" / "Synthetic Site"


@pytest.fixture
def config_for_tests():
    """
    Get test configs
    """
    with open(TESTS_FIXTURES_PATH / "test_configs.json", "r") as read_file:
        configs = json.load(read_file)

    configs["data_input"]["data_dir"] = str(TESTS_DATA_PATH)
    configs["data_input"]["data_config"] = "Synthetic Site Config.json"

    return configs


def test_prep_for_rnn(config_for_tests, tmpdir):
    # patch configs and create temporary, unquie output file
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    config_for_tests["data_output"]["exp_dir"] = str(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # get data
    data, config_for_tests = read_dataset_from_file(config_for_tests)

    # creat data frame
    train_df, val_df = prep_for_rnn(config_for_tests, data)

    excepted_data_columns = []
    # add weather columns
    for pred in config_for_tests["data_input"]["predictor_columns"]:
        excepted_data_columns += [f"{pred}_{m}" for m in ["max", "min", "mean"]]
        excepted_data_columns += [
            f"{pred}_{m}_lag{lag + 1}"
            for m in ["max", "min", "mean"]
            for lag in range(
                config_for_tests["data_processing"]["feat_timelag"]["lag_count"]
            )
        ]

    # add year and hour columns
    excepted_data_columns += [
        f"{f}_{t}_{m}"
        for t in ["MOY", "HOD"]
        for f in ["cos", "sin"]
        for m in ["max", "min", "mean"]
    ]
    excepted_data_columns += [
        f"{f}_{t}_{m}_lag{lag + 1}"
        for t in ["MOY", "HOD"]
        for f in ["cos", "sin"]
        for m in ["max", "min", "mean"]
        for lag in range(
            config_for_tests["data_processing"]["feat_timelag"]["lag_count"]
        )
    ]

    # add week columns
    excepted_data_columns += [
        f"DOW_binary_reg_{i}_{m}" for i in range(0, 7) for m in ["max", "min", "mean"]
    ]
    excepted_data_columns += [
        f"DOW_binary_reg_{i}_{m}_lag{lag + 1}"
        for i in range(0, 7)
        for m in ["max", "min", "mean"]
        for lag in range(
            config_for_tests["data_processing"]["feat_timelag"]["lag_count"]
        )
    ]

    # add target var
    excepted_data_columns.append(config_for_tests["data_input"]["target_var"])

    assert set(train_df.columns) == set(excepted_data_columns)
    assert train_df.shape == (528, len(excepted_data_columns))

    assert set(val_df.columns) == set(excepted_data_columns)
    assert val_df.shape == (60, len(excepted_data_columns))

    test_df = pd.read_hdf(exp_dir / "internal_test.h5", key="df")
    assert set(test_df.columns) == set(excepted_data_columns)
    assert test_df.shape == (73, len(excepted_data_columns))
