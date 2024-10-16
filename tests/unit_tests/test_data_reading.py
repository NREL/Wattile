import json
import pathlib

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

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


def test_create_input_dataframe(config_for_tests, tmpdir):
    # patch configs and create temporary, unquie output file
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    config_for_tests["data_output"]["exp_dir"] = str(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # get data
    data = read_dataset_from_file(config_for_tests)

    assert data.index.inferred_type == "datetime64"
    assert set(data.columns) == set(
        config_for_tests["data_input"]["predictor_columns"]
        + [config_for_tests["data_input"]["target_var"]]
    )
    assert data.shape == (10080, 8)

    # assert predictors saved
    # open predictors_target_config
    with open(exp_dir / "predictors_target_config.json", "r") as read_file:
        predictors_target_config = json.load(read_file)
        saved_predictors = pd.DataFrame(predictors_target_config["predictors"])
        saved_target = pd.Series(predictors_target_config["target"])

    # open data config
    dataset_dir = pathlib.Path(config_for_tests["data_input"]["data_dir"])
    configs_file_inputdata = dataset_dir / config_for_tests["data_input"]["data_config"]
    with open(configs_file_inputdata, "r") as read_file:
        configs_input = json.load(read_file)

    # get used columns
    all_predictors = pd.DataFrame(configs_input["predictors"])
    expected_predictors = all_predictors[
        all_predictors["column"].isin(
            config_for_tests["data_input"]["predictor_columns"]
        )
    ]
    all_targets = pd.DataFrame(configs_input["targets"])
    expected_target = all_targets[
        all_targets.column == config_for_tests["data_input"]["target_var"]
    ].iloc[0]

    # assert
    assert_frame_equal(saved_predictors, expected_predictors)
    assert_series_equal(saved_target, expected_target, check_names=False)
