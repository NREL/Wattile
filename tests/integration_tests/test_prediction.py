import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest

import wattile.entry_point as epb
from wattile.entry_point import create_input_dataframe, run_model

TESTS_PATH = pathlib.Path(__file__).parents[1]
TESTS_FIXTURES_PATH = TESTS_PATH / "fixtures"
TEST_DATA_PATH = TESTS_PATH / "data" / "Synthetic Site"


@pytest.fixture
def config_for_tests():
    """
    Get test configs
    """
    with open(TESTS_FIXTURES_PATH / "test_configs.json", "r") as read_file:
        configs = json.load(read_file)

    return configs


def popluate_test_data_dir_with_prediction_data(data_dir):
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    data_config_file = TEST_DATA_PATH / "Synthetic Site Config.json"
    data_config = json.loads(data_config_file.read_text())
    predictor_files = [
        TEST_DATA_PATH / file_["filename"]
        for file_ in data_config["files"]
        if file_["contentType"] == "predictors"
    ]

    shutil.copy(data_config_file, data_dir)
    for pf in predictor_files:
        shutil.copy(pf, data_dir)


V4_EXP_DIR = TESTS_FIXTURES_PATH / "v4_exp_dir"
V4_CONFIG_PATCH = {"arch_version": 4}

V5_EXP_DIR = TESTS_FIXTURES_PATH / "v5_exp_dir"
V5_CONFIG_PATCH = {"arch_version": 5}


def test_prediction_v4(config_for_tests, tmpdir):
    # don't run training
    config_for_tests["arch_version"] = 4
    config_for_tests["use_case"] = "prediction"

    # use a temp result dir
    exp_dir = tmpdir / "train_results"
    config_for_tests["exp_dir"] = str(exp_dir)
    shutil.copytree(TESTS_FIXTURES_PATH / "v4_exp_dir", exp_dir)

    # create a temp data dir
    config_for_tests["data_dir"] = str(tmpdir / "data")
    data_dir = tmpdir / "data" / config_for_tests["building"]
    popluate_test_data_dir_with_prediction_data(data_dir)

    train_df, val_df = create_input_dataframe(config_for_tests)
    results = run_model(config_for_tests, train_df, val_df)

    assert results.shape == (val_df.shape[0], len(config_for_tests["qs"]))
    pd.testing.assert_index_equal(results.index, val_df.index)
    np.testing.assert_array_equal(results.columns, config_for_tests["qs"])

    assert (exp_dir / "output.out").exists()


def test_prediction_v5(config_for_tests, tmpdir):
    # don't run training
    config_for_tests["arch_version"] = 5
    config_for_tests["use_case"] = "prediction"

    # use a temp result dir
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    config_for_tests["exp_dir"] = str(exp_dir)
    shutil.copytree(TESTS_FIXTURES_PATH / "v5_exp_dir", exp_dir)

    # create a temp data dir
    config_for_tests["data_dir"] = str(tmpdir / "data")
    data_dir = tmpdir / "data" / config_for_tests["building"]
    popluate_test_data_dir_with_prediction_data(data_dir)

    results = epb.main(config_for_tests)

    num_timestamps = (
        config_for_tests["S2S_stagger"]["initial_num"]
        + config_for_tests["S2S_stagger"]["secondary_num"]
    )
    assert results.shape[1:] == (len(config_for_tests["qs"]), num_timestamps)

    assert (exp_dir / "output.out").exists()
