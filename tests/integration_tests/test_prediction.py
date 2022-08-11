import json
import pathlib
import shutil

import pytest

import wattile.entry_point as epb

TESTS_PATH = pathlib.Path(__file__).parents[1]
TESTS_FIXTURES_PATH = TESTS_PATH / "fixtures"
TEST_DATA_PATH = TESTS_PATH / "data" / "Synthetic Site"
TEST_DATA_CONFIG_PATH = TEST_DATA_PATH / "Synthetic Site Config.json"


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

    data_config = json.loads(TEST_DATA_CONFIG_PATH.read_text())
    predictor_files = [
        TEST_DATA_PATH / file_["filename"]
        for file_ in data_config["files"]
        if file_["contentType"] == "predictors"
    ]

    shutil.copy(TEST_DATA_CONFIG_PATH, data_dir)
    for pf in predictor_files:
        shutil.copy(pf, data_dir)


ALFA_EXP_DIR = TESTS_FIXTURES_PATH / "alfa_exp_dir"
ALFA_CONFIG_PATCH = {"arch_version": "alfa"}

BRAVO_EXP_DIR = TESTS_FIXTURES_PATH / "bravo_exp_dir"
BRAVO_CONFIG_PATCH = {"arch_version": "bravo"}


def test_prediction_alfa(config_for_tests, tmpdir):
    # don't run training
    config_for_tests["arch_version"] = "alfa"
    config_for_tests["use_case"] = "prediction"

    # use a temp result dir
    exp_dir = tmpdir / "train_results"
    config_for_tests["exp_dir"] = str(exp_dir)
    shutil.copytree(TESTS_FIXTURES_PATH / "alfa_exp_dir", exp_dir)

    # create a temp data dir
    config_for_tests["data_dir"] = str(tmpdir / "data")
    config_for_tests["data_config"] = "Synthetic Site Config.json"
    data_dir = tmpdir / "data"
    popluate_test_data_dir_with_prediction_data(data_dir)

    results = epb.main(config_for_tests)

    assert results.shape[1:] == (len(config_for_tests["qs"]),)
    assert (exp_dir / "output.out").exists()


def test_prediction_bravo(config_for_tests, tmpdir):
    # don't run training
    config_for_tests["arch_version"] = "bravo"
    config_for_tests["use_case"] = "prediction"

    # use a temp result dir
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    config_for_tests["exp_dir"] = str(exp_dir)
    shutil.copytree(TESTS_FIXTURES_PATH / "bravo_exp_dir", exp_dir)

    # create a temp data dir
    config_for_tests["data_dir"] = str(tmpdir / "data")
    config_for_tests["data_config"] = "Synthetic Site Config.json"
    data_dir = tmpdir / "data"
    popluate_test_data_dir_with_prediction_data(data_dir)

    results = epb.main(config_for_tests)

    num_timestamps = (
        config_for_tests["S2S_stagger"]["initial_num"]
        + config_for_tests["S2S_stagger"]["secondary_num"]
    )
    assert results.shape[1:] == (len(config_for_tests["qs"]), num_timestamps)

    assert (exp_dir / "output.out").exists()
