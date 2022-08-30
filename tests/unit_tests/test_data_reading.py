import json
import pathlib

import pytest

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

    configs["data_dir"] = str(TESTS_DATA_PATH)
    configs["data_config"] = "Synthetic Site Config.json"

    return configs


def test_create_input_dataframe(config_for_tests, tmpdir):
    # patch configs and create temporary, unquie output file
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    config_for_tests["exp_dir"] = str(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # get data
    data, config_for_tests = read_dataset_from_file(config_for_tests)

    assert data.index.inferred_type == "datetime64"
    assert set(data.columns) == set(
        config_for_tests["predictor_columns"] + [config_for_tests["target_var"]]
    )
    assert data.shape == (10080, 8)
