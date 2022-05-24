import json
import pathlib

import pytest

import intelcamp.entry_point as epb

TESTS_PATH = pathlib.Path(__file__).parents[1]
TESTS_FIXTURES_PATH = TESTS_PATH / "fixtures"
TESTS_DATA_PATH = TESTS_PATH / "data"


@pytest.fixture
def config_for_tests():
    """
    Get test configs
    """
    with open(TESTS_FIXTURES_PATH / "test_configs.json", "r") as read_file:
        configs = json.load(read_file)

    configs["data_dir"] = str(TESTS_DATA_PATH)

    return configs


# test model training config patches
ARCH_VERSION_4_RNN_CONFIG_PATCH = {"arch_version": 4, "arch_type_variant": "vanilla"}
ARCH_VERSION_4_LSTM_CONFIG_PATCH = {"arch_version": 4, "arch_type_variant": "lstm"}
ARCH_VERSION_4_STANDARD_TRANSFORMATION_CONFIG_PATCH = {
    "arch_version": 4,
    "transformation_method": "standard",
}

ARCH_VERSION_5_RNN_CONFIG_PATCH = {"arch_version": 5, "arch_type_variant": "vanilla"}
ARCH_VERSION_5_LSTM_CONFIG_PATCH = {"arch_version": 5, "arch_type_variant": "lstm"}
ARCH_VERSION_5_STANDARD_TRANSFORMATION_CONFIG_PATCH = {
    "arch_version": 5,
    "transformation_method": "standard",
}


@pytest.mark.parametrize(
    "config_patch",
    [
        ARCH_VERSION_4_RNN_CONFIG_PATCH,
        ARCH_VERSION_4_LSTM_CONFIG_PATCH,
        ARCH_VERSION_4_STANDARD_TRANSFORMATION_CONFIG_PATCH,
        ARCH_VERSION_5_RNN_CONFIG_PATCH,
        ARCH_VERSION_5_LSTM_CONFIG_PATCH,
        ARCH_VERSION_5_STANDARD_TRANSFORMATION_CONFIG_PATCH,
    ],
)
def test_model_trains(config_for_tests, tmpdir, config_patch):
    """
    Run training and verify results are made.
    """
    # patch configs and create temporary, unquie output file
    config_for_tests.update(config_patch)
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    config_for_tests["exp_dir"] = str(exp_dir)

    # train model
    epb.main(config_for_tests)

    # check result file were created
    assert (exp_dir / "output.out").exists()
    assert (exp_dir / "torch_model").exists()
    assert (exp_dir / "train_stats.json").exists()
