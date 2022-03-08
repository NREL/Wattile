import json
import pathlib
import entry_point as epb
import pytest
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


# test model training config patches
ARCH_VERSION_4_RNN_CONFIG_PATCH = {"arch_version": 4, "arch_type_variant": "vanilla"}
ARCH_VERSION_4_LSTM_CONFIG_PATCH = {"arch_version": 4, "arch_type_variant": "lstm"}
ARCH_VERSION_4_STANDARD_TRANSFORMATION_CONFIG_PATCH = {"arch_version": 4, "transformation_method": "standard"}

ARCH_VERSION_5_RNN_CONFIG_PATCH = {"arch_version": 5, "arch_type_variant": "vanilla"}
ARCH_VERSION_5_LSTM_CONFIG_PATCH = {"arch_version": 5, "arch_type_variant": "lstm"}
ARCH_VERSION_5_STANDARD_TRANSFORMATION_CONFIG_PATCH = {"arch_version": 5, "transformation_method": "standard"}



@pytest.mark.parametrize(
    "config_patch",
    [
        ARCH_VERSION_4_RNN_CONFIG_PATCH,
        ARCH_VERSION_4_LSTM_CONFIG_PATCH,
        ARCH_VERSION_4_STANDARD_TRANSFORMATION_CONFIG_PATCH,
        ARCH_VERSION_5_RNN_CONFIG_PATCH,
        ARCH_VERSION_5_LSTM_CONFIG_PATCH,
        ARCH_VERSION_5_STANDARD_TRANSFORMATION_CONFIG_PATCH
    ],
)
def test_model_trains(config_for_tests, tmpdir, config_patch):
    """
    Run training and verify results are made.
    """
    # patch configs and create temporary, unquie output file
    config_for_tests.update(config_patch)
    config_for_tests["results_dir"] = str(tmpdir / "train_results")

    # train model
    epb.main(config_for_tests)

    # check result file were created
    exp_dir = get_exp_dir(config_for_tests)

    assert (exp_dir / "output.out").exists()
    assert (exp_dir / "torch_model").exists()
    assert (exp_dir / "train_stats.json").exists()