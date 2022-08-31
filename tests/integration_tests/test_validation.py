import json
import pathlib
import shutil

import pytest

import wattile.entry_point as epb

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
    configs["plot_comparison"] = False
    configs["data_config"] = "Synthetic Site Config.json"

    return configs


ALFA_EXP_DIR = TESTS_FIXTURES_PATH / "alfa_exp_dir"
ALFA_CONFIG_PATCH = "alfa"

BRAVO_EXP_DIR = TESTS_FIXTURES_PATH / "bravo_exp_dir"
BRAVO_CONFIG_PATCH = "bravo"


@pytest.mark.parametrize(
    "config_patch, test_exp_dir",
    [
        (ALFA_CONFIG_PATCH, ALFA_EXP_DIR),
        (BRAVO_CONFIG_PATCH, BRAVO_EXP_DIR),
    ],
)
def test_validation(config_for_tests, tmpdir, config_patch, test_exp_dir):
    config_for_tests["learning_algorithm"]["arch_version"] = config_patch
    config_for_tests["learning_algorithm"]["use_case"] = "validation"

    exp_dir = pathlib.Path(tmpdir) / "train_results"
    config_for_tests["exp_dir"] = str(exp_dir)
    shutil.copytree(test_exp_dir, exp_dir)

    epb.main(config_for_tests)

    assert (exp_dir / "output.out").exists()
    assert (exp_dir / "error_stats_test.json").exists()
    assert (exp_dir / "QQ_data_Test.h5").exists()
