import pathlib
import shutil
import pytest
import json
from util import get_exp_dir 

import entry_point as epb

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

V4_EXP_DIR = TESTS_FIXTURES_PATH / "v4_exp_dir"
V4_CONFIG_PATCH = { "arch_version": 4}

V5_EXP_DIR = TESTS_FIXTURES_PATH / "v5_exp_dir"
V5_CONFIG_PATCH = { "arch_version": 5}


@pytest.mark.parametrize(
    "config_patch, test_exp_dir",
    [
        (V4_CONFIG_PATCH, V4_EXP_DIR),
        (V5_CONFIG_PATCH, V5_EXP_DIR),
    ],
)
def test_validation(config_for_tests, tmpdir, config_patch, test_exp_dir):
    config_for_tests.update(config_patch)
    config_for_tests["use_case"] = "validation"
    
    config_for_tests["results_dir"] = str(tmpdir / "train_results")
    exp_dir = get_exp_dir(config_for_tests)
    shutil.copytree(test_exp_dir, exp_dir)

    epb.main(config_for_tests)

    assert (exp_dir / "output.out").exists()
    assert (exp_dir / "error_stats_test.json").exists()
    assert (exp_dir / "QQ_data_Test.h5").exists()
