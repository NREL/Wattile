import json
import pathlib
import entry_point as epb
import pytest
from pathlib import Path
import os

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
VANILLA_CONFIG_PATCH = {}
ARCH_VERSION_4_CONFIG_PATCH = {"arch_version": 4}


@pytest.mark.parametrize(
    "config_patch",
    [VANILLA_CONFIG_PATCH, ARCH_VERSION_4_CONFIG_PATCH],
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

    # check if results were created


    
    results_dir = os.path.join(
        config_for_tests["results_dir"],
        config_for_tests["arch_type"]
        + "_M"
        + str(config_for_tests["target_var"].replace(" ", ""))
        + "_T"
        + str(config_for_tests["exp_id"]),
    )
    results_dir = Path(results_dir).resolve()

    assert results_dir / "output.out"
    assert results_dir / "torch_model"
    assert results_dir / "train_stats.json"
