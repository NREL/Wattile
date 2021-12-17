import json
import entry_point as epb
import pytest
from pathlib import Path 
import os


@pytest.fixture
def config_for_tests(tmp_path):
    """
    default configs moded with training cofigs.

    i.e, training data, results to tmp_path, less nodes & epochs...
    """
    with open("configs.json", "r") as read_file:
        configs = json.load(read_file)

    configs["year"] = [2018]  
    configs["results_dir"] = str(tmp_path / "results")
    configs["num_epochs"] = 1
    configs["hidden_nodes"] = 1

    return configs


# test model training config patches
VANILLA_CONFIG_PATCH = {}
ARCH_VERSION_4_CONFIG_PATCH = {"arch_version": 4}


@pytest.mark.parametrize("config_patch", [VANILLA_CONFIG_PATCH, ARCH_VERSION_4_CONFIG_PATCH])
def test_entrypoint(config_for_tests, config_patch):
    """
    Run training and verify results are made.
    """
    config_for_tests.update(config_patch)

    epb.main(config_for_tests)

    results_dir = os.path.join(config_for_tests["results_dir"], config_for_tests["arch_type"] + '_M' + str(
        config_for_tests["target_var"].replace(" ", "")) + '_T' + str(config_for_tests["exp_id"]))
    results_dir = Path(results_dir).resolve()

    assert results_dir / "output.out"
    assert results_dir / "torch_model"
    assert results_dir / "train_stats.json"



