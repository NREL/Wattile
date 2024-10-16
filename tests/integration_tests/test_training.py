import json
import pathlib

import pytest

from wattile.data_processing import prep_for_rnn
from wattile.data_reading import read_dataset_from_file
from wattile.models import ModelFactory
from wattile.models.charlie_model import CharlieModel

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


# test model training config patches
ARCH_VERSION_ALFA_RNN_CONFIG_PATCH = {
    "arch_version": "alfa",
    "arch_type_variant": "vanilla",
}
ARCH_VERSION_ALFA_LSTM_CONFIG_PATCH = {
    "arch_version": "alfa",
    "arch_type_variant": "lstm",
}
ARCH_VERSION_ALFA_STANDARD_TRANSFORMATION_CONFIG_PATCH = {
    "arch_version": "alfa",
    "transformation_method": "standard",
}

ARCH_VERSION_BRAVO_RNN_CONFIG_PATCH = {
    "arch_version": "bravo",
    "arch_type_variant": "vanilla",
}
ARCH_VERSION_BRAVO_LSTM_CONFIG_PATCH = {
    "arch_version": "bravo",
    "arch_type_variant": "lstm",
}
ARCH_VERSION_BRAVO_STANDARD_TRANSFORMATION_CONFIG_PATCH = {
    "arch_version": "bravo",
    "transformation_method": "standard",
}


@pytest.mark.parametrize(
    "config_patch",
    [
        ARCH_VERSION_ALFA_RNN_CONFIG_PATCH,
        ARCH_VERSION_ALFA_LSTM_CONFIG_PATCH,
        ARCH_VERSION_ALFA_STANDARD_TRANSFORMATION_CONFIG_PATCH,
        ARCH_VERSION_BRAVO_RNN_CONFIG_PATCH,
        ARCH_VERSION_BRAVO_LSTM_CONFIG_PATCH,
        ARCH_VERSION_BRAVO_STANDARD_TRANSFORMATION_CONFIG_PATCH,
    ],
)
def test_model_trains(config_for_tests, tmpdir, config_patch):
    """
    Run training and verify results are made.
    """
    # patch configs and create temporary, unquie output file
    config_for_tests["learning_algorithm"].update(config_patch)
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    config_for_tests["data_output"]["exp_dir"] = str(exp_dir)

    # train model
    model = ModelFactory.create_model(config_for_tests)
    data = read_dataset_from_file(config_for_tests)
    train_df, val_df = prep_for_rnn(config_for_tests, data)
    model.train(train_df, val_df)

    # check result file were created
    assert (exp_dir / "torch_model").exists()
    assert (exp_dir / "train_stats.json").exists()


def test_ensemble_model_trains(config_for_tests, tmpdir):
    """
    Run training and verify results are made.
    """
    # patch configs and create temporary, unquie output file
    config_for_tests["learning_algorithm"]["arch_version"] = "alfa_ensemble"
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    exp_dir.mkdir()
    config_for_tests["data_output"]["exp_dir"] = str(exp_dir)

    # train model
    data = read_dataset_from_file(config_for_tests)
    train_df, val_df = prep_for_rnn(config_for_tests, data)
    model = ModelFactory.create_model(config_for_tests)
    model.train(train_df, val_df)

    # check result file were created
    for target_lag in model.alfa_models_by_target_lag.keys():
        assert (exp_dir / target_lag.isoformat() / "torch_model").exists()
        assert (exp_dir / target_lag.isoformat() / "train_stats.json").exists()


def test_charlie_model_runs(config_for_tests, tmpdir):
    config_for_tests["learning_algorithm"]["arch_version"] = "charlie"
    exp_dir = pathlib.Path(tmpdir) / "train_results"
    exp_dir.mkdir()
    config_for_tests["data_output"]["exp_dir"] = str(exp_dir)

    data = read_dataset_from_file(config_for_tests)
    train_df, val_df = prep_for_rnn(config_for_tests, data)

    charlie = CharlieModel(configs=config_for_tests)
    charlie.main(train_df, val_df)

    # check result file were created
    assert (exp_dir / "torch_model").exists()
    assert (exp_dir / "configs.json").exists()
