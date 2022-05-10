import pandas as pd
import pytest

from intelcamp.error import ConfigsError
from intelcamp.buildings_processing import correct_predictor_columns


def test_correct_columns_too_few_columns():
    configs = {
        "predictor_columns": ["a", "b", "c"],
        "target_var": "target_var"
    }
    data = pd.DataFrame({"a": [], "b": [], "target_var": []})

    with pytest.raises(ConfigsError):
        correct_predictor_columns(configs, data)


def test_correct_columns_too_many_columns():
    configs = {
        "predictor_columns": ["a"],
        "target_var": "target_var"
    }
    data = pd.DataFrame({"a": [], "b": [], "target_var": []})

    data = correct_predictor_columns(configs, data)
    assert list(data.columns) == ["a", "target_var"]


def test_correct_columns_reorder_columns():
    configs = {
        "predictor_columns": ["a", "b"],
        "target_var": "target_var"
    }
    data = pd.DataFrame({"b": [], "a": [], "target_var": []})

    data = correct_predictor_columns(configs, data)
    assert list(data.columns) == ["a", "b", "target_var"]

