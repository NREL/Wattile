import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wattile.buildings_processing import (
    correct_predictor_columns,
    correct_timestamps,
    resample_or_rolling_stats,
)
from wattile.error import ConfigsError

TESTS_PATH = Path(__file__).parents[1]
TESTS_FIXTURES_PATH = TESTS_PATH / "fixtures"

JULY_14_CONFIG = {
    "data_handling": {
        "start_time": "1997-07-14T00:00:00-00:00",
        "end_time": "1997-07-15T00:00:00-00:00",
    }
}
JULY_14_MIDNIGHT = pd.Timestamp(year=1997, month=7, day=14, tz=dt.timezone.utc)


def test_correct_columns_too_few_columns():
    configs = {
        "data_handling": {
            "predictor_columns": ["a", "b", "c"],
            "target_var": "target_var",
        }
    }
    data = pd.DataFrame({"a": [], "b": [], "target_var": []})

    with pytest.raises(ConfigsError):
        correct_predictor_columns(configs, data)


def test_correct_columns_too_many_columns():
    configs = {
        "data_handling": {"predictor_columns": ["a"], "target_var": "target_var"}
    }
    data = pd.DataFrame({"a": [], "b": [], "target_var": []})

    data = correct_predictor_columns(configs, data)
    assert list(data.columns) == ["a", "target_var"]


def test_correct_columns_reorder_columns():
    configs = {
        "data_handling": {"predictor_columns": ["a", "b"], "target_var": "target_var"}
    }
    data = pd.DataFrame({"b": [], "a": [], "target_var": []})

    data = correct_predictor_columns(configs, data)
    assert list(data.columns) == ["a", "b", "target_var"]


def test_correct_timestamps_trim():
    configs = JULY_14_CONFIG
    data = pd.DataFrame(
        {},
        index=[
            JULY_14_MIDNIGHT,
            JULY_14_MIDNIGHT + pd.Timedelta(days=1, microseconds=1),
        ],
    )

    data = correct_timestamps(configs, data)
    assert_frame_equal(
        data,
        data.eq(
            pd.DataFrame(
                {},
                index=[
                    JULY_14_MIDNIGHT,
                ],
            )
        ),
    )


def test_correct_timestamps_no_data():
    configs = JULY_14_CONFIG
    data = pd.DataFrame(
        {},
        index=[
            JULY_14_MIDNIGHT + pd.Timedelta(days=1, microseconds=1),
        ],
    )

    with pytest.raises(ConfigsError):
        data = correct_timestamps(configs, data)


def test_correct_timestamps_reorder():
    configs = JULY_14_CONFIG
    data = pd.DataFrame(
        {},
        index=[
            JULY_14_MIDNIGHT + pd.Timedelta(hours=2),
            JULY_14_MIDNIGHT + pd.Timedelta(hours=3),
            JULY_14_MIDNIGHT + pd.Timedelta(hours=1),
        ],
    )

    data = correct_timestamps(configs, data)
    assert_frame_equal(
        data,
        data.eq(
            pd.DataFrame(
                {},
                index=[
                    JULY_14_MIDNIGHT + pd.Timedelta(hours=1),
                    JULY_14_MIDNIGHT + pd.Timedelta(hours=2),
                    JULY_14_MIDNIGHT + pd.Timedelta(hours=3),
                ],
            )
        ),
    )


def test_rolling_stats():
    input = pd.read_csv(TESTS_FIXTURES_PATH / "rolling_stats_input.csv")
    input["var1"] = pd.to_numeric(input["var1"], errors="coerce")
    input["var2"] = pd.to_numeric(input["var2"], errors="coerce")
    input["var1"] = input["var1"].astype(float)
    input["var2"] = input["var2"].astype(float)
    input["ts"] = pd.to_datetime(input["ts"], exact=False, utc=True)
    input = input.set_index("ts")

    output = resample_or_rolling_stats(
        input,
        configs={
            "target_var": "target_var",
            "resample_interval": "1min",
            "feat_stats": {
                "active": True,
                "window_width": "5min",
            },
        },
    )

    expected_output = pd.read_csv(TESTS_FIXTURES_PATH / "rolling_stats_output.csv")
    expected_output["ts"] = pd.to_datetime(expected_output["ts"], exact=False, utc=True)
    expected_output = expected_output.set_index("ts")
    expected_output = expected_output.asfreq("T")
    expected_output = expected_output.astype("float64")

    pd.testing.assert_frame_equal(output, expected_output)
