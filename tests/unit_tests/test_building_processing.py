import datetime as dt

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from intelcamp.buildings_processing import correct_predictor_columns, correct_timestamps
from intelcamp.error import ConfigsError

JULY_14_CONFIG = {
    "start_time": "1997-07-14T00:00:00-00:00",
    "end_time": "1997-07-15T00:00:00-00:00",
}
JULY_14_MIDNIGHT = pd.Timestamp(year=1997, month=7, day=14, tz=dt.timezone.utc)


def test_correct_columns_too_few_columns():
    configs = {"predictor_columns": ["a", "b", "c"], "target_var": "target_var"}
    data = pd.DataFrame({"a": [], "b": [], "target_var": []})

    with pytest.raises(ConfigsError):
        correct_predictor_columns(configs, data)


def test_correct_columns_too_many_columns():
    configs = {"predictor_columns": ["a"], "target_var": "target_var"}
    data = pd.DataFrame({"a": [], "b": [], "target_var": []})

    data = correct_predictor_columns(configs, data)
    assert list(data.columns) == ["a", "target_var"]


def test_correct_columns_reorder_columns():
    configs = {"predictor_columns": ["a", "b"], "target_var": "target_var"}
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
