import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from wattile.buildings_processing import (
    correct_predictor_columns,
    correct_timestamps,
    resample_or_rolling_stats,
    roll_predictors_target,
    timelag_predictors,
    timelag_predictors_target,
)
from wattile.error import ConfigsError

TESTS_PATH = Path(__file__).parents[1]
TESTS_FIXTURES_PATH = TESTS_PATH / "fixtures"

JULY_14_CONFIG = {
    "data_input": {
        "start_time": "1997-07-14T00:00:00-00:00",
        "end_time": "1997-07-15T00:00:00-00:00",
    }
}
JULY_14_MIDNIGHT = pd.Timestamp(year=1997, month=7, day=14, tz=dt.timezone.utc)
Y2K = pd.Timestamp(year=2000, month=1, day=1, tz=dt.timezone.utc)


def test_correct_columns_too_few_columns():
    configs = {
        "data_input": {
            "predictor_columns": ["a", "b", "c"],
            "target_var": "target_var",
        }
    }
    data = pd.DataFrame({"a": [], "b": [], "target_var": []})

    with pytest.raises(ConfigsError):
        correct_predictor_columns(configs, data)


def test_correct_columns_too_many_columns():
    configs = {"data_input": {"predictor_columns": ["a"], "target_var": "target_var"}}
    data = pd.DataFrame({"a": [], "b": [], "target_var": []})

    data = correct_predictor_columns(configs, data)
    assert list(data.columns) == ["a", "target_var"]


def test_correct_columns_reorder_columns():
    configs = {
        "data_input": {"predictor_columns": ["a", "b"], "target_var": "target_var"}
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
            "data_input": {"target_var": "target_var"},
            "data_processing": {
                "resample": {
                    "bin_interval": "1min",
                    "bin_closed": "right",
                    "bin_label": "right",
                },
                "feat_stats": {
                    "active": True,
                    "window_width": "5min",
                },
            },
        },
    )

    expected_output = pd.read_csv(TESTS_FIXTURES_PATH / "rolling_stats_output.csv")
    expected_output["ts"] = pd.to_datetime(expected_output["ts"], exact=False, utc=True)
    expected_output = expected_output.set_index("ts")
    expected_output = expected_output.asfreq("T")
    expected_output = expected_output.astype("float64")

    pd.testing.assert_frame_equal(output, expected_output)


def _get_time_lag_dummy_data(interval: pd.Timedelta) -> pd.DataFrame:
    """midnight to noon, every 10 min, var_1 = 1,2,3..., target_var = 100, 200, 300...

    :param interval: time freq
    :type interval: pd.Timedelta
    :return: dataframe
    :rtype: pd.DataFrame
    """
    data = pd.DataFrame(
        index=pd.date_range("2000-01-01 00:00:00", "2000-01-01 12:00:00", freq=interval)
    )
    data["var_1"] = np.arange(len(data), dtype=np.float32)
    data["target_var"] = np.arange(len(data), dtype=np.float64) * 100

    return data


TIMELAG_PREDICTORS_CONFIGS0 = {
    "data_processing": {
        "feat_timelag": {
            "lag_interval": "10Min",
            "lag_count": 4,
        },
        "input_output_window": {"window_width_futurecast": "60Min"},
    },
    "data_input": {"target_var": "target_var"},
}
TIMELAG_PREDICTORS_CONFIGS1 = {
    "data_processing": {
        "feat_timelag": {
            "lag_interval": "20Min",
            "lag_count": 3,
        },
        "input_output_window": {"window_width_futurecast": "40Min"},
    },
    "data_input": {"target_var": "target_var"},
}
TIMELAG_PREDICTORS_CONFIGS2 = {
    "data_processing": {
        "feat_timelag": {
            "lag_interval": "20Min",
            "lag_count": 3,
        },
        "input_output_window": {"window_width_futurecast": "0Min"},
    },
    "data_input": {"target_var": "target_var"},
}


@pytest.mark.parametrize(
    "configs",
    [
        TIMELAG_PREDICTORS_CONFIGS0,
        TIMELAG_PREDICTORS_CONFIGS1,
        TIMELAG_PREDICTORS_CONFIGS2,
    ],
)
def test_timelag_predictors(configs):
    # Setup
    cdp = configs["data_processing"]
    lag_interval = pd.Timedelta(cdp["feat_timelag"]["lag_interval"])
    lag_count = cdp["feat_timelag"]["lag_count"]
    window_width_futurecast = pd.Timedelta(
        cdp["input_output_window"]["window_width_futurecast"]
    )

    input = _get_time_lag_dummy_data(interval=lag_interval)

    # Action
    output = timelag_predictors(input, configs)

    # Assertion
    # assert columns are correct
    lag_columns = [f"var_1_lag{i}" for i in range(lag_count, 0, -1)]
    assert list(output.columns) == lag_columns + ["var_1", "target_var"]

    # assert indices are correct
    if window_width_futurecast > pd.Timedelta("0Min"):
        num_less_rows = int(window_width_futurecast / lag_interval)
        assert_index_equal(output.index, input.index[lag_count : -1 * (num_less_rows)])
    else:
        assert_index_equal(output.index, input.index[lag_count:])

    # assert lagged var_1 are correct
    # ie, assert var_1_lag{i} == var_1 lag_interval * i ago
    for i in range(1, lag_count + 1):
        var_1_with_lag = f"var_1_lag{i}"

        input_var_with_lag = input["var_1"].copy()
        lag = lag_interval * i
        input_var_with_lag.index += lag

        assert_series_equal(
            output[var_1_with_lag], input_var_with_lag[output.index], check_names=False
        )

    # assert var_1 is right
    assert_series_equal(output["var_1"], input["var_1"][output.index])

    # assert target_var is correect
    # ie, assert output target_var == input target_var in window_width_futurecast
    input_target_var_in_furture = input["target_var"].copy()
    input_target_var_in_furture.index -= window_width_futurecast
    assert_series_equal(output["target_var"], input_target_var_in_furture[output.index])


TIMELAG_PREDICTORS_TARGET_CONFIGS0 = {
    "data_processing": {
        "feat_timelag": {
            "lag_interval": "20Min",
            "lag_count": 3,
        },
        "input_output_window": {
            "window_width_futurecast": "60Min",
            "window_width_target": "40Min",
        },
        "resample": {
            "bin_interval": "10min",
        },
    },
    "data_input": {"target_var": "target_var"},
}
TIMELAG_PREDICTORS_TARGET_CONFIGS1 = {
    "data_processing": {
        "feat_timelag": {
            "lag_interval": "10Min",
            "lag_count": 3,
        },
        "input_output_window": {
            "window_width_futurecast": "30Min",
            "window_width_target": "60Min",
        },
        "resample": {
            "bin_interval": "10min",
        },
    },
    "data_input": {"target_var": "target_var"},
}
TIMELAG_PREDICTORS_TARGET_CONFIGS2 = {
    "data_processing": {
        "feat_timelag": {
            "lag_interval": "60Min",
            "lag_count": 3,
        },
        "input_output_window": {
            "window_width_futurecast": "0Min",
            "window_width_target": "20Min",
        },
        "resample": {
            "bin_interval": "20min",
        },
    },
    "data_input": {"target_var": "target_var"},
}


@pytest.mark.parametrize(
    "configs",
    [
        TIMELAG_PREDICTORS_TARGET_CONFIGS0,
        TIMELAG_PREDICTORS_TARGET_CONFIGS1,
        TIMELAG_PREDICTORS_TARGET_CONFIGS2,
    ],
)
def test_timelag_predictors_target(configs):
    cdp = configs["data_processing"]
    lag_interval = pd.Timedelta(cdp["feat_timelag"]["lag_interval"])
    lag_count = cdp["feat_timelag"]["lag_count"]
    window_width_futurecast = cdp["input_output_window"]["window_width_futurecast"]
    window_width_target = pd.Timedelta(
        cdp["input_output_window"]["window_width_target"]
    )
    bin_interval = pd.Timedelta(cdp["resample"]["bin_interval"])

    input = _get_time_lag_dummy_data(interval=bin_interval)

    # Action
    output = timelag_predictors_target(input.copy(), configs)

    # Assertion
    # assert columns are correct
    lag_columns = [f"var_1_lag{i}" for i in range(lag_count, 0, -1)]
    num_target_lags = int(window_width_target / bin_interval) + 1
    target_columns = [f"target_var_lag_{i}" for i in range(num_target_lags)]
    assert list(output.columns) == lag_columns + ["var_1"] + target_columns

    # assert indices are correct
    num_front_missing_rows = lag_count * int(lag_interval / bin_interval)
    # TODO: why is this right?
    num_back_missing_rows = int(
        (window_width_futurecast / bin_interval) + (window_width_target / bin_interval)
    )
    assert_index_equal(
        output.index, input.index[num_front_missing_rows : -1 * num_back_missing_rows]
    )

    # assert lagged var_1 are correct
    # ie, assert var_1_lag{i} == var_1 lag_interval * i ago
    for i in range(1, lag_count + 1):
        var_1_with_lag = f"var_1_lag{i}"

        input_var_with_lag = input["var_1"].copy()
        lag = lag_interval * i
        input_var_with_lag.index += lag

        assert_series_equal(
            output[var_1_with_lag], input_var_with_lag[output.index], check_names=False
        )

    # assert var_1 is right
    assert_series_equal(output["var_1"], input["var_1"][output.index])

    # assert lagged target_vars are right
    num_lagged_targets = int(window_width_target / bin_interval)
    for i in range(num_lagged_targets + 1):
        target_var_with_lag = f"target_var_lag_{i}"

        input_target_var_with_lag = input["target_var"].copy()
        lag = window_width_futurecast + bin_interval * i
        input_target_var_with_lag.index -= lag

        assert_series_equal(
            output[target_var_with_lag],
            input_target_var_with_lag[output.index],
            check_names=False,
        )


ROLL_PREDICTORS_TARGET_CONFIGS0 = {
    "data_processing": {
        "input_output_window": {
            "window_width_futurecast": "0Min",
            "window_width_target": "45Min",
            "window_width_source": "45min",
        },
        "resample": {
            "bin_interval": "15min",
        },
    },
    "data_input": {"target_var": "target_var"},
}
ROLL_PREDICTORS_TARGET_CONFIGS1 = {
    "data_processing": {
        "input_output_window": {
            "window_width_futurecast": "60Min",
            "window_width_target": "45Min",
            "window_width_source": "45min",
        },
        "resample": {
            "bin_interval": "15min",
        },
    },
    "data_input": {"target_var": "target_var"},
}
ROLL_PREDICTORS_TARGET_CONFIGS2 = {
    "data_processing": {
        "input_output_window": {
            "window_width_futurecast": "60Min",
            "window_width_target": "45Min",
            "window_width_source": "30min",
        },
        "resample": {
            "bin_interval": "15min",
        },
    },
    "data_input": {"target_var": "target_var"},
}
ROLL_PREDICTORS_TARGET_CONFIGS3 = {
    "data_processing": {
        "input_output_window": {
            "window_width_futurecast": "60Min",
            "window_width_target": "30Min",
            "window_width_source": "45min",
        },
        "resample": {
            "bin_interval": "15min",
        },
    },
    "data_input": {"target_var": "target_var"},
}


@pytest.mark.parametrize(
    "configs",
    [
        ROLL_PREDICTORS_TARGET_CONFIGS0,
        ROLL_PREDICTORS_TARGET_CONFIGS1,
        ROLL_PREDICTORS_TARGET_CONFIGS2,
        ROLL_PREDICTORS_TARGET_CONFIGS3,
    ],
)
def test_roll_predictors_target(configs):
    # Setup
    cdp = configs["data_processing"]
    window_width_futurecast = pd.Timedelta(
        cdp["input_output_window"]["window_width_futurecast"]
    )
    window_width_source = cdp["input_output_window"]["window_width_source"]
    window_width_target = pd.Timedelta(
        cdp["input_output_window"]["window_width_target"]
    )
    bin_interval = pd.Timedelta(cdp["resample"]["bin_interval"])

    input = _get_time_lag_dummy_data(interval=bin_interval)

    # Action
    output = roll_predictors_target(input.copy(), configs)

    # Assertion
    window_source_size_count = int(window_width_source / bin_interval)
    window_target_size_count = int(window_width_target / bin_interval)
    window_futurecast_size_count = int(window_width_futurecast / bin_interval)

    num_incompete_rows = (
        window_target_size_count
        + window_source_size_count
        + window_futurecast_size_count
    )
    num_rows = len(input.index) - num_incompete_rows

    assert output["predictor"].shape == (
        num_rows,
        window_source_size_count + 1,
        1,  # for var_1
    )
    assert output["target"].shape == (num_rows, window_target_size_count + 1, 1)
    assert output["timestamp"].shape == (num_rows,)

    assert_index_equal(
        output["timestamp"],
        input.iloc[
            window_source_size_count : -(
                window_target_size_count + window_futurecast_size_count
            )
        ].index,
    )

    for i, (timestamp, predictor, target) in enumerate(
        zip(output["timestamp"], output["predictor"], output["target"])
    ):
        # assert lagged predictors
        # ie, assert predictor is input from
        # now - window_source_size_count to
        # now
        needed_predictors_timestamps = [
            timestamp - (bin_interval * (j))
            for j in reversed(range(window_source_size_count + 1))
        ]
        assert_array_equal(
            predictor.flatten(), input["var_1"][needed_predictors_timestamps].to_numpy()
        )

        # assert target
        # ie, assert target is input from
        # now + window_width_futurecast to
        # now + window_width_futurecast + window_target_size_count
        needed_target_timestamps = [
            timestamp + window_width_futurecast + (bin_interval * j)
            for j in range(window_target_size_count + 1)
        ]
        except_target = input["target_var"][needed_target_timestamps].values.reshape(
            -1, 1
        )
        assert_array_equal(target, except_target)
