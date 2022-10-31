import datetime as dt

import pandas as pd
import pytest

from wattile.buildings_processing import _preprocess_data
from wattile.models import BravoModel


def get_dummy_data(start, end, iterval):
    data = pd.DataFrame(index=pd.date_range(start, end, freq=iterval))
    data["var_1"] = data.index.hour * 100 + data.index.minute
    data["target_var"] = -1 * (data.index.hour * 100 + data.index.minute)

    return data


CONFIGS = {
    "data_input": {"predictor_columns": ["var_1"], "target_var": "target_var"},
    "data_processing": {
        "feat_time": {
            "hour_of_day": [],
            "day_of_week": [],
            "month_of_year": [],
            "holidays": False,
        },
        "resample": {
            "bin_interval": "15min",
            "bin_closed": "right",
            "bin_label": "right",
        },
        "feat_stats": {
            "active": False,
            "window_width": "15min",
        },
    },
    "learning_algorithm": {
        "arch_version": "bravo",
        "use_case": "prediction",
    },
}

DATA_PROCESSING_CONFIGS0 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 0},
    "input_output_window": {
        "window_width_futurecast": "0min",
        "window_width_target": "0min",
    },
}
DATA_PROCESSING_CONFIGS1 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 4},
    "input_output_window": {
        "window_width_futurecast": "0min",
        "window_width_target": "0min",
    },
}
DATA_PROCESSING_CONFIGS2 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 4},
    "input_output_window": {
        "window_width_futurecast": "0min",
        "window_width_target": "15min",
    },
}
DATA_PROCESSING_CONFIGS3 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 4},
    "input_output_window": {
        "window_width_futurecast": "0min",
        "window_width_target": "45min",
    },
}
DATA_PROCESSING_CONFIGS4 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 4},
    "input_output_window": {
        "window_width_futurecast": "30min",
        "window_width_target": "45min",
    },
}


@pytest.mark.parametrize(
    "data_processing_configs",
    [
        DATA_PROCESSING_CONFIGS0,
        DATA_PROCESSING_CONFIGS1,
        DATA_PROCESSING_CONFIGS2,
        DATA_PROCESSING_CONFIGS3,
        DATA_PROCESSING_CONFIGS4,
    ],
)
def test_get_input_window_for_output_time(tmpdir, data_processing_configs):
    # SETUP
    configs = CONFIGS
    configs["data_output"] = {"exp_dir": tmpdir}
    configs["data_processing"].update(data_processing_configs)

    lag_interval = pd.Timedelta(data_processing_configs["feat_timelag"]["lag_interval"])
    window_width_futurecast = pd.Timedelta(
        data_processing_configs["input_output_window"]["window_width_futurecast"]
    )

    # ACTION
    bravo_model = BravoModel(configs)
    predict_for_time = pd.Timestamp(year=2020, month=1, day=1, tz=dt.timezone.utc)
    (
        prediction_window_start_time,
        prediction_window_end_time,
    ) = bravo_model.get_input_window_for_output_time(predict_for_time)

    # ASSERTION
    # When data with given start and end time is fed to _preprocess_data,
    # it should return one row, where the target_var is for the predict_for_time
    data = get_dummy_data(prediction_window_start_time, predict_for_time, lag_interval)
    configs["data_input"]["start_time"] = str(prediction_window_start_time)
    configs["data_input"]["end_time"] = str(prediction_window_end_time)
    preprocessed_data = _preprocess_data(configs, data.copy())

    assert preprocessed_data.shape[0] == 1
    assert preprocessed_data.index[0] == predict_for_time - window_width_futurecast


@pytest.mark.parametrize(
    "data_processing_configs",
    [
        DATA_PROCESSING_CONFIGS0,
        DATA_PROCESSING_CONFIGS1,
        DATA_PROCESSING_CONFIGS2,
        DATA_PROCESSING_CONFIGS3,
    ],
)
def test_get_prediction_vector_for_time(tmpdir, data_processing_configs):
    # SETUP
    configs = CONFIGS
    configs["data_output"] = {"exp_dir": tmpdir}
    configs["data_processing"].update(data_processing_configs)

    window_width_futurecast = pd.Timedelta(
        data_processing_configs["input_output_window"]["window_width_futurecast"]
    )
    window_width_target = pd.Timedelta(
        data_processing_configs["input_output_window"]["window_width_target"]
    )
    lag_interval = pd.Timedelta(data_processing_configs["feat_timelag"]["lag_interval"])

    # ACTION
    alfa_model = BravoModel(configs)
    results = alfa_model.get_prediction_vector_for_time()

    # ASSERT
    assert (
        results
        == pd.timedelta_range(
            start=pd.Timedelta("0 days"),
            end=window_width_futurecast + window_width_target,
            freq=lag_interval,
        ).to_list()
    )
