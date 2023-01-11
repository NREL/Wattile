import datetime as dt

import pandas as pd
import pytest

from wattile.buildings_processing import _preprocess_data
from wattile.models import AlfaModel


def get_dummy_data(start, end, iterval, label):
    data = pd.DataFrame(index=pd.date_range(start, end, freq=iterval, inclusive=label))
    data["var_1"] = data.index.hour * 100 + data.index.minute
    data["target_var"] = -1 * (data.index.hour * 100 + data.index.minute)

    return data


CONFIGS = {
    "data_input": {"predictor_columns": ["var_1"], "target_var": "target_var"},
    "data_processing": {
        "random_seed": 0,
        "feat_time": {
            "hour_of_day": [],
            "day_of_week": [],
            "month_of_year": [],
            "holidays": False,
        },
        "feat_stats": {
            "active": False,
            "window_width": "15min",
        },
    },
    "learning_algorithm": {
        "arch_version": "alfa",
        "use_case": "prediction",
    },
}

DATA_PROCESSING_CONFIGS0 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 0},
    "resample": {"bin_interval": "15min"},
    "input_output_window": {
        "window_width_futurecast": "0min",
    },
}
DATA_PROCESSING_CONFIGS1 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 4},
    "resample": {"bin_interval": "15min"},
    "input_output_window": {
        "window_width_futurecast": "0min",
    },
}
DATA_PROCESSING_CONFIGS2 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 4},
    "resample": {"bin_interval": "5min"},
    "input_output_window": {
        "window_width_futurecast": "30min",
    },
}
DATA_PROCESSING_CONFIGS3 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 4},
    "resample": {"bin_interval": "5min"},
    "input_output_window": {
        "window_width_futurecast": "30min",
    },
}


@pytest.mark.parametrize(
    "data_processing_configs",
    [
        DATA_PROCESSING_CONFIGS0,
        DATA_PROCESSING_CONFIGS1,
        DATA_PROCESSING_CONFIGS2,
        DATA_PROCESSING_CONFIGS3,
    ],
)
@pytest.mark.parametrize("bin_closed", ["left", "right"])
@pytest.mark.parametrize("bin_label", ["left", "right"])
@pytest.mark.parametrize("feat_stats_active", [True, False])
def test_get_input_window_for_output_time(
    tmpdir, data_processing_configs, bin_closed, bin_label, feat_stats_active
):
    # SETUP
    configs = CONFIGS
    configs["data_output"] = {"exp_dir": tmpdir}
    configs["data_processing"].update(data_processing_configs)
    configs["data_processing"]["resample"]["bin_closed"] = bin_closed
    configs["data_processing"]["resample"]["bin_label"] = bin_label
    configs["data_processing"]["feat_stats"]["active"] = feat_stats_active

    lag_interval = pd.Timedelta(data_processing_configs["feat_timelag"]["lag_interval"])

    # ACTION
    alfa_model = AlfaModel(configs)
    nominal_prediction_time = pd.Timestamp(
        year=2020, month=1, day=1, tz=dt.timezone.utc
    )
    (
        prediction_window_start_time,
        prediction_window_end_time,
    ) = alfa_model.get_input_window_for_output_time(nominal_prediction_time)

    # ASSERTION
    # When data with given start and end time is fed to _preprocess_data,
    # it should return one row, where the index is nominal_prediction_time
    data = get_dummy_data(
        prediction_window_start_time,
        prediction_window_end_time,
        lag_interval / 5,
        bin_label,
    )
    configs["data_input"]["start_time"] = str(prediction_window_start_time)
    configs["data_input"]["end_time"] = str(prediction_window_end_time)
    preprocessed_data = _preprocess_data(configs, data.copy())

    assert preprocessed_data.shape[0] == 1
    assert preprocessed_data.index[0] == nominal_prediction_time


@pytest.mark.parametrize(
    "data_processing_configs",
    [
        DATA_PROCESSING_CONFIGS0,
        DATA_PROCESSING_CONFIGS1,
        DATA_PROCESSING_CONFIGS2,
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

    # ACTION
    alfa_model = AlfaModel(configs)
    results = alfa_model.get_prediction_vector_for_time()

    assert results == [window_width_futurecast]
