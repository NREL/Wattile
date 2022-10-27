import datetime as dt

import pandas as pd
import pytest

from wattile.buildings_processing import _preprocess_data
from wattile.models import AlfaModel


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
        "arch_version": "alfa",
        "use_case": "prediction",
    },
}

DATA_PROCESSING_CONFIGS0 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 0},
    "input_output_window": {
        "window_width_futurecast": "0min",
    },
}
DATA_PROCESSING_CONFIGS1 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 4},
    "input_output_window": {
        "window_width_futurecast": "0min",
    },
}
DATA_PROCESSING_CONFIGS2 = {
    "feat_timelag": {"lag_interval": "15min", "lag_count": 4},
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
    alfa_model = AlfaModel(configs)
    predict_for_time = pd.Timestamp(year=2020, month=1, day=1, tz=dt.timezone.utc)
    (
        prediction_window_start_time,
        prediction_window_end_time,
    ) = alfa_model.get_input_window_for_output_time(predict_for_time)

    # ASSERTION
    # When data with given start and end time is fed to _preprocess_data,
    # it should return one row, where the target_var is for the predict_for_time
    data = get_dummy_data(prediction_window_start_time, predict_for_time, lag_interval)
    configs["data_input"]["start_time"] = str(prediction_window_start_time)
    configs["data_input"]["end_time"] = str(prediction_window_end_time)
    preprocessed_data = _preprocess_data(configs, data.copy())

    print("++ data_processing_configs ++")
    print(data_processing_configs)
    print("++ predict_for_time ++")
    print(predict_for_time)
    print("++  prediction_window start/end time ++")
    print(prediction_window_start_time, prediction_window_end_time)
    print("++ unprocessed data ++")
    print(data)
    print("++ unprocessed data predict_for_time ++")
    print(data["target_var"][predict_for_time])
    print("++ processed_data ++")
    print(preprocessed_data)

    assert preprocessed_data.shape[0] == 1
    assert preprocessed_data.index[0] == predict_for_time - window_width_futurecast

    assert False


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
