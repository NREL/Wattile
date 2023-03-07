import pandas as pd
from pandas.testing import assert_frame_equal

from wattile.models.alfa_ensemble_model import AlfaEnsembleModelDataBuilder


def test_alfa_ensemble_model_data_builder():
    # Set Up
    target_var = "target_var"
    time_lags = [i * pd.Timedelta(hours=1) for i in range(1, 5)]

    target_columns = [f"{target_var} {time_lag.isoformat()}" for time_lag in time_lags]
    predictor_columns = ["pred_1, pred_2, pred_3"]
    index = pd.date_range("2000-01-01 00:00:00", "2000-01-01 12:00:00", freq="H")
    data = pd.DataFrame(index=index, columns=predictor_columns + target_columns)

    aedb = AlfaEnsembleModelDataBuilder(data, target_var)

    # Assertion
    for time_lag in time_lags:
        target_column = f"{target_var} {time_lag.isoformat()}"
        assert_frame_equal(
            aedb.get_dataset(time_lag),
            pd.DataFrame(index=index, columns=predictor_columns + [target_column]),
        )
