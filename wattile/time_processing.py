import datetime as dt

import numpy as np
import pandas as pd

from wattile.holidays import HOLIDAYS


def _add_hour_based_columns(data, configs):
    """Add hour based columns to data based on "HOD" configs. Config options are

    sincos: Cyclic time variables are used (one sin column and one cos column)
    binary_reg: Binary indicator variables, one column for each hour.
    binary_fuzzy: Same as binary_reg, but binary edges are smoothed.

    :param configs: configs
    :type configs: dict
    :param data: data
    :type data: pd.dataframe
    :return: data
    :rtype: pd.dataframe
    """
    HOD_configs = configs["feat_time"]["hour_of_day"]

    if "sincos" in HOD_configs:
        num_seconds_in_day = 24 * 60 * 60
        time_in_seconds = (
            data.index.hour * 3600 + data.index.minute * 60 + data.index.second
        ).values

        data["sin_HOD"] = np.sin(2 * np.pi * time_in_seconds / num_seconds_in_day)
        data["cos_HOD"] = np.cos(2 * np.pi * time_in_seconds / num_seconds_in_day)

    if "binary_reg" in HOD_configs:
        for i in range(0, 24):
            data[f"HOD_binary_reg_{i}"] = (data.index.hour == i).astype(int)

    if "binary_fuzzy" in HOD_configs:
        time_as_float = data.index.hour + data.index.minute / 60

        for HOD in range(0, 24):
            data[f"HOD_binary_fuzzy_{HOD}"] = np.maximum(
                1 - abs(time_as_float - HOD), 0
            )

    return data


def _add_day_based_columns(data, configs):
    """Add day based columns to data based on "DOW" configs. Config options are

    binary_reg: Binary indicator variables, one column for each day.
    binary_fuzzy: Same as binary_reg, but binary edges are smoothed.

    :param configs: configs
    :type configs: dict
    :param data: data
    :type data: pd.dataframe
    :return: data
    :rtype: pd.dataframe
    """
    day_of_week_configs = configs["feat_time"]["day_of_week"]

    if "binary_reg" in day_of_week_configs:
        for i in range(0, 7):
            data[f"DOW_binary_reg_{i}"] = (data.index.weekday == i).astype(int)

    if "binary_fuzzy" in day_of_week_configs:
        for DOW in range(0, 7):
            day_as_float = data.index.weekday + data.index.hour / 24

            data[f"DOW_binary_fuzzy_{DOW}"] = np.maximum(1 - abs(day_as_float - DOW), 0)

    return data


def _add_month_based_columns(data, configs):
    """Add hour based columns to data based on "MOY" configs. Config options are

    sincos: Cyclic time variables are used (one sin column and one cos column)

    Also, if configs.get("Holidays"), add binary "Holiday" and "Holiday_forward".

    :param configs: configs
    :type configs: dict
    :param data: data
    :type data: pd.dataframe
    :return: data
    :rtype: pd.dataframe
    """
    if "sincos" in configs["feat_time"]["month_of_year"]:
        data["sin_MOY"] = np.sin(2 * np.pi * (data.index.dayofyear).values / (365))
        data["cos_MOY"] = np.cos(2 * np.pi * (data.index.dayofyear).values / (365))

    if configs["feat_time"]["Holidays"]:
        data["Holiday"] = pd.to_datetime(data.index.date).isin(HOLIDAYS).astype(int)

        next_day = pd.to_datetime(data.index.date + dt.timedelta(days=1))
        data["Holiday_forward"] = next_day.isin(HOLIDAYS).astype(int)

    return data


def add_processed_time_columns(data, configs):
    """add time based features.

    :param configs: configs
    :type configs: dict
    :param data: data
    :type data: pd.dataframe
    :return: data
    :rtype: pd.dataframe
    """
    data = _add_hour_based_columns(data, configs)
    data = _add_day_based_columns(data, configs)
    data = _add_month_based_columns(data, configs)

    return data
