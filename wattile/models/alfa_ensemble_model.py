from copy import deepcopy
from pathlib import Path

import pandas as pd
import xarray as xr

from wattile.models.alfa_model import AlfaModel
from wattile.models.base_model import BaseModel


class AlfaEnsembleModelDataBuilder:
    """From bravo like inputs, get_dataset returns alfa like inputs for a spefic target lag"""

    def __init__(self, data_frame: pd.DataFrame, target_var: str):
        """Seperate target and predictors"""
        # find target column names
        self.target_var = target_var
        target_column_names = [
            c for c in data_frame.columns if c.startswith(target_var)
        ]

        # seperate predictors and targets
        self.target_df = data_frame[target_column_names]
        self.predictors_df = data_frame.drop(target_column_names, axis=1)

    def get_dataset(self, target_lag: pd.Timedelta):
        target_column_name = f"{self.target_var} {target_lag.isoformat()}"
        target_column = self.target_df[target_column_name]

        dataset = self.predictors_df.copy()
        dataset[target_column_name] = target_column

        return dataset


class AlfaEnsembleModel(BaseModel):
    """A collection of alfa models, one for each target lag in the multi horizon config."""

    def __init__(self, configs):
        """Create an alfa model for each lagged target with approiate configs"""
        self.configs = configs

        # get items from config
        cdp = configs["data_processing"]
        bin_interval = pd.Timedelta(cdp["resample"]["bin_interval"])
        window_width_target = pd.Timedelta(
            cdp["input_output_window"]["window_width_target"]
        )

        # compute target_lags
        num_target_lags = int(window_width_target // bin_interval) + 1
        target_lags = [bin_interval * i for i in range(num_target_lags)]

        # create alfa models
        self.alfa_models_by_target_lag = {
            target_lag: AlfaModel(self._configs_for_alfa_model(configs, target_lag))
            for target_lag in target_lags
        }

    def __del__(self):
        for model in self.alfa_models_by_target_lag.values():
            model.writer.flush()
            model.writer.close()

    def _configs_for_alfa_model(self, base_configs, target_lag):
        """Edit base configs to have custom futurecast and expdir according to taraget lag."""
        alfa_model_configs = deepcopy(base_configs)

        # update arch version
        alfa_model_configs["learning_algorithm"]["arch_version"] = "alfa"

        # put exp_dir one folder deeper
        alfa_model_configs["data_output"]["exp_dir"] = str(
            Path(base_configs["data_output"]["exp_dir"]) / target_lag.isoformat()
        )

        # add target lag to furturecast
        window_width_futurecast = alfa_model_configs["data_processing"][
            "input_output_window"
        ]["window_width_futurecast"]
        alfa_model_configs["data_processing"]["input_output_window"][
            "window_width_futurecast"
        ] = str(window_width_futurecast + target_lag)

        # set target var to reflect lag
        target_var = alfa_model_configs["data_input"]["target_var"]
        alfa_model_configs["data_input"][
            "target_var"
        ] = f"{target_var} {target_lag.isoformat()}"

        # set run name and tags for registry logging
        if self.registry.log:
            alfa_model_configs["model_registry"][
                "run_name"
            ] = f"{self.registry.run_name} - {target_lag.isoformat()}"
            alfa_model_configs["model_registry"]["run_tags"] = {
                **self.registry.run_tags,
                "ensemble": "alfa",
                "ensemble_belongs_to": self.registry.run_name,
                "ensemble_target_lag": target_lag.isoformat(),
            }

        return alfa_model_configs

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Train all alfa models"""
        target_var = self.configs["data_input"]["target_var"]
        train_data_builder = AlfaEnsembleModelDataBuilder(train_df, target_var)
        val_data_builder = AlfaEnsembleModelDataBuilder(val_df, target_var)

        for target_lag, model in self.alfa_models_by_target_lag.items():
            train_data = train_data_builder.get_dataset(target_lag=target_lag)
            val_data = val_data_builder.get_dataset(target_lag=target_lag)

            model.train(train_data, val_data)

    def validate(self, val_df: pd.DataFrame) -> None:
        """validate all alfa models"""
        target_var = self.configs["data_input"]["target_var"]
        val_data_builder = AlfaEnsembleModelDataBuilder(val_df, target_var)

        for target_lag, model in self.alfa_models_by_target_lag.items():
            val_data = val_data_builder.get_dataset(target_lag=target_lag)

            model.validate(val_data)

    def predict(self, val_df: pd.DataFrame) -> None:
        """predict all alfa models"""
        target_var = self.configs["data_input"]["target_var"]
        val_data_builder = AlfaEnsembleModelDataBuilder(val_df, target_var)

        results = []
        for target_lag, model in self.alfa_models_by_target_lag.items():
            val_data = val_data_builder.get_dataset(target_lag=target_lag)

            results.append(model.predict(val_data))

        return xr.concat(results, dim="horizon")
