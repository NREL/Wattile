import json
import logging
import os
from abc import ABC
from pathlib import Path

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from wattile import version as wattile_version
from wattile.error import ConfigsError
from wattile.util import factors
from wattile.visualization import timeseries_comparison

global file_prefix
logger = logging.getLogger(str(os.getpid()))


class BaseModel(ABC):
    def __init__(self, configs):
        self.configs = configs

        # Create exp_dir and writer
        self.file_prefix = Path(configs["data_output"]["exp_dir"])
        self.file_prefix.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.file_prefix)

        # Setting random seed with constant
        torch.manual_seed(self.configs["data_processing"]["random_seed"])

        # log creation
        logger.info(
            f"{self.__class__.__name__} model created."
            f"Writing to {self.file_prefix}."
        )

    def __del__(self):
        self.writer.flush()
        self.writer.close()

    def size_the_batches(self, train_data, val_data):
        """
        Compute the batch sizes for training and val set

        :param train_data: (DataFrame)
        :param val_data: (DataFrame)
        :param tr_desired_batch_size: (int)
        :param te_desired_batch_size: (int)
        :return:
        """
        # TODO: maybe we want to break this up and call it in `to_data_loader`
        tr_desired_batch_size = self.configs["learning_algorithm"]["train_batch_size"]
        te_desired_batch_size = self.configs["learning_algorithm"]["val_batch_size"]

        # calcuate train batch size
        train_bth = factors(train_data.shape[0])
        train_num_batches = min(train_bth, key=lambda x: abs(x - tr_desired_batch_size))
        train_bt_size = int(train_data.shape[0] / train_num_batches)

        # calcuate validation batch size
        val_bth = factors(val_data.shape[0])
        val_num_batches = min(val_bth, key=lambda x: abs(x - te_desired_batch_size))
        val_bt_size = int(val_data.shape[0] / val_num_batches)

        logger.info(
            f"Available train batch factors: {sorted(train_bth)}"
            f"Requested number of batches per epoch - Train: \
                {tr_desired_batch_size}, val: {te_desired_batch_size}"
            f"Actual number of batches per epoch - Train: \
                {train_num_batches}, val: {val_num_batches}"
            f"Number of data samples in each batch - Train: {train_bt_size}, val: {val_bt_size}"
        )

        return train_bt_size, val_bt_size

    def create_normalization(self, train_data: pd.DataFrame) -> None:
        """writes train stats needed for normatization to disk.

        :param train_data: data used for training
        :type train_data: pd.DataFrame
        """
        train_stats = {
            "train_max": train_data.max().to_dict(),
            "train_min": train_data.min().to_dict(),
            "train_mean": train_data.mean(axis=0).to_dict(),
            "train_std": train_data.std(axis=0).to_dict(),
        }

        train_stats_path = self.file_prefix / "train_stats.json"
        with open(train_stats_path, "w") as fp:
            json.dump(train_stats, fp)

    def apply_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to data using the train stats on disk and the transformation_method
        in the configs

        :param data: data
        :type data: pd.DataFrame
        :raises ConfigsError: if transformation_method is not supported
        :return: normzalized data
        :rtype: pd.DataFrame
        """
        train_stats_path = self.file_prefix / "train_stats.json"
        with open(train_stats_path, "r") as f:
            train_stats = json.load(f)
            # print("train_stats", train_stats)

        transformation_method = self.configs["learning_algorithm"][
            "transformation_method"
        ]
        if transformation_method == "minmaxscale":
            train_max = pd.DataFrame(train_stats["train_max"], index=[1]).iloc[0]
            train_min = pd.DataFrame(train_stats["train_min"], index=[1]).iloc[0]
            print("data pre minmax", data.isna().values.any())

            data = (data - train_min) / (train_max - train_min)

            print("data post minmax", data.isna().values.any())
            print("data cols with nans", data.columns[data.isna().any()].tolist())

        elif transformation_method == "standard":
            train_mean = pd.DataFrame(train_stats["train_mean"], index=[1]).iloc[0]
            train_std = pd.DataFrame(train_stats["train_std"], index=[1]).iloc[0]

            data = (data - train_mean) / train_std

        else:
            raise ConfigsError(
                f"{transformation_method} is not a supported form of data normalization"
            )

        return data

    def write_metadata(self):
        """Write the metadata to a json file"""
        path = os.path.join(self.file_prefix, "metadata.json")
        metadata = {"wattile_version": wattile_version}
        with open(path, "w") as fp:
            json.dump(metadata, fp, indent=1)

    def main(self, train_df, val_df):
        """
        Main executable for prepping data for input to RNN model.

        This function is on it's way out. use train, validate, or predict.

        :param train_df: (DataFrame)
        :param val_df: (DataFrame)
        :return: None
        """
        use_case = self.configs["learning_algorithm"]["use_case"]
        if use_case == "train":
            self.train(train_df, val_df)

        elif use_case == "validation":
            self.validate(val_df)

        elif use_case == "prediction":
            return self.predict(val_df)

        else:
            raise ValueError

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        # Normalization transformation
        print("train_df.shape", train_df.shape)
        print("is nans in train_df", train_df.isna().values.any())
        self.create_normalization(train_df)
        train_data = self.apply_normalization(train_df)
        val_data = self.apply_normalization(val_df.copy())
        print("train_data.shape", train_data.shape)
        print("is nans in train_data", train_data.isna().values.any())

        # Put data into DataLoaders
        train_batch_size, val_batch_size = self.size_the_batches(train_data, val_data)
        train_loader = self.to_data_loader(train_data, train_batch_size, shuffle=True)
        val_loader = self.to_data_loader(val_data, val_batch_size, shuffle=True)

        self.run_training(train_loader, val_loader, val_df)
        self.write_metadata()

        # Create visualization
        if self.configs["data_output"]["plot_comparison"]:
            timeseries_comparison(self.configs, 0)

    def validate(self, val_df: pd.DataFrame) -> None:
        val_data = self.apply_normalization(val_df.copy())

        # only 1 batch during validation
        val_loader = self.to_data_loader(
            val_data, batch_size=val_data.shape[0], shuffle=False
        )

        self.run_validation(val_loader, val_df)

        # Create visualization
        if self.configs["data_output"]["plot_comparison"]:
            timeseries_comparison(self.configs, 0)

    def predict(self, val_df: pd.DataFrame) -> None:
        val_data = self.apply_normalization(val_df.copy())

        # only 1 batch during prediction
        val_loader = self.to_data_loader(
            val_data, batch_size=val_data.shape[0], shuffle=False
        )

        return self.run_prediction(val_loader, val_df)
