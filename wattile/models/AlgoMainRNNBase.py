import json
import logging
import os
from abc import ABC
from pathlib import Path

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from wattile.error import ConfigsError
from wattile.util import factors
from wattile.visualization import timeseries_comparison

global file_prefix
logger = logging.getLogger(str(os.getpid()))


class AlgoMainRNNBase(ABC):
    def __init__(self, configs):
        self.file_prefix = Path(configs["exp_dir"])
        self.file_prefix.mkdir(parents=True, exist_ok=True)

    def size_the_batches(
        self,
        train_data,
        val_data,
        tr_desired_batch_size,
        te_desired_batch_size,
        configs,
    ):
        """
        Compute the batch sizes for training and val set

        :param train_data: (DataFrame)
        :param val_data: (DataFrame)
        :param tr_desired_batch_size: (int)
        :param te_desired_batch_size: (int)
        :return:
        """

        if configs["use_case"] == "train":
            # Find factors of the length of train and val df's
            # and pick the closest one to the requested batch sizes
            train_bth = factors(train_data.shape[0])
            train_num_batches = min(
                train_bth, key=lambda x: abs(x - tr_desired_batch_size)
            )
            train_bt_size = int(train_data.shape[0] / train_num_batches)

            val_bth = factors(val_data.shape[0])
            val_num_batches = min(val_bth, key=lambda x: abs(x - te_desired_batch_size))
            val_bt_size = int(val_data.shape[0] / val_num_batches)

            num_train_data = train_data.shape[0]

            # logger.info(
            #     "Train size: {}, val size: {}, split {}%:{}%".format(
            #         train_data.shape[0], val_data.shape[0], train_ratio, val_ratio
            #     )
            # )
            logger.info("Available train batch factors: {}".format(sorted(train_bth)))
            logger.info(
                "Requested number of batches per epoch - Train: {}, val: {}".format(
                    tr_desired_batch_size, te_desired_batch_size
                )
            )
            logger.info(
                "Actual number of batches per epoch - Train: {}, val: {}".format(
                    train_num_batches, val_num_batches
                )
            )
            logger.info(
                "Number of data samples in each batch - Train: {}, val: {}".format(
                    train_bt_size, val_bt_size
                )
            )
        else:
            val_bt_size = val_data.shape[0]
            train_bt_size = 0
            num_train_data = 0

        return train_bt_size, val_bt_size, num_train_data

    def data_transform(self, train_data, val_data, transformation_method, run_train):
        """
        Normalize the training and val data according to a user-defined criteria

        :param train_data: DataFrame
        :param val_data: DataFrame
        :param transformation_method: str
        :param run_train: Boolean
        :return:
        """
        if run_train:
            # For the result de-normalization purpose, saving the max and min values of the
            # STM_Xcel_Meter columns
            train_stats = {}
            train_stats["train_max"] = train_data.max().to_dict()
            train_stats["train_min"] = train_data.min().to_dict()
            train_stats["train_mean"] = train_data.mean(axis=0).to_dict()
            train_stats["train_std"] = train_data.std(axis=0).to_dict()
            path = os.path.join(self.file_prefix, "train_stats.json")
            with open(path, "w") as fp:
                json.dump(train_stats, fp)

            if transformation_method == "minmaxscale":
                train_data = (train_data - train_data.min()) / (
                    train_data.max() - train_data.min()
                )

            elif transformation_method == "standard":
                train_data = (train_data - train_data.mean(axis=0)) / train_data.std(
                    axis=0
                )

            else:
                raise ConfigsError(
                    "{} is not a supported form of data normalization".format(
                        transformation_method
                    )
                )

        # Reading back the train stats for normalizing val data w.r.t to train data
        file_loc = os.path.join(self.file_prefix, "train_stats.json")
        with open(file_loc, "r") as f:
            train_stats = json.load(f)

        # get statistics for training data
        train_max = pd.DataFrame(train_stats["train_max"], index=[1]).iloc[0]
        train_min = pd.DataFrame(train_stats["train_min"], index=[1]).iloc[0]
        train_mean = pd.DataFrame(train_stats["train_mean"], index=[1]).iloc[0]
        train_std = pd.DataFrame(train_stats["train_std"], index=[1]).iloc[0]

        # Normalize data
        if transformation_method == "minmaxscale":
            val_data = (val_data - train_min) / (train_max - train_min)
        elif transformation_method == "standard":
            val_data = (val_data - train_mean) / train_std
        else:
            raise ConfigsError(
                "{} is not a supported form of data normalization".format(
                    transformation_method
                )
            )

        return train_data, val_data

    def plot_mid_train_stats(self):
        data = pd.read_hdf(
            os.path.join(self.file_prefix, "mid_train_error_stats.h5"), key="df"
        )
        data.plot(x="n_iter", subplots=True)

    def main(self, train_df, val_df, configs):
        """
        Main executable for prepping data for input to RNN model.

        :param train_df: (DataFrame)
        :param val_df: (DataFrame)
        :param configs: (Dictionary)
        :return: None
        """

        transformation_method = configs["transformation_method"]
        run_train = configs["use_case"] == "train"
        num_epochs = configs["num_epochs"]
        run_resume = configs["run_resume"]
        tr_desired_batch_size = configs["train_batch_size"]
        te_desired_batch_size = configs["val_batch_size"]

        # Create writer object for TensorBoard
        writer_path = str(self.file_prefix)
        writer = SummaryWriter(writer_path)
        logger.info("Writer path: {}".format(writer_path))

        # Reset DataFrame index
        if run_train:
            train_data = train_df.copy(deep=True)
            train_data.reset_index(drop=True, inplace=True)
            train_df.reset_index(drop=True, inplace=True)
        else:
            train_data = train_df
        val_data = val_df.copy(deep=True)
        val_data.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        # Normalization transformation
        train_data, val_data = self.data_transform(
            train_data, val_data, transformation_method, run_train
        )
        logger.info(
            "Data transformed using {} as transformation method".format(
                transformation_method
            )
        )

        # Size the batches
        (train_batch_size, val_batch_size, num_train_data,) = self.size_the_batches(
            train_data, val_data, tr_desired_batch_size, te_desired_batch_size, configs
        )

        # Already did sequential padding: Convert to iterable dataset (DataLoaders)
        if configs["train_val_split"] == "Random":
            train_loader, val_loader = self.data_iterable_random(
                train_data,
                val_data,
                run_train,
                train_batch_size,
                val_batch_size,
                configs,
            )
        logger.info("Data converted to iterable dataset")

        if configs["use_case"] == "train":
            self.run_training(
                train_loader,
                val_loader,
                val_df,
                num_epochs,
                run_resume,
                writer,
                transformation_method,
                configs,
                train_batch_size,
                val_batch_size,
                configs["window"] + 1,
                num_train_data,
            )

            # When training is done, wrap up the tensorboard files
            writer.flush()
            writer.close()

            # Create visualization
            if configs["plot_comparison"]:
                timeseries_comparison(configs)

        elif configs["use_case"] == "validation":
            self.run_validation(
                val_loader,
                val_df,
                writer,
                transformation_method,
                configs,
                val_batch_size,
                configs["window"] + 1,
            )

        elif configs["use_case"] == "prediction":
            return self.run_prediction(
                val_loader,
                val_df,
                writer,
                transformation_method,
                configs,
                val_batch_size,
                configs["window"] + 1,
            )

        else:
            raise ValueError
