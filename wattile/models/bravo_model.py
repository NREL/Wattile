import csv
import json
import logging
import os
import pathlib
import timeit

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.utils.data as data_utils
from psutil import virtual_memory
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

from wattile.error import ConfigsError
from wattile.models.AlgoMainRNNBase import AlgoMainRNNBase
from wattile.models.utils import init_model, load_model, save_model

logger = logging.getLogger(str(os.getpid()))


class BravoModel(AlgoMainRNNBase):
    def data_iterable_random(
        self, train_data, val_data, run_train, train_batch_size, val_batch_size
    ):
        """
        Converts train and val data to torch data types (used only if splitting training and val set
        randomly)

        :param train_data: (DataFrame)
        :param val_data: (DataFrame)
        :param run_train: (Boolean)
        :param train_batch_size: (int)
        :param val_batch_size: (int)
        :return:
        """

        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.configs["device"] = device
        logger.info("Training on {}".format(device))

        # # Set default floating point precision
        # if torch.cuda.is_available():
        #     default_dtype = torch.cuda.FloatTensor
        #     torch.set_default_tensor_type(default_dtype)
        # else:
        #     default_dtype = torch.FloatTensor
        #     torch.set_default_tensor_type(default_dtype)

        if run_train:
            # Define input feature matrix
            X_train = train_data.drop(
                train_data.filter(like=self.configs["target_var"], axis=1).columns,
                axis=1,
            ).values.astype(dtype="float32")

            # Output variable (batch*72), then (batch*(72*num of qs))
            y_train = train_data[
                train_data.filter(like=self.configs["target_var"], axis=1).columns
            ].values.astype(dtype="float32")
            y_train = np.tile(y_train, len(self.configs["qs"]))

            # Convert to iterable tensors
            train_feat_tensor = torch.from_numpy(X_train).to(device)
            train_target_tensor = torch.from_numpy(y_train).to(device)
            train = data_utils.TensorDataset(train_feat_tensor, train_target_tensor)
            train_loader = data_utils.DataLoader(
                train, batch_size=train_batch_size, shuffle=True
            )  # Contains features and targets

        else:
            train_loader = []

        # Do the same as above, but for the val set
        # X_val = val_data.drop(self.configs['target_var'], axis=1).values.astype(dtype='float32')
        X_val = val_data.drop(
            val_data.filter(like=self.configs["target_var"], axis=1).columns, axis=1
        ).values.astype(dtype="float32")

        y_val = val_data[
            val_data.filter(like=self.configs["target_var"], axis=1).columns
        ].values.astype(dtype="float32")
        y_val = np.tile(y_val, len(self.configs["qs"]))

        val_feat_tensor = torch.from_numpy(X_val).to(device)
        val_target_tensor = torch.from_numpy(y_val).to(device)

        val = data_utils.TensorDataset(val_feat_tensor, val_target_tensor)
        val_loader = DataLoader(dataset=val, batch_size=val_batch_size, shuffle=True)

        return train_loader, val_loader

    def pinball_np(self, output, target):
        num_future_time_instances = (
            self.configs["S2S_stagger"]["initial_num"]
            + self.configs["S2S_stagger"]["secondary_num"]
        )
        resid = target - output
        tau = np.repeat(self.configs["qs"], num_future_time_instances)
        alpha = self.configs["smoothing_alpha"]
        log_term = np.zeros_like(resid)
        log_term[resid < 0] = np.log(1 + np.exp(resid[resid < 0] / alpha)) - (
            resid[resid < 0] / alpha
        )
        log_term[resid >= 0] = np.log(1 + np.exp(-resid[resid >= 0] / alpha))
        loss = resid * tau + alpha * log_term

        return loss

    def quantile_loss(self, output, target, device):
        """
        Computes loss for quantile methods.

        :param output: (Tensor)
        :param target: (Tensor)
        :return: (Tensor) Loss for this study (single number)
        """

        num_future_time_instances = (
            self.configs["S2S_stagger"]["initial_num"]
            + self.configs["S2S_stagger"]["secondary_num"]
        )
        resid = target - output

        # logger.info("Resid device: ".format(resid.device))

        # tau = np.repeat(self.configs["qs"], num_future_time_instances)
        tau = torch.tensor(self.configs["qs"], device=device).repeat_interleave(
            num_future_time_instances
        )

        alpha = self.configs["smoothing_alpha"]
        log_term = torch.zeros_like(resid, device=device)
        log_term[resid < 0] = torch.log(1 + torch.exp(resid[resid < 0] / alpha)) - (
            resid[resid < 0] / alpha
        )
        log_term[resid >= 0] = torch.log(1 + torch.exp(-resid[resid >= 0] / alpha))
        loss = resid * tau + alpha * log_term
        loss = torch.mean(torch.mean(loss, 0))

        # Extra statistics to return optionally
        # stats = [resid.data.numpy().min(), resid.data.numpy().max()]

        # See histogram of residuals
        # graph = pd.DataFrame(resid.data.numpy()).plot(
        #     kind="hist", alpha=0.5, bins=50, ec="black", stacked=True
        # )

        return loss

    def test_processing(  # noqa: C901 TODO: remove noqa
        self,
        val_df,
        val_loader,
        model,
        seq_dim,
        input_dim,
        val_batch_size,
        transformation_method,
        last_run,
        device,
    ):
        """
        Process the val set and report error statistics.

        :param val_df: (DataFrame)
        :param val_loader: (DataLoader)
        :param model: (Pytorch model)
        :param seq_dim: ()
        :param input_dim:
        :param val_batch_size:
        :param transformation_method:
        :return:
        """
        with torch.no_grad():
            # Plug the val set into the model
            num_timestamps = (
                self.configs["S2S_stagger"]["initial_num"]
                + self.configs["S2S_stagger"]["secondary_num"]
            )
            model.eval()
            preds = []
            targets = []
            for i, (feats, values) in enumerate(val_loader):
                features = Variable(feats.view(-1, seq_dim, input_dim))
                outputs = model(features)
                # preds.append(outputs.data.numpy())
                # targets.append(values.data.numpy())
                preds.append(outputs.cpu().numpy())
                targets.append(values.cpu().numpy())

            # (Normalized Data) Concatenate the predictions and targets for the whole val set
            semifinal_preds = np.concatenate(preds)
            semifinal_targs = np.concatenate(targets)

            # Calculate pinball loss (done on normalized data)
            loss = self.pinball_np(semifinal_preds, semifinal_targs)
            pinball_loss = np.mean(np.mean(loss, 0))

            # Loading the training data stats for de-normalization purpose
            file_loc = os.path.join(self.file_prefix, "train_stats.json")
            with open(file_loc, "r") as f:
                train_stats = json.load(f)

            # Get normalization statistics
            train_max = pd.DataFrame(train_stats["train_max"], index=[1]).iloc[0]
            train_min = pd.DataFrame(train_stats["train_min"], index=[1]).iloc[0]
            train_mean = pd.DataFrame(train_stats["train_mean"], index=[1]).iloc[0]
            train_std = pd.DataFrame(train_stats["train_std"], index=[1]).iloc[0]

            # Do de-normalization process on predictions and targets from val set
            if transformation_method == "minmaxscale":
                maxs = np.tile(
                    train_max[
                        train_max.filter(like=self.configs["target_var"], axis=0).index
                    ].values,
                    len(self.configs["qs"]),
                )
                mins = np.tile(
                    train_min[
                        train_min.filter(like=self.configs["target_var"], axis=0).index
                    ].values,
                    len(self.configs["qs"]),
                )
                final_preds = (
                    (maxs - mins) * semifinal_preds
                ) + mins  # (batch x (num time predictions * num q's)))
                final_targs = (
                    (maxs - mins) * semifinal_targs
                ) + mins  # (batch x (num time predictions * num q's)))
            elif transformation_method == "standard":
                stds = np.tile(
                    train_std[
                        train_std.filter(like=self.configs["target_var"], axis=0).index
                    ].values,
                    len(self.configs["qs"]),
                )
                means = np.tile(
                    train_mean[
                        train_mean.filter(like=self.configs["target_var"], axis=0).index
                    ].values,
                    len(self.configs["qs"]),
                )
                final_preds = (semifinal_preds * stds) + means
                final_targs = (semifinal_targs * stds) + means
            else:
                raise ConfigsError(
                    "{} is not a supported form of data normalization".format(
                        transformation_method
                    )
                )

            # (De-Normalized Data) Assign target and output variables
            target = final_targs
            output = final_preds

            # Do quantile-related (q != 0.5) error statistics
            # QS (single point)
            loss = self.pinball_np(output, target)
            QS = loss.mean()
            # PICP (single point for each bound)
            target_1D = target[:, range(num_timestamps)]
            bounds = np.zeros((target.shape[0], int(len(self.configs["qs"]) / 2)))
            PINC = []
            split_arrays = np.split(output, len(self.configs["qs"]), axis=1)
            for i, q in enumerate(self.configs["qs"]):
                if q == 0.5:
                    break
                filtered_low = split_arrays[i]
                filtered_high = split_arrays[-(i + 1)]
                low_check = filtered_low < target_1D
                high_check = filtered_high > target_1D
                check_across_time = np.logical_and(low_check, high_check)
                time_averaged = check_across_time.mean(axis=1)
                bounds[:, i] = time_averaged
                # Calculate theoretical PI
                PINC.append(self.configs["qs"][-(i + 1)] - self.configs["qs"][i])
            PINC = np.array(PINC)
            PICP = bounds.mean(axis=0)
            # ACE (single point)
            ACE = np.sum(np.abs(PICP - PINC))
            # IS (single point)
            ISs = []
            # Iterate through Prediction Intervals (pair of quantiles)
            for i, q in enumerate(self.configs["qs"]):
                if q == 0.5:
                    break
                low = split_arrays[i]  # (batches * time) for a single quantile
                high = split_arrays[-(i + 1)]  # (batches * time) for a single quantile
                x = target_1D  # (batches * time) for nominal results
                alph = 1 - (
                    self.configs["qs"][-(i + 1)] - self.configs["qs"][i]
                )  # Single float
                IS = (
                    (high - low)
                    + (2 / alph) * (low - x) * (x < low)
                    + (2 / alph) * (x - high) * (x > high)
                )  # (batches * time) for a single quantile
                IS = IS.mean(axis=1)  # (batches * 1)
                ISs.append(IS)
            IS = np.concatenate(ISs).mean()  # Mean of all values in ISs

            # Compare theoretical and actual Q's
            temp_actual = []
            for i, q in enumerate(self.configs["qs"]):
                act_prob = (split_arrays[i] > target_1D).mean()
                temp_actual.append(act_prob)
            Q_vals = pd.DataFrame()
            Q_vals["q_requested"] = self.configs["qs"]
            Q_vals["q_actual"] = temp_actual

            # Do quantile-related (q == 0.5) error statistics
            # Get the predictions for the q=0.5 case
            final_preds_median = np.split(final_preds, len(self.configs["qs"]), axis=1)[
                int(len(self.configs["qs"]) / 2)
            ]
            output = pd.DataFrame(final_preds_median).values.squeeze()
            target = np.split(final_targs, len(self.configs["qs"]), axis=1)[
                int(len(self.configs["qs"]) / 2)
            ]
            # Set "Number of adjustable model parameters" for each type of error statistic
            p_nmbe = 0
            p_cvrmse = 1
            # Calculate different error metrics
            rmse = np.sqrt(np.mean((output - target) ** 2))
            nmbe = (
                (1 / (np.mean(target)))
                * (np.sum(target - output))
                / (len(target) - p_nmbe)
            )
            cvrmse = (1 / (np.mean(target))) * np.sqrt(
                np.sum((target - output) ** 2) / (len(target) - p_cvrmse)
            )
            gof = (np.sqrt(2) / 2) * np.sqrt(cvrmse**2 + nmbe**2)

            # If this is the last val run of training,
            # get histogram data of residuals for each quantile (use normalized data)
            if last_run:
                resid = semifinal_targs - semifinal_preds
                split_arrays = np.split(resid, len(self.configs["qs"]), axis=1)
                hist_data = pd.DataFrame()
                for i, q in enumerate(self.configs["qs"]):
                    tester = np.histogram(split_arrays[i], bins=200)
                    y_vals = tester[0]
                    x_vals = 0.5 * (tester[1][1:] + tester[1][:-1])
                    hist_data["{}_x".format(q)] = x_vals
                    hist_data["{}_y".format(q)] = y_vals
            else:
                hist_data = []

            # Add different error statistics to a dictionary
            errors = {
                "pinball_loss": pinball_loss,
                "rmse": rmse,
                "nmbe": nmbe,
                "cvrmse": cvrmse,
                "gof": gof,
                "qs": QS,
                "ace": ACE,
                "is": IS,
            }

        return final_preds, errors, target, Q_vals

    def run_training(  # noqa: C901 TODO: remove noqa
        self,
        train_loader,
        val_loader,
        val_df,
        num_epochs,
        run_resume,
        writer,
        transformation_method,
        train_batch_size,
        val_batch_size,
        seq_dim,
        num_train_data,
    ):
        """
        Contains main training process for RNN

        :param train_loader: (Pytorch DataLoader)
        :param val_loader: (Pytorch DataLoader)
        :param val_df: (DataFrame)
        :param num_epochs: (int)
        :param run_resume: (Boolean)
        :param writer: (SummaryWriter object)
        :param transformation_method: (str)
        :param train_batch_size: (Float)
        :param val_batch_size: (Float)
        :param seq_dim: (Int)
        :param num_train_data: (Float)
        :return: None
        """

        weight_decay = float(self.configs["weight_decay"])
        input_dim = self.configs["input_dim"]

        # Write the configurations used for this training process to a json file
        path = os.path.join(self.file_prefix, "configs.json")
        with open(path, "w") as fp:
            json.dump(self.configs, fp, indent=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initializing lists to store losses over epochs:
        train_loss = []
        train_iter = []
        # val_loss = []
        val_iter = []
        # val_rmse = []

        # TODO: make resumed model and new model use the same number of epochs
        if run_resume:
            model, resume_num_epoch, resume_n_iter = load_model(self.configs)
            epoch_range = np.arange(resume_num_epoch + 1, num_epochs + 1)

            logger.info(f"Model loaded from: {self.file_prefix}")

        else:
            model = init_model(self.configs)
            epoch_range = np.arange(num_epochs)

            logger.info(
                f"A new {self.configs['arch_type_variant']} {self.configs['arch_type']} "
                "model instantiated"
            )

        # Move model and data to GPU, if availiable
        model.to(device)

        # Instantiate Optimizer Class
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.configs["lr_config"]["base"],
            weight_decay=weight_decay,
        )

        # Set up learning rate scheduler
        if not self.configs["lr_config"]["schedule"]:
            pass
        elif (
            self.configs["lr_config"]["schedule"]
            and self.configs["lr_config"]["type"] == "performance"
        ):
            # Patience (for our case) is # of iterations, not epochs,
            # but self.configs specification is num epochs
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.configs["lr_config"]["factor"],
                min_lr=self.configs["lr_config"]["min"],
                patience=int(
                    self.configs["lr_config"]["patience"]
                    * (num_train_data / train_batch_size)
                ),
                verbose=True,
            )
        elif (
            self.configs["lr_config"]["schedule"]
            and self.configs["lr_config"]["type"] == "absolute"
        ):
            # scheduler = StepLR(
            #     optimizer,
            #     step_size=int(
            #         self.configs["lr_config"]["step_size"] * (num_train_data / train_batch_size)
            #     ),
            #     gamma=self.configs["lr_config"]["factor"],
            # )
            pass
        else:
            raise ConfigsError(
                "{} is not a supported method of LR scheduling".format(
                    self.configs["lr_config"]["type"]
                )
            )

        # Computing platform
        num_logical_processors = psutil.cpu_count(logical=True)
        num_cores = psutil.cpu_count(logical=False)
        mem = virtual_memory()
        mem = {
            "total": mem.total / 10**9,
            "available": mem.available / 10**9,
            "percent": mem.percent,
            "used": mem.used / 10**9,
            "free": mem.free / 10**9,
        }
        logger.info("Number of cores available: {}".format(num_cores))
        logger.info(
            "Number of logical processors available: {}".format(num_logical_processors)
        )
        logger.info("Initial memory statistics (GB): {}".format(mem))

        if len(epoch_range) == 0:
            epoch = resume_num_epoch + 1
            logger.info(
                f"the previously saved model was at epoch= {resume_num_epoch}, which is same as "
                "num_epochs. So, not training"
            )

        if run_resume:
            n_iter = resume_n_iter
            epoch_num = resume_num_epoch
        else:
            n_iter = 0
            epoch_num = 1

        # Start training timer
        train_start_time = timeit.default_timer()

        # Initialize re-trainable matrix
        # train_y_at_t = torch.zeros(train_batch_size, seq_dim, 1)  # 960 x 5 x 1

        mid_train_error_stats = pd.DataFrame()

        logger.info("Starting to train the model for {} epochs!".format(num_epochs))

        # Loop through epochs
        for epoch in epoch_range:

            # Do manual learning rate scheduling, if requested
            if (
                self.configs["lr_config"]["schedule"]
                and self.configs["lr_config"]["type"] == "absolute"
                and epoch_num % self.configs["lr_config"]["step_size"] == 0
            ):
                for param_group in optimizer.param_groups:
                    old_lr = param_group["lr"]
                    param_group["lr"] = (
                        param_group["lr"] * self.configs["lr_config"]["factor"]
                    )
                    new_lr = param_group["lr"]
                logger.info(
                    "Changing learning rate from {} to {}".format(old_lr, new_lr)
                )

            # This loop returns elements from the dataset batch by batch.
            # Contains features AND targets
            for i, (feats, values) in enumerate(train_loader):
                model.train()
                # feats: (# samples in batch) x (unrolled features) (tensor)
                # values: (# samples in batch) x (Output dimension) (tensor)
                time1 = timeit.default_timer()

                # (batches, timesteps, features)
                features = Variable(feats.view(-1, seq_dim, input_dim))
                target = Variable(values)  # size: batch size

                time2 = timeit.default_timer()

                # Clear gradients w.r.t. parameters (from previous epoch). Same as model.zero_grad()
                optimizer.zero_grad()

                # # Get memory statistics
                # if n_iter % self.configs["eval_frequency"] == 0:
                #     mem = virtual_memory()
                #     mem = {"total": mem.total / 10 ** 9, "available": mem.available / 10 ** 9,
                #            "used": mem.used / 10 ** 9, "free": mem.free / 10 ** 9}
                #     writer.add_scalars("Memory_GB", mem, n_iter)

                # FORWARD PASS to get output/logits.
                outputs = model(features)

                time3 = timeit.default_timer()

                # Calculate Loss
                loss = self.quantile_loss(outputs, target, device)

                # resid_stats.append(stats)
                # train_loss.append(loss.data.item())
                train_loss.append(loss)
                train_iter.append(n_iter)

                # Print to terminal and save training loss
                # writer.add_scalars("Loss", {'Train': loss.data.item()}, n_iter)
                writer.add_scalars("Loss", {"Train": loss}, n_iter)
                time4 = timeit.default_timer()

                # Does backpropogation and gets gradients, (the weights and bias).
                # Create computational graph
                loss.backward()

                time5 = timeit.default_timer()

                if (
                    self.configs["lr_config"]["schedule"]
                    and self.configs["lr_config"]["type"] == "performance"
                ):
                    scheduler.step(loss)

                # Updating the weights/parameters. Clear computational graph.
                optimizer.step()

                # Each iteration is one batch
                n_iter += 1

                # Compute time per iteration
                time6 = timeit.default_timer()
                writer.add_scalars(
                    "Iteration_time",
                    {
                        "Package_variables": time2 - time1,
                        "Evaluate_model": time3 - time2,
                        "Calc_loss": time4 - time3,
                        "Backprop": time5 - time4,
                        "Step": time6 - time5,
                    },
                    n_iter,
                )

                # Save the model every ___ iterations
                if n_iter % self.configs["eval_frequency"] == 0:
                    filepath = os.path.join(self.file_prefix, "torch_model")
                    save_model(model, epoch, n_iter, filepath)

                # Do a val batch every ___ iterations
                if n_iter % self.configs["eval_frequency"] == 0:
                    # Evaluate val set
                    (predictions, errors, measured, Q_vals,) = self.test_processing(
                        val_df,
                        val_loader,
                        model,
                        seq_dim,
                        input_dim,
                        val_batch_size,
                        transformation_method,
                        False,
                        device,
                    )

                    temp_holder = errors
                    temp_holder.update({"n_iter": n_iter, "epoch": epoch})
                    mid_train_error_stats = mid_train_error_stats.append(
                        temp_holder, ignore_index=True
                    )

                    val_iter.append(n_iter)
                    # val_loss.append(errors['mse_loss'])
                    # val_rmse.append(errors['rmse'])
                    writer.add_scalars("Loss", {"val": errors["pinball_loss"]}, n_iter)

                    # Save the final predictions to a file
                    # pd.DataFrame(predictions).to_hdf(
                    #     os.path.join(self.file_prefix, "predictions.h5"), key="df", mode="w"
                    # )
                    # pd.DataFrame(measured).to_hdf(
                    #     os.path.join(self.file_prefix, "measured.h5"), key="df", mode="w"
                    # )

                    # Save the QQ information to a file
                    # Q_vals.to_hdf(
                    #     os.path.join(self.file_prefix, "QQ_data.h5"), key="df", mode="w"
                    # )

                    # Add parody plot to TensorBoard
                    # fig2, ax2 = plt.subplots()
                    # ax2.scatter(predictions, val_df[self.configs["target_var"]], s=5, alpha=0.3)
                    # strait_line = np.linspace(
                    #     min(min(predictions), min(val_df[self.configs["target_var"]])),
                    #     max(max(predictions), max(val_df[self.configs["target_var"]])),
                    #     5,
                    # )
                    # ax2.plot(strait_line, strait_line, c="k")
                    # ax2.set_xlabel("Predicted")
                    # ax2.set_ylabel("Observed")
                    # ax2.axhline(y=0, color="k")
                    # ax2.axvline(x=0, color="k")
                    # ax2.axis("equal")
                    # writer.add_figure("Parody", fig2, n_iter)

                    # Add QQ plot to TensorBoard
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(Q_vals["q_requested"], Q_vals["q_actual"], s=20)
                    ax2.plot([0, 1], [0, 1], c="k", alpha=0.5)
                    ax2.set_xlabel("Requested")
                    ax2.set_ylabel("Actual")
                    ax2.set_xlim(left=0, right=1)
                    ax2.set_ylim(bottom=0, top=1)
                    writer.add_figure("QQ", fig2, n_iter)

                    # # Write information about CPU usage to tensorboard
                    # percentages = dict(zip(
                    #     list(np.arange(1, num_logical_processors + 1).astype(str)),
                    #     psutil.cpu_percent(interval=None, percpu=True)
                    # ))
                    # writer.add_scalars("CPU_utilization", percentages, n_iter)

                    logger.info(
                        "Epoch: {} Iteration: {}. Train_loss: {}. val_loss: {}, LR: {}".format(
                            epoch_num,
                            n_iter,
                            loss.data.item(),
                            errors["pinball_loss"],
                            optimizer.param_groups[0]["lr"],
                        )
                    )
            epoch_num += 1

        # Once model training is done, save the current model state
        filepath = os.path.join(self.file_prefix, "torch_model")
        save_model(model, epoch, n_iter, filepath)

        # Once model is done training, process a final val set
        predictions, errors, measured, Q_vals = self.test_processing(
            val_df,
            val_loader,
            model,
            seq_dim,
            input_dim,
            val_batch_size,
            transformation_method,
            True,
            device,
        )

        # Save the residual distribution to a file
        # hist_data.to_hdf(
        #     os.path.join(self.file_prefix, "residual_distribution.h5"), key='df', mode='w'
        # )

        # Save the final predictions to a file
        pd.DataFrame(predictions).to_hdf(
            os.path.join(self.file_prefix, "predictions.h5"), key="df", mode="w"
        )
        pd.DataFrame(measured).to_hdf(
            os.path.join(self.file_prefix, "measured.h5"), key="df", mode="w"
        )

        # Save the QQ information to a file
        Q_vals.to_hdf(
            os.path.join(self.file_prefix, "QQ_data_Train.h5"), key="df", mode="w"
        )

        # Save the mid-train error statistics to a file
        mid_train_error_stats.to_hdf(
            os.path.join(self.file_prefix, "mid_train_error_stats.h5"),
            key="df",
            mode="w",
        )

        # End training timer
        train_end_time = timeit.default_timer()
        train_time = train_end_time - train_start_time

        # If a training history csv file does not exist, make one
        if not pathlib.Path("Training_history.csv").exists():
            with open(r"Training_history.csv", "a") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(
                    [
                        "File Path",
                        "RMSE",
                        "CV(RMSE)",
                        "NMBE",
                        "GOF",
                        "QS",
                        "ACE",
                        "IS",
                        "Train time",
                    ]
                )
        # Save the errors statistics to a file once everything is done
        with open(r"Training_history.csv", "a") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(
                [
                    self.file_prefix,
                    errors["rmse"],
                    errors["cvrmse"],
                    errors["nmbe"],
                    errors["gof"],
                    errors["qs"],
                    errors["ace"],
                    errors["is"],
                    train_time,
                ]
            )

        # Write error statistics to a local json file
        errors["train_time"] = train_time
        for k in errors:
            errors[k] = str(errors[k])
        path = os.path.join(self.file_prefix, "error_stats_train.json")
        with open(path, "w") as fp:
            json.dump(errors, fp, indent=1)

    def run_validation(
        self,
        val_loader,
        val_df,
        writer,
        transformation_method,
        val_batch_size,
        seq_dim,
    ):
        """run prediction

        :param val_loader: (Pytorch DataLoader)
        :param val_df: (DataFrame)
        :param writer: (SummaryWriter object)
        :param transformation_method: (str)
        :param val_batch_size: (Float)
        :param seq_dim: (Int)
        :return: None
        """
        model, _, _ = load_model(self.configs)
        logger.info("Loaded model from file, given run_train=False\n")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictions, errors, measured, Q_vals = self.test_processing(
            val_df,
            val_loader,
            model,
            seq_dim,
            self.configs["input_dim"],
            val_batch_size,
            transformation_method,
            False,
            device,
        )

        # Save the QQ information to a file
        Q_vals.to_hdf(
            os.path.join(self.file_prefix, "QQ_data_Test.h5"), key="df", mode="w"
        )

        # Save the errors to a file
        for k in errors:
            errors[k] = str(errors[k])
        path = os.path.join(self.file_prefix, "error_stats_test.json")
        with open(path, "w") as fp:
            json.dump(errors, fp, indent=1)

        # If a training history csv file does not exist, make one
        train_history_path = self.file_prefix / "Testing_history.csv"
        if not train_history_path.exists():
            with open(train_history_path, "a") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(
                    ["File Path", "RMSE", "CV(RMSE)", "NMBE", "GOF", "QS", "ACE", "IS"]
                )

        # Save the errors statistics to a central results csv once everything is done
        with open(train_history_path, "a") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(
                [
                    self.file_prefix,
                    errors["rmse"],
                    errors["cvrmse"],
                    errors["nmbe"],
                    errors["gof"],
                    errors["qs"],
                    errors["ace"],
                    errors["is"],
                ]
            )

        if self.configs.get("plot_results", True):
            self._plot_results(measured, predictions)

    def run_prediction(
        self,
        val_loader,
        val_df,
        writer,
        transformation_method,
        val_batch_size,
        seq_dim,
    ):
        model, _, _ = load_model(self.configs)
        model.eval()

        logger.info("Loaded model from file, given run_train=False\n")

        with torch.no_grad():
            preds = []
            for (feats, v) in val_loader:
                features = Variable(feats.view(-1, seq_dim, self.configs["input_dim"]))
                outputs = model(features)
                preds.append(outputs.cpu().numpy())

            file_path = pathlib.Path(self.configs["exp_dir"]) / "train_stats.json"
            with open(file_path, "r") as f:
                train_stats = json.load(f)

            semifinal_preds = np.concatenate(preds)

            # Get normalization statistics
            train_max = pd.DataFrame(train_stats["train_max"], index=[1]).iloc[0]
            train_min = pd.DataFrame(train_stats["train_min"], index=[1]).iloc[0]
            train_mean = pd.DataFrame(train_stats["train_mean"], index=[1]).iloc[0]
            train_std = pd.DataFrame(train_stats["train_std"], index=[1]).iloc[0]

            # Do de-normalization process on predictions and targets from val set
            if transformation_method == "minmaxscale":
                maxs = np.tile(
                    train_max[
                        train_max.filter(like=self.configs["target_var"], axis=0).index
                    ].values,
                    len(self.configs["qs"]),
                )
                mins = np.tile(
                    train_min[
                        train_min.filter(like=self.configs["target_var"], axis=0).index
                    ].values,
                    len(self.configs["qs"]),
                )
                final_preds = (
                    (maxs - mins) * semifinal_preds
                ) + mins  # (batch x (num time predictions * num q's)))
            elif transformation_method == "standard":
                stds = np.tile(
                    train_std[
                        train_std.filter(like=self.configs["target_var"], axis=0).index
                    ].values,
                    len(self.configs["qs"]),
                )
                means = np.tile(
                    train_mean[
                        train_mean.filter(like=self.configs["target_var"], axis=0).index
                    ].values,
                    len(self.configs["qs"]),
                )
                final_preds = (semifinal_preds * stds) + means
            else:
                raise ConfigsError(
                    "{} is not a supported form of data normalization".format(
                        transformation_method
                    )
                )

        num_timestamps = (
            self.configs["S2S_stagger"]["initial_num"]
            + self.configs["S2S_stagger"]["secondary_num"]
        )
        final_preds = np.array(final_preds)
        final_preds = final_preds.reshape(
            (final_preds.shape[0], len(self.configs["qs"]), num_timestamps)
        )

        return final_preds

    def _plot_results(self, measured, predictions):
        """
        Plot some stats about the predictions
        """  # # Plotting
        if self.configs["test_method"] == "external":
            file = os.path.join(
                self.configs["data_dir"],
                self.configs["building"],
                "{}_external_test.h5".format(self.configs["target_var"]),
            )
            test_data = pd.read_hdf(file, key="df")
        else:
            test_data = pd.read_hdf(
                os.path.join(self.file_prefix, "internal_test.h5"), key="df"
            )

        num_timestamps = (
            self.configs["S2S_stagger"]["initial_num"]
            + self.configs["S2S_stagger"]["secondary_num"]
        )
        data = np.array(predictions)
        data = data.reshape((data.shape[0], len(self.configs["qs"]), num_timestamps))

        # Plotting the test set with ALL of the sequence forecasts
        fig, ax1 = plt.subplots()
        plt.rc("font", family="serif")
        # for j in range(0, test_data.shape[0]-1):
        # for j in range(1515, 1528):
        for j in range(1, 100):
            time_index = pd.date_range(
                start=test_data.index[j],
                periods=self.configs["S2S_stagger"]["initial_num"],
                freq="{}min".format(self.configs["resample_freq"]),
            )
            ax1.plot(time_index, measured[j, :], color="black", lw=1, zorder=5)
            if j == 70:
                ax1.plot(
                    time_index,
                    measured[j, :],
                    label="Load",
                    color="black",
                    lw=1,
                    zorder=5,
                )
                ax1.plot(
                    time_index,
                    data[j, int(len(self.configs["qs"]) / 2), :],
                    label="Median Predicted Load",
                    color="Blue",
                    zorder=5,
                )
        #         for i in range(int(len(self.configs["qs"]) / 2) - 1, -1, -1):
        #             q = self.configs["qs"][i]
        #             ax1.plot(
        #                 time_index, data[j, i, :], color="black", lw=0.3, alpha=1, zorder=i
        #             )
        #             ax1.plot(
        #                 time_index,
        #                 data[j, -(i + 1), :],
        #                 color="black",
        #                 lw=0.3,
        #                 alpha=1,
        #                 zorder=i,
        #             )
        #             ax1.fill_between(
        #                 time_index,
        #                 data[j, i, :],
        #                 data[j, -(i + 1), :],
        #                 color=cmap(q),
        #                 alpha=1,
        #                 label="{}% PI".format(round((self.configs["qs"][-(i + 1)] - q) * 100)),
        #                 zorder=i,
        #             )

        # plt.axvline(test_data.index[1522], c="black", ls="--", lw=1, zorder=6)
        # plt.text(
        #     test_data.index[1522],
        #     50,
        #     r" Forecast generation time $t_{gen}$",
        #     rotation=0,
        #     fontsize=8,
        # )

        # plt.xticks(rotation=45, ha="right", va="top", fontsize=8)
        myFmt = mdates.DateFormatter("%H:%M:%S\n%m/%d/%y")
        ax1.xaxis.set_major_formatter(myFmt)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel("Cafe Main Power (kW)", fontsize=8)
        plt.xlabel("Date & Time", fontsize=8)
        ax1.legend(loc="upper center", fontsize=8)

        plt.show()

        # Plotting residuals vs time-step-ahead forecast
        # residuals = data[:, 3, :] - measured
        # fig, ax2 = plt.subplots()
        # for i in range(0, residuals.shape[1]):
        #     ax2.scatter(
        #         np.ones_like(residuals[:, i]) * i, residuals[:, i], s=0.5, color="black"
        #     )
        # ax2.set_xlabel("Forecast steps ahead")
        # ax2.set_ylabel("Residual")
        # plt.show()

        # # Plot residuals for all times in test set
        # fig, ax3 = plt.subplots()
        # ax3.scatter(test_data.index, residuals[:, -1], s=0.5, alpha=0.5, color="blue")
        # ax3.set_ylabel("Residual of 18hr ahead forecast")
        # ax3.scatter(
        #     processed.index[
        #         np.logical_and(processed.index.weekday == 5, processed.index.hour == 12)
        #     ],
        #     residuals[:, -1][
        #         np.logical_and(processed.index.weekday == 5, processed.index.hour == 12)
        #     ],
        #     s=20,
        #     alpha=0.5,
        #     color="black",
        # )