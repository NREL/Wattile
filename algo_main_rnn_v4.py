import os
import numpy as np
import pandas as pd
import json
from util import prtime, factors, tile
import rnn
import lstm

import torch
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data_utils
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import timeit
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import csv
import pathlib
import psutil
from psutil import virtual_memory
import buildings_processing as bp
import logging


file_prefix = '/default'


class ConfigsError(Exception):
    """Base class for exceptions in this module."""
    pass


def size_the_batches(train_data, val_data, tr_desired_batch_size, te_desired_batch_size, configs):
    """
    Compute the batch sizes for training and val set

    :param train_data: (DataFrame)
    :param val_data: (DataFrame)
    :param tr_desired_batch_size: (int)
    :param te_desired_batch_size: (int)
    :return:
    """

    if configs["run_train"]:
        # Find factors of the length of train and val df's and pick the closest one to the requested batch sizes
        train_bth = factors(train_data.shape[0])
        train_num_batches = min(train_bth, key=lambda x: abs(x - tr_desired_batch_size))
        train_bt_size = int(train_data.shape[0] / train_num_batches)

        val_bth = factors(val_data.shape[0])
        val_num_batches = min(val_bth, key=lambda x: abs(x - te_desired_batch_size))
        val_bt_size = int(val_data.shape[0]/val_num_batches)

        train_ratio = round(train_data.shape[0] * 100 / (train_data.shape[0] + val_data.shape[0]),1)
        val_ratio = 100 - train_ratio
        num_train_data = train_data.shape[0]

        logging.info("Train size: {}, val size: {}, split {}%:{}%".format(train_data.shape[0], val_data.shape[0], train_ratio,
                                                                  val_ratio))
        logging.info("Available train batch factors: {}".format(sorted(train_bth)))
        logging.info("Requested number of batches per epoch - Train: {}, val: {}".format(tr_desired_batch_size, te_desired_batch_size))
        logging.info("Actual number of batches per epoch - Train: {}, val: {}".format(train_num_batches, val_num_batches))
        logging.info("Number of data samples in each batch - Train: {}, val: {}".format(train_bt_size, val_bt_size))
    else:
        val_bt_size = val_data.shape[0]
        train_bt_size = 0
        num_train_data = 0

    return train_bt_size, val_bt_size, num_train_data


def data_transform(train_data, val_data, transformation_method, run_train):
    """
    Normalize the training and val data according to a user-defined criteria

    :param train_data: DataFrame
    :param val_data: DataFrame
    :param transformation_method: str
    :param run_train: Boolean
    :return:
    """
    if run_train:
        # For the result de-normalization purpose, saving the max and min values of the STM_Xcel_Meter columns
        train_stats = {}
        train_stats['train_max'] = train_data.max().to_dict()
        train_stats['train_min'] = train_data.min().to_dict()
        train_stats['train_mean'] = train_data.mean(axis=0).to_dict()
        train_stats['train_std'] = train_data.std(axis=0).to_dict()
        path = os.path.join(file_prefix, "train_stats.json")
        with open(path, 'w') as fp:
            json.dump(train_stats, fp)

        if transformation_method == "minmaxscale":
            train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        elif transformation_method == "standard":
            train_data = (train_data - train_data.mean(axis=0)) / train_data.std(axis=0)

        else:
            raise ConfigsError("{} is not a supported form of data normalization".format(transformation_method))

    # Reading back the train stats for normalizing val data w.r.t to train data
    file_loc = os.path.join(file_prefix, "train_stats.json")
    with open(file_loc, 'r') as f:
        train_stats = json.load(f)

    # get statistics for training data
    train_max = pd.DataFrame(train_stats['train_max'], index=[1]).iloc[0]
    train_min = pd.DataFrame(train_stats['train_min'], index=[1]).iloc[0]
    train_mean = pd.DataFrame(train_stats['train_mean'], index=[1]).iloc[0]
    train_std = pd.DataFrame(train_stats['train_std'], index=[1]).iloc[0]

    # Normalize data
    if transformation_method == "minmaxscale":
        val_data = (val_data - train_min) / (train_max - train_min)
    elif transformation_method == "standard":
        val_data = ((val_data - train_mean) / train_std)
    else:
        raise ConfigsError("{} is not a supported form of data normalization".format(transformation_method))

    return train_data, val_data


def data_iterable_random(train_data, val_data, run_train, train_batch_size, val_batch_size, configs):
    """
    Converts train and val data to torch data types (used only if splitting training and val set randomly)

    :param train_data: (DataFrame)
    :param val_data: (DataFrame)
    :param run_train: (Boolean)
    :param train_batch_size: (int)
    :param val_batch_size: (int)
    :param configs: (Dictionary)
    :return:
    """

    if run_train:
        # Define input feature matrix
        X_train = train_data.drop(configs['target_var'], axis=1).values.astype(dtype='float32')

        # Output variable
        y_train = train_data[configs['target_var']]
        y_train = y_train.values.astype(dtype='float32')
        y_train = np.tile(y_train, (len(configs['qs']), 1))
        y_train = np.transpose(y_train)

        # Convert to iterable tensors
        train_feat_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
        train_target_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
        train = data_utils.TensorDataset(train_feat_tensor, train_target_tensor)
        train_loader = data_utils.DataLoader(train, batch_size=train_batch_size,
                                             shuffle=True)  # Contains features and targets

    else:
        train_loader = []

    # Do the same as above, but for the val set
    X_val = val_data.drop(configs['target_var'], axis=1).values.astype(dtype='float32')

    y_val = val_data[configs['target_var']]
    y_val = y_val.values.astype(dtype='float32')
    y_val = np.tile(y_val, (len(configs['qs']), 1))
    y_val = np.transpose(y_val)

    val_feat_tensor = torch.from_numpy(X_val).type(torch.FloatTensor)
    val_target_tensor = torch.from_numpy(y_val).type(torch.FloatTensor)

    val = data_utils.TensorDataset(val_feat_tensor, val_target_tensor)
    val_loader = DataLoader(dataset=val, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader


def save_model(model, epoch, n_iter):
    """
    Save a PyTorch model to a file

    :param model: (Pytorch model)
    :param epoch: (int)
    :param n_iter: (int)
    :return: None
    """
    model_dict = {'epoch_num': epoch, 'n_iter': n_iter, 'torch_model': model}
    torch.save(model_dict, os.path.join(file_prefix, "torch_model"))


def pinball_np(output, target, configs):
    resid = target - output
    tau = np.array(configs["qs"])
    alpha = configs["smoothing_alpha"]
    log_term = np.zeros_like(resid)
    log_term[resid < 0] = (np.log(1+np.exp(resid[resid < 0]/alpha)) - (resid[resid < 0]/alpha))
    log_term[resid >= 0] = np.log(1 + np.exp(-resid[resid >= 0]/alpha))
    loss = resid * tau + alpha * log_term

    return loss


def quantile_loss(output, target, configs):
    """
    Computes loss for quantile methods.

    :param output: (Tensor)
    :param target: (Tensor)
    :param configs: (Dictionary)
    :return: (Tensor) Loss for this study (single number)
    """

    resid = target - output
    tau = torch.FloatTensor(configs["qs"])
    alpha = configs["smoothing_alpha"]
    log_term = torch.zeros_like(resid)
    log_term[resid < 0] = (torch.log(1+torch.exp(resid[resid < 0]/alpha)) - (resid[resid < 0]/alpha))
    log_term[resid >= 0] = torch.log(1 + torch.exp(-resid[resid >= 0]/alpha))
    loss = resid * tau + alpha * log_term
    loss = torch.mean(torch.mean(loss, 0))

    # Extra statistics to return optionally
    stats = [resid.data.numpy().min(), resid.data.numpy().max()]

    # See histogram of residuals
    # graph = pd.DataFrame(resid.data.numpy()).plot(kind="hist", alpha=0.5, bins=50, ec='black', stacked=True)

    return loss


def test_processing(val_df, val_loader, model, seq_dim, input_dim, val_batch_size, transformation_method, configs, last_run):
    """
    Process the val set and report error statistics.

    :param val_df: (DataFrame)
    :param val_loader: (DataLoader)
    :param model: (Pytorch model)
    :param seq_dim: ()
    :param input_dim:
    :param val_batch_size:
    :param transformation_method:
    :param configs: (Dictionary)
    :return:
    """

    # Plug the val set into the model
    model.eval()
    preds = []
    targets = []
    for i, (feats, values) in enumerate(val_loader):
        features = Variable(feats.view(-1, seq_dim, input_dim))
        outputs = model(features)
        preds.append(outputs.data.numpy().squeeze())
        targets.append(values.data.numpy())

    # (Normalized Data) Concatenate the predictions and targets for the whole val set
    semifinal_preds = np.concatenate(preds)
    semifinal_targs = np.concatenate(targets)

    # Calculate pinball loss (done on normalized data)
    loss = pinball_np(semifinal_preds, semifinal_targs, configs)
    pinball_loss = np.mean(np.mean(loss, 0))

    # Loading the training data stats for de-normalization purpose
    file_loc = os.path.join(file_prefix, "train_stats.json")
    with open(file_loc, 'r') as f:
        train_stats = json.load(f)

    # Get normalization statistics
    train_max = pd.DataFrame(train_stats['train_max'], index=[1]).iloc[0]
    train_min = pd.DataFrame(train_stats['train_min'], index=[1]).iloc[0]
    train_mean = pd.DataFrame(train_stats['train_mean'], index=[1]).iloc[0]
    train_std = pd.DataFrame(train_stats['train_std'], index=[1]).iloc[0]

    # Do de-normalization process on predictions and targets from val set

    if transformation_method == "minmaxscale":
        final_preds = ((train_max[configs['target_var']] - train_min[configs['target_var']]) * semifinal_preds) + \
                      train_min[configs['target_var']]
        final_targs = ((train_max[configs['target_var']] - train_min[configs['target_var']]) * semifinal_targs) + \
                      train_min[configs['target_var']]
    elif transformation_method == "standard":
        final_preds = ((semifinal_preds * train_std[configs['target_var']]) + train_mean[configs['target_var']])
        final_targs = ((semifinal_targs * train_std[configs['target_var']]) + train_mean[configs['target_var']])
    else:
        raise ConfigsError("{} is not a supported form of data normalization".format(transformation_method))

    # (De-Normalized Data) Assign target and output variables
    target = final_targs
    output = final_preds

    # Do quantile-related (q != 0.5) error statistics
    # QS (single point)
    loss = pinball_np(output, target, configs)
    QS = loss.mean()
    # PICP (single point for each bound)
    target_1D = target[:, 0]
    bounds = np.zeros((target.shape[0], int(len(configs["qs"])/2)))
    PINC = []
    for i, q in enumerate(configs["qs"]):
        if q == 0.5:
            break
        bounds[:, i] = np.logical_and(output[:, i] < target_1D, target_1D < output[:, -(i + 1)])
        PINC.append(configs["qs"][-(i+1)] - configs["qs"][i])
    PINC = np.array(PINC)
    PICP = bounds.mean(axis=0)
    # ACE (single point)
    ACE = np.sum(np.abs(PICP - PINC))
    # IS (single point)
    lower = output[:, :int(len(configs["qs"])/2)]
    upper = np.flip(output[:, int(len(configs["qs"])/2)+1:], 1)
    alph = 1-PINC
    x = target[:, :int(len(configs["qs"]) / 2)]
    IS = (upper - lower) + (2 / alph) * (lower - x) * (x < lower) + (2 / alph) * (x - upper) * (x > upper)
    IS = IS.mean()

    # Compare theoretical and actual Q's
    act_prob = (output > target).sum(axis=0)/(target.shape[0])
    Q_vals = pd.DataFrame()
    Q_vals["q_requested"] = configs["qs"]
    Q_vals["q_actual"] = act_prob

    # Do quantile-related (q == 0.5) error statistics
    # Only do reportable error statistics on the q=0.5 predictions. Crop np arrays accordingly
    final_preds_median = final_preds[:, int(semifinal_preds.shape[1] / 2)]
    final_targs_median = final_targs[:, int(semifinal_targs.shape[1] / 2)]
    predictions = pd.DataFrame(final_preds_median)
    output = final_preds_median
    target = final_targs_median
    # Set "Number of adjustable model parameters" for each type of error statistic
    p_nmbe = 0
    p_cvrmse = 1
    # Calculate different error metrics
    rmse = np.sqrt(np.mean((output - target) ** 2))
    nmbe = (1 / (np.mean(target))) * (np.sum(target - output)) / (len(target) - p_nmbe)
    cvrmse = (1 / (np.mean(target))) * np.sqrt(np.sum((target - output) ** 2) / (len(target) - p_cvrmse))
    gof = (np.sqrt(2) / 2) * np.sqrt(cvrmse ** 2 + nmbe ** 2)

    # If this is the last val run of training, get histogram data of residuals for each quantile
    if last_run:
        #resid = target - output
        resid = semifinal_targs - semifinal_preds
        hist_data = pd.DataFrame()
        for i, q in enumerate(configs["qs"]):
            tester = np.histogram(resid[:, i], bins=200)
            y_vals = tester[0]
            x_vals = 0.5*(tester[1][1:]+tester[1][:-1])
            hist_data["{}_x".format(q)] = x_vals
            hist_data["{}_y".format(q)] = y_vals
    else:
        hist_data = []

    # Add different error statistics to a dictionary
    errors = {"pinball_loss": pinball_loss,
              "rmse": rmse,
              "nmbe": nmbe,
              "cvrmse": cvrmse,
              "gof": gof,
              "qs": QS,
              "ace": ACE,
              "is": IS}

    predictions = pd.DataFrame(final_preds)
    targets = pd.DataFrame(final_targs)
    return predictions, targets, errors, Q_vals, hist_data


def process(train_loader, val_loader, val_df, num_epochs, run_train, run_resume, writer, transformation_method,
            configs, train_batch_size, val_batch_size, seq_dim, num_train_data):
    """
    Contains main training process for RNN

    :param train_loader: (Pytorch DataLoader)
    :param val_loader: (Pytorch DataLoader)
    :param val_df: (DataFrame)
    :param num_epochs: (int)
    :param run_train: (Boolean)
    :param run_resume: (Boolean)
    :param writer: (SummaryWriter object)
    :param transformation_method: (str)
    :param configs: (Dictionary)
    :param train_batch_size: (Float)
    :param val_batch_size: (Float)
    :param seq_dim: (Int)
    :param num_train_data: (Float)
    :return: None
    """

    num_epochs = num_epochs
    hidden_dim = int(configs['hidden_nodes'])
    output_dim = len(configs["qs"])
    weight_decay = float(configs['weight_decay'])
    input_dim = configs['input_dim']
    layer_dim = configs['layer_dim']

    # Write the configurations used for this training process to a json file, only if training is happening
    if configs["run_train"]:
        path = os.path.join(file_prefix, "configs.json")
        with open(path, 'w') as fp:
            json.dump(configs, fp, indent=1)

    # initializing lists to store losses over epochs:
    train_loss = []
    train_iter = []
    # val_loss = []
    val_iter = []
    # val_rmse = []

    # If you want to continue training the model:
    if run_train:
        # If you are resuming from a previous training session
        if run_resume:
            try:
                torch_model = torch.load(os.path.join(file_prefix, 'torch_model'))
                model = torch_model['torch_model']
                resume_num_epoch = torch_model['epoch_num']
                resume_n_iter = torch_model['n_iter']

                epoch_range = np.arange(resume_num_epoch + 1, num_epochs + 1)
            except FileNotFoundError:
                logging.info("model does not exist in the given folder for resuming the training. Exiting...")
                exit()
            logging.info("run_resume=True, model loaded from: {}".format(file_prefix))

        # If you want to start training a model from scratch
        else:
            # RNN layer
            # input_dim: The number of expected features in the input x
            # hidden_dim: the number of features in the hidden state h
            # layer_dim: Number of recurrent layers. i.e. if 2, it is stacking two RNNs together to form a stacked RNN
            # Initialize the model
            if configs["arch_type_variant"] == "vanilla":
                model = rnn.RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
            elif configs["arch_type_variant"] == "lstm":
                model = lstm.LSTM_Model(input_dim, hidden_dim, layer_dim, output_dim)
            else:
                raise ConfigsError(
                    "{} is not a supported architecture variant".format(configs["arch_type_variant"]))
            epoch_range = np.arange(num_epochs)
            logging.info("A new {} {} model instantiated, with run_train=True".format(configs["arch_type"], configs["arch_type_variant"]))

        # Instantiate Optimizer Class
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr_config']['base'], weight_decay=weight_decay)

        # Set up learning rate scheduler
        if not configs["lr_config"]["schedule"]:
            pass
        elif configs["lr_config"]["schedule"] and configs["lr_config"]["type"] == "performance":
            # Patience (for our case) is # of iterations, not epochs, but configs specification is num epochs
            scheduler = ReduceLROnPlateau(optimizer,
                                          mode='min',
                                          factor=configs['lr_config']['factor'],
                                          min_lr=configs['lr_config']['min'],
                                          patience=int(
                                              configs['lr_config']['patience'] * (num_train_data / train_batch_size)),
                                          verbose=True)
        elif configs["lr_config"]["schedule"] and configs["lr_config"]["type"] == "absolute":
            # scheduler = StepLR(optimizer,
            #                    step_size=int(configs['lr_config']["step_size"]*(num_train_data/train_batch_size)),
            #                    gamma=configs['lr_config']['factor'])
            pass
        else:
            raise ConfigsError("{} is not a supported method of LR scheduling".format(configs["lr_config"]["type"]))

        # Computing platform
        num_logical_processors = psutil.cpu_count(logical=True)
        num_cores = psutil.cpu_count(logical=False)
        mem = virtual_memory()
        mem = {"total": mem.total / 10 ** 9, "available": mem.available / 10 ** 9, "percent": mem.percent,
               "used": mem.used / 10 ** 9, "free": mem.free / 10 ** 9}
        logging.info("Number of cores available: {}".format(num_cores))
        logging.info("Number of logical processors available: {}".format(num_logical_processors))
        logging.info("Initial memory statistics (GB): {}".format(mem))

        # Check for GPU
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            logging.info("GPU is available for training")
        else:
            logging.info("GPU is not available for training")

        if (len(epoch_range) == 0):
            epoch = resume_num_epoch + 1
            logging.info("The previously saved model was at epoch= {}, which is same as num_epochs. So, not training"
                   .format(resume_num_epoch))

        if run_resume:
            n_iter = resume_n_iter
            epoch_num = resume_num_epoch
        else:
            n_iter = 0
            epoch_num = 1

        # Start training timer
        train_start_time = timeit.default_timer()

        # Residual diagnostics
        resid_stats = []

        # Initialize re-trainable matrix
        # train_y_at_t = torch.zeros(train_batch_size, seq_dim, 1)  # 960 x 5 x 1

        mid_train_error_stats = pd.DataFrame()

        logging.info("Starting to train the model for {} epochs!".format(num_epochs))

        # Loop through epochs
        for epoch in epoch_range:

            # Do manual learning rate scheduling, if requested
            if configs["lr_config"]["schedule"] and configs["lr_config"]["type"] == "absolute" and epoch_num % configs['lr_config']["step_size"] == 0:
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = param_group['lr'] * configs['lr_config']['factor']
                    new_lr = param_group['lr']
                logging.info("Changing learning rate from {} to {}".format(old_lr, new_lr))

            # This loop returns elements from the dataset batch by batch. Contains features AND targets
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

                # Get memory statistics
                if n_iter % configs["eval_frequency"] == 0:
                    mem = virtual_memory()
                    mem = {"total": mem.total / 10 ** 9, "available": mem.available / 10 ** 9, "used": mem.used / 10 ** 9, "free": mem.free / 10 ** 9}
                    writer.add_scalars("Memory_GB", mem, n_iter)

                # FORWARD PASS to get output/logits.
                # train_y_at_t is (#batches x timesteps x 1)
                # features is     (#batches x timesteps x features)
                # This command: (960x5x7) --> 960x1
                # outputs = model(torch.cat((features, train_y_at_t.detach_()), dim=2))
                outputs = model(features)

                time3 = timeit.default_timer()

                # tiling the 2nd axis of y_at_t from 1 to 5
                # train_y_at_t = tile(outputs.unsqueeze(2), 1, 5)
                # train_y_at_t_nump = train_y_at_t.detach().numpy()

                # Calculate Loss
                loss = quantile_loss(outputs, target, configs)

                # resid_stats.append(stats)
                train_loss.append(loss.data.item())
                train_iter.append(n_iter)

                # Print to terminal and save training loss
                writer.add_scalars("Loss", {'Train': loss.data.item()}, n_iter)

                time4 = timeit.default_timer()

                # Does backpropogation and gets gradients, (the weights and bias). Create computational graph
                loss.backward()

                time5 = timeit.default_timer()

                if configs["lr_config"]["schedule"] and configs["lr_config"]["type"] == "performance":
                    scheduler.step(loss)

                # Updating the weights/parameters. Clear computational graph.
                optimizer.step()

                # Each iteration is one batch
                n_iter += 1

                # Compute time per iteration
                time6 = timeit.default_timer()
                writer.add_scalars("Iteration_time", {"Package_variables": time2 - time1,
                                                      "Evaluate_model": time3 - time2,
                                                      "Calc_loss": time4 - time3,
                                                      "Backprop": time5 - time4,
                                                      "Step": time6 - time5}, n_iter)

                # Save the model every ___ iterations
                if n_iter % configs["eval_frequency"] == 0:
                    save_model(model, epoch, n_iter)

                # Do a val batch every ___ iterations
                if n_iter % configs["eval_frequency"] == 0:
                    # Evaluate val set
                    predictions, targets, errors, Q_vals, hist_data = test_processing(val_df, val_loader, model, seq_dim, input_dim,
                                                          val_batch_size, transformation_method, configs, False)
                    predictions = predictions.iloc[:, int(predictions.shape[1] / 2)]
                    temp_holder = errors
                    temp_holder.update({"n_iter": n_iter, "epoch": epoch})
                    mid_train_error_stats = mid_train_error_stats.append(temp_holder, ignore_index=True)

                    val_iter.append(n_iter)
                    writer.add_scalars("Loss", {"val": errors['pinball_loss']}, n_iter)

                    # Add parody plot to TensorBoard
                    # fig2, ax2 = plt.subplots()
                    # ax2.scatter(predictions, val_df[configs['target_var']], s=5, alpha=0.3)
                    # strait_line = np.linspace(min(min(predictions), min(val_df[configs['target_var']])),
                    #                           max(max(predictions), max(val_df[configs['target_var']])), 5)
                    # ax2.plot(strait_line, strait_line, c='k')
                    # ax2.set_xlabel('Predicted')
                    # ax2.set_ylabel('Observed')
                    # ax2.axhline(y=0, color='k')
                    # ax2.axvline(x=0, color='k')
                    # ax2.axis('equal')
                    # writer.add_figure('Parody', fig2, n_iter)

                    #Add QQ plot to TensorBoard
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(Q_vals["q_requested"], Q_vals["q_actual"], s=20)
                    ax2.plot([0, 1], [0, 1], c='k', alpha=0.5)
                    ax2.set_xlabel('Requested')
                    ax2.set_ylabel('Actual')
                    ax2.set_xlim(left=0, right=1)
                    ax2.set_ylim(bottom=0, top=1)
                    writer.add_figure('QQ', fig2, n_iter)

                    # Write information about CPU usage to tensorboard
                    percentages = dict(zip(list(np.arange(1,num_logical_processors+1).astype(str)), psutil.cpu_percent(interval=None, percpu=True)))
                    writer.add_scalars("CPU_Utilization", percentages, n_iter)

                    logging.info('Epoch: {} Iteration: {}. Train_loss: {}. val_loss: {}, LR: {}'.format(epoch_num, n_iter,
                                                                                                loss.data.item(),
                                                                                                errors['pinball_loss'],
                                                                                                optimizer.param_groups[
                                                                                                    0]['lr']))
            epoch_num += 1

        # Once model training is done, save the current model state
        save_model(model, epoch, n_iter)

        # Once model is done training, process a final val set
        predictions, targets, errors, Q_vals, hist_data = test_processing(val_df, val_loader, model, seq_dim, input_dim,
                                              val_batch_size, transformation_method, configs, True)

        # Save the residual distribution to a file
        hist_data.to_hdf(os.path.join(file_prefix, "residual_distribution.h5"), key='df', mode='w')

        # Save the final predictions to a file
        #predictions.to_csv(file_prefix + '/predictions.csv', index=False)
        pd.DataFrame(predictions).to_hdf(os.path.join(file_prefix, "predictions.h5"), key='df', mode='w')

        # Save the QQ information to a file
        Q_vals.to_hdf(os.path.join(file_prefix, "QQ_data.h5"), key='df', mode='w')

        # Save the mid-train error statistics to a file
        mid_train_error_stats.to_hdf(os.path.join(file_prefix, "mid_train_error_stats.h5"), key='df', mode='w')

        # End training timer
        train_end_time = timeit.default_timer()
        train_time = train_end_time - train_start_time

        # Plot residual stats
        # fig3, ax3 = plt.subplots()
        # ax3.plot(np.array(resid_stats)[:, 0], label="Min")
        # ax3.plot(np.array(resid_stats)[:, 1], label="Max")
        # plt.show()

        # If a training history csv file does not exist, make one
        if not pathlib.Path("Training_history.csv").exists():
            with open(r'Training_history.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(["File Path", "RMSE", "CV(RMSE)", "NMBE", "GOF", "QS", "ACE", "IS", "Train time"])

        # Save the errors statistics to a central results csv once everything is done
        with open(r'Training_history.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([file_prefix,
                             errors["rmse"],
                             errors["cvrmse"],
                             errors["nmbe"],
                             errors["gof"],
                             errors["qs"],
                             errors["ace"],
                             errors["is"],
                             train_time])

        # Write error statistics to a local json file
        errors["train_time"] = train_time
        for k in errors:
            errors[k] = str(errors[k])
        path = os.path.join(file_prefix, "error_stats.json")
        with open(path, 'w') as fp:
            json.dump(errors, fp, indent=1)


    # If you just want to immediately val the model on the existing (saved) model
    else:
        torch_model = torch.load(os.path.join(file_prefix, 'torch_model'))
        model = torch_model['torch_model']
        logging.info("Loaded model from file, given run_train=False\n")

        # Run val
        predictions, targets, errors, Q_vals, hist_data = test_processing(val_df, val_loader, model, seq_dim, input_dim,
                                              val_batch_size, transformation_method, configs, True)

        building = configs["building"]
        year = configs["external_test"]["year"]
        month = configs["external_test"]["month"]
        file = os.path.join(configs["data_dir"], "{}_external_test.h5".format(configs["target_var"]))
        index = pd.read_hdf(file, key='df').index

        # Plot the results of the test set
        cmap = plt.get_cmap('Reds')
        fig, ax1 = plt.subplots(figsize=(20, 4))
        ax1.plot(index, targets.iloc[:,0], label="Actual Demand", color='black')
        ax1.plot(index, predictions.iloc[:, int(len(configs["qs"]) / 2)], label='q = 0.5 forecast', color="red")
        for i, q in enumerate(configs["qs"]):
            if q == 0.5:
                break
            ax1.fill_between(index, predictions.iloc[:, i], predictions.iloc[:, -(i+1)], color=cmap(q), alpha=1,
                             label="{}% PI".format(round((configs["qs"][-(i + 1)] - q) * 100)), lw=0)
        plt.xticks(rotation=0)
        ax1.set_ylabel(configs["target_var"])
        #ax1.set_xlim([pd.to_datetime('2019-01-07 00:00:00'), pd.to_datetime('2019-01-14 00:00:00')])
        #ax1.set_title("Cafe Main Meter LSTM Predictions")
        ax1.legend()
        plt.show()
        #fig.savefig(os.path.join(configs["results_dir"], "{}_test.png".format(configs["target_var"])))

        # Plotting residuals for the test set
        residuals = predictions.iloc[:, int(len(configs["qs"])/2)] - targets.iloc[:, int(len(configs["qs"])/2)]
        file = os.path.join(configs["data_dir"], "{}_external_test.h5".format(configs["target_var"]))
        processed = pd.read_hdf(file, key='df')
        fig, ax2 = plt.subplots()
        ax2.scatter(processed.index, residuals.values, s=0.5, alpha=0.3, color="blue")
        ax2.set_ylabel('Median Forecast Residual (kW)')
        plt.show()
        print("Done")

        # Plotting out of bounds
        fig, ax3 = plt.subplots()
        mask = ~np.logical_and(predictions.iloc[:,0] < targets.iloc[:, int(len(configs["qs"])/2)], predictions.iloc[:,-1] > targets.iloc[:, int(len(configs["qs"])/2)])
        dates = processed.index[mask]
        plt.vlines(dates, 0, 1, alpha=0.05)


def eval_trained_model(file_prefix, train_data, train_batch_size, configs):
    """
    Pass the entire training set through the trained model and get the predictions.
    Compute the residual and save to a DataFrame.

    :param file_prefix: (str)
    :param train_data: (DataFrame)
    :param train_batch_size: (Float)
    :param configs: (Dictionary)
    :return: None
    """

    # Evaluate the training model
    torch_model = torch.load(os.path.join(file_prefix, 'torch_model'))
    model = torch_model['torch_model']
    X_train = train_data.drop(configs['target_var'], axis=1).values.astype(dtype='float32')
    y_train = train_data[configs['target_var']]
    y_train = y_train.values.astype(dtype='float32')
    y_train = np.tile(y_train, (len(configs['qs']), 1))
    y_train = np.transpose(y_train)
    train_feat_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
    train_target_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
    train = data_utils.TensorDataset(train_feat_tensor, train_target_tensor)
    train_loader = DataLoader(dataset=train, batch_size=train_batch_size, shuffle=False)

    # Pass the training data into the trained model
    model.eval()
    preds = []
    targets = []
    for i, (feats, values) in enumerate(train_loader):
        features = Variable(feats.view(-1, configs['window']+1, configs['input_dim']))
        outputs = model(features)
        preds.append(outputs.data.numpy().squeeze())
        targets.append(values.data.numpy())
    semifinal_preds = np.concatenate(preds)
    semifinal_targs = np.concatenate(targets)

    # Get the saved binary mask from file
    mask_file = os.path.join(file_prefix, "mask.h5")
    mask = pd.read_hdf(mask_file, key='df')
    msk = mask["msk"] == 0


    # Adjust the datetime index so it is in line with the EC data
    target_index = mask.index[msk] + pd.DateOffset(
        minutes=(configs["EC_future_gap"] * configs["resample_freq"]))
    processed_data = pd.DataFrame(index=target_index)

    # Stick data into a DataFrame to be accessed later
    i = 0
    for q in configs["qs"]:
        processed_data["{}_fit".format(q)] = semifinal_preds[:, i]
        i = i + 1
    processed_data['Target'] = semifinal_targs[:, 0]
    processed_data['Residual'] = semifinal_targs[:, 0] - semifinal_preds[:, int(semifinal_preds.shape[1] / 2)]

    # Save DataFrame to file
    processed_data.to_hdf(os.path.join(file_prefix, "evaluated_training_model.h5"), key='df', mode='w')


def plot_processed_model(file_prefix):
    """
    Plot the trained model, along with the residuals for the trained model.
    The plot will show what time periods are not being captured by the model.

    :param file_prefix: (str) Relative path to the results folder for the model you want to study
    :return: None
    """

    # Read in training data and config file from results directory
    processed_data = pd.read_hdf(os.path.join(file_prefix, "evaluated_training_model.h5"), key='df')
    with open(os.path.join(file_prefix, "configs.json"), "r") as read_file:
        configs = json.load(read_file)

    # Plot data
    cmap = plt.get_cmap('Reds')
    f, axarr = plt.subplots(2, sharex=True)
    for i, q in enumerate(configs["qs"]):
        if q == 0.5:
            break
        axarr[0].fill_between(processed_data.index, processed_data["{}_fit".format(q)],
                              processed_data["{}_fit".format(configs["qs"][-(i + 1)])],
                              alpha=1,
                              color=cmap(q),
                              lw=0,
                              label="{}% PI".format(round((configs["qs"][-(i + 1)] - q) * 100)))

    axarr[0].plot(processed_data["Target"], label='Target', color='black')
    axarr[0].plot(processed_data["0.5_fit"], label='q = 0.5', color="red")
    axarr[0].set_ylabel("target variable")
    axarr[0].legend()
    axarr[1].plot(processed_data['Residual'], label='Targets')
    axarr[1].set_ylabel("Residual")
    axarr[1].axhline(y=0, color='k')
    plt.show()


def plot_QQ(file_prefix):
    """
    Plots a QQ plot for a specific study specified by an input file directory string.

    :param file_prefix: (str) Relative path to the training results directory in question.
    :return: None
    """
    QQ_data = pd.read_hdf(os.path.join(file_prefix, "QQ_data.h5"), key='df')
    fig2, ax2 = plt.subplots()
    ax2.scatter(QQ_data["q_requested"], QQ_data["q_actual"], s=20)
    # strait_line = np.linspace(min(min(QQ_data["q_requested"]), min(QQ_data["q_actual"])),
    #                           max(max(QQ_data["q_requested"]), max(QQ_data["q_actual"])), 5)
    ax2.plot([0, 1], [0, 1], c='k', alpha=0.5)
    ax2.set_xlabel('Requested')
    ax2.set_ylabel('Actual')
    ax2.set_xlim(left=0, right=1)
    ax2.set_ylim(bottom=0, top=1)
    plt.show()


def plot_training_history(x):
    """
    Platform for plotting results recorded in the shared csv results file. In development.

    :param x:
    :return:
    """
    data = pd.read_csv("Training_history.csv")
    data = data.iloc[55:, :]
    data["iterable"] = x
    data.plot(x="iterable", subplots=True)


def plot_resid_dist(study_path, building, alphas, q):
    """
    Plot the residual distribution over the smooth approximations for different values of the alpha smoothing parameter.

    :param study_path: (str) Relative path to the study directory in question.
    :param building: (str) Name of the building in question
    :param alphas: (list) Numerical values of alphas to consider for plotting
    :param q: (float) Quantile value to consider for plotting.
    :return: None
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    resid = np.linspace(-1, 1, 1000)
    max_dist = 0

    colors = ["red", "green", "blue"]
    for i, alpha in enumerate(alphas):
        c = colors[i]
        # Plot the residual distribution for this alpha value
        sub_study_path = "RNN_M{}_Tsmoothing_alpha_{}".format(building, alpha)
        data = pd.read_hdf(os.path.join(study_path, sub_study_path, "residual_distribution.h5"), key='df')
        # ax2.plot(data["{}_x".format(q)], data["{}_y".format(q)], c=c, alpha=0.5)
        ax2.fill_between(data["{}_x".format(q)], data["{}_y".format(q)], 0, alpha=0.4, color=c, zorder=1)

        # Store data to later scale axis
        if data["{}_y".format(q)].max() > max_dist:
            max_dist = data["{}_y".format(q)].max()

        # Plot the PLF for this alpha value
        log_term = np.zeros_like(resid)
        log_term[resid < 0] = (np.log(1 + np.exp(resid[resid < 0] / alpha)) - (resid[resid < 0] / alpha))
        log_term[resid >= 0] = np.log(1 + np.exp(-resid[resid >= 0] / alpha))
        loss = resid * q + alpha * log_term
        ax1.plot(resid, loss, c=c, label="$\\alpha$={}".format(alpha), zorder=10)

    ax1.set_xlabel('Normalized Residual')
    ax2.set_xlim(left=-0.5, right=0.5)
    ax1.set_ylim(top=0.6)
    ax1.set_ylabel('Smoothed PLF')
    #ax1.set_ylim(bottom=-0.5)
    ax2.set_ylim(top=4*max_dist)
    ax2.set_ylabel('Frequency')
    ax1.legend()
    plt.show()


def plot_mid_train_stats(file_prefix):
    data = pd.read_hdf(os.path.join(file_prefix, "mid_train_error_stats.h5"), key='df')
    data.plot(x="n_iter", subplots=True)


def predict(data, file_prefix):
    # Get rid of this eventually
    # file_prefix = "EnergyForecasting_Results\RNN_MCafeMainPower(kW)_Tdev"

    # Read configs from results directory
    with open(os.path.join(file_prefix, "configs.json"), "r") as read_file:
        configs = json.load(read_file)

    # Get rid of this eventually
    # data = pd.read_hdf("sample_predict.h5", key='df').drop([configs["target_var"]], axis=1)

    # Check if the supplied data matches the sequence length that the model was trained on
    if not data.shape[0] == configs["window"]+1:
        raise ConfigsError("Input data has sequence length {}. Expected sequence length of {}".format(data.shape[0], configs["window"]+1))

    # Data should be resampled, cleaned by this point. No nans.

    # Convert data to rolling average (except output) and create min, mean, and max columns
    if configs["rolling_window"]["active"]:
        target = data[configs["target_var"]]
        X_data = data.drop(configs["target_var"], axis=1)
        mins = X_data.rolling(window=configs["rolling_window"]["minutes"]+1).min().add_suffix("_min")
        means = X_data.rolling(window=configs["rolling_window"]["minutes"]+1).mean().add_suffix("_mean")
        maxs = X_data.rolling(window=configs["rolling_window"]["minutes"]+1).max().add_suffix("_max")
        data = pd.concat([mins, means, maxs], axis=1)
        data[configs["target_var"]] = target

    # Add time-based variables
    data = bp.time_dummies(data, configs)

    # Do sequential padding of the inputs
    data_orig = data
    for i in range(1, configs["window"]+1):
        shifted = data_orig.shift(i)
        shifted = shifted.join(data, lsuffix="_lag{}".format(i))
        data = shifted
    data = data.iloc[-1, :]

    # Transpose dataframe
    data = pd.DataFrame(data).transpose()

    # Reset index
    data.reset_index(drop=True, inplace=True)

    # Do normalization
    # Reading back the train stats for normalizing test data w.r.t to train data
    file_loc = os.path.join(file_prefix, "train_stats.json")
    with open(file_loc, 'r') as f:
        train_stats = json.load(f)

    # get statistics for training data
    train_max = pd.DataFrame(train_stats['train_max'], index=[1]).iloc[0].drop(configs["target_var"])
    train_min = pd.DataFrame(train_stats['train_min'], index=[1]).iloc[0].drop(configs["target_var"])
    train_mean = pd.DataFrame(train_stats['train_mean'], index=[1]).iloc[0].drop(configs["target_var"])
    train_std = pd.DataFrame(train_stats['train_std'], index=[1]).iloc[0].drop(configs["target_var"])

    # Normalize data
    if configs["transformation_method"] == "minmaxscale":
        data = (data - train_min) / (train_max - train_min)
    elif configs["transformation_method"] == "standard":
        data = ((data - train_mean) / train_std)
    else:
        raise ConfigsError("{} is not a supported form of data normalization".format(configs["transformation_method"]))

    # Convert to iterable dataset
    data = data.values.astype(dtype='float32')
    train_feat_tensor = torch.from_numpy(data).type(torch.FloatTensor)

    # Load model
    torch_model = torch.load(os.path.join(file_prefix, 'torch_model'))
    model = torch_model['torch_model']

    # Evaluate model
    model.eval()
    features = Variable(train_feat_tensor.view(-1, configs['window']+1, configs["input_dim"]))
    outputs = model(features)
    semifinal_preds = outputs.data.numpy()

    # Denormalize
    # Get normalization statistics
    train_max = pd.DataFrame(train_stats['train_max'], index=[1]).iloc[0]
    train_min = pd.DataFrame(train_stats['train_min'], index=[1]).iloc[0]
    train_mean = pd.DataFrame(train_stats['train_mean'], index=[1]).iloc[0]
    train_std = pd.DataFrame(train_stats['train_std'], index=[1]).iloc[0]

    # Do de-normalization process on predictions and targets from test set

    if configs["transformation_method"] == "minmaxscale":
        final_preds = ((train_max[configs['target_var']] - train_min[configs['target_var']]) * semifinal_preds) + \
                      train_min[configs['target_var']]
    elif configs["transformation_method"] == "standard":
        final_preds = ((semifinal_preds * train_std[configs['target_var']]) + train_mean[configs['target_var']])
    else:
        raise ConfigsError("{} is not a supported form of data normalization".format(configs["transformation_method"]))

    return final_preds


def main(train_df, val_df, configs):
    """
    Main executable for prepping data for input to RNN model.

    :param train_df: (DataFrame)
    :param val_df: (DataFrame)
    :param configs: (Dictionary)
    :return: None
    """

    transformation_method = configs['transformation_method']
    run_train = configs['run_train']
    num_epochs = configs['num_epochs']
    run_resume = configs['run_resume']
    tr_desired_batch_size = configs['train_batch_size']
    te_desired_batch_size = configs['val_batch_size']
    building_ID = configs["building"]
    exp_id = configs['exp_id']
    arch_type = configs['arch_type']
    results_dir = configs["results_dir"]

    # Make results directory
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    global file_prefix
    file_prefix = os.path.join(results_dir, arch_type + '_M' + str(configs["target_var"].replace(" ", "")) + '_T' + str(
        exp_id))

    # Create writer object for TensorBoard
    writer_path = file_prefix
    writer = SummaryWriter(writer_path)
    logging.info("Writer path: {}".format(writer_path))

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
    train_data, val_data = data_transform(train_data, val_data, transformation_method, run_train)
    logging.info("Data transformed using {} as transformation method".format(transformation_method))

    # Size the batches
    train_batch_size, val_batch_size, num_train_data = size_the_batches(train_data, val_data, tr_desired_batch_size,
                                                                         te_desired_batch_size, configs)

    # Already did sequential padding: Convert to iterable dataset (DataLoaders)
    if configs["train_val_split"] == 'Random':
        train_loader, val_loader = data_iterable_random(train_data, val_data, run_train, train_batch_size,
                                                         val_batch_size, configs)
    logging.info("Data converted to iterable dataset")

    # Start the training process
    process(train_loader, val_loader, val_df, num_epochs, run_train, run_resume, writer, transformation_method,
            configs, train_batch_size, val_batch_size, configs['window']+1, num_train_data)

