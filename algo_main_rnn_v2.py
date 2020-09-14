import os
import numpy as np
import pandas as pd
import json
from util import prtime, factors, tile
import rnn

import torch
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data_utils
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import timeit
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
import pathlib

file_prefix = '/default'


def seq_pad(a, window):
    """
    Append time-lagged versions of exogenous variables in input array.
    :param a: np.array
    :param window: int
    :return: np.array
    """
    # Create lagged versions of exogenous variables
    rows = a.shape[0]
    cols = a.shape[1]
    b = np.zeros((rows, window * cols))

    # Make new columns for the time-lagged values. Lagged spaces filled with zeros.
    for i in range(window):
        # The first window isnt lagged and is just a copy of "a"
        if i == 0:
            b[:, 0:cols] = a
        # For all remaining windows, just paste a slightly cropped version (so it fits) of "a" into "b"
        else:
            b[i:, i * cols:(i + 1) * cols] = a[:-i, :]

    # The zeros are replaced with a copy of the first n rows
    for i in list(np.arange(window - 1)):
        j = (i * cols) + cols
        b[i, j:] = np.tile(b[i, 0:cols], window - i - 1)

    return b


def size_the_batches(train_data, val_data, tr_desired_batch_size, te_desired_batch_size):
    """
    Compute the batch sizes for training and val set
    :param train_data: DataFrame
    :param val_data: DataFrame
    :param tr_desired_batch_size: int
    :param te_desired_batch_size: int
    :return:
    """
    # Find factors of the length of train and val df's and pick the closest one to the requested batch sizes
    train_bth = factors(train_data.shape[0])
    train_bt_size = min(train_bth, key=lambda x: abs(x - tr_desired_batch_size))

    val_bth = factors(val_data.shape[0])
    val_bt_size = min(val_bth, key=lambda x: abs(x - te_desired_batch_size))

    train_ratio = int(train_data.shape[0] * 100 / (train_data.shape[0] + val_data.shape[0]))
    val_ratio = 100 - train_ratio
    num_train_data = train_data.shape[0]
    print("Train size: {}, val size: {}, split {}:{}".format(train_data.shape[0], val_data.shape[0], train_ratio,
                                                              val_ratio))
    print("Available train batch sizes: {}".format(sorted(train_bth)))
    print("Requested size of batches - Train: {}, val: {}".format(tr_desired_batch_size, te_desired_batch_size))
    print("Actual size of batches - Train: {}, val: {}".format(train_bt_size, val_bt_size))
    print("Number of batches in 1 epoch - Train: {}, val: {}".format(train_data.shape[0] / train_bt_size,
                                                                      val_data.shape[0] / val_bt_size))

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

        # for the result de-normalization purpose, saving the max and min values of the STM_Xcel_Meter columns
        train_stats = {}
        train_stats['train_max'] = train_data.max().to_dict()
        train_stats['train_min'] = train_data.min().to_dict()
        train_stats['train_mean'] = train_data.mean(axis=0).to_dict()
        train_stats['train_std'] = train_data.std(axis=0).to_dict()
        path = file_prefix + '/train_stats.json'
        with open(path, 'w') as fp:
            json.dump(train_stats, fp)

        if transformation_method == "minmaxscale":
            train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())


        else:
            train_data = (train_data - train_data.mean(axis=0)) / train_data.std(axis=0)

    # reading back the train stats for normalizing val data w.r.t to train data
    file_loc = file_prefix + '/train_stats.json'
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

    else:
        val_data = ((val_data - train_mean) / train_std)

    return train_data, val_data


def data_iterable(train_data, val_data, run_train, train_batch_size, val_batch_size, configs):
    """
    Create lagged variables and convert train and val data to torch data types
    :param train_data: DataFrame
    :param val_data: DataFrame
    :param run_train: Boolean
    :param train_batch_size: int
    :param val_batch_size: int
    :param configs: dict
    :return:
    """

    if run_train:
        # Create lagged INPUT variables, i.e. columns: w1_(t-1), w1_(t-2)...
        # Does this for all input variables for times up to "window"
        X_train = train_data.drop(configs['target_var'], axis=1).values.astype(dtype='float32')
        X_train = seq_pad(X_train, configs['window'])

        # Lag output variable, i.e. input for t=5 maps to EC at t=10, t=6 maps to EC t=11, etc.
        y_train = train_data[configs['target_var']].shift(-configs['EC_future_gap']).fillna(method='ffill')
        y_train = y_train.values.astype(dtype='float32')

        # Convert to iterable tensors
        train_feat_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
        train_target_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
        train = data_utils.TensorDataset(train_feat_tensor, train_target_tensor)
        train_loader = data_utils.DataLoader(train, batch_size=train_batch_size,
                                             shuffle=True)  # Contains features and targets
        print("data train made iterable")

    else:
        train_loader = []

    # Do the same as above for the val set
    X_val = val_data.drop(configs['target_var'], axis=1).values.astype(dtype='float32')
    X_val = seq_pad(X_val, configs['window'])

    y_val = val_data[configs['target_var']].shift(-configs['EC_future_gap']).fillna(method='ffill')
    y_val = y_val.values.astype(dtype='float32')

    val_feat_tensor = torch.from_numpy(X_val).type(torch.FloatTensor)
    val_target_tensor = torch.from_numpy(y_val).type(torch.FloatTensor)

    val = data_utils.TensorDataset(val_feat_tensor, val_target_tensor)
    val_loader = DataLoader(dataset=val, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, train_batch_size, val_batch_size


def data_iterable_random(train_data, val_data, run_train, train_batch_size, val_batch_size, configs):
    """
    Converts train and val data to torch data types (used only if splitting training and val set randomly)
    :param train_data: DataFrame
    :param val_data: DataFrame
    :param run_train: Boolean
    :param train_batch_size: int
    :param val_batch_size: int
    :param configs: dict
    :return:
    """

    if run_train:
        # Create lagged INPUT variables, i.e. columns: w1_(t-1), w1_(t-2)...
        # Does this for all input variables for times up to "window"
        X_train = train_data.drop(configs['target_var'], axis=1).values.astype(dtype='float32')

        # Output variable
        y_train = train_data[configs['target_var']]
        y_train = y_train.values.astype(dtype='float32')

        # Convert to iterable tensors
        train_feat_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
        train_target_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
        train = data_utils.TensorDataset(train_feat_tensor, train_target_tensor)
        train_loader = data_utils.DataLoader(train, batch_size=train_batch_size,
                                             shuffle=True)  # Contains features and targets
        print("data train made iterable")

    else:
        train_loader = []

    # Do the same as above for the val set
    X_val = val_data.drop(configs['target_var'], axis=1).values.astype(dtype='float32')

    y_val = val_data[configs['target_var']]
    y_val = y_val.values.astype(dtype='float32')

    val_feat_tensor = torch.from_numpy(X_val).type(torch.FloatTensor)
    val_target_tensor = torch.from_numpy(y_val).type(torch.FloatTensor)

    val = data_utils.TensorDataset(val_feat_tensor, val_target_tensor)
    val_loader = DataLoader(dataset=val, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, train_batch_size, val_batch_size


def save_model(model, epoch, n_iter):
    """
    Save a PyTorch model to a file
    :param model: Pytorch model
    :param epoch: int
    :param n_iter: int
    :return:
    """
    model_dict = {'epoch_num': epoch, 'n_iter': n_iter, 'torch_model': model}
    torch.save(model_dict, file_prefix + '/torch_model')

    # prtime("RNN model checkpoint saved")


def test_processing(val_df, val_loader, model, seq_dim, input_dim, val_batch_size, transformation_method, configs):
    """
    Process the val set and report error statistics
    :param val_df: DataFrame
    :param val_loader: Data Loader
    :param model:
    :param seq_dim:
    :param input_dim:
    :param val_batch_size:
    :param transformation_method:
    :param configs:
    :return:
    """
    # val_df, val_loader, model, seq_dim, input_dim, val_batch_size, transformation_method
    model.eval()
    # val_y_at_t = torch.zeros(val_batch_size, seq_dim, 1)
    preds = []
    targets = []
    for i, (feats, values) in enumerate(val_loader):
        # features = Variable(feats.view(-1, seq_dim, input_dim - 1))
        # outputs = model(torch.cat((features, val_y_at_t), dim=2))
        # val_y_at_t = tile(outputs.unsqueeze(2), 1, 5)
        features = Variable(feats.view(-1, seq_dim, input_dim))
        outputs = model(features)
        preds.append(outputs.data.numpy().squeeze())
        targets.append(values.data.numpy())

    # concatenating the preds and targets for the whole epoch (iterating over val_loader once)
    semifinal_preds = np.concatenate(preds).ravel()
    semifinal_targs = np.concatenate(targets).ravel()
    mse_loss = np.mean((semifinal_targs - semifinal_preds) ** 2)

    # loading the training data stats for de-normalization purpose
    file_loc = file_prefix + '/train_stats.json'
    with open(file_loc, 'r') as f:
        train_stats = json.load(f)

    train_max = pd.DataFrame(train_stats['train_max'], index=[1]).iloc[0]
    train_min = pd.DataFrame(train_stats['train_min'], index=[1]).iloc[0]
    train_mean = pd.DataFrame(train_stats['train_mean'], index=[1]).iloc[0]
    train_std = pd.DataFrame(train_stats['train_std'], index=[1]).iloc[0]

    # Do do-normalization process on predictions from val set
    if transformation_method == "minmaxscale":
        final_preds = ((train_max[configs['target_var']] - train_min[configs['target_var']]) * semifinal_preds) + \
                      train_min[
                          configs['target_var']]

    else:
        final_preds = ((semifinal_preds * train_std[configs['target_var']]) + train_mean[configs['target_var']])

    predictions = pd.DataFrame(final_preds)
    predicted = predictions.values.squeeze()
    measured = val_df[configs['target_var']].values
    p_nmbe = 0  # Number of "adjustable model parameters"
    p_cvrmse = 1

    # Calculate different error metrics
    rmse = np.sqrt(np.mean((predicted - measured) ** 2))
    nmbe = (1 / (np.mean(measured))) * (np.sum(measured - predicted)) / (len(measured) - p_nmbe)
    cvrmse = (1 / (np.mean(measured))) * np.sqrt(np.sum((measured-predicted)**2) / (len(measured) - p_cvrmse))
    gof = (np.sqrt(2) / 2)  * np.sqrt(cvrmse**2 + nmbe**2)

    # Add different error statistics to a dictionary
    errors = {"mse_loss": mse_loss, "rmse": rmse, "nmbe": nmbe, "cvrmse": cvrmse, "gof": gof}

    return predictions, errors


def process(train_loader, val_loader, val_df, num_epochs, run_train, run_resume, writer, transformation_method,
            configs, train_batch_size, val_batch_size, seq_dim, num_train_data):
    """

    :param train_loader:
    :param val_loader:
    :param val_df:
    :param num_epochs:
    :param run_train:
    :param run_resume:
    :param writer:
    :param transformation_method:
    :param configs:
    :param train_batch_size:
    :param val_batch_size:
    :param seq_dim:
    :param num_train_data:
    :return:
    """
    # ___ Hyper-parameters
    # Input_dim: Determined automatically
    num_epochs = num_epochs
    lr_schedule = configs['lr_schedule']
    hidden_dim = int(configs['hidden_nodes'])
    output_dim = configs["output_dim"]
    weight_decay = float(configs['weight_decay'])

    input_dim = configs['input_dim']
    #configs['hidden_dim'] = hidden_dim
    layer_dim = configs['layer_dim']

    # Write the configurations used for this training process to a json file
    path = file_prefix + '/configs.json'
    with open(path, 'w') as fp:
        json.dump(configs, fp, indent=1)

    # initializing lists to store losses over epochs:
    train_loss = []
    train_iter = []
    #val_loss = []
    val_iter = []
    #val_rmse = []

    # If you want to continue training the model
    if run_train:
        # If you are resuming from a previous training session
        if run_resume:
            try:
                torch_model = torch.load(file_prefix + '/torch_model')
                model = torch_model['torch_model']
                resume_num_epoch = torch_model['epoch_num']
                resume_n_iter = torch_model['n_iter']

                epoch_range = np.arange(resume_num_epoch + 1, num_epochs + 1)
            except FileNotFoundError:
                print("model does not exist in the given folder for resuming the training. Exiting...")
                exit()
            prtime("rune_resume=True, model loaded from: {}".format(file_prefix))
        # If you want to start training a model from scratch
        else:
            # RNN layer
            # input_dim: The number of expected features in the input x
            # hidden_dim: the number of features in the hidden state h
            # layer_dim: Number of recurrent layers. i.e. if 2, it is stacking two RNNs together to form a stacked RNN
            # Initialize the model
            model = rnn.RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
            epoch_range = np.arange(num_epochs)
            # prtime("A new {} model instantiated, with run_train=True".format("rnn"))
            print("A new {} model instantiated, with run_train=True".format("rnn"))

        # Check if gpu support is available
        cuda_avail = torch.cuda.is_available()

        # Instantiating Loss Class
        criterion = nn.MSELoss()

        # Instantiate Optimizer Class
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr_config']['base'], weight_decay=weight_decay)

        if lr_schedule:
            # Set up learning rate scheduler.
            # Patience (for our case) is # of iterations, not epochs, but configs specification is num epochs
            scheduler = ReduceLROnPlateau(optimizer,
                                          mode='min',
                                          factor=configs['lr_config']['factor'],
                                          min_lr=configs['lr_config']['min'],
                                          patience=int(configs['lr_config']['patience'] * (num_train_data / train_batch_size)),
                                          verbose=True)

        prtime("Preparing model to train")
        prtime("starting to train the model for {} epochs!".format(num_epochs))

        if (len(epoch_range) == 0):
            epoch = resume_num_epoch + 1
            prtime("the previously saved model was at epoch= {}, which is same as num_epochs. So, not training"
                   .format(resume_num_epoch))

        if run_resume:
            n_iter = resume_n_iter
        else:
            n_iter = 0

        # Initialize re-trainable matrix
        # train_y_at_t = torch.zeros(train_batch_size, seq_dim, 1)  # 960 x 5 x 1
        # Loop through epochs
        for epoch in epoch_range:
            # This loop returns elements from the dataset batch by batch. Contains features AND targets
            for i, (feats, values) in enumerate(train_loader):
                model.train()
                # feats: 960x30 (tensor)
                # values: 960x1 (tensor)
                time1 = timeit.default_timer()

                # batch size x 5 x 6. -1 means "I don't know". Middle dimension is time.
                # (batches, timesteps, features)
                # features = Variable(feats.view(-1, seq_dim, input_dim - 1)) # size: (960x5x6)
                features = Variable(feats.view(-1, seq_dim, input_dim))
                target = Variable(values)  # size: batch size

                time2 = timeit.default_timer()

                # Clear gradients w.r.t. parameters (from previous epoch). Same as model.zero_grad()
                optimizer.zero_grad()

                # FORWARD PASS to get output/logits.
                # train_y_at_t is 960 x 5 x 1
                # features is     960 x 5 x 6
                # This command: (960x5x7) --> 960x1
                # outputs = model(torch.cat((features, train_y_at_t.detach_()), dim=2))
                outputs = model(features)

                time3 = timeit.default_timer()

                # tiling the 2nd axis of y_at_t from 1 to 5
                # train_y_at_t = tile(outputs.unsqueeze(2), 1, 5)
                # train_y_at_t_nump = train_y_at_t.detach().numpy()

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs.squeeze(), target)

                train_loss.append(loss.data.item())
                train_iter.append(n_iter)

                # Print to terminal and save training loss
                # prtime('Epoch: {} Iteration: {} TrainLoss: {}'.format(epoch, n_iter, train_loss[-1]))
                writer.add_scalars("Loss", {'Train': loss.data.item()}, n_iter)

                time4 = timeit.default_timer()

                # Does backpropogation and gets gradients, (the weights and bias). Create graph
                loss.backward()

                time5 = timeit.default_timer()

                if lr_schedule:
                    scheduler.step(loss)

                # Updating the weights/parameters. Clear computational graph.
                optimizer.step()

                # Each iteration is one batch
                n_iter += 1

                # Compute time per iteration
                time6 = timeit.default_timer()
                writer.add_scalars("Iteration time", {"Package_variables": time2 - time1,
                                                      "Evaluate_model": time3 - time2,
                                                      "Calc_loss": time4 - time3,
                                                      "Backprop": time5 - time4,
                                                      "Step": time6 - time5}, n_iter)

                # save the model every ___ iterations
                if n_iter % 200 == 0:
                    save_model(model, epoch, n_iter)

                # Do a val batch every ___ iterations
                if n_iter % 200 == 0:
                    # Evaluate val set
                    predictions, errors = test_processing(val_df, val_loader, model, seq_dim, input_dim,
                                                          val_batch_size, transformation_method, configs)
                    val_iter.append(n_iter)
                    #val_loss.append(errors['mse_loss'])
                    #val_rmse.append(errors['rmse'])
                    writer.add_scalars("Loss", {"val": errors['mse_loss']}, n_iter)

                    # Add matplotlib plot to TensorBoard to compare actual val set vs predicted
                    fig1, ax1 = plt.subplots(figsize=(20, 5))
                    ax1.plot(val_df[configs['target_var']], label='Actual', lw=0.5)
                    ax1.plot(predictions, label='Prediction', lw=0.5)
                    ax1.legend()
                    writer.add_figure('Predictions', fig1, n_iter)

                    # Add parody plot to TensorBoard
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(predictions, val_df[configs['target_var']], s=5, alpha=0.3)
                    strait_line = np.linspace(min(min(predictions), min(val_df[configs['target_var']])),
                                              max(max(predictions), max(val_df[configs['target_var']])), 5)
                    ax2.plot(strait_line, strait_line, c='k')
                    ax2.set_xlabel('Predicted')
                    ax2.set_ylabel('Observed')
                    ax2.axhline(y=0, color='k')
                    ax2.axvline(x=0, color='k')
                    ax2.axis('equal')
                    writer.add_figure('Parody', fig2, n_iter)

                    print('Epoch: {} Iteration: {}. Train_MSE: {}. val_MSE: {}, LR: {}'.format(epoch, n_iter,
                                                                                                loss.data.item(),
                                                                                                errors['mse_loss'],
                                                                                                optimizer.param_groups[
                                                                                                    0]['lr']))

        # Once model training is done, save the current model state
        save_model(model, epoch, n_iter)

        # Once model is done training, process a final val set
        predictions, errors = test_processing(val_df, val_loader, model, seq_dim, input_dim,
                                              val_batch_size, transformation_method, configs)

        # Save the final predictions and error statistics to a file
        predictions.to_csv(file_prefix + '/predictions.csv', index=False)
        #np.savetxt(file_prefix + '/final_rmse.csv', errors['rmse'], delimiter=",")

    # If you just want to immediately val the model on the existing (saved) model
    else:
        torch_model = torch.load(file_prefix + '/torch_model')
        model = torch_model['torch_model']
        prtime("Loaded model from file, given run_train=False\n")

        predictions, errors = test_processing(val_df, val_loader, model, seq_dim, input_dim,
                                              val_batch_size, transformation_method, configs)
        #val_loss.append(errors['mse_loss'])
        #val_rmse.append(errors['rmse'])
        writer.add_scalars("Loss", {"val": errors['mse_loss']})
        prtime('val_MSE: {}'.format(errors['mse_loss']))

        # Save the final predictions and error statistics to a file
        predictions.to_csv(file_prefix + '/predictions.csv', index=False)
        #np.savetxt(file_prefix + '/final_rmse.csv', errors['rmse'], delimiter=",")

    # If a training history csv file does not exist, make one
    if not pathlib.Path("Training_history.csv").exists():
        with open(r'Training_history.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["File Path", "RMSE", "CV(RMSE)", "NMBE", "GOF"])

    # Save the errors statistics to a file once everything is done
    with open(r'Training_history.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([file_prefix, errors["rmse"], errors["cvrmse"], errors["nmbe"], errors["gof"]])



def eval_trained_model(file_prefix, train_data, train_batch_size, configs):
    """
    Pass the entire training set through the trained model and get the predictions. Compute the residual and save to a DataFrame.
    :param file_prefix:
    :param train_data:
    :param train_batch_size:
    :param configs:
    :return:
    """
    # Evaluate the training model
    torch_model = torch.load(file_prefix + '/torch_model')
    model = torch_model['torch_model']

    X_train = train_data.drop(configs['target_var'], axis=1).values.astype(dtype='float32')
    y_train = train_data[configs['target_var']]
    y_train = y_train.values.astype(dtype='float32')
    train_feat_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
    train_target_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
    train = data_utils.TensorDataset(train_feat_tensor, train_target_tensor)
    train_loader = DataLoader(dataset=train, batch_size=train_batch_size, shuffle=False)

    model.eval()

    preds = []
    targets = []
    for i, (feats, values) in enumerate(train_loader):
        # features = Variable(feats.view(-1, seq_dim, input_dim - 1))
        features = Variable(feats.view(-1, configs['window'], configs['input_dim']))
        # outputs = model(torch.cat((features, val_y_at_t), dim=2))
        outputs = model(features)
        # val_y_at_t = tile(outputs.unsqueeze(2), 1, 5)
        preds.append(outputs.data.numpy().squeeze())
        targets.append(values.data.numpy())

    # concatenating the preds and targets for the whole epoch (iterating over val_loader once)
    semifinal_preds = np.concatenate(preds).ravel()
    semifinal_targs = np.concatenate(targets).ravel()  # Last 5 entries are nan

    # mask_file = os.path.join("data", "mask_{}_{}.json".format(configs['building'], "-".join(configs['year'])))
    # with open(mask_file, "r") as read_file:
    #     msk = json.load(read_file)

    # Get the saved binary mask from file
    mask_file = os.path.join("data", "mask_{}_{}.json".format(configs['building'], "-".join(configs['year'])))
    mask = pd.read_hdf(mask_file, key='df')
    msk = mask["msk"]

    # Adjust the datetime index so it is in line with the EC data
    # target_index = data_time_index[msk] + pd.DateOffset(
    #     minutes=(configs["EC_future_gap"] * configs["resample_freq"]))
    # processed_data = pd.DataFrame(index=target_index)

    # Adjust the datetime index so it is in line with the EC data
    target_index = mask.index[msk] + pd.DateOffset(
        minutes=(configs["EC_future_gap"] * configs["resample_freq"]))
    processed_data = pd.DataFrame(index=target_index)

    processed_data['Training fit'] = semifinal_preds
    processed_data['Target'] = semifinal_targs
    processed_data['Residual'] = semifinal_targs - semifinal_preds
    processed_data.to_hdf(os.path.join(file_prefix, "evaluated_training_model.h5"), key='df', mode='w')


def plot_processed_model(file_prefix):
    """
    Plot the trained model, along with the residuals for the trained model.
    The plot will show what time periods are not being captured by the model.
    :param file_prefix:
    :return:
    """
    processed_data = pd.read_hdf(os.path.join(file_prefix, "evaluated_training_model.h5"), key='df')
    # plt.figure(figsize=(9, 3))
    # plt.plot(processed_data["Target"], label='Targets')
    # plt.plot(processed_data["Training fit"], label='Training fit')
    # plt.ylabel("target variable")
    # plt.legend()
    # plt.show()

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(processed_data["Target"], label='Targets')
    axarr[0].plot(processed_data["Training fit"], label='Training fit')
    axarr[0].set_ylabel("target variable")
    axarr[0].legend()
    axarr[1].plot(processed_data['Residual'], label='Targets')
    axarr[1].set_ylabel("Residual")
    axarr[1].axhline(y=0, color='k')
    plt.show()


def main(train_df, val_df, configs):
    """
    Main executable for prepping data for input to RNN model.
    :param train_df:
    :param val_df:
    :param configs:
    :return:
    """
    transformation_method = configs['transformation_method']
    run_train = configs['run_train']
    num_epochs = configs['num_epochs']
    run_resume = configs['run_resume']
    tr_desired_batch_size = configs['tr_batch_size']
    te_desired_batch_size = configs['te_batch_size']

    building_ID = configs["building"]
    exp_id = configs['exp_id']
    arch_type = configs['arch_type']

    results_dir = configs["results_dir"]
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    global file_prefix
    file_prefix = os.path.join(results_dir, arch_type + '_M' + str(configs["target_var"].replace(" ", "")) + '_T' + str(
        exp_id))

    writer_path = file_prefix
    writer = SummaryWriter(writer_path)
    print("Writer path: {}".format(writer_path))

    # Get rid of datetime index
    if run_train:
        train_data = train_df.copy(deep=True)
        # train_data = train_data.drop('Date_time_MT', axis=1)
        train_data.reset_index(drop=True, inplace=True)
        train_df.reset_index(drop=True, inplace=True)
    else:
        train_data = train_df

    val_data = val_df.copy(deep=True)
    # val_data = val_data.drop('Date_time_MT', axis=1)
    val_data.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    # Normalization transformation
    train_data, val_data = data_transform(train_data, val_data, transformation_method, run_train)
    # prtime("data transformed using {} as transformation method".format(transformation_method))
    print("Data transformed using {} as transformation method".format(transformation_method))

    # Size the batches
    train_batch_size, val_batch_size, num_train_data = size_the_batches(train_data, val_data, tr_desired_batch_size,
                                                                         te_desired_batch_size)

    if configs["train_val_split"] == 'Sequential':
        # Normal: Convert to iterable dataset (DataLoaders)
        train_loader, val_loader, train_batch_size, val_batch_size = data_iterable(train_data, val_data, run_train,
                                                                                     train_batch_size,
                                                                                     val_batch_size, configs)

    elif configs["train_val_split"] == 'Random':
        # Already did sequential padding: Convert to iterable dataset (DataLoaders)
        train_loader, val_loader, train_batch_size, val_batch_size = data_iterable_random(train_data, val_data,
                                                                                            run_train,
                                                                                            train_batch_size,
                                                                                            val_batch_size, configs)
    prtime("data converted to iterable dataset")

    # Start the training process
    process(train_loader, val_loader, val_df, num_epochs, run_train, run_resume, writer, transformation_method,
            configs, train_batch_size, val_batch_size, configs['window'], num_train_data)

    # Evaluate the trained model with the training set to diagnose training ability and plot residuals
    # TODO: Currently only supported for random val/train split
    if configs["train_val_split"] == 'Random':
        eval_trained_model(file_prefix, train_data, train_batch_size, configs)
        # plot_processed_model(file_prefix)

