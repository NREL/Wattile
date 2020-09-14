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

file_prefix = '/default'

# Create lagged versions of exogenous variables
def seq_pad(a, window):
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
    train_bth = factors(train_data.shape[0])
    train_bt_size = min(train_bth, key=lambda x: abs(x - tr_desired_batch_size))

    val_bth = factors(val_data.shape[0])
    val_bt_size = min(val_bth, key=lambda x: abs(x - te_desired_batch_size))

    return train_bt_size, val_bt_size


# Normalization
def data_transform(train_data, val_data, transformation_method, run_train):
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


# Create lagged variables and convert train and val data to torch data types
def data_iterable(train_data, val_data, run_train, window, tr_desired_batch_size, te_desired_batch_size):
    train_batch_size, val_batch_size = size_the_batches(train_data, val_data, tr_desired_batch_size,
                                                         te_desired_batch_size)

    if run_train:

        # Create lagged INPUT variables, i.e. columns: w1_(t-1), w1_(t-2)...
        # Does this for all input variables for times up to "window"
        X_train = train_data.drop('STM_Xcel_Meter', axis=1).values.astype(dtype='float32')
        X_train = seq_pad(X_train, window)

        # Lag output variable, i.e. input for t=5 maps to EC at t=10, t=6 maps to EC t=11, etc.

        y_train = train_data['STM_Xcel_Meter'].shift(-window).fillna(method='ffill')
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
    X_val = val_data.drop('STM_Xcel_Meter', axis=1).values.astype(dtype='float32')
    X_val = seq_pad(X_val, window)

    y_val = val_data['STM_Xcel_Meter'].shift(-window).fillna(method='ffill')
    y_val = y_val.values.astype(dtype='float32')

    val_feat_tensor = torch.from_numpy(X_val).type(torch.FloatTensor)
    val_target_tensor = torch.from_numpy(y_val).type(torch.FloatTensor)

    val = data_utils.TensorDataset(val_feat_tensor, val_target_tensor)
    val_loader = DataLoader(dataset=val, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, train_batch_size, val_batch_size


def save_model(model, epoch, n_iter):
    model_dict = {'epoch_num': epoch, 'n_iter': n_iter, 'torch_model': model}
    torch.save(model_dict, file_prefix + '/torch_model')

    prtime("RNN model checkpoint saved")


def test_processing(val_df, val_loader, model, seq_dim, input_dim, val_batch_size, transformation_method):
    # val_df, val_loader, model, seq_dim, input_dim, val_batch_size, transformation_method
    model.eval()
    val_y_at_t = torch.zeros(val_batch_size, seq_dim, 1)
    preds = []
    targets = []
    for i, (feats, values) in enumerate(val_loader):
        features = Variable(feats.view(-1, seq_dim, input_dim - 1))
        outputs = model(torch.cat((features, val_y_at_t), dim=2))
        val_y_at_t = tile(outputs.unsqueeze(2), 1, 5)
        preds.append(outputs.data.numpy().squeeze())
        targets.append(values.data.numpy())

    # concatenating the preds and targets for the whole epoch (iterating over val_loader once)
    semifinal_preds = np.concatenate(preds).ravel()
    semifinal_targs = np.concatenate(targets).ravel()  # Last 5 entries are nan
    mse = np.mean((semifinal_targs - semifinal_preds) ** 2)

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
        final_preds = ((train_max['STM_Xcel_Meter'] - train_min['STM_Xcel_Meter']) * semifinal_preds) + train_min[
            'STM_Xcel_Meter']

    else:
        final_preds = ((semifinal_preds * train_std['STM_Xcel_Meter']) + train_mean['STM_Xcel_Meter'])

    predictions = pd.DataFrame(final_preds)
    denormalized_rmse = np.array(np.sqrt(np.mean((predictions.values.squeeze() - val_df.STM_Xcel_Meter.values) ** 2)),
                                 ndmin=1)
    return predictions, denormalized_rmse, mse


def process(train_loader, val_loader, val_df, num_epochs, run_train, run_resume, writer, transformation_method,
            configs, train_batch_size, val_batch_size, seq_dim):
    # hyper-parameters
    num_epochs = num_epochs
    #learning_rate = 0.0005
    learning_rate = 0.00005
    input_dim = 7  # Fixed
    hidden_dim = int(configs['hidden_nodes'])
    output_dim = 1  # one prediction - energy consumption
    layer_dim = 1
    weight_decay = float(configs['weight_decay'])

    configs['learning_rate'] = learning_rate
    configs['input_dim'] = input_dim
    configs['hidden_dim'] = hidden_dim
    configs['output_dim'] = output_dim
    configs['layer_dim'] = layer_dim
    configs['weight_decay'] = weight_decay

    path = file_prefix + '/configs.json'
    with open(path, 'w') as fp:
        json.dump(configs, fp)

    # initializing lists to store losses over epochs:
    train_loss = []
    train_iter = []
    val_loss = []
    val_iter = []
    val_rmse = []

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
            if configs["arch_type_variant"] == "vanilla":
                model = rnn.RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
            elif configs["arch_type_variant"] == "lstm":
                model = lstm.LSTM_Model(input_dim, hidden_dim, layer_dim, output_dim)
            epoch_range = np.arange(num_epochs)
            prtime("A new {} {} model instantiated, with run_train=True".format(configs["arch_type_variant"], configs["arch_type"]))

        # Check if gpu support is available
        cuda_avail = torch.cuda.is_available()

        # Instantiating Loss Class
        criterion = nn.MSELoss()

        # Instantiate Optimizer Class
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
        train_y_at_t = torch.zeros(train_batch_size, seq_dim, 1)  # 960 x 5 x 1
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
                features = Variable(feats.view(-1, seq_dim, input_dim - 1)) # size: (960x5x6)
                target = Variable(values)  # size: batch size

                time2 = timeit.default_timer()

                # Clear gradients w.r.t. parameters (from previous epoch). Same as model.zero_grad()
                optimizer.zero_grad()

                # FORWARD PASS to get output/logits.
                # train_y_at_t is 960 x 5 x 1
                # features is     960 x 5 x 6
                # This command: (960x5x7) --> 960x1
                outputs = model(torch.cat((features, train_y_at_t.detach_()), dim=2))

                time3 = timeit.default_timer()

                # tiling the 2nd axis of y_at_t from 1 to 5
                train_y_at_t = tile(outputs.unsqueeze(2), 1, 5)
                #train_y_at_t_nump = train_y_at_t.detach().numpy()

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs.squeeze(), target)

                train_loss.append(loss.data.item())
                train_iter.append(n_iter)

                # Print to terminal and save training loss
                prtime('Epoch: {} Iteration: {} TrainLoss: {}'.format(epoch, n_iter, train_loss[-1]))
                writer.add_scalars("Loss", {'Train': loss.data.item()}, n_iter)

                time4 = timeit.default_timer()

                # Does backpropogation and gets gradients, (the weights and bias). Create graph
                loss.backward()

                time5 = timeit.default_timer()

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

                # save the model every 50 iterations
                if n_iter % 50 == 0:
                    save_model(model, epoch, n_iter)

                # Do a val batch every 100 iterations
                if n_iter % 100 == 0:
                    predictions, denorm_rmse, mse = test_processing(val_df, val_loader, model, seq_dim, input_dim,
                                                                    val_batch_size, transformation_method)
                    val_iter.append(n_iter)
                    val_loss.append(mse)
                    val_rmse.append(denorm_rmse)
                    writer.add_scalars("Loss", {"val": mse}, n_iter)
                    # Add matplotlib plot to compare actual val set vs predicted
                    if (epoch == 199):
                        fig = plt.figure()
                        ax1 = fig.add_subplot(2, 1, 1)
                        ax1.plot(predictions, label='Prediction')
                        ax2 = fig.add_subplot(2, 1, 2)
                        ax2.plot(val_df['STM_Xcel_Meter'], label='Actual')
                        fig.savefig(file_prefix + '/test_graph.png')
                        writer.add_figure('Predictions', fig)

                        plt.plot(predictions, label='Prediction')
                        plt.plot(val_df['STM_Xcel_Meter'], label='Actual')
                        plt.savefig(file_prefix + '/overlayed-comparison.png')

                    print('Epoch: {} Iteration: {}. Train_MSE: {}. val_MSE: {}'.format(epoch, n_iter, loss.data.item(),mse))

        # Once model training is done, save it
        save_model(model, epoch, n_iter)

        predictions, denorm_rmse, mse = test_processing(val_df, val_loader, model, seq_dim, input_dim, val_batch_size, transformation_method)

        actual_values = pd.DataFrame(val_df['STM_Xcel_Meter'])
        preds_targets = pd.concat([actual_values, predictions], axis=1)
        preds_targets.columns = ['actual_consumption', 'predictions']
        # writing both predictions and target values to the csv
        preds_targets.to_csv(file_prefix + '/predictions.csv', index=False)
        np.savetxt(file_prefix + '/final_rmse.csv', denorm_rmse, delimiter=",")

    # If you just want to immediately val the model on the existing (saved) model
    else:
        torch_model = torch.load(file_prefix + '/torch_model')
        model = torch_model['torch_model']
        prtime("Loaded model from file, given run_train=False\n")

        predictions, denorm_rmse, mse = test_processing(val_df, val_loader, model, seq_dim, input_dim,
                                                        val_batch_size, transformation_method)
        val_loss.append(mse)
        val_rmse.append(denorm_rmse)
        writer.add_scalars("Loss", {"val": mse})

        prtime('val_MSE: {}'.format(mse))
        predictions.to_csv(file_prefix + '/predictions.csv', index=False)
        np.savetxt(file_prefix + '/final_rmse.csv', denorm_rmse, delimiter=",")


def main(train_df, val_df, configs):
    transformation_method = configs['transformation_method']
    run_train = configs['run_train']
    num_epochs = configs['num_epochs']
    run_resume = configs['run_resume']
    tr_desired_batch_size = configs['tr_batch_size']
    te_desired_batch_size = configs['te_batch_size']

    train_exp_id = configs['train_exp_id']
    exp_id = configs['exp_id']
    arch_type = configs['arch_type']

    results_dir = "EnergyForecasting_Results"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    global file_prefix
    file_prefix = os.path.join(results_dir, arch_type + '_M' + str(train_exp_id) + '_T' + str(
        exp_id))

    writer_path = file_prefix
    writer = SummaryWriter(writer_path)
    print(writer_path)

    # dropping the datetime_str column. Causes problem with normalization
    if run_train:
        train_data = train_df.copy(deep=True)
        train_data = train_data.drop('Date_time_MT', axis=1)

    else:
        train_data = train_df

    val_data = val_df.copy(deep=True)
    val_data = val_data.drop('Date_time_MT', axis=1)

    # Normalization transformation
    train_data, val_data = data_transform(train_data, val_data, transformation_method, run_train)
    prtime("data transformed using {} as transformation method".format(transformation_method))

    # Convert to iterable dataset (DataLoaders)
    window = 5  # window is synonymous to the "sequence length" dimension
    train_loader, val_loader, train_batch_size, val_batch_size = data_iterable(train_data, val_data, run_train,
                                                                                 window, tr_desired_batch_size,
                                                                                 te_desired_batch_size)
    prtime("data converted to iterable dataset")

    process(train_loader, val_loader, val_df, num_epochs, run_train, run_resume, writer, transformation_method,
            configs, train_batch_size, val_batch_size, seq_dim=window)
