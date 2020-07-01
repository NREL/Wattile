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

file_prefix = '/default'


def seq_pad(a, window):
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


def size_the_batches(train_data, test_data, tr_desired_batch_size, te_desired_batch_size):
    # Find factors of the length of train and test df's and pick the closest one to the requested batch sizes
    train_bth = factors(train_data.shape[0])
    train_bt_size = min(train_bth, key=lambda x: abs(x - tr_desired_batch_size))

    test_bth = factors(test_data.shape[0])
    test_bt_size = min(test_bth, key=lambda x: abs(x - te_desired_batch_size))

    train_ratio = int(train_data.shape[0]*100/(train_data.shape[0]+test_data.shape[0]))
    test_ratio = 100-train_ratio
    print("Train size: {}, Test size: {}, split {}:{}".format(train_data.shape[0], test_data.shape[0], train_ratio, test_ratio))
    print("Available train batch sizes: {}".format(sorted(train_bth)))
    print("Requested size of batches - Train: {}, Test: {}".format(tr_desired_batch_size, te_desired_batch_size))
    print("Actual size of batches - Train: {}, Test: {}".format(train_bt_size, test_bt_size))
    print("Number of batches in 1 epoch - Train: {}, Test: {}".format(train_data.shape[0]/train_bt_size, test_data.shape[0]/test_bt_size))

    return train_bt_size, test_bt_size


# Normalization
def data_transform(train_data, test_data, transformation_method, run_train):
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

    # reading back the train stats for normalizing test data w.r.t to train data
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
        test_data = (test_data - train_min) / (train_max - train_min)

    else:
        test_data = ((test_data - train_mean) / train_std)

    return train_data, test_data


# Create lagged variables and convert train and test data to torch data types
def data_iterable(train_data, test_data, run_train, train_batch_size, test_batch_size, configs):

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

    # Do the same as above for the test set
    X_test = test_data.drop(configs['target_var'], axis=1).values.astype(dtype='float32')
    X_test = seq_pad(X_test, configs['window'])

    y_test = test_data[configs['target_var']].shift(-configs['EC_future_gap']).fillna(method='ffill')
    y_test = y_test.values.astype(dtype='float32')

    test_feat_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)
    test_target_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)

    test = data_utils.TensorDataset(test_feat_tensor, test_target_tensor)
    test_loader = DataLoader(dataset=test, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, train_batch_size, test_batch_size

def data_iterable_random(train_data, test_data, run_train, train_batch_size, test_batch_size, configs):

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

    # Do the same as above for the test set
    X_test = test_data.drop(configs['target_var'], axis=1).values.astype(dtype='float32')

    y_test = test_data[configs['target_var']]
    y_test = y_test.values.astype(dtype='float32')

    test_feat_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)
    test_target_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)

    test = data_utils.TensorDataset(test_feat_tensor, test_target_tensor)
    test_loader = DataLoader(dataset=test, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, train_batch_size, test_batch_size

def save_model(model, epoch, n_iter):
    model_dict = {'epoch_num': epoch, 'n_iter': n_iter, 'torch_model': model}
    torch.save(model_dict, file_prefix + '/torch_model')

    #prtime("RNN model checkpoint saved")


def test_processing(test_df, test_loader, model, seq_dim, input_dim, test_batch_size, transformation_method, configs):
    # test_df, test_loader, model, seq_dim, input_dim, test_batch_size, transformation_method
    model.eval()
    #test_y_at_t = torch.zeros(test_batch_size, seq_dim, 1)
    preds = []
    targets = []
    for i, (feats, values) in enumerate(test_loader):
        #features = Variable(feats.view(-1, seq_dim, input_dim - 1))
        features = Variable(feats.view(-1, seq_dim, input_dim))
        #outputs = model(torch.cat((features, test_y_at_t), dim=2))
        outputs = model(features)
        #test_y_at_t = tile(outputs.unsqueeze(2), 1, 5)
        preds.append(outputs.data.numpy().squeeze())
        targets.append(values.data.numpy())

    # concatenating the preds and targets for the whole epoch (iterating over test_loader once)
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

    # Do do-normalization process on predictions from test set
    if transformation_method == "minmaxscale":
        final_preds = ((train_max[configs['target_var']] - train_min[configs['target_var']]) * semifinal_preds) + train_min[
            configs['target_var']]

    else:
        final_preds = ((semifinal_preds * train_std[configs['target_var']]) + train_mean[configs['target_var']])

    predictions = pd.DataFrame(final_preds)
    # denormalized_rmse = np.array(np.sqrt(np.mean((predictions.values.squeeze() - test_df.STM_Xcel_Meter.values) ** 2)),
    #                              ndmin=1)
    denormalized_rmse = np.array(np.sqrt(np.mean((predictions.values.squeeze() - test_df[configs['target_var']].values) ** 2)),
                                 ndmin=1)

    return predictions, denormalized_rmse, mse


def process(train_loader, test_loader, test_df, num_epochs, run_train, run_resume, writer, transformation_method,
            configs, train_batch_size, test_batch_size, seq_dim):

    # ___ Hyper-parameters
    # Input_dim: Determined automatically
    num_epochs = num_epochs
    learning_rate_base = configs['learning_rate_base']
    lr_schedule = configs['lr_schedule']
    hidden_dim = int(configs['hidden_nodes'])
    output_dim = 1  # one prediction - energy consumption
    layer_dim = 1
    weight_decay = float(configs['weight_decay'])

    configs['learning_rate_base'] = learning_rate_base
    #configs['input_dim'] = input_dim
    input_dim = configs['input_dim']
    configs['hidden_dim'] = hidden_dim
    configs['output_dim'] = output_dim
    configs['layer_dim'] = layer_dim
    configs['weight_decay'] = weight_decay

    # Write the configurations used for this training process to a json file
    path = file_prefix + '/configs.json'
    with open(path, 'w') as fp:
        json.dump(configs, fp, indent=1)

    # initializing lists to store losses over epochs:
    train_loss = []
    train_iter = []
    test_loss = []
    test_iter = []
    test_rmse = []

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
            #prtime("A new {} model instantiated, with run_train=True".format("rnn"))
            print("A new {} model instantiated, with run_train=True".format("rnn"))

        # Check if gpu support is available
        cuda_avail = torch.cuda.is_available()

        # Instantiating Loss Class
        criterion = nn.MSELoss()

        # Instantiate Optimizer Class
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_base, weight_decay=weight_decay)

        if lr_schedule:
            # Set up learning rate scheduler. patience (for our case) is # of iterations, not epochs
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, min_lr=1e-04, patience=2000, verbose=True)

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
        #train_y_at_t = torch.zeros(train_batch_size, seq_dim, 1)  # 960 x 5 x 1
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
                #features = Variable(feats.view(-1, seq_dim, input_dim - 1)) # size: (960x5x6)
                features = Variable(feats.view(-1, seq_dim, input_dim))
                target = Variable(values)  # size: batch size

                time2 = timeit.default_timer()

                # Clear gradients w.r.t. parameters (from previous epoch). Same as model.zero_grad()
                optimizer.zero_grad()

                # FORWARD PASS to get output/logits.
                # train_y_at_t is 960 x 5 x 1
                # features is     960 x 5 x 6
                # This command: (960x5x7) --> 960x1
                #outputs = model(torch.cat((features, train_y_at_t.detach_()), dim=2))
                outputs = model(features)

                time3 = timeit.default_timer()

                # tiling the 2nd axis of y_at_t from 1 to 5
                #train_y_at_t = tile(outputs.unsqueeze(2), 1, 5)
                #train_y_at_t_nump = train_y_at_t.detach().numpy()

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs.squeeze(), target)

                train_loss.append(loss.data.item())
                train_iter.append(n_iter)

                # Print to terminal and save training loss
                #prtime('Epoch: {} Iteration: {} TrainLoss: {}'.format(epoch, n_iter, train_loss[-1]))
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
                writer.add_scalars("Iteration time", {"dt1": time2-time1,
                                                      "dt2": time3-time2,
                                                      "dt3": time4-time3,
                                                      "dt4": time5-time4,
                                                      "dt5": time6-time5}, n_iter)

                # save the model every __ iterations
                if n_iter % 200 == 0:
                    save_model(model, epoch, n_iter)

                # Do a test batch every 100 iterations
                if n_iter % 200 == 0:
                    predictions, denorm_rmse, mse = test_processing(test_df, test_loader, model, seq_dim, input_dim,
                                                                    test_batch_size, transformation_method, configs)
                    test_iter.append(n_iter)
                    test_loss.append(mse)
                    test_rmse.append(denorm_rmse)
                    writer.add_scalars("Loss", {"Test": mse}, n_iter)

                    # Add matplotlib plot to compare actual test set vs predicted
                    fig1, ax1 = plt.subplots(figsize=(20, 5))
                    ax1.plot(test_df[configs['target_var']], label='Actual', lw=0.5)
                    ax1.plot(predictions, label='Prediction', lw=0.5)
                    ax1.legend()
                    writer.add_figure('Predictions', fig1, n_iter)

                    # Add parody plot
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(predictions, test_df[configs['target_var']], s=5, alpha=0.3)
                    strait_line = np.linspace(min(min(predictions), min(test_df[configs['target_var']])),
                                     max(max(predictions), max(test_df[configs['target_var']])), 5)
                    ax2.plot(strait_line, strait_line, c='k')
                    ax2.set_xlabel('Predicted')
                    ax2.set_ylabel('Observed')
                    ax2.axhline(y=0, color='k')
                    ax2.axvline(x=0, color='k')
                    ax2.axis('equal')
                    writer.add_figure('Parody', fig2, n_iter)

                    print('Epoch: {} Iteration: {}. Train_MSE: {}. Test_MSE: {}, LR: {}'.format(epoch, n_iter, loss.data.item(),
                                                                                        mse, optimizer.param_groups[0]['lr']))

        # Once model training is done, save it
        save_model(model, epoch, n_iter)

        predictions, denorm_rmse, mse = test_processing(test_df, test_loader, model, seq_dim, input_dim,
                                                        test_batch_size, transformation_method, configs)
        predictions.to_csv(file_prefix + '/predictions.csv', index=False)
        np.savetxt(file_prefix + '/final_rmse.csv', denorm_rmse, delimiter=",")

    # If you just want to immediately test the model on the existing (saved) model
    else:
        torch_model = torch.load(file_prefix + '/torch_model')
        model = torch_model['torch_model']
        prtime("Loaded model from file, given run_train=False\n")

        predictions, denorm_rmse, mse = test_processing(test_df, test_loader, model, seq_dim, input_dim,
                                                        test_batch_size, transformation_method, configs)
        test_loss.append(mse)
        test_rmse.append(denorm_rmse)
        writer.add_scalars("Loss", {"Test": mse})

        prtime('Test_MSE: {}'.format(mse))
        predictions.to_csv(file_prefix + '/predictions.csv', index=False)
        np.savetxt(file_prefix + '/final_rmse.csv', denorm_rmse, delimiter=",")


def main(train_df, test_df, configs):
    transformation_method = configs['transformation_method']
    run_train = configs['run_train']
    num_epochs = configs['num_epochs']
    run_resume = configs['run_resume']
    tr_desired_batch_size = configs['tr_batch_size']
    te_desired_batch_size = configs['te_batch_size']

    train_exp_num = configs['train_exp_num']
    test_exp_num = configs['test_exp_num']
    arch_type = configs['arch_type']

    #configs['input_dim'] = train_df.shape[1] - 1

    #results_dir = "EnergyForecasting_Results"
    results_dir = configs["results_dir"]
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    global file_prefix
    file_prefix = os.path.join(results_dir, arch_type + '_M' + str(train_exp_num) + '_T' + str(
        test_exp_num))

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

    test_data = test_df.copy(deep=True)
    # test_data = test_data.drop('Date_time_MT', axis=1)
    test_data.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Normalization transformation
    train_data, test_data = data_transform(train_data, test_data, transformation_method, run_train)
    #prtime("data transformed using {} as transformation method".format(transformation_method))
    print("Data transformed using {} as transformation method".format(transformation_method))

    # Size the batches
    train_batch_size, test_batch_size = size_the_batches(train_data, test_data, tr_desired_batch_size,
                                                         te_desired_batch_size)

    if configs["TrainTestSplit"] == 'Sequential':
        # Normal: Convert to iterable dataset (DataLoaders)
        train_loader, test_loader, train_batch_size, test_batch_size = data_iterable(train_data, test_data, run_train,
                                                                                     train_batch_size,
                                                                                     test_batch_size, configs)

    elif configs["TrainTestSplit"] == 'Random':
        # Already did sequential padding: Convert to iterable dataset (DataLoaders)
        train_loader, test_loader, train_batch_size, test_batch_size = data_iterable_random(train_data, test_data, run_train,
                                                                                     train_batch_size,
                                                                                     test_batch_size, configs)
    # Everything should be the same fron this point on
    prtime("data converted to iterable dataset")


    process(train_loader, test_loader, test_df, num_epochs, run_train, run_resume, writer, transformation_method,
            configs, train_batch_size, test_batch_size, seq_dim=configs['window'])
