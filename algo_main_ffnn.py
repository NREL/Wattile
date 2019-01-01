import os
import numpy as np
import pandas as pd
import json
from util import prtime, factors
import ffnn

import torch
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data_utils
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

file_prefix = "default/"

def size_the_batches(train_data, test_data, tr_desired_batch_size, te_desired_batch_size):

    train_bth = factors(train_data.shape[0])
    train_bt_size = min(train_bth, key=lambda x:abs(x-tr_desired_batch_size))

    test_bth = factors(test_data.shape[0])
    test_bt_size = min(test_bth, key=lambda x:abs(x-te_desired_batch_size))

    return train_bt_size, test_bt_size

def data_transform(train_data, test_data, transformation_method, run_train):

    if run_train:

        train_stats = {}
        train_stats['train_max'] = train_data.max().to_dict()
        train_stats['train_min'] = train_data.min().to_dict()
        train_stats['train_mean'] = train_data.mean(axis=0).to_dict()
        train_stats['train_std'] = train_data.std(axis=0).to_dict()

        path = file_prefix+ '/train_stats.json'
        with open(path, 'w') as fp:
            json.dump(train_stats, fp)

        if transformation_method == "minmaxscale":
            #min_max_scaler = preprocessing.MinMaxScaler()
            #temp_cols1 = train_data.columns.values
            #train_data = pd.DataFrame(min_max_scaler.fit_transform(train_data.values), columns=temp_cols1)
            train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        else:
            #stand_scaler = preprocessing.StandardScaler()
            #temp_cols1 = train_data.columns.values
            #train_data = pd.DataFrame(stand_scaler.fit_transform(train_data.values), columns=temp_cols1)
            train_data = (train_data - train_data.mean(axis=0)) / train_data.std(axis=0)



    # reading back the train stats for normalizing test data w.r.t to train data
    file_loc = file_prefix + '/train_stats.json'
    with open(file_loc, 'r') as f:
        train_stats = json.load(f)

    train_max = pd.DataFrame(train_stats['train_max'], index=[1]).iloc[0]
    train_min = pd.DataFrame(train_stats['train_min'], index=[1]).iloc[0]
    train_mean = pd.DataFrame(train_stats['train_mean'], index=[1]).iloc[0]
    train_std = pd.DataFrame(train_stats['train_std'], index=[1]).iloc[0]

    if transformation_method == "minmaxscale":
        test_data = (test_data - train_min) / (train_max - train_min)

    else:
        test_data = ((test_data - train_mean) / train_std)

    return train_data, test_data

def data_iterable(train_data, test_data, run_train, tr_desired_batch_size, te_desired_batch_size):

    train_batch_size, test_batch_size = size_the_batches(train_data, test_data, tr_desired_batch_size, te_desired_batch_size)
    #test_batch_size = 200

    if run_train:
        X_train = train_data.drop('EC', axis=1).values.astype(dtype='float32')

        y_train = train_data['EC'].shift(1).fillna(method='bfill')
        y_train = y_train.values.astype(dtype='float32')

        train_feat_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
        train_target_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)

        train = data_utils.TensorDataset(train_feat_tensor, train_target_tensor)
        train_loader = data_utils.DataLoader(train, batch_size=train_batch_size, shuffle=True)
        print("data train made iterable")

    else:
        train_loader = []

    X_test = test_data.drop('EC', axis=1).values.astype(dtype='float32')

    y_test = test_data['EC'].shift(1).fillna(method='bfill')
    y_test = y_test.values.astype(dtype='float32')

    test_feat_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)
    test_target_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)

    test = data_utils.TensorDataset(test_feat_tensor, test_target_tensor)
    test_loader = DataLoader(dataset=test, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, train_batch_size, test_batch_size

def save_model(model, epoch, n_iter):
    model_dict = {'epoch_num': epoch, 'n_iter':n_iter, 'torch_model':model}
    torch.save(model_dict, file_prefix + '/torch_model')
    prtime("FFNN Model checkpoint saved")

def test_processing(test_df, test_loader, model, input_dim, transformation_method):
    model.eval()
    preds = []
    targets = []
    for i, (feats, values) in enumerate(test_loader):
        features = Variable(feats.view(-1, input_dim))
        outputs = model(features)
        preds.append(outputs.data.numpy().squeeze())
        targets.append(values.data.numpy())

    # concatenating the preds and targets for the whole epoch (iterating over test_loader once)
    semifinal_preds = np.concatenate(preds).ravel()
    semifinal_targs = np.concatenate(targets).ravel()
    mse = np.mean((semifinal_targs - semifinal_preds) ** 2)

    # loading the training data stats for de-normalization purpose
    file_loc = file_prefix + '/train_stats.json'
    with open(file_loc, 'r') as f:
        train_stats = json.load(f)

    train_max = pd.DataFrame(train_stats['train_max'], index=[1]).iloc[0]
    train_min = pd.DataFrame(train_stats['train_min'], index=[1]).iloc[0]
    train_mean = pd.DataFrame(train_stats['train_mean'], index=[1]).iloc[0]
    train_std = pd.DataFrame(train_stats['train_std'], index=[1]).iloc[0]

    if transformation_method == "minmaxscale":
        final_preds = ((train_max['EC'] - train_min['EC']) * semifinal_preds) + train_min['EC']

    else:
        final_preds = ((semifinal_preds * train_std['EC']) + train_mean['EC'])

    predictions = pd.DataFrame(final_preds)
    denormalized_rmse = np.array(np.sqrt(np.mean((predictions.values.squeeze() - test_df.EC.values) ** 2)), ndmin=1)

    return predictions, denormalized_rmse, mse


def process(train_loader, test_loader, test_df, num_epochs, run_train, run_resume, writer, transformation_method, configs):

    # hyper-parameters
    num_epochs = num_epochs
    learning_rate = 0.0005
    input_dim = 15  # Fixed
    hidden_dim = int(configs['hidden_nodes'])
    output_dim = 1  # one prediction - energy consumption
    layer_dim = 1
    weight_decay = float(configs['weight_decay'])

    configs['learning_rate'] = learning_rate
    configs['input_dim'] = input_dim
    configs['hidden_dim'] = hidden_dim
    configs['output_dim'] = output_dim
    configs['layer_dim']= layer_dim
    configs['weight_decay'] = weight_decay

    path = file_prefix + '/configs.json'
    with open(path, 'w') as fp:
        json.dump(configs, fp)

    # initializing lists to store losses over epochs:
    train_loss = []
    train_iter = []
    test_loss = []
    test_iter = []
    test_rmse = []

    if run_train:

        if run_resume:
            torch_model = torch.load(file_prefix + '/torch_model')
            model = torch_model['torch_model']
            resume_num_epoch = torch_model['epoch_num']
            resume_n_iter = torch_model['n_iter']

            epoch_range = np.arange(resume_num_epoch+1, num_epochs+1)
            prtime("rune_resume=True, model loaded from: {}".format(file_prefix))
        else:
            model = ffnn.FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
            epoch_range = np.arange(num_epochs)
            prtime("A new {} model instantiated, with run_train=True".format("ffnn"))

        # Check if gpu support is available
        cuda_avail = torch.cuda.is_available()

        # Instantiating Loss Class
        criterion = nn.MSELoss()

        # Instantiate Optimizer Class
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

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

        for epoch in epoch_range:
            model.train()

            for i, (feats, values) in enumerate(train_loader):

                features = Variable(feats.view(-1, input_dim))
                target = Variable(values)

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Forward pass to get output/logits
                outputs = model(features)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs.squeeze(), target)
                #my_loss = np.mean((outputs.data.numpy().squeeze() - target.data.numpy())**2)

                train_loss.append(loss.data.item())
                train_iter.append(n_iter)

                prtime('Epoch: {} Iteration: {} TrainLoss: {}'.format(epoch, n_iter, train_loss[-1]))
                writer.add_scalar("/train_loss", loss.data.item(), n_iter)

                # Getting gradients w.r.t. parameters
                loss.backward(retain_graph=True)

                # Updating parameters
                optimizer.step()

                n_iter += 1

                # save the model every few iterations
                if n_iter %25 == 0:
                    save_model(model, epoch, n_iter)

                if n_iter % 100 == 0:
                    predictions, denorm_rmse, mse = test_processing(test_df, test_loader, model, input_dim, transformation_method)
                    test_iter.append(n_iter)
                    test_loss.append(mse)
                    test_rmse.append(denorm_rmse)

                    writer.add_scalar("test_loss", mse, n_iter)
                    writer.add_scalar("denorm_test_rmse", denorm_rmse, n_iter)

                    print('Epoch: {} Iteration: {}. Train_MSE: {}. Test_MSE: {}'.format(epoch, n_iter, loss.data.item(), mse))


        save_model(model, epoch, n_iter)

        predictions, denorm_rmse, mse = test_processing(test_df, test_loader, model, input_dim, transformation_method)

        predictions.to_csv(file_prefix + '/predictions.csv', index=False)
        np.savetxt(file_prefix + '/final_rmse.csv', denorm_rmse, delimiter=",")


    else:
        torch_model = torch.load(file_prefix + '/torch_model')
        model = torch_model['torch_model']
        prtime("Loaded model from file, given run_train=False\n")


        predictions, denorm_rmse, mse = test_processing(test_df, test_loader, model, input_dim, transformation_method)
        test_loss.append(mse)
        test_rmse.append(denorm_rmse)
        writer.add_scalar("/test_loss", mse)
        writer.add_scalar("/denorm_test_rmse", denorm_rmse)

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

    results_dir = "EnergyForecasting_Results"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    global file_prefix
    file_prefix = os.path.join(results_dir, arch_type + '_M' + str(train_exp_num) + '_T' + str(
        test_exp_num))

    writer_path = file_prefix
    writer = SummaryWriter(writer_path)
    print(writer_path)

    # dropping the datetime_str column. Causes problem with normalization
    if run_train:
        train_data = train_df.copy(deep=True)
        train_data = train_data.drop('datetime_str', axis=1)

    else:
        train_data = train_df

    test_data = test_df.copy(deep=True)
    test_data = test_data.drop('datetime_str', axis=1)

    train_data, test_data = data_transform(train_data, test_data, transformation_method, run_train)
    prtime("data transformed using {} as transformation method".format(transformation_method))

    train_loader, test_loader, train_batch_size, test_batch_size = data_iterable(train_data, test_data, run_train, tr_desired_batch_size, te_desired_batch_size)
    prtime("data converted to iterable dataset")

    process(train_loader, test_loader, test_df, num_epochs, run_train, run_resume, writer, transformation_method, configs)