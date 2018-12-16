import numpy as np
import pandas as pd
from sklearn import preprocessing
from util import prtime
import ffnn
import rnn

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.utils.data as data_utils
import torch.nn as nn
from torchvision import transforms, utils
from torch.autograd import Variable
from tensorboardX import SummaryWriter

train_exp_num = 1  # increment this number everytime a new model is trained
test_exp_num = 1   # increment this number when the tests are run on an existing model (run_train = False)
writer = SummaryWriter('LoadForecasting_Results/Model_' + str(train_exp_num) + '/Test_num_' +  str(test_exp_num)+'/logs/FFNN_test')


def seq_pad(a, window):
    rows = a.shape[0]
    cols = a.shape[1]

    b = np.zeros((rows, window * cols))

    for i in range(window):
        if i == 0:
            b[:, 0:cols] = a
        else:
            b[i:, i * cols:(i + 1) * cols] = a[:-i, :]

    for i in list(np.arange(window - 1)):
        j = (i * cols) + cols
        b[i, j:] = np.tile(b[i, 0:cols], window - i-1)

    return b

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def data_transform(train_data, test_data, transformation_method, run_train):

    if run_train:
        if transformation_method == "minmaxscaling":
            min_max_scaler = preprocessing.MinMaxScaler()
            temp_cols1 = train_data.columns.values
            train_data = pd.DataFrame(min_max_scaler.fit_transform(train_data.values), columns=temp_cols1)

        elif transformation_method == "standardize":
            stand_scaler = preprocessing.StandardScaler()
            temp_cols1 = train_data.columns.values
            train_data = pd.DataFrame(stand_scaler.fit_transform(train_data.values), columns=temp_cols1)

        else:
            temp_cols1 = train_data.columns.values
            train_data = pd.DataFrame(preprocessing.normalize(train_data.values), columns=temp_cols1)

    if transformation_method == "minmaxscaling":
        min_max_scaler = preprocessing.MinMaxScaler()
        temp_cols1 = test_data.columns.values
        test_data = pd.DataFrame(min_max_scaler.fit_transform(test_data.values), columns=temp_cols1)

    elif transformation_method == "standardize":
        stand_scaler = preprocessing.StandardScaler()
        temp_cols1 = test_data.columns.values
        test_data = pd.DataFrame(stand_scaler.fit_transform(test_data.values), columns=temp_cols1)

    else:
        temp_cols1 = test_data.columns.values
        test_data = pd.DataFrame(preprocessing.normalize(test_data.values), columns=temp_cols1)

    return train_data, test_data

def data_iterable(train_data, test_data, run_train, window):

    batch_size = 100
    if run_train:
        X_train = train_data.drop('EC', axis=1).values.astype(dtype='float32')
        X_train = seq_pad(X_train, window)

        y_train = train_data['EC'].shift(window).fillna(method='bfill')
        y_train = y_train.values.astype(dtype='float32')

        train_feat_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
        train_target_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)

        train = data_utils.TensorDataset(train_feat_tensor, train_target_tensor)
        train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=False)
        print("data train made iterable")

    else:
        train_loader = []

    X_test = test_data.drop('EC', axis=1).values.astype(dtype='float32')
    X_test = seq_pad(X_test, window)

    y_test = test_data['EC'].shift(window).fillna(method='bfill')
    y_test = y_test.values.astype(dtype='float32')

    test_feat_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)
    test_target_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)

    test = data_utils.TensorDataset(test_feat_tensor, test_target_tensor)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def process(train_loader, test_loader, num_epochs, run_train):

    # hyper-parameters
    num_epochs = num_epochs
    learning_rate = 0.00005
    input_dim = 15  # Fixed
    hidden_dim = 28
    output_dim = 1  # one prediction - energy consumption
    layer_dim = 1
    seq_dim = 5


    # initializing lists to store losses over epochs:
    train_loss = []
    train_iter = []
    test_loss = []
    test_iter = []
    preds = []


    if run_train:

        #model = ffnn.FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
        model = rnn.RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
        prtime("{} model module executed to instantiate the FFNN model, with run_train=True".format("rnn"))

        # Instantiating Loss Class
        criterion = nn.MSELoss()

        # Instantiate Optimizer Class
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


        prtime("Preparing model to train")
        prtime("starting to train the model for {} epochs!".format(num_epochs))

        n_iter = 0
        #y_at_t = torch.FloatTensor()
        y_at_t = torch.zeros(100,seq_dim,1)
        for epoch in range(num_epochs):
            for i, (feats, values) in enumerate(train_loader):

                features = Variable(feats.view(-1, seq_dim, input_dim-1))
                target = Variable(values)

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Forward pass to get output/logits
                outputs = model(torch.cat((features, y_at_t), dim=2))

                #
                y_at_t = tile(outputs.unsqueeze(2), 1,5)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs.squeeze(), target)

                train_loss.append(loss.data.item())
                train_iter.append(n_iter)

                prtime('Epoch: {} Iteration: {} TrainLoss: {}'.format(epoch, n_iter, train_loss[-1]))
                writer.add_scalar("/train_loss", loss.data.item(), n_iter)

                # Getting gradients w.r.t. parameters
                loss.backward(retain_graph=True)

                # Updating parameters
                optimizer.step()

                n_iter += 1

                if n_iter % 200 == 0:
                    for i, (feats, values) in enumerate(test_loader):
                        features = Variable(feats).view(-1, 14)
                        target = Variable(values)

                        outputs = model(features)

                        mse = np.sqrt(
                            np.mean((target.data.numpy() - outputs.data.numpy().squeeze()) ** 2) / len(target))

                        test_iter.append(n_iter)
                        test_loss.append(mse)
                        preds.append(outputs.data.numpy().squeeze())
                        writer.add_scalar("/test_loss", mse, n_iter)

                    print('Epoch: {} Iteration: {}. Train_MSE: {}. Test_MSE: {}'.format(epoch, n_iter, loss.data.item(), mse))
        semifinal_preds = np.concatenate(preds).ravel()

        torch.save(model, 'LoadForecasting_Results/Model_' + str(train_exp_num) + '/torch_model')



    else:
        model = torch.load('LoadForecasting_Results/Model_' + str(train_exp_num) + '/torch_model')
        prtime("Loaded model from file, given run_train=False\n")

        for i, (feats, values) in enumerate(test_loader):
            features = Variable(feats).view(-1, 14)
            target = Variable(values)

            outputs = model(features)

            mse = np.sqrt(np.mean((target.data.numpy() - outputs.data.numpy().squeeze()) ** 2) / len(target))

            test_loss.append(mse)

            writer.add_scalar("/test_loss", mse)
            preds.append(outputs.data.numpy().squeeze())
        semifinal_preds = np.concatenate(preds).ravel()

        prtime('Test_MSE: {}'.format(mse))

    return train_loss, test_loss, preds




def main(train_df, test_df, transformation_method, run_train, num_epochs):


    # dropping the datetime_str column. Causes problem with normalization
    if run_train:
        train_data = train_df.copy()
        train_data = train_data.drop('datetime_str', axis=1)

    else:
        train_data = train_df

    test_data = test_df.copy()
    test_data = test_data.drop('datetime_str', axis=1)

    train_data, test_data = data_transform(train_data, test_data, transformation_method, run_train)
    prtime("data transformed using {} as transformation method".format(transformation_method))

    window = 5    # window is synonomus to the "sequence length" dimension
    train_loader, test_loader = data_iterable(train_data, test_data, run_train, window)
    prtime("data converted to iterable dataset")

    test_loss, train_loss, preds = process(train_loader, test_loader, num_epochs, run_train)
    print(preds)


















