import numpy as np
import pandas as pd
from sklearn import preprocessing
from util import prtime
import ffnn

import torch
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data_utils
import torch.nn as nn
from torchvision import transforms, utils
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# get this path to the tensorboardX results corrected
# writer = SummaryWriter("logs/FFNN_test_"+str(TEST_VERSION))


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

def data_iterable(train_data, test_data, run_train):

    batch_size = 100
    if run_train:
        X_train = train_data.drop('EC', axis=1).values.astype(dtype='float32')
        y_train = train_data['EC'].values.astype(dtype='float32')

        train_feat_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
        train_target_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)

        train = data_utils.TensorDataset(train_feat_tensor, train_target_tensor)
        train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=False)

        X_pred = test_data.drop('EC', axis=1).values.astype(dtype='float32')
        y_pred = test_data['EC'].values.astype(dtype='float32')

        pred_feat_tensor = torch.from_numpy(X_pred).type(torch.FloatTensor)
        pred_target_tensor = torch.from_numpy(y_pred).type(torch.FloatTensor)

        pred = data_utils.TensorDataset(pred_feat_tensor, pred_target_tensor)
        pred_loader = DataLoader(dataset=pred, batch_size=batch_size, shuffle=False)

        return train_loader, pred_loader

    else:
        X_pred = test_data.drop('EC', axis=1).values.astype(dtype='float32')
        y_pred = test_data['EC'].values.astype(dtype='float32')

        pred_feat_tensor = torch.from_numpy(X_pred).type(torch.FloatTensor)
        pred_target_tensor = torch.from_numpy(y_pred).type(torch.FloatTensor)

        pred = data_utils.TensorDataset(pred_feat_tensor, pred_target_tensor)
        pred_loader = DataLoader(dataset=pred, batch_size=batch_size, shuffle=False)

        return pred_loader


def main(train_df, test_df, transformation_method, run_train, num_epochs, train_exp_num):


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

    train_loader, test_loader = data_iterable(train_data, test_data, run_train)
    prtime("data converted to iterable dataset")

    

    # hyper-parameters
    num_epochs = num_epochs
    learning_rate = 0.00005
    input_dim = 14  # Fixed
    hidden_dim = 28
    output_dim = 1  # one prediction - energy consumption
    # Test set
    test_loss = []
    test_iter = []

    if run_train:

        model = ffnn.FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
        prtime("ffnn model module executed to instantiate the FFNN model, with run_train=True")

        # Instantiating Loss Class
        criterion = nn.MSELoss()

        # Instantiate Optimizer Class
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # initializing lists to store losses over epochs:
        train_loss = []
        train_iter = []

        prtime("Preparing model to train")
        prtime("starting to train the model for {} epochs!".format(num_epochs))

        n_iter = 0
        for epoch in range(num_epochs):
            for i, (feats, values) in enumerate(train_loader):

                features = Variable(feats.view(-1, 14))
                target = Variable(values)

                # Forward pass to get output/logits
                outputs = model(features)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs.squeeze(), target)

                train_loss.append(loss.data.item())
                train_iter.append(n_iter)

                print('Epoch: {} Iteration: {} TrainLoss: {}'.format(epoch, n_iter, train_loss[-1]))
                # get this path corrected
                # writer.add_scalar(str(freq) + "/train_loss", loss.data.item(), n_iter)

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

                if n_iter % 200 == 0:
                    for i, (feats, values) in enumerate(test_loader):
                        features = Variable(feats).view(-1, 14)
                        target = Variable(values)

                        outputs = model(features)

                        mse = np.sqrt(
                            np.mean((target.data.numpy() - outputs.data.numpy().squeeze()) ** 2) / len(target))

                        test_iter.append(n_iter)
                        test_loss.append(mse)
                        # get this corrected
                        # #writer.add_scalar(str(freq) + "/val_loss", mse, n_iter)

                    print('Epoch: {} Iteration: {}. Train_MSE: {}. Test_MSE: {}'.format(epoch, n_iter, loss.data.item(),
                                                                                        mse))

                n_iter += 1

    else:
        model = torch.load('LoadForecasting_Results/Model_' + str(train_exp_num) + '/torch_model')
        prtime("Loaded model from file, given run_train=False\n")













