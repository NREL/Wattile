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

train_exp_num = 1  # increment this number everytime a new model is trained
test_exp_num = 1   # increment this number when the tests are run on an existing model (run_train = False)
writer = SummaryWriter('LoadForecasting_Results/Model_' + str(train_exp_num) + '/Test_num_' +  str(test_exp_num)+'/logs/FFNN_test')


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

    else:
        train_loader = []

    X_test = test_data.drop('EC', axis=1).values.astype(dtype='float32')
    y_test = test_data['EC'].values.astype(dtype='float32')

    test_feat_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)
    test_target_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)

    test = data_utils.TensorDataset(test_feat_tensor, test_target_tensor)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def process(train_loader, test_loader, num_epochs, run_train):

    # hyper-parameters
    num_epochs = num_epochs
    learning_rate = 0.00005
    input_dim = 14  # Fixed
    hidden_dim = 28
    output_dim = 1  # one prediction - energy consumption


    # initializing lists to store losses over epochs:
    train_loss = []
    train_iter = []
    test_loss = []
    test_iter = []
    preds = []


    if run_train:

        model = ffnn.FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
        prtime("ffnn model module executed to instantiate the FFNN model, with run_train=True")

        # Instantiating Loss Class
        criterion = nn.MSELoss()

        # Instantiate Optimizer Class
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


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

                prtime('Epoch: {} Iteration: {} TrainLoss: {}'.format(epoch, n_iter, train_loss[-1]))
                writer.add_scalar("/train_loss", loss.data.item(), n_iter)

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Getting gradients w.r.t. parameters
                loss.backward()

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
                        preds = outputs.data.numpy().squeeze()
                        writer.add_scalar("/test_loss", mse, n_iter)

                    print('Epoch: {} Iteration: {}. Train_MSE: {}. Test_MSE: {}'.format(epoch, n_iter, loss.data.item(), mse))


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

    train_loader, test_loader = data_iterable(train_data, test_data, run_train)
    prtime("data converted to iterable dataset")

    test_loss, train_loss, preds = process(train_loader, test_loader, num_epochs, run_train)


















