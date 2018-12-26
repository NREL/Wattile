import numpy as np
import pandas as pd
from sklearn import preprocessing
from util import prtime, factors, tile
import ffnn

import torch
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data_utils
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter


def size_the_batches(train_data, test_data, desired_batch_size):

    train_bth = factors(train_data.shape[0])
    train_bt_size = min(train_bth, key=lambda x:abs(x-desired_batch_size))

    test_bth = factors(test_data.shape[0])
    test_bt_size = min(test_bth, key=lambda x:abs(x-desired_batch_size))

    return train_bt_size, test_bt_size

def data_transform(train_data, test_data, transformation_method, run_train, arch_type, train_exp_num):

    if run_train:

        train_max = train_data.EC.max()
        train_min = train_data.EC.min()
        min_max = pd.DataFrame({'train_min': train_min, 'train_max': train_max}, index=[1])
        min_max.to_csv('EnergyForecasting_Results/' + arch_type + '/Model_' + str(train_exp_num) + + '/min_max.csv')

        if transformation_method == "minmaxscale":
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



    if transformation_method == "minmaxscale":
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

    desired_batch_size = 100
    train_batch_size, test_batch_size = size_the_batches(train_data, test_data, desired_batch_size)

    if run_train:
        X_train = train_data.drop('EC', axis=1).values.astype(dtype='float32')

        y_train = train_data['EC'].values.astype(dtype='float32')

        train_feat_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
        train_target_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)

        train = data_utils.TensorDataset(train_feat_tensor, train_target_tensor)
        train_loader = data_utils.DataLoader(train, batch_size=train_batch_size, shuffle=False)
        print("data train made iterable")

    else:
        train_loader = []

    X_test = test_data.drop('EC', axis=1).values.astype(dtype='float32')

    y_test = test_data['EC'].values.astype(dtype='float32')

    test_feat_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)
    test_target_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)

    test = data_utils.TensorDataset(test_feat_tensor, test_target_tensor)
    test_loader = DataLoader(dataset=test, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, train_batch_size, test_batch_size

def save_model(model,arch_type, train_exp_num):
    torch.save(model, 'EnergyForecasting_Results/' + arch_type + '/Model_' + str(train_exp_num) + '/torch_model')
    prtime("FFNN Model checkpoint saved")

def post_processing(test_df, test_loader, model, input_dim, arch_type, train_exp_num):
    model.eval()
    preds = []
    for i, (feats, values) in enumerate(test_loader):
        features = Variable(feats.view(-1, input_dim))
        output = model(features)
        preds.append(output.data.numpy().squeeze())
    # concatenating the preds done in
    semifinal_preds = np.concatenate(preds).ravel()
    # loading the min and max values of the energy consumption column of the train data
    min_max = pd.read_csv('EnergyForecasting_Results/' + arch_type + '/Model_' + str(train_exp_num) + '/min_max.csv')

    final_preds = (((min_max['train_max'].values[0] - min_max['train_min'].values[0]) * semifinal_preds) + min_max['train_min'].values[0]).tolist()
    predictions = pd.DataFrame(np.array(final_preds))
    denormalized_mse = np.array(np.mean((predictions.values.squeeze() - test_df.EC.values) ** 2), ndmin=1)
    predictions.to_csv('predictions.csv')
    np.savetxt('result_mse.csv', denormalized_mse, delimiter=",")


def process(train_loader, test_loader, test_df, num_epochs, run_train, train_batch_size, test_batch_size, run_resume, arch_type, train_exp_num, writer):

    # hyper-parameters
    num_epochs = num_epochs
    learning_rate = 0.0005
    input_dim = 14  # Fixed
    hidden_dim = 28
    output_dim = 1  # one prediction - energy consumption
    layer_dim = 1


    # initializing lists to store losses over epochs:
    train_loss = []
    train_iter = []
    test_loss = []
    test_iter = []


    if run_train:

        #model = ffnn.FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

        if run_resume:
            model = torch.load('EnergyForecasting_Results/' + arch_type + '/Model_' + str(train_exp_num) + '/torch_model')
            prtime("model {} loaded, with run_resume=True and run_train=True".format('Model_' + str(train_exp_num)))
        else:
            model = ffnn.FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
            prtime("A new {} model instantiated, with run_train=True".format("ffnn"))

        # Check if gpu support is available
        cuda_avail = torch.cuda.is_available()

        # Instantiating Loss Class
        criterion = nn.MSELoss()

        # Instantiate Optimizer Class
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


        prtime("Preparing model to train")
        prtime("starting to train the model for {} epochs!".format(num_epochs))

        n_iter = 0
        for epoch in range(num_epochs):
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
                    save_model(model, arch_type, train_exp_num)

                if n_iter % 150 == 0:
                    model.eval()
                    for i, (feats, values) in enumerate(test_loader):
                        features = Variable(feats.view(-1, input_dim))
                        target = Variable(values)

                        outputs = model(features)

                        mse = np.sqrt(
                            np.mean((target.data.numpy() - outputs.data.numpy().squeeze()) ** 2))

                        test_iter.append(n_iter)
                        test_loss.append(mse)
                        writer.add_scalar("/test_loss", mse, n_iter)

                    print('Epoch: {} Iteration: {}. Train_MSE: {}. Test_MSE: {}'.format(epoch, n_iter, loss.data.item(), mse))

        #torch.save(model, 'EnergyForecasting_Results/' + arch_type + '/Model_' + str(train_exp_num) + '/torch_model')
        save_model(model, arch_type, train_exp_num)

        post_processing(test_df, test_loader, model, input_dim, arch_type, train_exp_num)




    else:
        model = torch.load('EnergyForecasting_Results/' + arch_type + '/Model_' + str(train_exp_num) + '/torch_model')
        prtime("Loaded model from file, given run_train=False\n")

        for i, (feats, values) in enumerate(test_loader):
            features = Variable(feats.view(-1,input_dim))
            target = Variable(values)

            outputs = model(features)

            mse = np.sqrt(np.mean((target.data.numpy() - outputs.data.numpy().squeeze()) ** 2) / len(target))

            test_loss.append(mse)

            writer.add_scalar("/test_loss", mse)


        prtime('Test_MSE: {}'.format(mse))
        post_processing(test_df, test_loader, model, input_dim, arch_type, train_exp_num)



def main(train_df, test_df, configs):
    transformation_method = configs['transformation_method']
    run_train = configs['run_train']
    num_epochs = configs['num_epochs']
    run_resume = configs['run_resume']

    train_exp_num = configs['train_exp_num']
    test_exp_num = configs['test_exp_num']
    arch_type = configs['arch_type']
    writer_path = 'EnergyForecasting_Results/' + arch_type + '/Model_' + str(train_exp_num) + '/TestNum_' + str(
        test_exp_num) + '/logs/'
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

    train_data, test_data = data_transform(train_data, test_data, transformation_method, run_train, arch_type, train_exp_num)
    prtime("data transformed using {} as transformation method".format(transformation_method))

    train_loader, test_loader, train_batch_size, test_batch_size = data_iterable(train_data, test_data, run_train)
    prtime("data converted to iterable dataset")

    process(train_loader, test_loader, test_df, num_epochs, run_train, train_batch_size, test_batch_size, run_resume, arch_type, train_exp_num, writer)