import numpy as np
import pandas as pd
from sklearn import preprocessing


def data_transform(train_data, test_Data, transformation_method, run_train):

    if run_train:
        if transformation_method == "normalize":
            min_max_scaler = preprocessing.MinMaxScaler()
            temp_cols1 = train_data.columns.values
            train_data = pd.DataFrame(min_max_scaler.fit_transform(train_data.values), columns=temp_cols1)

        elif transformation_method == "standardize":
            do that

        else:
            minmaxscaling


    if transformation_method == "normalize":
        min_max_scaler = preprocessing.MinMaxScaler()
        temp_cols1 = train_data.columns.values
        train_data = pd.DataFrame(min_max_scaler.fit_transform(train_data.values), columns=temp_cols1)

    elif transformation_method == "standardize":
        do
        that

    else:
        minmaxscaling



def main(train_df, test_df, transformation_method, run_train, num_epochs):


    # dropping the datetime_str column. Causes problem with normalization
    if run_train:
        train_data = train_df.copy()
        train_data = train_data.drop('datetime_str', axis=1)

    test_data = test_df.copy()
    test_data = test_data.drop('datetime_str', axis=1)

    train_data, test_data = data_transform(train_data, test_data, transformation_method, run_train)



