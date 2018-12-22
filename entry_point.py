import data_preprocessing
import algo_main_rnn
import algo_main_ffnn
import algo_main_lstm
import algo_main_gru
from util import get_arguments
import pandas as pd


train_start_date, train_end_date, test_start_date, test_end_date, transformation_method, run_train, num_epochs, run_resume, preprocess, arch_type = get_arguments()

if preprocess:
    train_df, test_df= data_preprocessing.main(train_start_date, train_end_date, test_start_date, test_end_date, run_train)

    # save the data fetched from API and piped through data_preprocessing (i.e. train_df and test_df)
    train_df.to_csv('train_data.csv')
    test_df.to_csv('test_data.csv')

# read the pre-processed data from  csvs
train_df = pd.read_csv('train_data.csv')
train_df.drop('Unnamed: 0',axis=1, inplace=True)
test_df = pd.read_csv('test_data.csv')
test_df.drop('Unnamed: 0',axis=1, inplace=True)
print("data read from csv")

if arch_type == 'FFNN':
    algo_main_ffnn.main(train_df, test_df, transformation_method,run_train, num_epochs, run_resume)
elif arch_type == 'RNN':
    algo_main_rnn.main(train_df, test_df, transformation_method,run_train, num_epochs, run_resume)
elif arch_type == 'LSTM':
    algo_main_lstm.main(train_df, test_df, transformation_method,run_train, num_epochs, run_resume)
elif arch_type == 'RNN':
    algo_main_gru.main(train_df, test_df, transformation_method,run_train, num_epochs, run_resume)

print('done!')