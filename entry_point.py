import data_preprocessing
import algo_main_rnn
import algo_main_ffnn
import algo_main_lstm
import algo_main_gru
import pandas as pd
import argparser
import seaborn as sns


configs = argparser.get_arguments()


if configs['preprocess']:
    train_df, test_df, configs= data_preprocessing.main(configs)

    # save the data fetch_n_parseed from API and piped through data_preprocessing (i.e. train_df and test_df)
    train_df.to_csv('./data/STM_Train_Data.csv')
    test_df.to_csv('./data/STM_Test_Data.csv')

else:
    # preprocessing module defines target_feat_name list and sends it back.
    configs['target_feat_name'] =  ['STM_Xcel_Meter']

# read the pre-processed data from  csvs
train_df = pd.read_csv('./data/STM_Train_Data_processed.csv')
#train_df.drop('Unnamed: 0',axis=1, inplace=True)
test_df = pd.read_csv('./data/STM_Test_Data_processed.csv')
#test_df.drop('Unnamed: 0',axis=1, inplace=True)
print("data read from csv")

#sns.pairplot(train_df)


if configs['arch_type'] == 'FFNN':
    algo_main_ffnn.main(train_df, test_df, configs)
elif configs['arch_type'] == 'RNN':
    algo_main_rnn.main(train_df, test_df, configs)
elif configs['arch_type'] == 'LSTM':
    algo_main_lstm.main(train_df, test_df, configs)
elif configs['arch_type'] == 'RNN':
    algo_main_gru.main(train_df, test_df, configs)

train_exp_num = configs['train_exp_num']
test_exp_num = configs['test_exp_num']
arch_type = configs['arch_type']
print('Run with arch: {}, train_num= {}, test_num= {} and target= {} is done!'.format(arch_type, train_exp_num, test_exp_num,configs['target_feat_name'] ))