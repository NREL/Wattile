import sys
import data_preprocessing
import algo_main_rnn_v2
import algo_main_ffnn
import algo_main_lstm
import algo_main_gru
import pandas as pd
import argparser
import json

# Import shared files from other project
shared_dir = 'C:\\dev\\intelligentcampus-2020summer\\loads\\Stats_Models_Loads'
sys.path.append(shared_dir)
import buildings_processing as bp


def main(configs):
    # Preprocess if needed
    if configs['preprocess']:
        train_df, test_df, configs = data_preprocessing.main(configs)

        # save the data fetch_n_parseed from API and piped through data_preprocessing (i.e. train_df and test_df)
        train_df.to_csv('./data/STM_Train_Data.csv')
        test_df.to_csv('./data/STM_Test_Data.csv')
    else:
        # preprocessing module defines target_feat_name list and sends it back.
        configs['target_feat_name'] = [configs['target_var']]

    # For individual building use:
    train_df, test_df = bp.prep_for_rnn(configs)

    # Choose what ML architecture to use and execute the corresponding script
    if configs['arch_type'] == 'FFNN':
        algo_main_ffnn.main(train_df, test_df, configs)
    elif configs['arch_type'] == 'RNN':
        algo_main_rnn_v2.main(train_df, test_df, configs)
    elif configs['arch_type'] == 'LSTM':
        algo_main_lstm.main(train_df, test_df, configs)
    elif configs['arch_type'] == 'RNN':
        algo_main_gru.main(train_df, test_df, configs)

    train_exp_num = configs['train_exp_num']
    test_exp_num = configs['test_exp_num']
    arch_type = configs['arch_type']
    print('Run with arch: {}, train_num= {}, test_num= {} and target= {} is done!'.format(arch_type, train_exp_num,
                                                                                      test_exp_num,
                                                                                      configs['target_var']))


if __name__ == "__main__":
    # Read in configs from json
    with open("configs.json", "r") as read_file:
        configs = json.load(read_file)
    main(configs)
