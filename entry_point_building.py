import data_preprocessing
import algo_main_rnn_v2
import algo_main_ffnn
import algo_main_lstm
import algo_main_gru
import pandas as pd
import argparser
import seaborn as sns
import buildings_processing as bp

# Get inputs from command-line
configs = argparser.get_arguments()

# Temp holding place for new configs:
configs['LAN_path'] = "Z:\\Data"
configs['building'] = 'RSF'
configs['year'] = ['2019']
configs['target_var'] = 'RSF Main Power (kW)'
configs['weather_include'] = ['SRRL BMS Global Horizontal Irradiance (W/m²_irr)',
                              'SRRL BMS Wind Speed at 19\' (mph)',
                              'SRRL BMS Dry Bulb Temperature (°F)',
                              'SRRL BMS Total Cloud Cover (%)']
configs['TrainTestSplit'] = 'Sequential'  # Random or Sequential
configs['test_start'] = '10/01/2019 00:00:00'  # Only matters if TrainTestSplit = Sequential
configs['test_end'] = '12/31/2019 23:45:00'  # Only matters if TrainTestSplit = Sequential
configs['resample'] = True
configs['resample_bin_min'] = 15  # In Minutes
configs['HOD'] = True
configs['DOW'] = True
configs['MOY'] = False
configs['Holidays'] = True
configs['HOD_indicator'] = 'sincos'  # sincos or fuzzy or regDummy

if configs['preprocess']:
    train_df, test_df, configs = data_preprocessing.main(configs)

    # save the data fetch_n_parseed from API and piped through data_preprocessing (i.e. train_df and test_df)
    train_df.to_csv('./data/STM_Train_Data.csv')
    test_df.to_csv('./data/STM_Test_Data.csv')

else:
    # preprocessing module defines target_feat_name list and sends it back.
    configs['target_feat_name'] = [configs['target_var']]

# For individual building use:
# Get full dataset from LAN for specific building
data_full = bp.get_full_data(configs)

# Remove all data columns we dont care about
important_vars = configs['weather_include'] + [configs['target_var']]
data = data_full[important_vars]

# Resample
resample_bin_size = "{}T".format(configs['resample_bin_min'])
data = data.resample(resample_bin_size).mean()

# Clean
data = bp.clean_data(data, configs)

# Deal with NANs
data = data.dropna(how='any')

# Add time-based dummy variables
data = bp.time_dummies(data, configs)

# Split into training and test dataframes
train_df, test_df = bp.train_test_split(data, configs)

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
                                                                                      configs['target_feat_name']))
