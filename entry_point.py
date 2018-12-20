import data_preprocessing
import algorithm_implement
import argparser
import pandas as pd


train_start_date, train_end_date, test_start_date, test_end_date, transformation_method, run_train, num_epochs, run_resume = argparser.get_arguments()
train_df, test_df= data_preprocessing.main(train_start_date, train_end_date, test_start_date, test_end_date, run_train)

#save the data fetched from API and piped through data_preprocessing (i.e. train_df and test_df)
train_df.to_csv('train_data.csv')
test_df.to_csv('test_data.csv')

# read it back
train_df = pd.read_csv('train_data.csv')
train_df.drop('Unnamed: 0',axis=1, inplace=True)
test_df = pd.read_csv('test_data.csv')
test_df.drop('Unnamed: 0',axis=1, inplace=True)
print("read from csv")
algorithm_implement.main(train_df, test_df, transformation_method,run_train, num_epochs, run_resume)

print('done')