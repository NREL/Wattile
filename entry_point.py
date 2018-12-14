import data_preprocessing
import algorithm_implement
import argparser


train_start_date, train_end_date, test_start_date, test_end_date, transformation_method, run_train, num_epochs = argparser.get_arguments()
train_df, test_df= data_preprocessing.main(train_start_date, train_end_date, test_start_date, test_end_date, run_train)
algorithm_implement.main(train_df, test_df, transformation_method,run_train, num_epochs)

print('done')