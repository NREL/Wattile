import data_preprocessing
import algorithm_implement


train_df, test_df = data_preprocessing.main()
algorithm_implement.main(train_df, test_df)
print('done')