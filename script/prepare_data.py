'''
Script to create train-valid-test sets for Amzn and Quora data. Partitions sets into two different tables.
Blocking is required to generate candidate pairs.
'''
from utilities import *
os.chdir(os.path.dirname(os.path.abspath(__file__)))
amz_g_seed = 420
quora_seed = 80085

amz_goog = Amazon_Google(seed = amz_g_seed, diff_sub_sample = 100, difficult_cutoff = 0.1, prop_train = 0.8)
quora = Quora(seed = quora_seed, diff_sub_sample = 10000, difficult_cutoff = 0.1, prop_train = 0.8)


# Save To Disk
# Saving to disk so that Magelan can read in the data as a catalog
file_directories = ["../data/processed_amazon_google/", "../data/processed_quora/"]
data_set_names = ["amz_google","quora"]


for i, dataset in enumerate([amz_goog, quora]):

    current_filepath = file_directories[i] + data_set_names[i]
    # Process Input Data Frames
    # Train
    lhs_train, rhs_train  = partition_data_set(dataset.data.train_valid_sets[0], dataset.data.id_names, dataset.data.feature_cols)
    lhs_train.to_csv(current_filepath + "_X_train_lhs.csv")
    rhs_train.to_csv(current_filepath + "_X_train_rhs.csv")
    dataset.data.train_valid_sets[1][dataset.data.train_valid_sets[1].y == 1].to_csv(current_filepath + "_y_train.csv")

    # Valid
    lhs_valid, rhs_valid  = partition_data_set(dataset.data.train_valid_sets[2], dataset.data.id_names, dataset.data.feature_cols)
    lhs_valid.to_csv(current_filepath + "_X_valid_lhs.csv")
    rhs_valid.to_csv(current_filepath + "_X_valid_rhs.csv")
    dataset.data.train_valid_sets[3][dataset.data.train_valid_sets[3].y == 1].to_csv(current_filepath + "_y_valid.csv")

    # Test
    lhs_test, rhs_test  = partition_data_set(dataset.data.test_sets[0], dataset.data.id_names, dataset.data.feature_cols)
    lhs_test.to_csv(current_filepath + "_X_test_lhs.csv")
    rhs_test.to_csv(current_filepath + "_X_test_rhs.csv")
    dataset.data.test_sets[1][dataset.data.test_sets[1].y == 1].to_csv(current_filepath + "_y_test.csv")




