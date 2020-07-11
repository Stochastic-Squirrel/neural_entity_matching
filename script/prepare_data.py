'''
Script to create train-valid-test sets for Amzn and Quora data
'''
from utilities import *

amz_g_seed = 420
quora_seed = 80085

amz_goog = Amazon_Google(seed = amz_g_seed, diff_sub_sample = 100, difficult_cutoff = 0.1, prop_train = 0.8)
quora = Quora(seed = quora_seed, diff_sub_sample = 10000, difficult_cutoff = 0.1, prop_train = 0.8)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Save To Disk
# Saving to disk so that Magelan can read in the data as a catalog
file_directories = ["../data/processed_amazon_google/", "../data/processed_quora/"]
data_set_names = ["amz_google","quora"]


for i, dataset in enumerate([amz_goog, quora]):

    current_filepath = file_directories[i] + data_set_names[i]
    
    (dataset.data.train_valid_sets[0]).to_csv(current_filepath + "_X_train.csv")
    pd.DataFrame({"y":dataset.data.train_valid_sets[1]}).to_csv(current_filepath + "_y_train.csv")
    (dataset.data.train_valid_sets[2]).to_csv(current_filepath + "_X_valid.csv")
    pd.DataFrame({"y":dataset.data.train_valid_sets[3]}).to_csv(current_filepath + "_y_valid.csv")

    (dataset.data.test_sets[0]).to_csv(current_filepath + "_X_test.csv")
    pd.DataFrame({"y":dataset.data.test_sets[1]}).to_csv(current_filepath + "_y_test.csv")





