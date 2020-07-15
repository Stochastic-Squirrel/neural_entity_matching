'''
Entity matching model using Automatic Feature creation of magellan and tests a host
of algorithms.
'''
import os
import py_entitymatching as em
from py_entitymatching import XGBoostMatcher
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from blocking_algorithms import *
import re
import numpy as np
# https://nbviewer.jupyter.org/github/anhaidgroup/py_entitymatching/blob/master/notebooks/guides/step_wise_em_guides/Selecting%20the%20Best%20Learning%20Matcher.ipynb




def automatic_feature_gen(candidate_table, feature_cols, id_names, id_names_phrase):
    '''
    NB!
    The automatic function creates pairwise features. Consequently, it will convert
    internally the colnames in lhs and rhs portions of feature cols to the SAME name.
    It does this by trimming the `id_names_phrase` portion (suffix or prefix) from each column name
    It assumes that the id names are of the form id_{id_names_phrase} e.g. id_amzn

    Replaces Nans in candidate table with empty strings

    Takes in a single DataFrame object (lhs_table and rhs_table concatenated) and
    splits it into two tables then generates features on each of the sub tables.

    Inputs:
            candidate_table: single Pandas DataFrame (typically output of blocking_algorithms.py functions)

    Outputs:
    '''

    em.del_catalog()
    candidate_table = candidate_table.reset_index()

    lhs_table = candidate_table.loc[:, feature_cols[0] + [id_names[0]]]
    rhs_table = candidate_table.loc[:, feature_cols[1] + [id_names[1]]]

    lhs_colnames = []
    for colname in lhs_table:
        if colname != id_names[0]:
            lhs_colnames.append(re.sub(id_names_phrase[0],"",colname))
        else:
            lhs_colnames.append(colname)
    rhs_colnames = []
    for colname in rhs_table:
        if colname != id_names[1]:
            rhs_colnames.append(re.sub(id_names_phrase[1],"",colname))
        else:
            rhs_colnames.append(colname)

    lhs_table.columns = lhs_colnames
    rhs_table.columns = rhs_colnames
    # To circumvent the same product ID coming up again (due to it being in multiple candidate comparisons)
    lhs_table["index_num_lhs"] = np.arange(lhs_table.shape[0])
    rhs_table["index_num_rhs"] = np.arange(rhs_table.shape[0])

    em.set_key(lhs_table, "index_num_lhs") # changed from id_names
    em.set_key(rhs_table,  "index_num_rhs")
    # Generate List Of Features
    matching_features = em.get_features_for_matching(lhs_table.drop(id_names[0], axis = 1), rhs_table.drop(id_names[1], axis = 1), validate_inferred_attr_types= False)
    # Extract feature vectors and save as a  DF
    # Set primary keys and foreign keys for candidate table
    candidate_table["index"] = np.arange(candidate_table.shape[0])
    # Add foreign keys to candidate table
    candidate_table["index_num_lhs"] = np.arange(lhs_table.shape[0])
    candidate_table["index_num_rhs"] = np.arange(rhs_table.shape[0])

    em.set_key(candidate_table, "index")
    em.set_fk_ltable(candidate_table, "index_num_lhs")
    em.set_fk_rtable(candidate_table, "index_num_rhs")
    em.set_ltable(candidate_table, lhs_table)
    em.set_rtable(candidate_table, rhs_table)

    matching_features_df = em.extract_feature_vecs(candidate_table, 
                            feature_table = matching_features, 
                            show_progress = False)

    matching_features_df = em.impute_table(matching_features_df, 
                exclude_attrs=['index',"index_num_lhs","index_num_rhs"],
                strategy='mean')
    # add back the amzn and google ids
    matching_features_df["id_amzn"] = candidate_table.id_amzn
    matching_features_df["id_g"] = candidate_table.id_g

    matching_features_df = matching_features_df.fillna(value = 0)
    
    # Check if any NaNs are left
    # if any(matching_features_df.isna()):
    #     print(matching_features_df.isna().apply(sum))
    #     raise ValueError("NaNs present in generated feature DF after imputation.")

    return matching_features_df





def run_magellan_models(sampler = "iterative", blocking = "sequential", **kwargs):


    '''
    1. Loads data from processed folder of dataset choice.
    2. Performs blocking according to given hyper-parameters
    3. For every given blocking set, generate automatic features
    4. Run suite of shallow learning algorithms on candidate sets

    Inputs:
            sampler: sampling technique that was used to generate data: iterative or naive
            blocking: blocking algorithm used: iterative or lsh
    Outputs:
            training_pred_dict, validation_pred_dict, test_pred_dict, pre_blocked_all_sets_labels, post_blocked_all_sets_labels
    
    '''
    if (sampler != "iterative") & (sampler != "naive"):
        raise ValueError("Sampler should be iterative or naive (completely random).")

    # Load Training Set according to sampler
    em.del_catalog()
    lhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_train_lhs.csv").rename(columns = {"Unnamed: 0":"id_lhs"})
    rhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_train_rhs.csv").rename(columns = {"Unnamed: 0":"id_rhs"})
    y_train = pd.read_csv("../data/processed_amazon_google/amz_google_" + sampler + "_y_train.csv")
    em.del_catalog()
    em.set_key(lhs_table, "id_lhs")
    em.set_key(rhs_table, "id_rhs")

    # Blocking
    blocking_cols = ["title_amzn","title_g"]
    min_shared_tokens = 3
    feature_cols  = [['title_amzn',
    'description_amzn',
    'manufacturer_amzn', 
    'price_amzn'],
    ['title_g',
    'description_g',
    'manufacturer_g',
    'price_g']]
    id_names = ["id_amzn","id_g"]
    cutoff_distance = 60

    if (blocking == "lsh"):
        candidates = lsh_blocking(lhs_table, rhs_table, 1, 5, ["id_amzn","id_g"], char_ngram = 2, seeds = 200, bands = 4)
    elif (blocking == "sequential"):
        # Initial Rough Blocking on Overlapped Attributes
        candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, min_shared_tokens , feature_cols, id_names)
        # Fine Grained Blocking on edit distance
        candidates = edit_distance_blocking(None, None, blocking_cols, cutoff_distance, True, candidates)
    else:
        raise ValueError("Blocking must be lsh or sequential")
    
    

    # Generate Features
    id_names_phrase = ["_amzn","_g"] # Trims away these suffixes from id columns
    feature_cols  = [['title_amzn',
    'description_amzn', # removed manufacturer due to missingess: produces features with nans
    'price_amzn'],
    ['title_g',
    'description_g',
    'price_g']]

    generated_df_train  =  automatic_feature_gen(candidates, feature_cols, id_names, id_names_phrase)
    generated_df_train = pd.merge(generated_df_train, y_train, left_on  = ["id_amzn","id_g"] , right_on = ["id_amzn","id_g"], how  = "left")
    generated_df_train.y = generated_df_train.y.map({1.0:int(1), np.nan: 0})

    # Store Training Column names. Ensures that if by chance a new column is generated in 
    # validation or test phase, these ones will be ignored
    model_features = generated_df_train.columns

    # Train Models on training set
    dt = em.DTMatcher(name='DecisionTree', random_state=0)
    svm = em.SVMMatcher(name='SVM', random_state=0)
    rf = em.RFMatcher(name='RF', random_state=0)
    lg = em.LogRegMatcher(name='LogReg', random_state=0)
    ln = em.LinRegMatcher(name='LinReg')
    xg = em.XGBoostMatcher(name = "Xg-Boost", random_state = 0)

    dt.fit(table = generated_df_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')
    svm.fit(table = generated_df_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')
    rf.fit(table = generated_df_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')
    lg.fit(table = generated_df_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')
    ln.fit(table = generated_df_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')
    xg.fit(table = generated_df_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')

    models = [dt, svm, rf,  lg, ln, xg]

    training_predictions = {}
    for model in models:
        training_predictions[model.name] = model.predict(table = generated_df_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs',"y"])
    # Load validation Set + Generate the feature columns
    em.del_catalog()
    lhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_valid_lhs.csv").rename(columns = {"Unnamed: 0":"id_lhs"})
    rhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_valid_rhs.csv").rename(columns = {"Unnamed: 0":"id_rhs"})
    y_valid = pd.read_csv("../data/processed_amazon_google/amz_google_" + sampler + "_y_valid.csv")
    em.del_catalog()
    em.set_key(lhs_table, "id_lhs")
    em.set_key(rhs_table, "id_rhs")

    if (blocking == "lsh"):
        candidates = lsh_blocking(lhs_table, rhs_table, 1, 5, ["id_amzn","id_g"], char_ngram = 2, seeds = 200, bands = 4)
    elif (blocking == "sequential"):
        # Initial Rough Blocking on Overlapped Attributes
        candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, min_shared_tokens , feature_cols, id_names)
        # Fine Grained Blocking on edit distance
        candidates = edit_distance_blocking(None, None, blocking_cols, cutoff_distance, True, candidates)
    else:
        raise ValueError("Blocking must be lsh or sequential")
    
    generated_df_valid  =  automatic_feature_gen(candidates, feature_cols, id_names, id_names_phrase)
    generated_df_valid = pd.merge(generated_df_valid, y_valid, left_on  = ["id_amzn","id_g"] , right_on = ["id_amzn","id_g"], how  = "left")
    generated_df_valid.y = generated_df_valid.y.map({1.0:int(1), np.nan: 0})
    generated_df_valid = generated_df_valid.loc[:,model_features]
    # Predict on Validation Set
    validation_predictions = {}
    for model in models:
        validation_predictions[model.name] = model.predict(table = generated_df_valid, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs',"y"])


    # Finally Generate Test Set Predictions
    em.del_catalog()
    lhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_test_lhs.csv").rename(columns = {"Unnamed: 0":"id_lhs"})
    rhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_test_rhs.csv").rename(columns = {"Unnamed: 0":"id_rhs"})
    y_test = pd.read_csv("../data/processed_amazon_google/amz_google_" + sampler + "_y_test.csv")
    em.del_catalog()
    em.set_key(lhs_table, "id_lhs")
    em.set_key(rhs_table, "id_rhs")

    if (blocking == "lsh"):
        candidates = lsh_blocking(lhs_table, rhs_table, 1, 5, ["id_amzn","id_g"], char_ngram = 2, seeds = 200, bands = 4)
    elif (blocking == "sequential"):
        # Initial Rough Blocking on Overlapped Attributes
        candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, min_shared_tokens , feature_cols, id_names)
        # Fine Grained Blocking on edit distance
        candidates = edit_distance_blocking(None, None, blocking_cols, cutoff_distance, True, candidates)
    else:
        raise ValueError("Blocking must be lsh or sequential")
    
    generated_df_test  =  automatic_feature_gen(candidates, feature_cols, id_names, id_names_phrase)
    generated_df_test = pd.merge(generated_df_test, y_test, left_on  = ["id_amzn","id_g"] , right_on = ["id_amzn","id_g"], how  = "left")
    generated_df_test.y = generated_df_test.y.map({1.0:int(1), np.nan: 0})
    generated_df_test = generated_df_test.loc[:,model_features]
    # Predict on test Set
    test_predictions = {}
    for model in models:
        test_predictions[model.name] = model.predict(table = generated_df_test, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs',"y"])


    # Create pre_blocked_all_sets_labels to store truth of candidate tuples after BLOCKING
    pre_blocked_all_sets_labels = {"train":y_train, "valid":y_valid, "test":y_test}
    post_blocked_all_sets_labels = {"train":generated_df_train[["id_amzn","id_g","y"]],
                                    "valid":generated_df_valid[["id_amzn","id_g","y"]], 
                                    "test":generated_df_test[["id_amzn","id_g","y"]]}




    return (training_predictions, validation_predictions, test_predictions, pre_blocked_all_sets_labels, post_blocked_all_sets_labels)


#TODO: debug naive:lsh combination
# TODO return hyper-parameters for blocking
# TODO: fix nan issue training and forces validation predict to use a oclumn that has nan 

all_results = pd.DataFrame({"sampler":[], "block_algo":[],"result_obj":[]})

sampler_list = []
blocking_algo_list = []
result_obj_list = []

for sampler in ["iterative","naive"]:
    for block_algo in ["sequential","lsh"]:
        print("--------------------------------------------------")
        print(f"Running on configuration {sampler}:{block_algo}")
        print("--------------------------------------------------")
        sampler_list = sampler_list.append(sampler)
        blocking_algo_list = blocking_algo_list.append(blocking_algo)
        result_obj_list = result_obj_list.append(run_magellan_models(sampler,block_algo))


