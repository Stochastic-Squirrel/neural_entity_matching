'''
Entity matching model using Automatic Feature creation of magellan and tests a host
of algorithms.

Magellan Model all_result dictionary hierarchy

    Sampler; Blocking Algorithm; Result object

    For a given Sampler and Blocking algo:
        Result Object[experiment_id corresponding to Sampler-Blocking iteration][Set ID][Model Name when appropriate]
        Set Ids:
            0: Training Predictions
            1: Validation Predictions
            2: Test Set Predictions
            3: All Sets pre-blocked labels
            4: All sets post-blocked labels
            5: Experiment Meta Data


'''
import os
import py_entitymatching as em
from py_entitymatching import XGBoostMatcher
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from blocking_algorithms import *
import re
import numpy as np
from progressbar import progressbar
import pickle
import datetime
# https://nbviewer.jupyter.org/github/anhaidgroup/py_entitymatching/blob/master/notebooks/guides/step_wise_em_guides/Selecting%20the%20Best%20Learning%20Matcher.ipynb

# TODO: LSH is doing really badly, maybe need to hash more than one column....


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
    
    # print(matching_features_df.describe())
    # print(f"Number na {matching_features_df.isna().apply(sum)}")
    # print(f"Number null {matching_features_df.isnull().apply(sum)}")
    return matching_features_df





def run_magellan_models(sampler = "iterative", blocking = "lsh", lsh_args = None, sequential_args = None):


    '''
    1. Loads data from processed folder of dataset choice.
    2. Performs blocking according to given hyper-parameters
    3. For every given blocking set, generate automatic features
    4. Run suite of shallow learning algorithms on candidate sets

    Inputs:
            sampler: sampling technique that was used to generate data: iterative or naive
            blocking: blocking algorithm used: iterative or lsh
            lsh_args = dictionary: seeds, char_ngrams, bands --> dictionary
            sequential_args: cutoff_distance , min_shared_tokens
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

    n_train  = lhs_table.shape[0]

    # Blocking
    blocking_cols = ["title_amzn","title_g"]
    #min_shared_tokens = 3
    feature_cols  = [['title_amzn',
    'description_amzn',
    'manufacturer_amzn', 
    'price_amzn'],
    ['title_g',
    'description_g',
    'manufacturer_g',
    'price_g']]
    id_names = ["id_amzn","id_g"]
    #cutoff_distance = 60

    if (blocking == "lsh"):
        candidates = lsh_blocking(lhs_table, rhs_table, 1, 5, ["id_amzn","id_g"], char_ngram = lsh_args["char_ngram"], seeds = lsh_args["seeds"], bands = lsh_args["bands"])
    elif (blocking == "sequential"):
        # Initial Rough Blocking on Overlapped Attributes
        candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, sequential_args["min_shared_tokens"] , feature_cols, id_names)
        # Fine Grained Blocking on edit distance
        candidates = edit_distance_blocking(None, None, blocking_cols, sequential_args["cutoff_distance"] , True, candidates)
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

    n_valid  = lhs_table.shape[0]

    if (blocking == "lsh"):
        candidates = lsh_blocking(lhs_table, rhs_table, 1, 5, ["id_amzn","id_g"], char_ngram = lsh_args["char_ngram"], seeds = lsh_args["seeds"], bands = lsh_args["bands"])
    elif (blocking == "sequential"):
        # Initial Rough Blocking on Overlapped Attributes
        candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, sequential_args["min_shared_tokens"] , feature_cols, id_names)
        # Fine Grained Blocking on edit distance
        candidates = edit_distance_blocking(None, None, blocking_cols, sequential_args["cutoff_distance"] , True, candidates)
    else:
        raise ValueError("Blocking must be lsh or sequential")
    
    generated_df_valid  =  automatic_feature_gen(candidates, feature_cols, id_names, id_names_phrase)
    generated_df_valid = pd.merge(generated_df_valid, y_valid, left_on  = ["id_amzn","id_g"] , right_on = ["id_amzn","id_g"], how  = "left")
    generated_df_valid.y = generated_df_valid.y.map({1.0:int(1), np.nan: 0})
    generated_df_valid = generated_df_valid.loc[:,model_features]
    ## TODO: think of a better idea!! it is because we enforce all generated data sets to have same columns as training set
    generated_df_valid = generated_df_valid.fillna(0)
    # Predict on Validation Set
    validation_predictions = {}
    for model in models:
        validation_predictions[model.name] = model.predict(table = generated_df_valid, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs',"y"])

    # Retrain on all data
    generated_final_train = pd.concat([generated_df_train,generated_df_valid], axis = 0)
    
    dt.fit(table = generated_final_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')
    svm.fit(table = generated_final_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')
    rf.fit(table = generated_final_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')
    lg.fit(table = generated_final_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')
    ln.fit(table = generated_final_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')
    xg.fit(table = generated_final_train, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs'],
            target_attr='y')




    # Finally Generate Test Set Predictions
    em.del_catalog()
    lhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_test_lhs.csv").rename(columns = {"Unnamed: 0":"id_lhs"})
    rhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_test_rhs.csv").rename(columns = {"Unnamed: 0":"id_rhs"})
    y_test = pd.read_csv("../data/processed_amazon_google/amz_google_" + sampler + "_y_test.csv")
    em.del_catalog()
    em.set_key(lhs_table, "id_lhs")
    em.set_key(rhs_table, "id_rhs")

    n_test  = lhs_table.shape[0]

    if (blocking == "lsh"):
        candidates = lsh_blocking(lhs_table, rhs_table, 1, 5, ["id_amzn","id_g"], char_ngram = lsh_args["char_ngram"], seeds = lsh_args["seeds"], bands = lsh_args["bands"])
    elif (blocking == "sequential"):
        # Initial Rough Blocking on Overlapped Attributes
        candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, sequential_args["min_shared_tokens"] , feature_cols, id_names)
        # Fine Grained Blocking on edit distance
        candidates = edit_distance_blocking(None, None, blocking_cols, sequential_args["cutoff_distance"] , True, candidates)
    else:
        raise ValueError("Blocking must be lsh or sequential")
    
    generated_df_test  =  automatic_feature_gen(candidates, feature_cols, id_names, id_names_phrase)
    generated_df_test = pd.merge(generated_df_test, y_test, left_on  = ["id_amzn","id_g"] , right_on = ["id_amzn","id_g"], how  = "left")
    generated_df_test.y = generated_df_test.y.map({1.0:int(1), np.nan: 0})
    generated_df_test = generated_df_test.loc[:,model_features]
    generated_df_test = generated_df_test.fillna(0)

    
    # Predict on test Set
    test_predictions = {}
    for model in models:
        print(model.name)
        test_predictions[model.name] = model.predict(table = generated_df_test, 
            exclude_attrs=['index', 'id_amzn','id_g','index_num_lhs', 'index_num_rhs',"y"])


    # Create pre_blocked_all_sets_labels to store truth of candidate tuples after BLOCKING
    pre_blocked_all_sets_labels = {"train":y_train, "valid":y_valid, "test":y_test}
    post_blocked_all_sets_labels = {"train":generated_df_train[["id_amzn","id_g","y"]],
                                    "valid":generated_df_valid[["id_amzn","id_g","y"]], 
                                    "test":generated_df_test[["id_amzn","id_g","y"]]}


    if(blocking == "lsh"):
        metadata  = lsh_args
    else:
        metadata = sequential_args
    print(f"Finished Experiment using {sampler} and {blocking} with params: {metadata}")
    # Add in sample sizes
    metadata["n_train"] = n_train
    metadata["n_valid"] = n_valid
    metadata["n_test"] = n_test

    return (training_predictions, validation_predictions, test_predictions, pre_blocked_all_sets_labels, post_blocked_all_sets_labels, metadata)

sampler_list = []
blocking_algo_list = []
result_obj_list = []


def expand_grid(data_dict):
    # Produces dictionary objects across exploration space
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys()).to_dict("records")

lsh_exploration_space = {"seeds":[200], "char_ngram":[2,3], "bands":[10,25,50]}
sequential_exploration_space = {"cutoff_distance":[50,60,70], "min_shared_tokens":[1,2,3]}

lsh_args = expand_grid(lsh_exploration_space)
sequential_args = expand_grid(sequential_exploration_space)

total_num_experiments = 2*(len(lsh_args)) + 2*len(sequential_args)


for sampler in ["iterative","naive"]:
    for block_algo in ["lsh","sequential"]:
        print("--------------------------------------------------")
        print(f"Running on configuration {sampler}:{block_algo}")
        print("--------------------------------------------------")
        if (block_algo == "sequential"):
            for arg_dic in sequential_args:
                # If entire set of blocked candidates consists of one label only 1 = positive match or 0 = no match, matcher will fail
                # This is why the try block is here.
                # TODO: what about cases where blocker is so good we don't need a matcher?
                try:
                    result_obj_list.append(run_magellan_models(sampler,block_algo, sequential_args = arg_dic))
                    sampler_list.append(sampler)
                    blocking_algo_list.append(block_algo)
                except:
                    continue 
        else:
            for arg_dic in lsh_args:
                try:
                    result_obj_list.append(run_magellan_models(sampler,block_algo, lsh_args = arg_dic))
                    sampler_list.append(sampler)
                    blocking_algo_list.append(block_algo)
                except:
                    continue

all_results = {"sampler":sampler_list, "blocking_algo":blocking_algo_list,"result_obj":result_obj_list}

pickle.dump( all_results, open( "../results/magellan_"+datetime.datetime.today().strftime("%h_%d_%H%M")+".p", "wb" ))
