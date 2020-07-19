'''
General
https://nbviewer.jupyter.org/github/anhaidgroup/deepmatcher/blob/master/examples/getting_started.ipynb
Data Processing
https://nbviewer.jupyter.org/github/anhaidgroup/deepmatcher/blob/master/examples/data_processing.ipynb
Matching Algorithm
https://nbviewer.jupyter.org/github/anhaidgroup/deepmatcher/blob/master/examples/matching_models.ipynb


Entity matching model using Automatic Feature creation of magellan and tests a host
of algorithms.

 all_result dictionary hierarchy

    Sampler; Blocking Algorithm; Result object

    Since training deep models is expensive, only the best set of candidate blocking solutions are picked

    For a given Sampler and Blocking algo:
        Result Object[experiment_id corresponding to Sampler-Blocking iteration][Set ID][Model Name when appropriate]
        Set Ids:
            0: Training Predictions
            1: Validation Predictions
            2: Test Set Predictions
            3: (All Sets pre-blocked labels) = None  because blocking performance is analysed by model_magellan_ml.py result objects
            4: All sets post-blocked labels
            5: Experiment Meta Data




https://github.com/anhaidgroup/deepmatcher
'''
import deepmatcher as dm
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from blocking_algorithms import *
import re
import numpy as np
from progressbar import progressbar
import pickle
import datetime
import spacy
# TODO: need to decide on meaningful hyper-paramters for this method
# TODO: think of a better way of pushing through blocking hyperparameters




#Important Note: Be aware that creating a matching model (MatchingModel) object does not immediately instantiate all its components - deepmatcher uses a lazy initialization paradigm where components are instantiated just before training. Hence, code examples in this tutorial manually perform this initialization to demonstrate model customization meaningfully.

# Deep matcher learns DIFFERENT attribute similarity weights per attribute

# We are able to specify custom functions for attrbute summariser and attribute comparison

# Customising Attribute Summarisation
## Can specify separate components for the 3 sub modules in the attribute summarisation
## This is in contrast to you just definining an entire attribute summarisation class
## you can instead choose to build sub-module components to fit within the larger system

### Contextualizer
### Comparator
### Aggregator 

sampler = "iterative"
blocking = "lsh"
tokenizer = "spacy"




lsh_args = [{"seeds":200, "char_ngram":2, "bands": 4},
{"seeds":200, "char_ngram":2, "bands": 10}]
sequential_args = [{"cutoff_distance":60, "min_shared_tokens":4},
    {"cutoff_distance":80, "min_shared_tokens":2}
]

deepmatcher_args = {"sampler":["naive","iterative"], 
                    "blocking_algo":["lsh", "sequential"],
                    "model_args":[{"attr_summarizer":"hybrid"},{"attr_summarizer":"rnn"}]}

sampler_list = []
blocking_algo_list = []
result_obj_list = []




def prepare_data_deepmatcher(blocking, sampler, lsh_args, sequential_args):
    '''
    Function which prepares Google-Amazon data for deep matcher.
    Accepts a blocking algo, sampler and the associated arguments
    '''
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
    feature_cols  = [['title_amzn',
    'description_amzn',
    'manufacturer_amzn', 
    'price_amzn'],
    ['title_g',
    'description_g',
    'manufacturer_g',
    'price_g']]
    id_names = ["id_amzn","id_g"]


    if (blocking == "lsh"):
        candidates = lsh_blocking(lhs_table, rhs_table, 1, 5, ["id_amzn","id_g"], char_ngram = lsh_args["char_ngram"], seeds = lsh_args["seeds"], bands = lsh_args["bands"])
    elif (blocking == "sequential"):
    # Initial Rough Blocking on Overlapped Attributes
        candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, sequential_args["min_shared_tokens"] , feature_cols, id_names)
    # Fine Grained Blocking on edit distance
        candidates = edit_distance_blocking(None, None, blocking_cols, sequential_args["cutoff_distance"] , True, candidates)
    else:
        raise ValueError("Blocking must be lsh or sequential")

    # Append lhs_ and rhs_ prefix appropriately to work with deepmatcher expected format

    candidates.columns = ["lhs_id_lhs","lhs_title","lhs_description","lhs_manufacturer","lhs_price","rhs_id_rhs","rhs_title","rhs_description","rhs_manufacturer","rhs_price"]
    candidates = candidates.reset_index()


    candidates = pd.merge(candidates, y_train, left_on  = ["id_amzn","id_g"] , right_on = ["id_amzn","id_g"], how  = "left")
    candidates.y = candidates.y.map({1.0:int(1), np.nan: 0})
    candidates = candidates.fillna(0)
    candidates["id"] = np.arange(candidates.shape[0])
    candidates.set_index("id",inplace = True)
    candidates = candidates.rename(columns = {"id_amzn":"lhs_code", "id_g":"rhs_code"})
    candidates = candidates.drop(columns = ['lhs_id_lhs', 'rhs_id_rhs','lhs_code','rhs_code'])
    candidates.to_csv("../data/tmp/train.csv")

    # Load validation set
    em.del_catalog()
    lhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_valid_lhs.csv").rename(columns = {"Unnamed: 0":"id_lhs"})
    rhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_valid_rhs.csv").rename(columns = {"Unnamed: 0":"id_rhs"})
    y_valid = pd.read_csv("../data/processed_amazon_google/amz_google_" + sampler + "_y_valid.csv")
    em.del_catalog()
    em.set_key(lhs_table, "id_lhs")
    em.set_key(rhs_table, "id_rhs")

    n_valid  = lhs_table.shape[0]

    # Blocking
    blocking_cols = ["title_amzn","title_g"]
    feature_cols  = [['title_amzn',
    'description_amzn',
    'manufacturer_amzn', 
    'price_amzn'],
    ['title_g',
    'description_g',
    'manufacturer_g',
    'price_g']]
    id_names = ["id_amzn","id_g"]


    if (blocking == "lsh"):
        candidates = lsh_blocking(lhs_table, rhs_table, 1, 5, ["id_amzn","id_g"], char_ngram = lsh_args["char_ngram"], seeds = lsh_args["seeds"], bands = lsh_args["bands"])
    elif (blocking == "sequential"):
    # Initial Rough Blocking on Overlapped Attributes
        candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, sequential_args["min_shared_tokens"] , feature_cols, id_names)
    # Fine Grained Blocking on edit distance
        candidates = edit_distance_blocking(None, None, blocking_cols, sequential_args["cutoff_distance"] , True, candidates)
    else:
        raise ValueError("Blocking must be lsh or sequential")

    # Append lhs_ and rhs_ prefix appropriately to work with deepmatcher expected format

    candidates.columns = ["lhs_id_lhs","lhs_title","lhs_description","lhs_manufacturer","lhs_price","rhs_id_rhs","rhs_title","rhs_description","rhs_manufacturer","rhs_price"]
    candidates = candidates.reset_index()


    candidates = pd.merge(candidates, y_valid, left_on  = ["id_amzn","id_g"] , right_on = ["id_amzn","id_g"], how  = "left")
    candidates.y = candidates.y.map({1.0:int(1), np.nan: 0})
    candidates = candidates.fillna(0)
    candidates["id"] = np.arange(candidates.shape[0])
    candidates.set_index("id",inplace = True)
    candidates = candidates.rename(columns = {"id_amzn":"lhs_code", "id_g":"rhs_code"})
    candidates = candidates.drop(columns = ['lhs_id_lhs', 'rhs_id_rhs','lhs_code','rhs_code'])
    candidates.to_csv("../data/tmp/valid.csv")

    # Load Test
    em.del_catalog()
    lhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_test_lhs.csv").rename(columns = {"Unnamed: 0":"id_lhs"})
    rhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_test_rhs.csv").rename(columns = {"Unnamed: 0":"id_rhs"})
    y_test = pd.read_csv("../data/processed_amazon_google/amz_google_" + sampler + "_y_test.csv")
    em.del_catalog()
    em.set_key(lhs_table, "id_lhs")
    em.set_key(rhs_table, "id_rhs")

    n_test  = lhs_table.shape[0]

    # Blocking
    blocking_cols = ["title_amzn","title_g"]
    feature_cols  = [['title_amzn',
    'description_amzn',
    'manufacturer_amzn', 
    'price_amzn'],
    ['title_g',
    'description_g',
    'manufacturer_g',
    'price_g']]
    id_names = ["id_amzn","id_g"]


    if (blocking == "lsh"):
        candidates = lsh_blocking(lhs_table, rhs_table, 1, 5, ["id_amzn","id_g"], char_ngram = lsh_args["char_ngram"], seeds = lsh_args["seeds"], bands = lsh_args["bands"])
    elif (blocking == "sequential"):
    # Initial Rough Blocking on Overlapped Attributes
        candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, sequential_args["min_shared_tokens"] , feature_cols, id_names)
    # Fine Grained Blocking on edit distance
        candidates = edit_distance_blocking(None, None, blocking_cols, sequential_args["cutoff_distance"] , True, candidates)
    else:
        raise ValueError("Blocking must be lsh or sequential")

    # Append lhs_ and rhs_ prefix appropriately to work with deepmatcher expected format

    candidates.columns = ["lhs_id_lhs","lhs_title","lhs_description","lhs_manufacturer","lhs_price","rhs_id_rhs","rhs_title","rhs_description","rhs_manufacturer","rhs_price"]
    candidates = candidates.reset_index()


    candidates = pd.merge(candidates, y_test, left_on  = ["id_amzn","id_g"] , right_on = ["id_amzn","id_g"], how  = "left")
    candidates.y = candidates.y.map({1.0:int(1), np.nan: 0})
    candidates = candidates.fillna(0)
    candidates["id"] = np.arange(candidates.shape[0])
    candidates.set_index("id",inplace = True)

    candidates = candidates.rename(columns = {"id_amzn":"lhs_code", "id_g":"rhs_code"})
    candidates = candidates.drop(columns = ['lhs_id_lhs', 'rhs_id_rhs','lhs_code','rhs_code'])

    candidates.to_csv("../data/tmp/test.csv")


def fit_deepmatcher(model_args, train, validation, test):
    '''
    Takes train-validation-test sets generated by dm.data.process() using temp written csv
    files created by perpare_data_deepmatcher

    Outputs:
        (train_predictions, valid_predictions, test_predictions, None, post_blocked_all_sets_labels)

        * The none is to keep it in the desired format for evaluation_functions.py
    '''
    # Confugre the Matching Algorithm 
    model = dm.MatchingModel(attr_summarizer=model_args["attr_summarizer"])
    # Fit the model and select best epoch version based on best validation accuracy
    model.run_train(
        train,
        validation,
        epochs=10,
        batch_size=16,
        best_save_path='../results/' + model_args["attr_summarizer"] + '.pth',
        pos_neg_ratio=2)
    # Create and store predictions
    ## Name of the model is the attr summarizer setting
    train_predictions = {model_args["attr_summarizer"]:np.round(model.run_prediction(train).match_score.values,0)}
    valid_predictions = {model_args["attr_summarizer"]:np.round(model.run_prediction(validation).match_score.values,0)}
    test_predictions = {model_args["attr_summarizer"]:np.round(model.run_prediction(test).match_score.values,0)}


    # Create source of truth to be used for evaluation_functions.py
    post_blocked_all_sets_labels = {"train":train.get_raw_table().y.values,
                        "valid":validation.get_raw_table().y.values, 
                        "test":test.get_raw_table().y.values}
    
    return (train_predictions, valid_predictions, test_predictions, None, post_blocked_all_sets_labels)


def run_deepmatcher_models(deepmatcher_args):

    for sampler in deepmatcher_args["sampler"]:

        if (sampler != "iterative") & (sampler != "naive"):
            raise ValueError("Sampler should be iterative or naive (completely random).")

        for blocker in deepmatcher_args["blocking_algo"]:

            if (blocker != "sequential") & (blocker != "lsh"):
                raise ValueError("Blocker should be sequential or lsh.")

            # Prepare Data For the entire sampler + blocking algo combination AND hyperparameter config
            if (blocker == "lsh"):
                for block_params in lsh_args:
                    print(f"Preparing Data under blocker: {blocker} and sampler: {sampler} with params: {block_params}")
                    prepare_data_deepmatcher(blocker, sampler, lsh_args = block_params, sequential_args = None)

                    # Fit all appropriate models for given set of prepared data
                    for model_args in deepmatcher_args["model_args"]:
                        print(f"Training Model Configuration {model_args}")    
                        sampler_list.append(sampler)
                        blocking_algo_list.append(blocker)

                        # Generate Data Sets from the current blocking method
                        train, validation, test = dm.data.process(
                            path='../data/tmp',
                            train='train.csv',
                            validation='valid.csv',
                            test='test.csv',
                            left_prefix='lhs_',
                            right_prefix='rhs_',
                            label_attr='y',
                            id_attr='id')
                        # Fit model and return predictions
                        result_obj_list.append(fit_deepmatcher(model_args,train, validation, test))
            if (blocker == "sequential"):
                for block_params in sequential_args:
                        print(f"Preparing Data under blocker: {blocker} and sampler: {sampler} with params: {block_params}")
                        prepare_data_deepmatcher(blocker, sampler, lsh_args = None, sequential_args = block_params)

                        # Fit all appropriate models for given set of prepared data
                        for model_args in deepmatcher_args["model_args"]:
                            print(f"Training Model Configuration {model_args}")    
                            sampler_list.append(sampler)
                            blocking_algo_list.append(blocker)

                            # Generate Data Sets from the current blocking method
                            train, validation, test = dm.data.process(
                                path='../data/tmp',
                                train='train.csv',
                                validation='valid.csv',
                                test='test.csv',
                                left_prefix='lhs_',
                                right_prefix='rhs_',
                                label_attr='y',
                                id_attr='id')
                            # Fit model and return predictions
                            result_obj_list.append(fit_deepmatcher(model_args,train, validation, test))


# Debug
sampler = deepmatcher_args["sampler"][0]
blocker = deepmatcher_args["blocking_algo"][0]
block_params  = lsh_args[0]
model_args = deepmatcher_args["model_args"][0]

run_deepmatcher_models(deepmatcher_args)

