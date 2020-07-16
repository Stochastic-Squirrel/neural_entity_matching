'''
Suite of evaluation functions to evaluate performance of blockers and matchers/models

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
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
import itertools


def evaluate_blocking(result):
    '''
    Measures blocking performance against 
            3: All Sets pre-blocked labels
            4: All sets post-blocked labels
            set ids on the following measures.

            - Recall of Positive Matches
            - Pruning Rate against maximal set 
    n_train, n_valid, n_test =  number of rows INCLUDING NEGATIVE AND POSITIVE MATCHES OF THE Sets

    KEY ASSUMPTION!
        Only Positive matches are stored in the truths for train, valid ,test i.e. y= 1

    Out
    '''
    # Extract pre-blocked data for train, valid, test
    ## Choice of experiment id is arbitrary as data is repeated in the PRE-blocking stage



    # Calculate Pruning Potential from Magellan paper
    ## Prune Rate = 1 - rows(Post_block)/rows(Pre_block)^2
    
    # Loop over each experiment ID as it represents a different samplinng-blocking combination
    train_pruning = []
    valid_pruning = []
    test_pruning = []

    train_recall = []
    valid_recall = []
    test_recall = []
    

    # Record the meta_data
    metadata = []

    # Cycle through experiments
    for i, obj in enumerate(result["result_obj"]):
        # Analyse Pruning Power
            pre_train_size = obj[5]["n_train"]
            pre_valid_size = obj[5]["n_valid"]
            pre_test_size = obj[5]["n_test"]
            # Extract post-blocked data for train, valid, test
            post_train_size = obj[4]["train"].shape[0]
            post_valid_size = obj[4]["valid"].shape[0]
            post_test_size =  obj[4]["test"].shape[0]

            train_pruning.append(1-(post_train_size/pre_train_size**2))
            valid_pruning.append(1-(post_valid_size/pre_valid_size**2))
            test_pruning.append(1-(post_test_size/pre_test_size**2))
            # Analyse Recall of Positive Matches
            ## Compute proportion of the ground truth number of POSITIVE matches found in post-blocked sets
            ## We aim to recall all of the positive matches 
            recalled_train = pd.merge(obj[3]["train"], obj[4]["train"], 
            left_on = ["id_amzn","id_g"], right_on = ["id_amzn","id_g"], how = "inner").shape[0]

            recalled_valid = pd.merge(obj[3]["valid"], obj[4]["valid"], 
            left_on = ["id_amzn","id_g"], right_on = ["id_amzn","id_g"], how = "inner").shape[0]

            recalled_test = pd.merge(obj[3]["test"], obj[4]["test"], 
            left_on = ["id_amzn","id_g"], right_on = ["id_amzn","id_g"], how = "inner").shape[0]
            
            # The stored truth is ONLY of positive matches so we can fetch the number of rows to denote
            # the total number of positives which should be recalled

            train_recall.append(recalled_train/obj[3]["train"].shape[0])
            valid_recall.append(recalled_valid/obj[3]["valid"].shape[0])
            test_recall.append(recalled_test/obj[3]["test"].shape[0])

            metadata.append(obj[5])

    return pd.DataFrame({"sampler":result["sampler"],
                        "blocking_algo":result["blocking_algo"],
                        "train_prune":train_pruning,
                        "valid_prune":valid_pruning,
                        "test_prune":test_pruning,
                        "train_recall":train_recall,
                        "valid_recall":valid_recall,
                        "test_recall":test_recall,
                        "metadata":metadata})

def evaluate_matcher(result):
    '''
    Given the post-blocked data sets, evaluate against the train, valid and test truth labels per matching algo
    Each row represents a Sampler-BlockingAlgo - Matcher combination
    Evaluates
        - Precision of Matcher
        - Recall of Matcher
        - F1 Score
    '''


 
    sampler_list = list(itertools.chain.from_iterable(itertools.repeat(x, len(result["result_obj"][0][0])) for x in result["sampler"]))
    blocking_algo_list = list(itertools.chain.from_iterable(itertools.repeat(x, len(result["result_obj"][0][0])) for x in result["blocking_algo"]))
   




    precision_list_train = []
    recall_list_train = []
    f1_score_list_train = []

    precision_list_valid = []
    recall_list_valid = []
    f1_score_list_valid = []

    precision_list_test = []
    recall_list_test = []
    f1_score_list_test = []

    metadata_list = []

    model_list = []
    # Cycle through experiments
    for i, obj in enumerate(result["result_obj"]):
        # Fetch Sources of Truth for this experiment which is POST BLOCKED DATA
        train_labels = obj[4]["train"]
        valid_labels = obj[4]["valid"]
        test_labels = obj[4]["test"]


        # For Each Model
        for model in obj[0]: #arbitrarily choose index 0 to extract model name
            model_list.append(model)

            metadata_list.append(obj[5])

            train_predictions = obj[0][model]
            valid_predictions = obj[1][model]
            test_predictions = obj[2][model]

            # Calculate and Store Metrics
            ## y_true, y_pred
            precision_list_train.append(precision_score(train_labels.y.values,train_predictions))
            recall_list_train.append(recall_score(train_labels.y.values,train_predictions))
            f1_score_list_train.append(f1_score(train_labels.y.values,train_predictions))

            precision_list_valid.append(precision_score(valid_labels.y.values,valid_predictions))
            recall_list_valid.append(recall_score(valid_labels.y.values,valid_predictions))
            f1_score_list_valid.append(f1_score(valid_labels.y.values,valid_predictions))

            precision_list_test.append(precision_score(test_labels.y.values,test_predictions))
            recall_list_test.append(recall_score(test_labels.y.values,test_predictions))
            f1_score_list_test.append(f1_score(test_labels.y.values,test_predictions))



    return pd.DataFrame({"sampler":sampler_list,
                    "blocking_algo":blocking_algo_list,
                    "model":model_list,
                    "train_precision":precision_list_train,
                    "train_recall":recall_list_train,
                    "train_f1":f1_score_list_train,
                    "valid_precision":precision_list_valid,
                    "valid_recall":recall_list_valid,
                    "valid_f1":f1_score_list_valid,
                    "test_precision":precision_list_test,
                    "test_recall":recall_list_test,
                    "test_f1":f1_score_list_test,
                    "metadata":metadata_list})















result = pickle.load(open("../results/magellan_Jul_16_1341.p","rb"))

blocking_results = evaluate_blocking(result)
matcher_results = evaluate_matcher(result)


assert blocking_results.shape[0]*6 == matcher_results.shape[0] 

sns.scatterplot(x = blocking_results.train_recall, y = blocking_results.valid_recall)


# blocking_results.loc[5,"metadata"]
# {'cutoff_distance': 80,
#  'min_shared_tokens': 2,
#  'n_train': 915,
#  'n_valid': 245,
#  'n_test': 250}


blocking_results.groupby(["sampler","blocking_algo"]).apply(np.mean)

matcher_results.groupby(["sampler","blocking_algo"]).apply(np.mean)