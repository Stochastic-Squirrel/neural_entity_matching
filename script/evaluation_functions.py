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
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score,roc_auc_score, auc
import itertools




#TODO: look at magellan models saving sampler blocking. fixed this by pulling the bropad object info but inner meta object seems to be wrong for sampler


#NB IMPORTANT POST ON INTERPRETING PRECISION RECALL
#https://datascience.stackexchange.com/questions/24990/irregular-precision-recall-curve

def count_off_diagonal(size):
    '''
    Counts upper or lower number of entries in a square matrix excluding the diagonal.
    Presumes a size x size sized matrix.
    '''
    return np.sum(np.arange(start = 1, stop = size))

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
    #Record missed matches by blocker for the TEST set only.
    #This will provide information will be added under evaluation_matcher functions so that test precision and recall
    # is calculated taking into account the MISSED positive matches with blocking
    missed_positive_matches_df_list = []

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

            train_pruning.append(1-(post_train_size/count_off_diagonal(pre_train_size)))
            valid_pruning.append(1-(post_valid_size/count_off_diagonal(pre_valid_size)))
            test_pruning.append(1-(post_test_size/count_off_diagonal(pre_test_size)))
            # Analyse Recall of Positive Matches
            ## Compute proportion of the ground truth number of POSITIVE matches found in post-blocked sets
            ## We aim to recall all of the positive matches 
            recalled_train = pd.merge(obj[3]["train"], obj[4]["train"], 
            left_on = ["id_amzn","id_g"], right_on = ["id_amzn","id_g"], how = "inner").shape[0]

            recalled_valid = pd.merge(obj[3]["valid"], obj[4]["valid"], 
            left_on = ["id_amzn","id_g"], right_on = ["id_amzn","id_g"], how = "inner").shape[0]

            recalled_test = pd.merge(obj[3]["test"], obj[4]["test"], 
            left_on = ["id_amzn","id_g"], right_on = ["id_amzn","id_g"], how = "inner").shape[0]
            # record which TEST set positive values did NOT make it through 
            # Identify what values are in TableB and not in TableA
            test_truth = obj[3]["test"].set_index(["id_amzn","id_g"])
            test_blocked = obj[4]["test"].set_index(["id_amzn","id_g"])
            # Find difference in keys between truth and blocked
            missed_positive_matches_index = set(test_truth.index).difference(test_blocked.index)
            missed_positive_matches_df_list.append(test_truth.loc[test_truth.index.isin(missed_positive_matches_index)])



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
                        "metadata":metadata,
                        "post_train_size":post_train_size,
                        "post_valid_size":post_valid_size,
                        "post_test_size":post_test_size,
                        "missed_positive_matches_df":missed_positive_matches_df_list})

def evaluate_matcher(result, missed_positive_matches_df):
    '''
    Given the post-blocked data sets, evaluate against the train, valid and test truth labels per matching algo
    Each row represents a Sampler-BlockingAlgo - Matcher combination

    Assumes predictions for each model are a MATCHING score between 0 to 1.
    1 being a predicted certainty of a match.

    Evaluates for the POSITIVE CLASS ONLY
        - Precision of Matcher
        - Recall of Matcher
        - F1 Score


       Result is of the magellan models.
       missed_positive_matches_df is the panda series of attributes from evaluate_blocking_with meta.
       This is used to adjust the test set recall and precision to take into account positive matches MISSED
       by the blocker 
    '''


    sampler_list = []
    blocking_algo_list = []



    precision_list_train = []
    recall_list_train = []
    #f1_score_list_train = []
    average_precision_train = []

    precision_list_valid = []
    recall_list_valid = []
    #f1_score_list_valid = []
    average_precision_valid = []

    precision_list_test = []
    recall_list_test = []
    #f1_score_list_test = []
    average_precision_test = []

    metadata_list = []

    threshold_list = []

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
            sampler_list.append(result["sampler"][i])
            blocking_algo_list.append(result["blocking_algo"][i])

            # Redudant but we need to calculate the blocker ID for this experiment to link up with missed_positive_df values
            cutoff_list = "NA"
            min_shared_list = "NA"
            char_ngram_list = "NA"
            bands_list = "NA"

            # print(obj[5])
            # print(obj[5]["blocking"])

            if (obj[5]["blocking"] == "sequential"):
                cutoff_list= str(obj[5]["cutoff_distance"])
                min_shared_list = str(obj[5]["min_shared_tokens"])
            else:
                char_ngram_list = str(obj[5]['char_ngram'])
                bands_list = str(obj[5]['bands'])

            id_col = obj[5]["sampler"] + obj[5]["blocking"] + cutoff_list + min_shared_list + char_ngram_list + bands_list


            # Gather predictions on a probability scale
            train_predictions = obj[0][model][1]
            valid_predictions = obj[1][model][1]
            test_predictions = obj[2][model][1]

            # Calculate and Store Metrics
            ## y_true, y_pred
            train_precision, train_recall, train_thresh = precision_recall_curve(train_labels.y, train_predictions)
            precision_list_train.append(train_precision)
            recall_list_train.append(train_recall)
            average_precision_train.append(average_precision_score(train_labels.y, train_predictions))

            valid_precision, valid_recall, valid_thresh = precision_recall_curve(valid_labels.y, valid_predictions)
            precision_list_valid.append(valid_precision)
            recall_list_valid.append(valid_recall)
            average_precision_valid.append(average_precision_score(valid_labels.y, valid_predictions))

            # Adjust TEST set values to take into account MISSED positive matches by the blocker
            # Since the matching algorithm did not have a chance to encounter these values, matcher predicted probabilities
            # would be exactly zero for these models
            # need to do a join with the ID string

            adjustment_df = missed_positive_matches_df.loc[id_col]

            test_labels_adjusted = pd.concat([test_labels.y, adjustment_df.y])
            test_predictions_adjusted = np.concatenate( (test_predictions,[0]*adjustment_df.shape[0]),axis = 0)
            #print(test_labels_adjusted)

            test_precision, test_recall, test_thresh = precision_recall_curve(test_labels_adjusted, test_predictions_adjusted)
            precision_list_test.append(test_precision)
            recall_list_test.append(test_recall)
            average_precision_test.append(average_precision_score(test_labels_adjusted, test_predictions_adjusted))



            # test_precision, test_recall, test_thresh = precision_recall_curve(test_labels.y, test_predictions)
            # precision_list_test.append(test_precision)
            # recall_list_test.append(test_recall)
            # average_precision_test.append(average_precision_score(test_labels.y, test_predictions))

            threshold_list.append([train_thresh, valid_thresh, test_thresh])





    return pd.DataFrame({"sampler":sampler_list,
                    "blocking_algo":blocking_algo_list,
                    "model":model_list,
                    "train_precision":precision_list_train,
                    "train_recall":recall_list_train,
                    "train_average_precision":average_precision_train,
                    "valid_precision":precision_list_valid,
                    "valid_recall":recall_list_valid,
                    "valid_average_precision":average_precision_valid,
                    "test_precision":precision_list_test,
                    "test_recall":recall_list_test,
                    "test_average_precision":average_precision_test,
                    "thresholds":threshold_list,
                    "metadata":metadata_list})

def evaluate_matcher_deepmatcher(result, missed_positive_matches_df):
    '''
    Accepts a result object from CLOUD_model_deepmatcher.py or Kaggle_deepmatcher.py or model_deepmatcher.py
    '''
    sampler_list = []
    blocking_algo_list = []
   
    precision_list_train = []
    recall_list_train = []
    #f1_score_list_train = []
    average_precision_train = []

    precision_list_valid = []
    recall_list_valid = []
    #f1_score_list_valid = []
    average_precision_valid = []

    precision_list_test = []
    recall_list_test = []
    #f1_score_list_test = []
    average_precision_test = []

    metadata_list = []

    threshold_list = []

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

            # Only used this configuration
            metadata_list.append({"char_ngram":8,"seeds":10000,"bands":5000})

            sampler_list.append(result["sampler"][i])
            blocking_algo_list.append(result["blocking_algo"][i])

            # Redudant but we need to calculate the blocker ID for this experiment to link up with missed_positive_df values
            cutoff_list = "NA"
            min_shared_list = "NA"
            char_ngram_list = "8"
            bands_list = "5000"

            id_col = result["sampler"][i] + "lsh" + cutoff_list + min_shared_list + char_ngram_list + bands_list



            # Gather predictions on a probability scale
            train_predictions = obj[0][model]
            valid_predictions = obj[1][model]
            test_predictions = obj[2][model]

            # Calculate and Store Metrics
            ## y_true, y_pred
            train_precision, train_recall, train_thresh = precision_recall_curve(train_labels, train_predictions)
            precision_list_train.append(train_precision)
            recall_list_train.append(train_recall)
            average_precision_train.append(average_precision_score(train_labels, train_predictions))

            valid_precision, valid_recall, valid_thresh = precision_recall_curve(valid_labels, valid_predictions)
            precision_list_valid.append(valid_precision)
            recall_list_valid.append(valid_recall)
            average_precision_valid.append(average_precision_score(valid_labels, valid_predictions))

            # Adjust TEST set values to take into account MISSED positive matches by the blocker
            # Since the matching algorithm did not have a chance to encounter these values, matcher predicted probabilities
            # would be exactly zero for these models
            adjustment_df = missed_positive_matches_df.loc[id_col]

            #TODO: CHECK THIS IF TEST SET RESULTS FOR DEEP LEARNING IS FUCKY
            test_labels_adjusted = np.concatenate((test_labels, adjustment_df.y), axis = 0)
            test_predictions_adjusted = np.concatenate( (test_predictions,[0]*adjustment_df.shape[0]),axis = 0)
            #print(test_labels_adjusted)

            test_precision, test_recall, test_thresh = precision_recall_curve(test_labels_adjusted, test_predictions_adjusted)
            precision_list_test.append(test_precision)
            recall_list_test.append(test_recall)
            average_precision_test.append(average_precision_score(test_labels_adjusted, test_predictions_adjusted))

            threshold_list.append([train_thresh, valid_thresh, test_thresh])





    return pd.DataFrame({"sampler":sampler_list,
                    "blocking_algo":blocking_algo_list,
                    "model":model_list,
                    "train_precision":precision_list_train,
                    "train_recall":recall_list_train,
                    "train_average_precision":average_precision_train,
                    "valid_precision":precision_list_valid,
                    "valid_recall":recall_list_valid,
                    "valid_average_precision":average_precision_valid,
                    "test_precision":precision_list_test,
                    "test_recall":recall_list_test,
                    "test_average_precision":average_precision_test,
                    "thresholds":threshold_list,
                    "metadata":metadata_list})


def matcher_results_with_meta(result, missed_positive_matches_df, is_deep_matcher = False):
    '''
    Adds in useful metadata columns and creates ID col to be able to merge with the blocking results.
    Expects output of evaluate_matcher or evaluate_matcher_deepmatcher
    '''

    if is_deep_matcher:
        matcher_results = evaluate_matcher_deepmatcher(result,missed_positive_matches_df)
    else:
        matcher_results = evaluate_matcher(result, missed_positive_matches_df)


    nrow = matcher_results.shape[0]
    cutoff_list = ["NA"] * nrow
    min_shared_list = ["NA"] * nrow
    char_ngram_list = ["NA"] *nrow
    bands_list = ["NA"] *nrow



    id_col = [""]*nrow




    for i, blocking in enumerate(matcher_results.blocking_algo):
        if (blocking == "sequential"):
            cutoff_list[i] = str(matcher_results["metadata"][i]["cutoff_distance"])
            min_shared_list[i] = str(matcher_results["metadata"][i]["min_shared_tokens"])
        else:
            char_ngram_list[i] = str(matcher_results["metadata"][i]["char_ngram"])
            bands_list[i] = str(matcher_results["metadata"][i]["bands"])


    for i in np.arange(nrow):
        id_col[i] = matcher_results.sampler[i] + matcher_results.blocking_algo[i] + cutoff_list[i] + min_shared_list[i] + char_ngram_list[i] + bands_list[i]


    matcher_results['cutoff_distance'] = cutoff_list
    matcher_results['min_shared_tokens'] = min_shared_list
    matcher_results['char_ngram'] = char_ngram_list
    matcher_results['bands'] = bands_list
    matcher_results['id'] = id_col
    matcher_results.set_index("id", inplace = True)

    return matcher_results


def blocking_results_with_meta(result):
    '''
    Expects object of result. Calls evaluate_blocker then adds in metadata
    '''

    blocking_results = evaluate_blocking(result)

    blocking_samplers = result["sampler"]
    blocking_blocks = result["blocking_algo"]
    nrow = blocking_results.shape[0]

    cutoff_list = ["NA"] * nrow
    min_shared_list = ["NA"] * nrow
    char_ngram_list = ["NA"] *nrow
    bands_list = ["NA"] *nrow

    for i, blocking in enumerate(blocking_blocks):
        if (blocking == "sequential"):
            cutoff_list[i] = str(blocking_results["metadata"][i]["cutoff_distance"])
            min_shared_list[i] = str(blocking_results["metadata"][i]["min_shared_tokens"])
        else:
            char_ngram_list[i] = str(blocking_results["metadata"][i]["char_ngram"])
            bands_list[i] = str(blocking_results["metadata"][i]["bands"])

    id_col = [""]*nrow

    for i in np.arange(nrow):
        id_col[i] = blocking_samplers[i] + blocking_blocks[i] + cutoff_list[i] + min_shared_list[i] + char_ngram_list[i] + bands_list[i]

    # blocking_results['sampler'] = blocking_samplers
    # blocking_results['blocking_algo'] = blocking_blocks
    blocking_results['cutoff_distance'] = cutoff_list
    blocking_results['min_shared_tokens'] = min_shared_list
    blocking_results['char_ngram'] = char_ngram_list
    blocking_results['bands'] = bands_list
    blocking_results['id']= id_col
    blocking_results.set_index("id", inplace = True)

    return blocking_results






