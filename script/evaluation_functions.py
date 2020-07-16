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



def evaluate_blocking(result):
    '''
    Measures blocking performance against 
            3: All Sets pre-blocked labels
            4: All sets post-blocked labels
            set ids on the following measures.

            - Recall of Positive Matches
            - Pruning Rate against maximal set 


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
    
    for i, obj in enumerate(result["result_obj"]):
        # Analyse Pruning Power
            pre_train = obj[5]["n_train"]
            pre_valid = obj[5]["n_valid"]
            pre_test = obj[5]["n_test"]
            # Extract post-blocked data for train, valid, test
            post_train = obj[4]["train"].shape[0]
            post_valid = obj[4]["valid"].shape[0]
            post_test =  obj[4]["test"].shape[0]

            train_pruning.append(1-(post_train/pre_train**2))
            valid_pruning.append(1-(post_valid/pre_valid**2))
            test_pruning.append(1-(post_test/pre_test**2))
        # Analyse Recall of Positive Matches


    raise NotImplementedError



def evaluate_matcher(result):
    '''
    Given the post-blocked data sets, evaluate against the train, valid and test truth labels per matching algo

    Evaluates
        - Precision of Matcher
        - Recall of Matcher
        - F1 Score
    '''

    raise NotImplementedError

result = pickle.load(open("../results/magellan_Jul_16_1341.p","rb"))