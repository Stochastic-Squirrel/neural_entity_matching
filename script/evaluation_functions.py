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
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
#from sklearn.metrics import plot_precision_recall_curve
import itertools


# TODO: work out how to genearted matching results when using LSH as a matcher

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

    #TODO: check pruning power calculations

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

    Assumes predictions for each model are a MATCHING score between 0 to 1.
    1 being a predicted certainty of a match.

    Evaluates for the POSITIVE CLASS ONLY
        - Precision of Matcher
        - Recall of Matcher
        - F1 Score
    '''


 
    sampler_list = list(itertools.chain.from_iterable(itertools.repeat(x, len(result["result_obj"][0][0])) for x in result["sampler"]))
    blocking_algo_list = list(itertools.chain.from_iterable(itertools.repeat(x, len(result["result_obj"][0][0])) for x in result["blocking_algo"]))
   




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

            test_precision, test_recall, test_thresh = precision_recall_curve(test_labels.y, test_predictions)
            precision_list_test.append(test_precision)
            recall_list_test.append(test_recall)
            average_precision_test.append(average_precision_score(test_labels.y, test_predictions))

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




#result = pickle.load(open("../results/magellan_Jul_20_2017.p","rb"))
result = pickle.load(open("../results/deep_matcher_Jul_21_1513.p","rb"))


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
















matcher_results = evaluate_matcher(result)
# matcher_samplers = [ x["sampler"] for x in matcher_results["metadata"]]
# matcher_blocks = [x["blocking"] for x in matcher_results["metadata"]]
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

# Visualise some results
sns.scatterplot(x = blocking_results.train_recall, y = blocking_results.valid_recall, style = blocking_results.blocking_algo, hue = blocking_results.sampler)
plt.show()

sns.scatterplot(x = blocking_results.train_recall, y = blocking_results.test_recall, style = blocking_results.blocking_algo, hue = blocking_results.sampler)
plt.show()


blocking_results.groupby(["sampler","blocking_algo"]).apply(np.mean)
matcher_results.groupby(["sampler","blocking_algo"]).apply(np.mean)
matcher_results.groupby(["model","sampler","blocking_algo"]).apply(np.mean)


all_results = pd.merge(matcher_results, blocking_results,left_index = True, right_index = True, suffixes = ("_m","_b"))

all_results["best_case_overall_recall"] = all_results.test_recall_b * all_results.test_recall_m.apply(np.max)

sns.scatterplot(x = all_results.model, y = all_results.best_case_overall_recall)

print(all_results[["sampler_m","blocking_algo_m","model","best_case_overall_recall"]].sort_values("best_case_overall_recall", ascending = False))


# Plot top 3 and bottom 3 precision recall curves
top_3 = all_results[["sampler_m","blocking_algo_m","model","best_case_overall_recall","test_precision","test_recall_m"]].nlargest(3,"best_case_overall_recall")
bottom_3 = all_results[["sampler_m","blocking_algo_m","model","best_case_overall_recall","test_precision","test_recall_m"]].nsmallest(3,"best_case_overall_recall")

#Note the use of explode
plotting_data = pd.concat([top_3,bottom_3])
plotting_data["id"] = [1,2,3,4,5,6]
#plotting_data = pd.melt(plotting_data, id_vars = ["id","sampler_m","blocking_algo_m","model"], value_vars=["test_precision","test_recall_m"]).explode("value")
# plotting_data.reset_index(inplace = True)
# plotting_data.value = plotting_data.value.astype(float)


#tt = plotting_data.pivot_table(index=[plotting_data.index,plotting_data.id],columns='variable',values='value',fill_value=0)

#sns.lineplot(data =tt, x = "test_precision", y = "test_recall_m", hue = np.array(tt.index.get_level_values(1)))



for id_val in np.arange(6):
    if (id_val <=2):
        colour = "green"
    else:
        colour = "red"
    ax = sns.lineplot(x = "test_precision" , y = "test_recall_m", data = plotting_data.iloc[id_val,:], color = colour)
plt.show()

