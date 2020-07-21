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
import numpy as np
import pickle
import datetime




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


# Best Blocker was lsh at 5000 bands. Investigate this under both naive and iterative samplers

lsh_args = [{"seeds":10000, "char_ngram":8, "bands":5000}]

deepmatcher_args = {"sampler":["naive","iterative"], 
                    "blocking_algo":["lsh"],
                    "model_args":[{"attr_summarizer":"sif"}]}

#,{"attr_summarizer":"rnn"}
sampler_list = []
blocking_algo_list = []
result_obj_list = []


def fit_deepmatcher(model_args, train, validation, test, batch_size = 16):
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
        batch_size= batch_size,
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
                

                    # Fit all appropriate models for given set of prepared data
                    for model_args in deepmatcher_args["model_args"]:
                        print(f"Training Model Configuration {model_args}")    
                        sampler_list.append(sampler)
                        blocking_algo_list.append(blocker)

                        # Generate Data Sets from the current blocking method
                        #For cloud version just read it in
                        train, validation, test = dm.data.process(
                            path=f'../data/tmp/{sampler}_set',
                            train='train.csv',
                            validation='valid.csv',
                            test='test.csv',
                            left_prefix='lhs_',
                            right_prefix='rhs_',
                            label_attr='y',
                            id_attr='id')
                        # Fit model and return predictions
                        result_obj_list.append(fit_deepmatcher(model_args,train, validation, test))

                        all_results = {"sampler":sampler_list, "blocking_algo":blocking_algo_list,"result_obj":result_obj_list}

                        pickle.dump( all_results, open( "../results/deep_matcher_"+datetime.datetime.today().strftime("%h_%d_%H%M")+".p", "wb" ))
                        print("WIP Results have been saved.")


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
run_deepmatcher_models(deepmatcher_args)

all_results = {"sampler":sampler_list, "blocking_algo":blocking_algo_list,"result_obj":result_obj_list}

pickle.dump( all_results, open( "../results/deep_matcher_"+datetime.datetime.today().strftime("%h_%d_%H%M")+".p", "wb" ))
print("Results have been saved.")

# Debug
# sampler = deepmatcher_args["sampler"][0]
# blocker = deepmatcher_args["blocking_algo"][0]
# block_params  = lsh_args[0]
# model_args = deepmatcher_args["model_args"][0]

