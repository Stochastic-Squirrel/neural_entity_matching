'''
Purpose is to test suitability of LSH at longer text form data to assess its performance.
'''
import os
import py_entitymatching as em
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from blocking_algorithms import *
import re
import numpy as np
from progressbar import progressbar
import pickle
import datetime



def run_quora_blocking(sampler = "iterative", lsh_args = None, sequential_args = None):
    if (sampler != "iterative") & (sampler != "naive"):
        raise ValueError("Sampler should be iterative or naive (completely random).")

    # Load Training Set according to sampler
    em.del_catalog()
    lhs_table = em.read_csv_metadata("../data/processed_quora/quora_" + sampler + "_X_train_lhs.csv").rename(columns = {"Unnamed: 0":"id_lhs"}).sample(n = 1500, random_state = 52)
    rhs_table = em.read_csv_metadata("../data/processed_quora/quora_" + sampler + "_X_train_rhs.csv").rename(columns = {"Unnamed: 0":"id_rhs"}).sample(n = 1500, random_state = 52)
    y_train = pd.read_csv("../data/processed_quora/quora_" + sampler + "_y_train.csv")
    em.del_catalog()
    em.set_key(lhs_table, "id_lhs")
    em.set_key(rhs_table, "id_rhs")

    n_train  = lhs_table.shape[0]

    # Blocking
    blocking_cols = ["question1","question2"]
    feature_cols  = [["question1"],
    ['question2']]
    id_names = ["qid1","qid2"]
    lsh_blocking_col_ids = 1

    print("Blocking Train Set of Quora using LSH only.") 
    candidates = lsh_blocking(lhs_table, rhs_table, lsh_blocking_col_ids, 2, ["qid1","qid2"], char_ngram = lsh_args["char_ngram"], seeds = lsh_args["seeds"], bands = lsh_args["bands"])
    print(f"Generated Candidate size has {candidates.shape[0]} rows")
    
    
    return NotImplementedError



def expand_grid(data_dict):
    # Produces dictionary objects across exploration space
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys()).to_dict("records")


lsh_exploration_space = {"seeds":[10000], "char_ngram":[4,5], "bands":[500,1000,1250,2500]}

lsh_args = expand_grid(lsh_exploration_space)

total_num_experiments = 2*(len(lsh_args))