'''
Entity matching model using Automatic Feature creation of magellan and tests a host
of algorithms.
'''
import os
import py_entitymatching as em
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from blocking_algorithms import *
import re


def automatic_feature_gen(candidate_table):
    '''
    Takes in a single DataFrame object (lhs_table and rhs_table concatenated) and
    splits it into two tables then generates features on each of the sub tables.

    Inputs:
            candidate_table: single Pandas DataFrame (typically output of blocking_algorithms.py functions)

    Outputs:

    '''



# Import Data and Block according to a chosen method
# Magellan Way
# TODO: need to wrap this in a function
# Read in Data into memory
em.del_catalog()
lhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_X_train_lhs.csv").rename(columns = {"Unnamed: 0":"id_lhs"})
rhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_X_train_rhs.csv").rename(columns = {"Unnamed: 0":"id_rhs"})
em.del_catalog()
em.set_key(lhs_table, "id_lhs")
em.set_key(rhs_table, "id_rhs")
blocking_cols = ["title_amzn","title_g"]
min_shared_tokens = 3
feature_cols  = [['title_amzn',
'description_amzn',
'manufacturer_amzn',
'price_amzn',
'id_amzn'],
['title_g',
'description_g',
'manufacturer_g',
'price_g',
'id_g']]
cutoff_distance = 60
# Sequential Blocking
## Initial Rough Block
candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, 2, feature_cols)

## Take the candidates and block on top of it
# Note the use of None 
overlapped_attribute_blocking(None, None, blocking_cols, 12, feature_cols, True, candidates)
second_blocking = edit_distance_blocking(None, None, blocking_cols, 60, True, candidates)

# LSH Hashing
lhs_table = pd.read_csv("../data/processed_amazon_google/amz_google_X_train_lhs.csv")
rhs_table = pd.read_csv("../data/processed_amazon_google/amz_google_X_train_rhs.csv")

candidate_pairs = lsh_blocking(lhs_table, rhs_table, 1, 5, ["id_amzn","id_g"])

# NOW try some automatic feature gen

candidate_pairs



