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


def automatic_feature_gen(candidate_table, feature_cols, id_names, id_names_phrase):
    '''
    NB!
    The automatic function creates pairwise features. Consequently, it will convert
    internally the colnames in lhs and rhs portions of feature cols to the SAME name.
    It does this by trimming the `id_names_phrase` portion (suffix or prefix) from each column name
    It assumes that the id names are of the form id_{id_names_phrase} e.g. id_amzn

    Takes in a single DataFrame object (lhs_table and rhs_table concatenated) and
    splits it into two tables then generates features on each of the sub tables.

    Inputs:
            candidate_table: single Pandas DataFrame (typically output of blocking_algorithms.py functions)

    Outputs:
    '''
    em.del_catalog()
    candidate_table = candidate_table.reset_index()
    
    #id_names_phrase = ["_amzn","_g"]

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

    em.set_key(lhs_table, id_names[0])
    em.set_key(rhs_table, id_names[1])
    # Generate List Of Features
    matching_features = em.get_features_for_matching(lhs_table.drop(id_names[0], axis = 1), rhs_table.drop(id_names[1], axis = 1), validate_inferred_attr_types= False)
    # Extract feature vectors and save as a  DF
    # Set primary keys and foreign keys for candidate table
    candidate_table["index"] = np.arange(candidate_table.shape[0])

    em.set_key(candidate_table, "index")
    em.set_fk_ltable(candidate_table, id_names[0])
    em.set_fk_rtable(candidate_table, id_names[1])
    em.set_ltable(candidate_table, lhs_table)
    em.set_rtable(candidate_table, rhs_table)

    matching_features_df = em.extract_feature_vecs(candidate_table, 
                            feature_table = matching_features, 
                            show_progress = False)

    matching_features_df = em.impute_table(matching_features_df, 
                exclude_attrs=['index', 'id_amzn','id_g'],
                strategy='mean')


    return matching_features_df


# Import Data and Block according to a chosen method
# Magellan Way
# TODO: need to wrap this in a function
# Read in Data into memory
em.del_catalog()
lhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_iterative_X_train_lhs.csv").rename(columns = {"Unnamed: 0":"id_lhs"})
rhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_iterative_X_train_rhs.csv").rename(columns = {"Unnamed: 0":"id_rhs"})
em.del_catalog()
em.set_key(lhs_table, "id_lhs")
em.set_key(rhs_table, "id_rhs")
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
# Sequential Blocking
## Initial Rough Block
candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, 2, feature_cols, id_names)

## Take the candidates and block on top of it
# Note the use of None 
overlapped_attribute_blocking(None, None, blocking_cols, 12, feature_cols, id_names, True, candidates)
second_blocking = edit_distance_blocking(None, None, blocking_cols, 60, True, candidates)

# LSH Hashing
lhs_table = pd.read_csv("../data/processed_amazon_google/amz_google_iterative_X_train_lhs.csv")
rhs_table = pd.read_csv("../data/processed_amazon_google/amz_google_iterative_X_train_rhs.csv")
y_train = pd.read_csv("../data/processed_amazon_google/amz_google_iterative_y_train.csv")
candidate_pairs = lsh_blocking(lhs_table, rhs_table, 1, 5, ["id_amzn","id_g"], char_ngram = 2, seeds = 200, bands = 4)


id_names_phrase = ["_amzn","_g"]
feature_cols  = [['title_amzn',
'description_amzn', # removed manufacturer due to missingess: produces features with nans
'price_amzn'],
['title_g',
'description_g',
'price_g']]


# generate regression features based off a blocking set
generated_df  =  automatic_feature_gen(candidate_pairs, feature_cols, id_names, ["_amzn","_g"])
# join up with generated DF

generated_df = pd.merge(generated_df, y_train, left_on  = ["id_amzn","id_g"] , right_on = ["id_amzn","id_g"], how  = "left")
generated_df.y = generated_df.y.map({1.0:int(1), np.nan: 0})
# TODO: need to deal with NAN entries
# LOTS OF NANAS BECAUSE OF BLANK ENTRIES FOR MANUFACTURERES
# Removed problematic manufacturer column for generating values




# https://nbviewer.jupyter.org/github/anhaidgroup/py_entitymatching/blob/master/notebooks/guides/step_wise_em_guides/Selecting%20the%20Best%20Learning%20Matcher.ipynb


# TODO: set up a modelling pipeline

dt = em.DTMatcher(name='DecisionTree', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
rf = em.RFMatcher(name='RF', random_state=0)
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')
xg = em.XGBoostMatcher(name = "Xg-Boost")

# Select the best ML matcher using CV
result = em.select_matcher([dt, rf, svm, ln, lg, xg], table= generated_df, 
        exclude_attrs=['index', 'id_amzn','id_g'],
        k=5,
        target_attr='y', metric_to_select_matcher='f1', random_state=0)
result['cv_stats']
result.keys()