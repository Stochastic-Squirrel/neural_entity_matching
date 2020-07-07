import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

amz_train = pd.read_csv("../data/amazon_google/Amazon_train.csv",index_col="id")
g_train = pd.read_csv("../data/amazon_google/Google_train.csv",index_col="id")

matches_train = pd.read_csv("../data/amazon_google/AG_perfect_matching_train.csv",index_col = ["idAmazon","idGoogleBase"])
matches_test = pd.read_csv("../data/amazon_google/AG_perfect_matching_test.csv",index_col = ["idAmazon","idGoogleBase"])

amz_train = amz_train.rename(columns = {'title':'title_amzn', "description":"description_amzn","manufacturer":"manufacturer_amzn"} )
g_train = g_train.rename(columns = {'name':'title_g', "description":"description_g","manufacturer":"manufacturer_g"})

# Generate a perfect match table
## Join Amazon Products
perfect_matches = pd.merge(amz_train, matches_train, how = 'inner', left_index=  True , right_on = matches_train.index.get_level_values("idAmazon"), suffixes = ("amzn_x","amzn_y"))
perfect_matches = perfect_matches.iloc[:,2:5]
#perfect_matches.columns = perfect_matches.columns.map(lambda x: x + "_amzn")
## Join Google Products
perfect_matches = pd.merge(g_train, perfect_matches, how = 'inner', left_index=  True , right_on = perfect_matches.index.get_level_values("idGoogleBase"), suffixes = ("g_x","g_y"))
perfect_matches = perfect_matches.rename(columns = {'name':'title_g', "description":"description_g","manufacturer":"manufacturer_g"})
## Remove junk columns
perfect_matches.drop(columns = ['key_0', 'Unnamed: 0'],inplace = True)
# Generate Negative Match table
## These rows are not involved in a match AT ALL. Need to reconstruct convincing negative pairs via edit distance
negative_amzn = amz_train[~amz_train.index.isin(perfect_matches.index.get_level_values("idAmazon"))]
negative_g = g_train[~g_train.index.isin(perfect_matches.index.get_level_values("idGoogleBase"))]




def calculate_edit_distance(x,cols):
    return fuzz.ratio("".join(str(x[cols[0]])), "".join(str(x[cols[1]])))

def generate_distance_samples(n, true_matches, negative_matches, distance_cols, plot = True):
    '''
    Inputs:
        n: Sample Size of true and positive match pairs to take WITHOUT replacement
        true_matches/negative_matches: DataFrame of true/negative match pairs
        distance_cols: column names in true/negative match tables used to calculate edit distance.
                        This is a nested list of length 2. The first nested list contains names of columns
                        to be used within each match table comparison for the first candidate entity. Second
                        list is second candidate entity columns. Columns are concatenated together to form a single
                        edit comparison string. [candidate_entity_1_cols, candidate_entity_2_cols]
                        All column names should reside in both the true_matches and negative_matches tables

    Outputs:
        (true_matches similarities, negative_matches similarities) 
    '''

    true_matches_sample = true_matches.sample(n = n)
    negative_matches_sample = negative_matches.sample(n = n)
    true_matches_sample["similarity"]= true_matches_sample.apply(axis = 1, func = calculate_edit_distance, cols = distance_cols)
    negative_matches_sample["similarity"]= negative_matches_sample.apply(axis = 1, func = calculate_edit_distance, cols = distance_cols)

    if plot:
        sns.distplot(true_matches_sample.similarity)
        sns.distplot(negative_matches_sample.similarity, color = "red")

    return (true_matches_sample.similarity, negative_matches_sample.similarity)


def generate_pos_neg_matches(positive_matching_table, table_list):
    raise NotImplementedError

def generate_em_train_valid_split(rules, prop_train = 0.8):
    raise NotImplementedError