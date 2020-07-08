import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os



amz_train = pd.read_csv("../data/amazon_google/Amazon_train.csv")
g_train = pd.read_csv("../data/amazon_google/Google_train.csv")

matches_train = pd.read_csv("../data/amazon_google/AG_perfect_matching_train.csv")
matches_test = pd.read_csv("../data/amazon_google/AG_perfect_matching_test.csv")

matches_train.columns = ["unknown","id_amzn","id_g"]

amz_train = amz_train.rename(columns = {"price":"price_amzn","id":"id_amzn",'title':'title_amzn', "description":"description_amzn","manufacturer":"manufacturer_amzn"} )
g_train = g_train.rename(columns = {"price":"price_g","id":"id_g",'name':'title_g', "description":"description_g","manufacturer":"manufacturer_g"})

# Generate a perfect match table
## Join Amazon Products
# perfect_matches = pd.merge(amz_train, matches_train, how = 'inner', left_index=  True , right_on = matches_train.index.get_level_values("idAmazon"), suffixes = ("amzn_x","amzn_y"))
# perfect_matches = perfect_matches.iloc[:,2:5]



# ## Join Google Products
# perfect_matches = pd.merge(g_train, perfect_matches, how = 'inner', left_index=  True , right_on = perfect_matches.index.get_level_values("idGoogleBase"), suffixes = ("g_x","g_y"))
# perfect_matches = perfect_matches.rename(columns = {'name':'title_g', "description":"description_g","manufacturer":"manufacturer_g"})
# ## Remove junk columns
# perfect_matches.drop(columns = ['key_0', 'Unnamed: 0'],inplace = True)
# # Generate Negative Match table
# ## These rows are not involved in a match AT ALL. Need to reconstruct convincing negative pairs via edit distance
# negative_amzn = amz_train[~amz_train.index.isin(perfect_matches.index.get_level_values("idAmazon"))]
# negative_g = g_train[~g_train.index.isin(perfect_matches.index.get_level_values("idGoogleBase"))]




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


def generate_pos_neg_matches(positive_matching_table, table_list, id_names, feature_cols, table_names = None):
    '''
    Inputs:
            positive_matching_table: a two column DataFrame where the first column is the index of the first
            table in table list and second column is the index of the second table in table_list

            table_list: a list of length 2 consisting of DataFrames

            id_names: Names of the ID columns found in positive_matching_table and in table list

            feature_cols: names of the features across both tables in table_lists which need to be returned
    Outputs:
            (positive_matches, negative_matches)
    '''

    # Make Sure Nested Indexing is present. First poisition of Id names is the Id of the left table
    positive_matching_table.set_index(keys = id_names, inplace = True)
    
    # Make sure that tables to be matched are indexed correctly
    for i, i_id_name in enumerate(id_names):
        table_list[i].set_index(keys = i_id_name, inplace = True)

    # Generate Positive Matches
    ## Join First Table
    positive_matches = pd.merge(table_list[0], positive_matching_table, \
                       how = 'inner', left_index = True , right_on = positive_matching_table.index.get_level_values(0))
    positive_matches = positive_matches.reset_index()
    ## Join Second Table
    positive_matches = pd.merge(table_list[1], positive_matches, \
                       how = 'inner', left_on = table_list[1].index , right_on = id_names[1])
    positive_matches = positive_matches.loc[:,id_names + feature_cols]
    positive_matches.set_index(keys = id_names, inplace = True)
    # Generate Negative Matches only including necessary feature columns
    negative_matches_list = []
    negative_matches_list.append(table_list[0][~table_list[0].index.isin(positive_matches.index.get_level_values(0))].loc[:,feature_cols])
    negative_matches_list.append(table_list[1][~table_list[1].index.isin(positive_matches.index.get_level_values(1))].loc[:,feature_cols])

    return (positive_matches, negative_matches_list)

def generate_em_train_valid_split(rules, prop_train = 0.8):
    raise NotImplementedError


table_list = [amz_train, g_train]
id_names = ["id_amzn","id_g"]
feature_cols = ['title_amzn', 'description_amzn',
       'manufacturer_amzn', 'price_amzn', 'title_g', 'description_g', 'manufacturer_g',
       'price_g']
positive_matching_table = matches_train

generated_matches = generate_pos_neg_matches(matches_train, [amz_train, g_train], id_names, feature_cols)

