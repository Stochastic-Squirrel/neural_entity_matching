import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
from functools import reduce



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

# TODO: Need to think of a better way to handle sampling when the one table is MUCH smaller than the other
# TODO: add in a naive method which simply takes a complete random sample of gen_neg_pos() without iteratively achieving proportions


def calculate_edit_distance(x,cols):
    return fuzz.ratio("".join(str(x[cols[0]])), "".join(str(x[cols[1]])))

def generate_distance_samples(pos_n, neg_n, true_matches, negative_matches, id_names, distance_cols, plot = True, return_sim = False, search_within = False):
    '''
    Inputs:
        pos_n: Sample Size of true positive match pairs to take WITHOUT replacement
        neg_n: Sample size of true negative match pairs to take WITHOUT replacement
        true_matches: A single DataFrame of true matches
        negative_matches: A LIST of length 2 where each entry is a DataFrame
        distance_cols: column names in true/negative match tables used to calculate edit distance.
                        This is a nested list of length 2. The first nested list contains names of columns
                        to be used within each match table comparison for the first candidate entity. Second
                        list is second candidate entity columns. Columns are concatenated together to form a single
                        edit comparison string. [candidate_entity_1_cols, candidate_entity_2_cols]
                        All column names should reside in both the true_matches and negative_matches tables
        search_within: Default False. If True, it combines the negative matches table together and returns negative pairs from the combined table.
                        Consequently, negative pair outputs could be formed by looking within the same table.

    Outputs:
        (true_matches similarities, negative_matches similarities) 
    '''

    #TODO: need to hand case when a 0 sample size is passed. need to think about this error more
    # if (neg_n > negative_matches[0].shape[0] or neg_n > negative_matches[1].shape[0] or pos_n > true_matches.shape[0]):
    #     raise ValueError('Sample size n is greater in length than at least one of the tables.')

    negative_matches_sample = [None, None]

    if (neg_n > 0):
        negative_matches_sample[0] = negative_matches[0].sample(n = neg_n).reset_index(drop = False)
        negative_matches_sample[1] = negative_matches[1].sample(n = neg_n).reset_index(drop = False)
        negative_matches_sample = pd.concat(negative_matches_sample, axis = 1)
        negative_matches_sample = negative_matches_sample.set_index(id_names)
        # Calculate pairwise similarities across each row
        negative_matches_sample["similarity"] = negative_matches_sample.apply(axis = 1, func = calculate_edit_distance, cols = distance_cols)
    else:
        negative_matches_sample = pd.DataFrame({"similarity":[-1000]})

    if (pos_n > 0):
        true_matches_sample = true_matches.sample(n = pos_n)
        true_matches_sample["similarity"] = true_matches_sample.apply(axis = 1, func = calculate_edit_distance, cols = distance_cols)
    else:
        true_matches_sample = pd.DataFrame({"similarity":[-1000]})

    if plot:
        sns.distplot(true_matches_sample.similarity)
        sns.distplot(negative_matches_sample.similarity, color = "red")
    if return_sim:
        return (true_matches_sample.similarity, negative_matches_sample.similarity)

    


# # DEBUG distance function
# true_matches = generated_matches[0]
# negative_matches = generated_matches[1]
# distance_cols = [["title_g","description_g"],["title_amzn","description_amzn"]]
# n = 100
# id_names = ["id_amzn","id_g"]

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
    positive_matching_table = positive_matching_table.set_index(keys = id_names)
    
    # Make sure that tables to be matched are indexed correctly
    for i, i_id_name in enumerate(id_names):
        table_list[i] = table_list[i].set_index(keys = i_id_name)

    # Generate Positive Matches
    ## Join First Table
    positive_matches = pd.merge(table_list[0], positive_matching_table, \
                       how = 'inner', left_index = True , right_on = positive_matching_table.index.get_level_values(0))
    positive_matches = positive_matches.reset_index()
    ## Join Second Table
    positive_matches = pd.merge(table_list[1], positive_matches, \
                       how = 'inner', left_on = table_list[1].index , right_on = id_names[1])
    positive_matches = positive_matches.loc[:,id_names + feature_cols]
    positive_matches = positive_matches.set_index(keys = id_names)
    # Generate Negative Matches only including necessary feature columns
    negative_matches_list = []

    feature_cols_subset = table_list[0].columns.isin(feature_cols)
    negative_matches_list.append(table_list[0][~table_list[0].index.isin(positive_matches.index.get_level_values(0))].loc[:,feature_cols_subset])
    
    feature_cols_subset = table_list[1].columns.isin(feature_cols)
    negative_matches_list.append(table_list[1][~table_list[1].index.isin(positive_matches.index.get_level_values(1))].loc[:,feature_cols_subset])

    return (positive_matches, negative_matches_list)

# # Debug generate pos neg matches
# table_list = [amz_train, g_train]
# id_names = ["id_amzn","id_g"]
# feature_cols = ['title_amzn', 'description_amzn',
#        'manufacturer_amzn', 'price_amzn', 'title_g', 'description_g', 'manufacturer_g',
#        'price_g']
# positive_matching_table = matches_train

def generate_em_train_valid_split(generated_matches, id_names, difficult_cutoff = 0.1, prop_train = 0.8):

    '''
    Inputs:
            generated_matches: output of generate_pos_neg_matches()

            difficult_cutoff: proportion of distance values that are defined as difficult 

            prop_train: proportion of data allocated to training DataFrame
    Outputs: 
            (X_train, y_train, X_valid, y_valid)

    '''

    total_size = generated_matches[0].shape[0] + min(generated_matches[1][0].shape[0],generated_matches[1][1].shape[0]) 
    train_size = round(total_size * prop_train,0)
    valid_size = total_size - train_size

    # TODO: repeat this 5 times and take average
    # First, generate a cutoff rule in terms of similarity measure units according to difficult_cutoff
    pos_sim, neg_sim = generate_distance_samples(100, 100, generated_matches[0], generated_matches[1], id_names, [["title_g","description_g"],["title_amzn","description_amzn"]], False, True)
    # Lowest similarity is more difficult for positive matches and higher similarity
    # is more difficult for negative matches
    pos_sim_cutoff = np.quantile(pos_sim, difficult_cutoff)
    neg_sim_cutoff = np.quantile(neg_sim, 1-difficult_cutoff)

    # Iteratively Sample until we have the desired proportion of difficult negative and positive matches
    positive_matches = generated_matches[0].copy()
    negative_matches = generated_matches[1].copy()

    # Assign Indices in negative_matches
    # negative_matches[0] = negative_matches[0].set_index(keys = id_names[0])
    # negative_matches[1] = negative_matches[1].set_index(keys = id_names[1])

    pos_difficult_indices = []
    neg_difficult_indices = []


    

    # Calculate the target sample number of difficult examples
    pos_difficult_sample_target =  round(generated_matches[0].shape[0] * difficult_cutoff, 0)
    neg_difficult_sample_target = round(difficult_cutoff * min(generated_matches[1][0].shape[0],generated_matches[1][1].shape[0]), 0)

    sample_pos = True
    sample_neg = True
    iteration = 1
    while (sample_pos | sample_neg) & (iteration < 50):

        # Calculate Minimum Available sample size
        min_sample_pos = positive_matches.shape[0]
        min_sample_neg = min(negative_matches[0].shape[0], negative_matches[1].shape[0])

        if (min_sample_pos == 0):
            sample_pos = False
        if (min_sample_neg == 0):
            sample_neg = False

        if (min_sample_pos > 100):
            pos_n = 100
        elif (min_sample_pos > 0 & min_sample_pos <= 100):
            pos_n = min_sample_pos
        else:
            pos_n = 0

        if (min_sample_neg > 100):
            neg_n = 100
        elif (min_sample_neg > 0 & min_sample_neg <= 100):
            neg_n = min_sample_neg
        else:
            neg_n = 0
        # print(f"min samplepos is {min_sample_pos}, min sample neg is {min_sample_neg}")
        # print(f"Iteration {iteration} with pos_n:{pos_n} and neg_n {neg_n}")

        # Calculate Edit Distance for a random sample of positive and negative matches
        pos_indices, neg_indices = generate_distance_samples(pos_n, neg_n, positive_matches, negative_matches, id_names, [["title_g","description_g"],["title_amzn","description_amzn"]], False, True)
        
        if(sample_pos):
        # Add indices to to *_difficult_indices if they are beyond cutoff rules
        # therefore they are classified as being difficult examples
            pos_difficult_indices.append(pos_indices[pos_indices <= pos_sim_cutoff].index)
         # Remove these stored indices from the tables to avoid double sampling
            removed_pos_bool = positive_matches.index.isin(pos_indices[pos_indices <= pos_sim_cutoff].index)
            positive_matches = positive_matches[~removed_pos_bool]    

        if(sample_neg):
            neg_difficult_indices.append(neg_indices[neg_indices >= neg_sim_cutoff].index)

            removed_neg_bool = negative_matches[0].index.isin(neg_indices[neg_indices >= neg_sim_cutoff].index.get_level_values(0))
            negative_matches[0] = negative_matches[0][~removed_neg_bool]

            removed_neg_bool = negative_matches[1].index.isin(neg_indices[neg_indices >= neg_sim_cutoff].index.get_level_values(1))
            negative_matches[1] = negative_matches[1][~removed_neg_bool] 

        
        # Check if target density of difficuly examples have been achieved, if it has simply randomly sample the rest and don't calc distances 
        total_pos_difficult = np.sum(list(map(len,pos_difficult_indices)))
        total_neg_difficult = np.sum(list(map(len,neg_difficult_indices)))

        if(total_pos_difficult >= pos_difficult_sample_target): sample_pos = False
        if(total_neg_difficult >= neg_difficult_sample_target): sample_neg = False
        
        iteration += 1
        
    # Reinitialise positive and negative matches for pulling data
    positive_matches = generated_matches[0].copy()
    negative_matches = generated_matches[1].copy()

    
    # collapse indices to a single list entry
    pos_difficult_indices = reduce(lambda l1, l2: l1.union(l2), pos_difficult_indices)
    neg_difficult_indices = reduce(lambda l1, l2: l1.union(l2), neg_difficult_indices)
    
    pos_easy_indices = positive_matches.index[~positive_matches.index.isin(pos_difficult_indices)]
    neg_easy_indices = [None] * 2
    neg_easy_indices[0] = negative_matches[0].index[~negative_matches[0].index.isin(neg_difficult_indices.get_level_values(0))]
    neg_easy_indices[1] = negative_matches[1].index[~negative_matches[1].index.isin(neg_difficult_indices.get_level_values(1))]
    # Need to figure out which table in the negative EASY indices is the smallest so we stick negative examples
    # from the other table onto it and discard the remainder
    # Below we actually pull the data 
    max_easy_example_size = min(neg_easy_indices[0].shape[0], neg_easy_indices[1].shape[0])
    neg_easy_samples = pd.concat([negative_matches[0].loc[neg_easy_indices[0]].sample(n = max_easy_example_size).reset_index(), \
                                   negative_matches[1].loc[neg_easy_indices[1]].sample(n = max_easy_example_size).reset_index() ],axis = 1)
    neg_easy_samples = neg_easy_samples.set_index(keys = id_names)

    # Also need to stitch together the difficult matches into a dataframe
    max_diff_example_size = len(neg_difficult_indices)
    neg_diff_samples = pd.concat([negative_matches[0].loc[neg_difficult_indices.get_level_values(0)].sample(n = max_diff_example_size).reset_index(), \
                                   negative_matches[1].loc[neg_difficult_indices.get_level_values(1)].sample(n = max_diff_example_size).reset_index() ],axis = 1)
    neg_diff_samples = neg_diff_samples.set_index(keys = id_names)
    
    # Now Create the Actual DataFrames for train and valid
    ## Create Indices for positive and negative matches taking random samples in proportion using prop_train
    train_pos_diff = pos_difficult_indices[0:np.int(pos_difficult_indices.shape[0]*prop_train)]
    train_pos_easy = pos_easy_indices[0:np.int(pos_easy_indices.shape[0]*prop_train)]

    valid_pos_diff = pos_difficult_indices[np.int(pos_difficult_indices.shape[0]*prop_train):]
    valid_pos_easy = pos_easy_indices[np.int(pos_easy_indices.shape[0]*prop_train):]

    train_neg_diff  = neg_diff_samples.index[0:np.int(max_diff_example_size*prop_train)]
    train_neg_easy = neg_easy_samples.index[0:np.int(max_easy_example_size*prop_train)]

    valid_neg_diff = neg_diff_samples.index[np.int(max_diff_example_size*prop_train):]
    valid_neg_easy = neg_easy_samples.index[np.int(max_easy_example_size*prop_train):]

    ## Compile Train and Validation
    X_train = pd.concat([positive_matches.loc[train_pos_diff.union(train_pos_easy),:], \
                        neg_diff_samples.loc[train_neg_diff,:], \
                        neg_easy_samples.loc[train_neg_easy,:]],axis = 0).reset_index()
    X_train = X_train.set_index(keys = id_names)
    
    number_matches_train = len(train_pos_diff.union(train_pos_easy))
    

    X_valid = pd.concat([positive_matches.loc[valid_pos_diff.union(valid_pos_easy),:], \
                        neg_diff_samples.loc[valid_neg_diff,:], \
                        neg_easy_samples.loc[valid_neg_easy,:]],axis = 0).reset_index()
    X_valid = X_valid.set_index(keys = id_names)

    number_matches_valid = len(valid_pos_diff.union(valid_pos_easy))

    ## Create Target Vectors
    Y_train = [1]*number_matches_train + [0]*(X_train.shape[0]-number_matches_train)
    Y_valid = [1]*number_matches_valid + [0]*(X_valid.shape[0]-number_matches_valid)

    return X_train, Y_train, X_valid, Y_valid




    


created_matches = generate_pos_neg_matches(matches_train,
                                            [amz_train, g_train], 
                                            ["id_amzn","id_g"], 
                                            ['title_amzn', 'description_amzn',
       'manufacturer_amzn', 'price_amzn', 'title_g', 'description_g', 'manufacturer_g',
       'price_g'])

# DEbug generate train valid
generated_matches = created_matches
id_names = ["id_amzn","id_g"]
prop_train = 0.8
difficult_cutoff = 0.1


X_train, y_train, X_valid, y_valid = generate_em_train_valid_split(created_matches, id_names)




generate_distance_samples(200, 200, generated_matches[0], generated_matches[1], ["id_amzn","id_g"],[["title_g","description_g"],["title_amzn","description_amzn"]])


