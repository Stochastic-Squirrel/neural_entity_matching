import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
from functools import reduce

# Script contains the abstract EM_Data class which produces training, valid and test sets
# Specific implementations of EM_Data for Amazon-Google product and Quora question pairs are included

class EM_Data:
    # Utility Functions
    def calculate_edit_distance(self, x,cols):
        return fuzz.ratio("".join(str(x[cols[0]])), "".join(str(x[cols[1]])))

    def generate_distance_samples(self, pos_n, neg_n, true_matches, negative_matches, id_names, distance_cols, plot, return_sim, seed):
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
            negative_matches_sample[0] = negative_matches[0].sample(n = neg_n, random_state = seed).reset_index(drop = False)
            negative_matches_sample[1] = negative_matches[1].sample(n = neg_n, random_state = seed).reset_index(drop = False)
            negative_matches_sample = pd.concat(negative_matches_sample, axis = 1)
            negative_matches_sample = negative_matches_sample.set_index(id_names)
            # Calculate pairwise similarities across each row
            negative_matches_sample["similarity"] = negative_matches_sample.apply(axis = 1, func = self.calculate_edit_distance, cols = distance_cols)
        else:
            negative_matches_sample = pd.DataFrame({"similarity":[-1000]})

        if (pos_n > 0):
            true_matches_sample = true_matches.sample(n = pos_n, random_state = seed)
            true_matches_sample["similarity"] = true_matches_sample.apply(axis = 1, func = self.calculate_edit_distance, cols = distance_cols)
        else:
            true_matches_sample = pd.DataFrame({"similarity":[-1000]})

        if plot:
            sns.distplot(true_matches_sample.similarity)
            sns.distplot(negative_matches_sample.similarity, color = "red")
        if return_sim:
            return (true_matches_sample.similarity, negative_matches_sample.similarity)

    def generate_pos_neg_matches(self, positive_matching_table, table_list, id_names, feature_cols):
        '''
        Inputs:
                positive_matching_table: a two column DataFrame where the first column is the index of the first
                table in table list and second column is the index of the second table in table_list

                table_list: a list of length 2 consisting of DataFrames

                id_names: Names of the ID columns found in positive_matching_table and in table list

                feature_cols: names of the features across both tables in table_lists which need to be returned. A single entry list NOT A LISTED ONE
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

    def generate_em_train_valid_split(self, generated_matches, id_names, feature_cols, difficult_cutoff, prop_train, diff_sub_sample, seed):

        '''
        Inputs:
                generated_matches: output of generate_pos_neg_matches()

                difficult_cutoff: proportion of distance values that are defined as difficult 
                diff_sub_sample: sample size the internal function uses to determine difficult and easy examples.
                                Larger is better but will take longer to run. This should scale with the dataset size.
                feature_cols: list of length 2 indicating which columns to use for distance measure
                prop_train: proportion of data allocated to training DataFrame
        Outputs:
                1 = Duplicate/Match; 0 = Non-match
                (X_train, y_train, X_valid, y_valid, meta_data_dictionary)

        '''

        total_size = generated_matches[0].shape[0] + min(generated_matches[1][0].shape[0],generated_matches[1][1].shape[0]) 
        train_size = round(total_size * prop_train,0)


        # TODO: repeat this 5 times and take average
        # First, generate a cutoff rule in terms of similarity measure units according to difficult_cutoff
        pos_sim, neg_sim = self.generate_distance_samples(diff_sub_sample, diff_sub_sample, generated_matches[0], generated_matches[1], id_names, feature_cols, False, True, seed)
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

            if (min_sample_pos > diff_sub_sample):
                pos_n = diff_sub_sample
            elif (min_sample_pos > 0 & min_sample_pos <= diff_sub_sample):
                pos_n = min_sample_pos
            else:
                pos_n = 0

            if (min_sample_neg > diff_sub_sample):
                neg_n = diff_sub_sample
            elif (min_sample_neg > 0 & min_sample_neg <= diff_sub_sample):
                neg_n = min_sample_neg
            else:
                neg_n = 0
            # print(f"min samplepos is {min_sample_pos}, min sample neg is {min_sample_neg}")
            # print(f"Iteration {iteration} with pos_n:{pos_n} and neg_n {neg_n}")

            # Calculate Edit Distance for a random sample of positive and negative matches
            pos_indices, neg_indices = self.generate_distance_samples(pos_n, neg_n, positive_matches, negative_matches, id_names, feature_cols, False, True, seed)
            
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
        neg_easy_samples = pd.concat([negative_matches[0].loc[neg_easy_indices[0]].sample(n = max_easy_example_size, random_state = seed).reset_index(), \
                                    negative_matches[1].loc[neg_easy_indices[1]].sample(n = max_easy_example_size, random_state = seed).reset_index() ],axis = 1)
        neg_easy_samples = neg_easy_samples.set_index(keys = id_names)

        # Also need to stitch together the difficult matches into a dataframe
        max_diff_example_size = len(neg_difficult_indices)
        neg_diff_samples = pd.concat([negative_matches[0].loc[neg_difficult_indices.get_level_values(0)].sample(n = max_diff_example_size, random_state = seed).reset_index(), \
                                    negative_matches[1].loc[neg_difficult_indices.get_level_values(1)].sample(n = max_diff_example_size, random_state = seed).reset_index() ],axis = 1)
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

        meta_data = {"pos_neg_cutoffs":[pos_sim_cutoff, neg_sim_cutoff], "prop_pos_difficult":total_pos_difficult/total_size, "prop_neg_difficult": total_neg_difficult/total_size, "seed":seed, "diff_sub_sample":diff_sub_sample, "diff_target_ratio":difficult_cutoff}

        return X_train, Y_train, X_valid, Y_valid, meta_data

    def generate_em_test_set(self, generated_matches, id_names, seed):
        '''
        Inputs:
            generated_matches: a generate_pos_neg_matches() object generated from test set data
            id_names: list of length 2 consisting of the string names of the left table and right table.
                    The order of the id_names is critical and must match the table_list order in generate_pos_neg_matches()

        Outputs:
            1 = Duplicate/Match; 0 = Non-match
            X_test, y_test  

        '''

        # Extract the positive and negative matches
        positive_matches = generated_matches[0].copy()
        negative_matches = generated_matches[1].copy()
        # Can only create number of  negative test set matches equal to min of samples from two tables
        max_negative_matches = min(negative_matches[0].shape[0], negative_matches[1].shape[0])

        test_set_negative = pd.concat([negative_matches[0].sample(n = max_negative_matches).reset_index(), negative_matches[1].sample(n = max_negative_matches, random_state = seed).reset_index()],axis = 1)
        test_set_negative = test_set_negative.set_index(keys = id_names)

        X_test = pd.concat([positive_matches,test_set_negative])
        Y_test = [1]* positive_matches.shape[0] + [0]*max_negative_matches

        return X_test, Y_test

    def __init__(self, positive_matching_table_train_valid, table_list_train_valid, positive_matching_table_test, table_list_test, id_names, distance_cols, feature_cols, seed, diff_sub_sample, difficult_cutoff, prop_train):
        # Run Matches for Train-Valid and Test
        self.generated_matches_train_valid = self.generate_pos_neg_matches(positive_matching_table_train_valid,\
                                                                        table_list_train_valid, \
                                                                        id_names, \
                                                                        feature_cols, \
                                                                        )

        self.generated_matches_test = self.generate_pos_neg_matches(positive_matching_table_test, \
                                                                table_list_test, \
                                                                id_names, \
                                                                feature_cols, \
                                                                )

        # Generate Train-Valid and Test
        self.train_valid_sets = self.generate_em_train_valid_split(self.generated_matches_train_valid,\
                                                                id_names, \
                                                                feature_cols, \
                                                                difficult_cutoff, \
                                                                prop_train, \
                                                                diff_sub_sample, \
                                                                seed)

        self.test_sets = self.generate_em_test_set(self.generated_matches_test,\
                                                                id_names, \
                                                                seed)
        # Store Some General Meta Data at a class level
        self.id_names = id_names
        self.feature_cols = feature_cols
class Amazon_Google:

    def __init__(self, seed, diff_sub_sample, difficult_cutoff, prop_train):
        amz_train = pd.read_csv("../data/amazon_google/Amazon_train.csv")
        g_train = pd.read_csv("../data/amazon_google/Google_train.csv")

        amz_test = pd.read_csv("../data/amazon_google/Amazon_test.csv")
        g_test = pd.read_csv("../data/amazon_google/Google_test.csv")


        matches_train = pd.read_csv("../data/amazon_google/AG_perfect_matching_train.csv")
        matches_test = pd.read_csv("../data/amazon_google/AG_perfect_matching_test.csv")

        matches_train.columns = ["unknown","id_amzn","id_g"]
        matches_test.columns = ["unknown","id_amzn","id_g"]

        amz_train = amz_train.rename(columns = {"price":"price_amzn","id":"id_amzn",'title':'title_amzn', "description":"description_amzn","manufacturer":"manufacturer_amzn"} )
        g_train = g_train.rename(columns = {"price":"price_g","id":"id_g",'name':'title_g', "description":"description_g","manufacturer":"manufacturer_g"})

        amz_test = amz_test.rename(columns = {"price":"price_amzn","id":"id_amzn",'title':'title_amzn', "description":"description_amzn","manufacturer":"manufacturer_amzn"} )
        g_test = g_test.rename(columns = {"price":"price_g","id":"id_g",'name':'title_g', "description":"description_g","manufacturer":"manufacturer_g"})

        self.data = EM_Data(matches_train,
                            [amz_train, g_train],
                            matches_test,
                            [amz_test, g_test],
                            ["id_amzn","id_g"],
                            [["title_g","description_g"],["title_amzn","description_amzn"]],
                            ['title_amzn', 'description_amzn',
       'manufacturer_amzn', 'price_amzn', 'title_g', 'description_g', 'manufacturer_g',
       'price_g'],
       seed,
       diff_sub_sample,
       difficult_cutoff,
       prop_train)
       
class Quora:
    def __init__(self, seed, diff_sub_sample, difficult_cutoff, prop_train):
        quora_train = pd.read_csv("../data/quora/quora_train.csv",index_col =  ["qid1","qid2"])
        quora_train = quora_train.drop(columns = ["Unnamed: 0","id"])


        quora_test = pd.read_csv("../data/quora/quora_test.csv",index_col =  ["qid1","qid2"])
        quora_test = quora_test.drop(columns = ["Unnamed: 0","id"])

        quora_matches_train = quora_train[quora_train.is_duplicate == 1].drop(columns = ["question1","question2","is_duplicate"]).reset_index()
        qid1_table_train = quora_train.loc[:,"question1"].reset_index().loc[:,["qid1","question1"]]
        qid2_table_train = quora_train.loc[:,"question2"].reset_index().loc[:,["qid2","question2"]]


        quora_matches_test = quora_test[quora_test.is_duplicate == 1].drop(columns = ["question1","question2","is_duplicate"]).reset_index()
        qid1_table_test = quora_test.loc[:,"question1"].reset_index().loc[:,["qid1","question1"]]
        qid2_table_test = quora_test.loc[:,"question2"].reset_index().loc[:,["qid2","question2"]]

        self.data = EM_Data(quora_matches_train,
                            [qid1_table_train, qid2_table_train],
                            quora_matches_test,
                            [qid1_table_test, qid2_table_test],
                            ["qid1","qid2"],
                            [["question1"],["question2"]],
                            ["question1","question2"],
                            seed,
                            diff_sub_sample,
                            difficult_cutoff,
                            prop_train)


def partition_data_set(data_set, id_names, feature_cols):
    
    '''
    Utility function which takes in a data_set i.e X_train, X_valid, X_test and splits it into two DataFrames
    in prepartaion for blocking functions which will generate candidate tuples.
    '''
    data_set = data_set.reset_index()
    column_bisection_index = int(len(feature_cols)/2)

    lhs_table_cols = feature_cols[0:column_bisection_index] + [id_names[0]]
    rhs_table_cols = feature_cols[column_bisection_index:] + [id_names[1]]

    return data_set.loc[:, lhs_table_cols], data_set.loc[:, rhs_table_cols]



# amz_goog = Amazon_Google(seed = 420, diff_sub_sample = 100, difficult_cutoff = 0.3, prop_train = 0.8)

# quora = Quora(seed = 420, diff_sub_sample = 10000, difficult_cutoff = 0.05, prop_train = 0.8)



# partition_data_set(amz_goog.data.train_valid_sets[0], amz_goog.data.id_names, amz_goog.data.feature_cols)