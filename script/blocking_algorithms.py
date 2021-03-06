''' 
This script includes all blocking algorithm implementation
Falls within the EM framework
 '''
import os
import py_entitymatching as em
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utilities import partition_data_set, calculate_edit_block_bool
import itertools
from lsh import cache, minhash # https://github.com/mattilyra/lsh
import pandas as pd

#https://onestopdataanalysis.com/lsh/
# https://anhaidgroup.github.io/py_entitymatching/v0.3.2/singlepage.html
#https://sites.google.com/site/anhaidgroup/projects/magellan/issues

# https://anhaidgroup.github.io/py_entitymatching/v0.3.x/user_manual/blocking.html#types-of-blockers-and-blocker-hierarchy
# https://nbviewer.jupyter.org/github/mattilyra/LSH/blob/master/examples/Introduction.ipynb

def overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, min_shared_tokens, feature_cols, id_names, verbose = True, candidates = None):
    '''
    Overlapp Blocking Algorithm

    Inputs: 
        blocking_cols: list of length 2 indicating which columns in LHS table and RHS table should be used to measure overlap
                        Columns presented should be in the same order as id_names when generating to data sets
        feature_cols: list of length 2 indicating which columns to KEEP for further analysis.
        id_names: list of length 2 with the names of the ids
    Outputs
        Candidate Tuples -- a dataframe of Candidate Tuples across lhs_table to rhs_table

    '''
 
    overlap = em.OverlapBlocker()
    # Add id names ot feature cols
    feature_cols[0] += [id_names[0]]
    feature_cols[1] += [id_names[1]]


    # Decide if you are blocking a pair of TABLES or if you are blocking upon already generated candidate tuples
    if candidates is not None:
        candidate_pairs = overlap.block_candset(candidates,
            blocking_cols[0], blocking_cols[1], 
            word_level = True, overlap_size = min_shared_tokens, allow_missing=True,
            show_progress = verbose)
    else:
        candidate_pairs = overlap.block_tables(lhs_table, rhs_table,
            blocking_cols[0], blocking_cols[1], 
            word_level = True, overlap_size = min_shared_tokens, allow_missing=True, 
            l_output_attrs = feature_cols[0], 
            r_output_attrs = feature_cols[1],
            l_output_prefix = "", r_output_prefix = "",
            show_progress = verbose)

    return candidate_pairs


def edit_distance_blocking(lhs_table, rhs_table, blocking_cols, cutoff_distance, verbose = True, candidates = None):
    '''
    Computes Levenstein edit distance. If similarity is below cutoff_distance, return blocking == False, otherwise return True
    Example how to use the function to block candidate pairs
        edit_distance_blocking(None, None, blocking_cols, 60, True, candidates)
    To block two tables
        candidates = edit_distance_blocking(lhs_table, rhs_table, blocking_cols, 60, True, None)
    Outputs 

        Candidate Tuples -- a dataframe of Candidate Tuples across lhs_table to rhs_table
    '''

    # Define Blackbox/Custom User Function blocking object
    bb = em.BlackBoxBlocker()
    bb.set_black_box_function(lambda lhs_table, rhs_table : calculate_edit_block_bool(lhs_table, rhs_table, blocking_cols, cutoff_distance) )

    # Decide if you are blocking a pair of TABLES or if you are blocking upon already generated candidate tuples
    if candidates is not None:
        candidate_pairs =  bb.block_candset(candidates) 
    else:
        candidate_pairs =  bb.block_tables(lhs_table, rhs_tablel_output_prefix = "", r_output_prefix = "")

    return candidate_pairs

# Credit: https://github.com/mattilyra/lsh for the minhash algorithm
def lsh_blocking(lhs_table, rhs_table, hashing_col_position, id_position, id_names, char_ngram=5, seeds=100, bands=5):
    '''
    https://www.youtube.com/watch?v=n3dCcwWV4_k
    https://nbviewer.jupyter.org/github/mattilyra/LSH/blob/master/examples/Introduction.ipynb
    
    hashing_col_position = 
    Bands = Number of Pieces or Bins
    The size of each bin is inferred

    Outputs:
        Returns a Dataframe of Candidate tuples 
    '''

    hasher = minhash.MinHasher(seeds=seeds, char_ngram=char_ngram, hashbytes=4)
    if seeds % bands != 0:
        raise ValueError('Seeds has to be a multiple of bands. {} % {} != 0'.format(seeds, bands))



    #Print Hashing information
    for _ in lhs_table.itertuples():
        print(f"Hashing Column name in LHS table is: {lhs_table.columns[hashing_col_position] }, RHS: {rhs_table.columns[hashing_col_position] }")
        print(f"Id Column name in LHS table is: {lhs_table.columns[id_position ]}, RHS {rhs_table.columns[id_position]}")
        break


    lshcache = cache.Cache(num_bands=bands, hasher=hasher)
    lshcache.clear()
    # NB! iterating over tuples puts the index in the FIRST position (adds a col in the beginning) therefore we scale forward the index
    # as specified by the usual column position by 1
    print("Adding Fingerprints")
    for x in rhs_table.itertuples():
        #document_string = x[hashing_col_position[0]+1] + " " +  str(x[hashing_col_position[1]+1]) 
        document_string = str(x[hashing_col_position + 1])
        # If the doc string is ShORTER than char_ngram will throw an error with no message
        if (len(document_string) < char_ngram ):
            document_string = document_string + " "*(char_ngram-len(document_string))
        docid  = x[id_position + 1]
        # add finger print for entity to the collection
        #print(f"docid {docid}" )
        lshcache.add_fingerprint(hasher.fingerprint(document_string.encode('utf8')), docid)

    for x in lhs_table.itertuples():
        #document_string = x[hashing_col_position[0]+1] + " " +  str(x[hashing_col_position[1]+1]) 
        document_string = str(x[hashing_col_position + 1])
        if (len(document_string) < char_ngram ):
            document_string = document_string + " "*(char_ngram-len(document_string)) 
        docid  = x[id_position + 1]
        lshcache.add_fingerprint(hasher.fingerprint(document_string.encode('utf8')), docid)
    

    print("Generating Possible Pairs")
    candidate_pairs = set()
    for b in lshcache.bins:
        for bucket_id in b:
            if len(b[bucket_id]) > 1:
                pairs_ = set(itertools.combinations(b[bucket_id], r=2))
                candidate_pairs.update(pairs_)
    
    # Assign Id_names for generating DataFrame
    lhs_table = lhs_table.set_index(id_names[0])
    rhs_table = rhs_table.set_index(id_names[1])

    print("Pruning and re-arranging possible pair indices.")
    appropriate_indices = set()
    #Faster Way than interating through all pairs
    candidate_indices = pd.Index(candidate_pairs)

    # Split indices into position 1 and 2 and then we check to see where it matches up
    candidate_indices_p1 = pd.Index([ x[0] for x in candidate_indices])
    candidate_indices_p2 = pd.Index([ x[1] for x in candidate_indices])
    # Central issue is that by default within table candidate pairs are given and sometimes the lhs and rhs indices are swapped

    # Check if correct lhs + rhs alignment is met and store those indices
    candidate_pairs_correct_alignment_bool = ((candidate_indices_p1.isin(lhs_table.index)) & (candidate_indices_p2.isin(rhs_table.index)))
    correct_alignment_indices = pd.MultiIndex.from_arrays([candidate_indices_p1[candidate_pairs_correct_alignment_bool],candidate_indices_p2[candidate_pairs_correct_alignment_bool]])
    # Now consider the fact that the indices for lhs are in the SECOND posiition in candidate indices and save accordingly
    # Check for VALID across table pairs BUT the order has just been switched
    candidate_pairs_switched_alignment_bool = ((candidate_indices_p2.isin(lhs_table.index)) & (candidate_indices_p1.isin(rhs_table.index)))
    switched_alignment_indices = pd.MultiIndex.from_arrays([candidate_indices_p2[candidate_pairs_switched_alignment_bool],candidate_indices_p1[candidate_pairs_switched_alignment_bool]])
    # Now merge the two sets of indices together
    appropriate_indices = correct_alignment_indices.union(switched_alignment_indices)
    appropriate_indices.names = id_names


    candidate_pair_df = pd.concat([lhs_table.loc[appropriate_indices.get_level_values(0)].reset_index(), rhs_table.loc[appropriate_indices.get_level_values(1)].reset_index()],axis = 1)
    candidate_pair_df = candidate_pair_df.set_index(keys = id_names)
    # Remove instances where id_names contain null entries
    non_null_entries = (~candidate_pair_df.index.get_level_values(0).isnull()) & (~candidate_pair_df.index.get_level_values(1).isnull())
    candidate_pair_df = candidate_pair_df.loc[non_null_entries, :]

    lshcache.clear()

    return candidate_pair_df




# # TODO: understand why blocking phase fails so much for LSh
# # debug this:
# seeds = 10000
# char_ngram = 8
# bands = 2500
# #bands = 1250
# sampler = "iterative"
# hashbytes = 4
# hashing_col_position = 1
# id_position = 5
# id_names = ["id_amzn","id_g"]
# lhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_test_lhs.csv").rename(columns = {"Unnamed: 0":"id_lhs"})
# #lhs_table = lhs_table.drop(246)
# rhs_table = em.read_csv_metadata("../data/processed_amazon_google/amz_google_" + sampler + "_X_test_rhs.csv").rename(columns = {"Unnamed: 0":"id_rhs"})

# # IT IS THE ER ROW!!!!
# # if you pick a char_ngram too large, it will crash as you are creating an empty set!!!!
# # 140 test naive rhs google

