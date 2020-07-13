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

def overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, min_shared_tokens, feature_cols, verbose = True, candidates = None):
    '''
    Overlapp Blocking Algorithm

    Inputs: 
        blocking_cols: list of length 2 indicating which columns in LHS table and RHS table should be used to measure overlap
                        Columns presented should be in the same order as id_names when generating to data sets
        feature_cols: list of length 2 indicating which columns to KEEP for further analysis
    
    Outputs
        Candidate Tuples -- a dataframe of Candidate Tuples across lhs_table to rhs_table

    '''
 
    overlap = em.OverlapBlocker()



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
def lsh_blocking(lhs_table, rhs_table, hashing_col_position, id_position, id_names, char_ngram=5, seeds=100, bands=5, hashbytes=4):
    '''
    https://www.youtube.com/watch?v=n3dCcwWV4_k
    https://nbviewer.jupyter.org/github/mattilyra/LSH/blob/master/examples/Introduction.ipynb
    

    Outputs:
        Returns a Dataframe of Candidate tuples 
    '''

    hasher = minhash.MinHasher(seeds=seeds, char_ngram=char_ngram, hashbytes=hashbytes)
    if seeds % bands != 0:
        raise ValueError('Seeds has to be a multiple of bands. {} % {} != 0'.format(seeds, bands))



    #Print Hashing information
    for _ in lhs_table.itertuples():
        print(f"Hashing Column name in LHS table is: {lhs_table.columns[hashing_col_position] }, RHS: {rhs_table.columns[hashing_col_position] }")
        print(f"Id Column name in LHS table is: {lhs_table.columns[id_position ]}, RHS {rhs_table.columns[id_position]}")
        break

    # NB! iterating over tuples puts the index in the FIRST position therefore we scale forward the index
    # as specified by the usual column position by 1
    hashing_col_position += 1
    id_position += 1
    lshcache = cache.Cache(num_bands=bands, hasher=hasher)

    for x in rhs_table.itertuples():
        document_string = x[hashing_col_position] 
        docid  = x[id_position]
        # add finger print for entity to the collection
        lshcache.add_fingerprint(hasher.fingerprint(document_string), docid)

    for x in lhs_table.itertuples():
        document_string = x[hashing_col_position] 
        docid  = x[id_position]
        lshcache.add_fingerprint(hasher.fingerprint(document_string), docid)
    


    candidate_pairs = set()
    for b in lshcache.bins:
        for bucket_id in b:
            if len(b[bucket_id]) > 1:
                pairs_ = set(itertools.combinations(b[bucket_id], r=2))
                candidate_pairs.update(pairs_)
    
    # Assign Id_names for generating DataFrame
    lhs_table = lhs_table.set_index(id_names[0])
    rhs_table = rhs_table.set_index(id_names[1])

    appropriate_indices = set()
    for i in candidate_pairs:
        # Create a hierarchical index 
        candidate_index = pd.Index([i])
        # First Try: Assume Correct Alignment
        if ((candidate_index.get_level_values(0).isin(lhs_table.index)) & (candidate_index.get_level_values(1).isin(rhs_table.index))):
            appropriate_indices.update({i})
        elif ((candidate_index.get_level_values(1).isin(lhs_table.index)) & (candidate_index.get_level_values(0).isin(rhs_table.index))): 
        # Assuming that the LHS index is actually in position one for this candidate pair
        # If this is the case, you need to swap the candidate_index values to align with lhs_table; rhs_table ordering
            appropriate_indices.update({(i[1],i[0])})
        # If a candidate index does NOT pass any of these tests, it means that it represents a WITHIN table possible duplicate
        # This is not the objective of this code

    # Convert set object to an index for easy pandas manipulation
    appropriate_indices = pd.Index(appropriate_indices)
    candidate_pair_df = pd.concat([lhs_table.loc[lhs_table.index.isin(appropriate_indices.get_level_values(0)),:].reset_index(), rhs_table.loc[rhs_table.index.isin(appropriate_indices.get_level_values(1)),:].reset_index()],axis = 1)
    candidate_pair_df = candidate_pair_df.set_index(keys = id_names)
    # Remove instances where id_names contain null entries
    non_null_entries = (~candidate_pair_df.index.get_level_values(0).isnull()) & (~candidate_pair_df.index.get_level_values(1).isnull())
    candidate_pairs_df = candidate_pair_df.loc[non_null_entries, :]


    return candidate_pair_df





