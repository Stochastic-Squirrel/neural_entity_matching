# This script includes all blocking algorithm implementation
# Falls within the EM framework

import os
import py_entitymatching as em

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utilties import partition_data_set

def overlapped_attribute_blocking(blocking_cols, min_shared_tokens, feature_cols, verbose = True):
    '''
    Overlapp Blocking Algorithm

    Inputs: 
        blocking_cols: list of length 2 indicating which columns in LHS table and RHS table should be used to measure overlap
                        Columns presented should be in the same order as id_names when generating to data sets
        feature_cols: list of length 2 indicating which columns to KEEP for further analysis
    
    Outputs
        Candidate Tuples

    '''
    overlap = em.OverlapBlocker()

    candidate_pairs = overlap.block_tables(A, B,
                    blocking_cols[0], blocking_cols[1], 
                    word_level = True, overlap_size = min_shared_tokens, allow_missing=True, 
                    l_output_attrs = feature_cols[0], 
                    r_output_attrs = feature_cols[1],
                    show_progress = verbose)

    return candidate_pairs

def locality_sensitive_hashing_blocking():

    raise NotImplementedError





#https://onestopdataanalysis.com/lsh/
# https://anhaidgroup.github.io/py_entitymatching/v0.3.2/singlepage.html
#https://sites.google.com/site/anhaidgroup/projects/magellan/issues
import py_entitymatching as em
em.OverlapBlocker()