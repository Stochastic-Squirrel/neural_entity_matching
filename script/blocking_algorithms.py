# This script includes all blocking algorithm implementation
# Falls within the EM framework

import os
import py_entitymatching as em

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utilities import partition_data_set, calculate_edit_distance_tuples

#https://onestopdataanalysis.com/lsh/
# https://anhaidgroup.github.io/py_entitymatching/v0.3.2/singlepage.html
#https://sites.google.com/site/anhaidgroup/projects/magellan/issues

# https://anhaidgroup.github.io/py_entitymatching/v0.3.x/user_manual/blocking.html#types-of-blockers-and-blocker-hierarchy


def overlapped_attribute_table_blocking(lhs_table, rhs_table, blocking_cols, min_shared_tokens, feature_cols, verbose = True):
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

    candidate_pairs = overlap.block_tables(lhs_table, rhs_table,
                    blocking_cols[0], blocking_cols[1], 
                    word_level = True, overlap_size = min_shared_tokens, allow_missing=True, 
                    l_output_attrs = feature_cols[0], 
                    r_output_attrs = feature_cols[1],
                    show_progress = verbose)

    return candidate_pairs


def edit_distance_blocking(lhs_table, rhs_table, blocking_cols, cutoff_distance, verbose = True):
    '''
    Computes Levenstein edit distance. If similarity is below cutoff_distance, return blocking == False, otherwise return True
    '''



# genreal function are .block_tables(),.block_tuples() and block_candidates()


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



candidates = overlapped_attribute_table_blocking(lhs_table, rhs_table, blocking_cols, 4, feature_cols)






def locality_sensitive_hashing_blocking():

    raise NotImplementedError





