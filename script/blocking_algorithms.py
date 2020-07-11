# This script includes all blocking algorithm implementation
# Falls within the EM framework

import os
import py_entitymatching as em

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utilities import partition_data_set

def overlapped_attribute_blocking(data_set, blocking_cols, min_shared_tokens, feature_cols, id_names, verbose = True):
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
   
    data_set = em.read_csv_metadata("../data/processed_amazon_google/amz_google_X_train.csv").reset_index()
    em.set_key(data_set, "index")
    blocking_cols = ["manufacturer_amzn","manufacturer_g"]
    min_shared_tokens = 2
    feature_cols  = [['title_amzn',
 'description_amzn',
 'manufacturer_amzn',
 'price_amzn'],
 ['title_g',
 'description_g',
 'manufacturer_g',
 'price_g']]
 verbose = True

    id_names =amz_goog.data.id_names 


    lhs_table, rhs_table = partition_data_set(data_set, id_names, feature_cols[0] + feature_cols[1])
    # Need to set key id as per the catalog functionality
    em.set_key(lhs_table.reset_index(), "index")
    em.set_key(rhs_table.reset_index(), "index")

    candidate_pairs = overlap.block_tables(lhs_table, rhs_table,
                    blocking_cols[0], blocking_cols[1], 
                    word_level = True, overlap_size = min_shared_tokens, allow_missing=True, 
                    l_output_attrs = feature_cols[0], 
                    r_output_attrs = feature_cols[1],
                    show_progress = verbose)

    return candidate_pairs



# overlapped_attribute_blocking(amz_goog.data.train_valid_sets[0], ["manufacturer_amzn","manufacturer_g"], 2, [['title_amzn',
#  'description_amzn',
#  'manufacturer_amzn',
#  'price_amzn'],
#  ['title_g',
#  'description_g',
#  'manufacturer_g',
#  'price_g']], amz_goog.data.id_names)


def locality_sensitive_hashing_blocking():

    raise NotImplementedError





#https://onestopdataanalysis.com/lsh/
# https://anhaidgroup.github.io/py_entitymatching/v0.3.2/singlepage.html
#https://sites.google.com/site/anhaidgroup/projects/magellan/issues
import py_entitymatching as em
em.OverlapBlocker()