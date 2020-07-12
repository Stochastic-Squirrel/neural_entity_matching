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
            show_progress = verbose)

    return candidate_pairs


def edit_distance_blocking(lhs_table, rhs_table, blocking_cols, cutoff_distance, verbose = True, candidates = None):
    '''
    Computes Levenstein edit distance. If similarity is below cutoff_distance, return blocking == False, otherwise return True
    
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
        candidate_pairs =  bb.block_tables(lhs_table, rhs_table)

    return candidate_pairs



# Credit: https://github.com/mattilyra/lsh
def shingles(text, char_ngram=5):
    return set(text[head:head + char_ngram] for head in range(0, len(text) - char_ngram))


    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def lsh_blocking(lhs_table, rhs_table, hashing_col_position, id_position, char_ngram=5, seeds=100, bands=5, hashbytes=4):
    '''
    https://www.youtube.com/watch?v=n3dCcwWV4_k
    
    

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

    for x in lhs_table.itertuples():
        document_string = x[hashing_col_position] 
        docid  = x[id_position]
        lshcache.add_fingerprint(hasher.fingerprint(document_string), docid)
    
    for x in rhs_table.itertuples():
        document_string = x[hashing_col_position] 
        docid  = x[id_position]
        lshcache.add_fingerprint(hasher.fingerprint(document_string), docid)

    candidate_pairs = set()
    for b in lshcache.bins:
        for bucket_id in b:
            if len(b[bucket_id]) > 1:
                pairs_ = set(itertools.combinations(b[bucket_id], r=2))
                candidate_pairs.update(pairs_)
    



    return candidate_pairs







# Magellan Way
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
cutoff_distance = 60



# Sequential Blocking
## Initial Rough Block
candidates = overlapped_attribute_blocking(lhs_table, rhs_table, blocking_cols, 2, feature_cols)

## Take the candidates and block on top of it
overlapped_attribute_blocking(None, None, blocking_cols, 12, feature_cols, True, candidates)
second_blocking = edit_distance_blocking(None, None, blocking_cols, 60, True, candidates)


## NOTE! Union based blocking does also exists 


# LSH Hashing
# https://nbviewer.jupyter.org/github/mattilyra/LSH/blob/master/examples/Introduction.ipynb

# # Seeds = number of hash functions = k
# # bands = number of groups
# hasher = minhash.MinHasher(seeds=100, char_ngram=5, hashbytes=4)
# lshcache = cache.Cache(bands=10, hasher=hasher)

# # read in the data file and add the first 100 documents to the LSH cache
# lhs_table = pd.read_csv("../data/processed_amazon_google/amz_google_X_train_lhs.csv").rename(columns = {"Unnamed: 0":"id_lhs"})


# # for every bucket in the LSH cache get the candidate duplicates
# candidate_pairs = set()
# for b in lshcache.bins:
#     for bucket_id in b:
#         if len(b[bucket_id]) > 1: # if the bucket contains more than a single document
#             pairs_ = set(itertools.combinations(b[bucket_id], r=2))
#             candidate_pairs.update(pairs_)


candidate_pairs = lsh_blocking(lhs_table, rhs_table, 1, 5)