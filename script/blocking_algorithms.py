# This script includes all blocking algorithm implementation


def shared_attributes_blocking(distance_cols):
    '''
    Blocking Algorithm which collects candidates with respect to sharing some selected attributes
    '''
    raise NotImplementedError

def locality_sensitive_hashing_blocking():

    raise NotImplementedError





#https://onestopdataanalysis.com/lsh/
# https://anhaidgroup.github.io/py_entitymatching/v0.3.2/singlepage.html
#https://sites.google.com/site/anhaidgroup/projects/magellan/issues
import py_entitymatching as em
em.OverlapBlocker()