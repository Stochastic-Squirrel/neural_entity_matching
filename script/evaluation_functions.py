'''
Suite of evaluation functions to evaluate performance of blockers and matchers/models

Magellan Model all_result dictionary hierarchy

    Sampler; Blocking Algorithm; Result object

    For a given Sampler and Blocking algo:
        Result Object[experiment_id corresponding to Sampler-Blocking iteration][Set ID][Model Name when appropriate]
        Set Ids:
            0: Training Predictions
            1: Validation Predictions
            2: Test Set Predictions
            3: All Sets pre-blocked labels
            4: All sets post-blocked labels
            5: Experiment Meta Data
'''



def evaluate_blocking():
    '''
    
    
    '''


    raise NotImplementedError



def evaluate_matcher():
    raise NotImplementedError

