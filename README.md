# Installation Notes
To get py_entitymatching to work with later version of cloudpickle (1.5 in my case),
under `extractfeatures.py` change `from cloudpickle import cloudpickle` to `import cloudpickle` on line 14.

To install deepmatcher, clone from the GitHub repo directly:
`git clone https://github.com/anhaidgroup/deepmatcher`
`cd deepmatcher`
`pip install .`

You also need to install XgBoost separately and pandastable if you wish to use that functionality of py_entitymatching

# Running Scripts

* prepare_data.py 
* magellan_model.py
* deepmatcher_model.py or the cloud variants
* analyse_results.ipynb

# Results
Results can be downloaded from the folliwing drive link:
https://drive.google.com/file/d/11DoEXQ-XCdKpCqg1fIi74gaJ5jDu1pKQ/view?usp=sharing
