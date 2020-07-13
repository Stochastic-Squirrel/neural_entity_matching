# Installation Notes
To get py_entitymatching to work with later version of cloudpickle (1.5 in my case),
under `extractfeatures.py` change `from cloudpickle import cloudpickle` to `import cloudpickle` on line 14.