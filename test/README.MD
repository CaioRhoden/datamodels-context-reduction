# Test

This document serves to describe how the tests are structured, what are the requirements and what are the assumptions for each part


## Tests Scheme:

- Setter
    - StratifiedSetter
        - test_stratified_setter_collection_creation
        - test_stratified_setter_number_of_samples
    - DatamodelsPipelines
        - DatamodelsNQPipeline
            - test_pre_collection_output_creation
            - test_pre_collection_output_size

## Getting Started

In order to run the test it's need some setup:
- Set a enviroment variable DATAMODELS_TEST_MODEL for the path of a small language model to be used for the test

With this done, it's only necessary to run the following command to execute all tests:
```
python -m unittest discover 
```

If only one specific script is desired it's possible to run it through
```
pytest {SCRIPT_TEST}.py
```


## Further Details

Here it will be detailed the assumptions assumed for each test:
1. Datamodels Pipeline:
    - It's assumed the existence of a DatamodelConfig for the creation  of the Pipeline
    - One of the confing paramters is the "datamodel_path", in this directory some files are required to start running all the Datamodels flow. Thy are: "test_set.csv", "train_set.csv", "train_collections.h5" and "test_collections.h5"
