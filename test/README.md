
# Test suite overview

This document lists the test subfolders and files in `test/` with a brief purpose statement so contributors can quickly find and understand tests.

Test folders and files

- `test_evaluators.py` — small unit tests for evaluator utilities and scoring logic.

- `datamodels_pipelines/` — tests for pipeline behavior (pre-collection, collection creation, training flows).
	- `test_pre_collections_creation.py` — validates pre-collection artifacts are created and saved.
	- `test_pre_collections_creation_batch.py` — batch-mode pre-collection creation tests.
	- `test_pre_collections_multiples_sentences.py` — exercises single and batch pre-collection pipelines across HF and vLLM backends (checks artifact presence, row counts, list lengths and dtypes).
	- `test_collections_creation.py` — tests collection-building logic.
	- `test_evaluate_datamodels.py` — integration-style tests for evaluating datamodel outputs.
	- `test_train_datamodels.py` — smoke tests for training datamodels pipeline.

- `datamodels_setter/` — tests and helpers for building train/test splits and setters.
	- `create_toy_datasets.py` — helper creating small toy datasets used by setter tests.
	- `toy_random.feather`, `toy_target.feather` — example toy datasets.
	- `test_index_based_setter.py` — tests index-based splitting.
	- `test_stratified_setter.py` — tests stratified splitting.
	- `test_naive_setter.py` — tests the naive/random setter implementation.

- `llm_models/` — tests for model wrappers and batch inference utilities.
	- `test_batch_model_instruct.py` — checks HF-based batch instruct wrapper.
	- `test_vllm_batch_model.py` — checks vLLM batch wrapper behaviors.
	- `test_generic_trhinking_instruct.py` — misc tests for instruct-style wrapper logic.

- `linear_models/` — tests for simple linear-model utilities.
	- `test_lasso_regressor.py` — Lasso regression tests.
	- `test_linear_regressor.py` — linear regressor tests.

Notes

- Most pipeline tests create temporary directories and files (feather/hdf5/index json) and remove them during teardown — they should not modify repository files.
- Some tests rely on a `DATAMODELS_TEST_MODEL` environment variable pointing to a local/test model for HF/vLLM wrappers; set it in environments where models are available.

