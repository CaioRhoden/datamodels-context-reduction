import pytest
import polars as pl
import numpy as np
import h5py
from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels.pipeline import BatchLLMPreCollectionsPipeline
from dmcr.datamodels.pipeline.DatamodelsPipelineData import DatamodelsPreCollectionsData
from dmcr.datamodels import DatamodelIndexBasedConfig
from dmcr.models import GenericInstructBatchHF
from dmcr.utils.test_utils import clean_temp_folders
import tempfile
import os
import shutil
import json
from langchain.prompts import PromptTemplate



class TestIndexBasedNQPipelineBatchPreCollection:

    @classmethod
    def setup_class(cls):
        def fill_prompt_template(idx_row: int, idx_test: int, rag_indexes: dict, datamodels: DatamodelsIndexBasedNQPipeline) -> str:
            template = """
                Documents:
                {context}

                Question: {input}\nAnswer: 
            """

            context = ""
            count = 0
            for collection_idx in datamodels.train_collections_idx[idx_row]:
                
                idx = rag_indexes[str(idx_test)][collection_idx]
                title = datamodels.train_set[idx]["title"].to_numpy().flatten()[0]
                text = datamodels.train_set[idx]["text"].to_numpy().flatten()[0]
                context += f"Document[{count}](Title: {title}){text}\n\n"
                count += 1

            print(f"test_set: {datamodels.test_set}")
            input = datamodels.test_set[idx_test]["question"].to_numpy().flatten()[0]

            prompt = PromptTemplate.from_template(template).format(context=context, input=input)

            return prompt

        ## CREATE TEMP DATA FOR PIPELINE
        clean_temp_folders()
        tmp_path = tempfile.mkdtemp(dir=os.getcwd())
        cls.datamodels_path = tmp_path

        train = {
            "text": ["train_text_1", "train_text_2", "train_text_3", "train_text_4", "train_text_5", "train_text_6", "train_text_7", "train_text_8", "train_text_9", "train_text_10"],
            "title": ["train_title_1", "train_title_2", "train_title_3", "train_title_4", "train_title_5", "train_title_6", "train_title_7", "train_title_8", "train_title_9", "train_title_10"],
        }

        test = {
            "question": ["test_question_1"],
            "answer": [["test_answer_1"]],
        }
        
        ## Create dfs

        train_df = pl.DataFrame(train).with_row_index("idx")
        test_df = pl.DataFrame(test).with_row_index("idx")

        ## Save temp data

        train_df.write_ipc(f"{tmp_path}/train_set.feather")
        test_df.write_ipc(f"{tmp_path}/test_set.feather")

        ## Create combinations for datamodel training
        with h5py.File(f"{tmp_path}/train_collection.h5", 'w') as hf:
            combinations = np.array([[0,1,2,3], [0,2,3,4]])
            hf.create_dataset('train_collection', data=combinations)

        with h5py.File(f"{tmp_path}/test_collection.h5", 'w') as hf:
            combinations = np.array([[1,2,3,4]])
            hf.create_dataset('test_collection', data=combinations)

        indexes = {
            "0": [0 ,1, 2, 3, 4],
            "1": [5, 6, 7, 8, 9],
        }

        with open(f"{tmp_path}/indexes.json", "w") as f:
            json.dump(indexes, f)

        

        ## INSTANTIATE PIPELINE
        model_configs = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 10
        }

        config = DatamodelIndexBasedConfig(
            k = 2,
            num_models= 1,
            datamodels_path = f"{tmp_path}",
            train_set_path= f"{tmp_path}/train_set.feather",
            test_set_path= f"{tmp_path}/test_set.feather",
        )

        pipe =DatamodelsIndexBasedNQPipeline(config)

        ## INSTANTIATE PRE-COLLECTION PIPELINE DATACLASS
        datamodels_data = DatamodelsPreCollectionsData(
            train_set=pipe.train_set,
            test_set=pipe.test_set,
            train_collections_idx=pipe.train_collections_idx,
            test_collections_idx=pipe.test_collections_idx,
            datamodels_path=pipe.datamodels_path,
        )
        print(tmp_path)
        pre_collection_pipeline = BatchLLMPreCollectionsPipeline(
            datamodels_data=datamodels_data,
            mode="train",
            instruction="Answer",
            model=GenericInstructBatchHF(os.environ["DATAMODELS_TEST_MODEL"], quantization=True),
            context_strategy=fill_prompt_template,
            rag_indexes_path=f"{tmp_path}/indexes.json",
            output_column="answer",
            batch_size=6,
            start_idx = 0,
            end_idx = -1,
            checkpoint = 1,
            log = False,
            log_config = None,
            model_configs=model_configs,
        )

        pipe.create_pre_collection(pre_collection_pipeline)
        pre_collection_pipeline = BatchLLMPreCollectionsPipeline(
            datamodels_data=datamodels_data,
            mode="test",
            instruction="Answer",
            model=GenericInstructBatchHF(os.environ["DATAMODELS_TEST_MODEL"], quantization=True),
            context_strategy=fill_prompt_template,
            rag_indexes_path=f"{tmp_path}/indexes.json",
            output_column="answer",
            batch_size=6,
            start_idx = 0,
            end_idx = -1,
            checkpoint = 50,
            log = False,
            log_config = None,
            model_configs=model_configs,
        )


        print("Creating pre-collections")
        pipe.create_pre_collection(pre_collection_pipeline)

        

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.datamodels_path)
        clean_temp_folders()


    def test_output_train_generated(self):
        assert os.path.exists(f"{self.datamodels_path}/pre_collections/train/pre_collection_1.feather")
        assert os.path.exists(f"{self.datamodels_path}/pre_collections/train/pre_collection_0.feather")
        
    def test_output_test_generated(self):
        assert os.path.exists(f"{self.datamodels_path}/pre_collections/test/pre_collection_0.feather")

    def test_output_length(self):
        df = pl.read_ipc(f"{self.datamodels_path}/pre_collections/train/pre_collection_1.feather")
        assert len(df) == 1
        df2 = pl.read_ipc(f"{self.datamodels_path}/pre_collections/train/pre_collection_0.feather")
        assert (len(df2)+len(df)) == 2

    def test_pre_collection_dtypes(self):
        df = pl.read_ipc(f"{self.datamodels_path}/pre_collections/train/pre_collection_1.feather")
        assert df["collection_idx"].dtype == pl.Int64
        assert df["test_idx"].dtype == pl.Int64
        assert df["input"].dtype == pl.Array
        assert df["predicted_output"].dtype == pl.String
        assert df["true_output"].dtype == pl.List(pl.String)

    


