from dmcr.models import GenericVLLMBatch
import os
from pathlib import Path
import json


class TestBatchVLLMBatch:
    @classmethod
    def setup_class(cls):
        path = Path(__file__).parent.parent.parent
        cls.model = GenericVLLMBatch(f"{path}/{os.environ['DATAMODELS_TEST_MODEL']}")

    def test_run(self):
        prompts = ["Hello, how are you?" for i in range(10)]
        instruction = "You are a helpful assistant. Answer ''Hello, how are you?'' in a friendly manner."
        config_params = {}
        response = self.model.run(prompts, instruction, config_params)
        json.dump(response, open("response.json", "w"))
        

        assert isinstance(response, list)
        assert type(response[0]) is list
        assert len(response) == len(prompts)
        assert type(response[0][0]) is dict
        assert "generated_text" in response[0][0].keys()


