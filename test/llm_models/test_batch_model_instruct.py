from dmcr.models import GenericInstructBatchHF

class TestBatchInstructModelHF:
    @classmethod
    def setup_class(cls):
        cls.model = GenericInstructBatchHF("Qwen/Qwen3-0.6B", thinking=True)

    def test_run(self):
        prompts = ["Hello, how are you?" for i in range(10)]
        instruction = "You are a helpful assistant. Answer ''Hello, how are you?'' in a friendly manner."
        config_params = {}
        response = self.model.run(prompts, instruction, config_params)

        assert type(response) is list
        assert type(response[0]) is list
        assert len(response) == len(prompts)
        assert type(response[0][0]) is dict
        assert "generated_text" in response[0][0].keys()
        assert response[0][0]["generated_text"].startswith("<think>")


