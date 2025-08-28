from dmcr.models import GenericInstructModelHF

class TestGenericThinkingInstructModelHF:
    @classmethod
    def setup_class(cls):
        cls.model = GenericInstructModelHF("Qwen/Qwen3-0.6B", thinking=True)

    def test_run(self):
        prompt = "Hello, how are you?"
        instruction = "You are a helpful assistant. Answer ''Hello, how are you?'' in a friendly manner."
        config_params = {}
        response = self.model.run(prompt, instruction, config_params)

        assert type(response) is list
        assert type(response[0]) is dict
        assert "generated_text" in response[0].keys()
        assert response[0]["generated_text"].startswith("<think>")


