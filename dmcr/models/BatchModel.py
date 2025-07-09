from abc import ABC, abstractmethod




class BatchModel(ABC):

    @abstractmethod
    def __init__(
            self,
            path: str,
            quantization = False,
        ) -> None:

        pass

    @abstractmethod
    def run(self, prompts: list[str],  instruction: str, config_params: dict) -> list[str]:
        """
        Run the model on a list of prompts and instruction.

        Args:
            prompts (list[str]): The list of prompts to be used as input.
            instruction (str): The instruction to be used as a system message.
            config_params (dict): A dictionary containing additional configuration parameters.

        Returns:
            list[str]: The generated text for each prompt.
        """
        
        pass
    