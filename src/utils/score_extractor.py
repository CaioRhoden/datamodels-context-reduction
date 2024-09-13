import torch
import transformers
from typing import Tuple


class ScoreExtractor:

    def __init__(
                    self, model:  transformers.AutoModelForCausalLM, 
                    tokenizer:    transformers.AutoTokenizer, 
                    device:       torch.device | str = "cpu", 
                    max_legth:    int = 1024,
                ):
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_legth = max_legth

    
    """
    	Extracts the log probability of a target output given an input context.

    	Parameters:
    	input_context (str): The input context to use for prediction.
    	target_output (str): The target output to calculate the log probability for.

    	Returns:
    	Tuple[float, torch.Tensor, torch.Tensor]: A tuple containing the total log probability, 
    	the logits output, and the log probabilities output.
    """
    def extract(
                    self, 
                    input_context: str, 
                    target_output: str,
                ) -> Tuple[float, torch.Tensor, torch.Tensor]:
    	

        input_tokens = self.tokenizer.encode(input_context, return_tensors="pt").to(self.device)
        target_tokens = self.tokenizer.encode(target_output, return_tensors="pt")[0].to(self.device)
        log_sum = 0


        logits_output = torch.tensor([])
        log_probs_output = torch.tensor([])


        for i in range(len(target_tokens)):
            # Predict with the given model
            with torch.no_grad():
                outputs = self.model.generate(input_tokens, max_new_tokens=1, output_logits=True, return_dict_in_generate=True, pad_token_id=50256)
                logit_predictions = outputs.logits[0]
        
            # Extract the log probability of the output token
            log_probs = torch.nn.functional.log_softmax(logit_predictions, dim=-1)


            out_token_logit = logit_predictions[0, target_tokens[i]]
            out_token_log_prob = log_probs[0, target_tokens[i]]


            logits_output = torch.cat([logits_output, out_token_logit.reshape(1,1)])
            log_probs_output = torch.cat([log_probs_output, out_token_log_prob.reshape(1,1)])



            log_sum += out_token_log_prob

            # Incrementally add an output token to the current sequence
            input_tokens = torch.cat([input_tokens, target_tokens[i].reshape(1,1)], dim=1)

        return log_sum, logits_output, log_probs_output
    

    def __call__(
                    self, 
                    input_context: str, 
                    target_output: str,
                ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        
        return self.extract(input_context, target_output)
    	
        





        
        