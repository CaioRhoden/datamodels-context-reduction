from src.dataloader.BaseDataloader import BaseDataloader
# from src.vector_store import BaseVectorStore
from langchain.embeddings import HuggingFaceBgeEmbeddings
from src.llms import BaseLLM
import torch
from langchain.prompts import PromptTemplate
import pandas as pd



from src.retriever import BaseRetriever
# from ..models import BaseModel




class BasePipeline():

    
    def __init__(
                    self, 
                    retriever: BaseRetriever,
                    llm: BaseLLM,
                    dataloader: BaseDataloader,
                    device: str | None = "cpu",
                ) -> None:
        
        torch.set_default_device(device)
        self.device = device

        self.dataloader = dataloader,
        self.documents = dataloader.get_documents()

        self.retriever = retriever
        self.llm = llm
    





        

        # self.retriever = retriever,
        # self.vector_store = vector_store,
        # self.model = model

    def setup_context(self, documents: list) -> PromptTemplate:

        context = "\n".join([f"{doc.page_content} \n" for doc in documents])



        return context



    
    def run(self, input: str, k: int) -> str:

        
        retrieved_documents = self.retriever.retrieve(self.documents, k, input)
        context = self.setup_context(retrieved_documents)

        template = """
            Fill the expected Output
        
            Examples:
            {context}

            Input:
            {input}

            Output:
        """
        prompt = PromptTemplate.from_template(template).format(context=context, input=input)

        


        return self.llm.pipe().invoke(prompt)



        
    
    def run_tests(self, test_data: pd.DataFrame, checkpoints_step: int, checkpoint: int, k: int, run_tag: str) -> None:

        tasks, inputs, outputs, predicted = [], [], [], []

        for i in range(checkpoint, len(test_data)):
            tasks.append( test_data.loc[i]["task"])
            inputs.append( test_data.loc[i]["input"])
            outputs.append( test_data.loc[i]["output"])


            predicted.append(self.run(test_data.loc[i]["input"], k))

            if i % checkpoints_step == 0 and i > checkpoint:

                df = pd.DataFrame({"task": tasks, "input": inputs, "output": outputs, "predicted": predicted})
                df.to_pickle(f"../data/runs_id/{run_tag}/{i - checkpoints_step}_{i}.pickle")
                tasks, inputs, outputs, predicted = [], [], [], []






