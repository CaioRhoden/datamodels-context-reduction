from ..dataloader import BaseDataloader
from ..retriever import BaseRetriever
from ..models import BaseModel
from ..vector_stores import BaseVectorStore




class BasePipeline:

    def __init__(
                    self,
                    dataloader: BaseDataloader,
                    vector_store: BaseVectorStore,
                    retriever: BaseRetriever,
                    model: BaseModel,
                    device: str,
                ) -> None:
        

        self.dataloader = dataloader,
        self.retriever = retriever,
        self.vector_store = vector_store,
        self.model = model




    
    def run(input: str) -> str:

        return "AHoy"