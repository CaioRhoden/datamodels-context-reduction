from src.dataloader.BaseDataloader import BaseDataloader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma


# from ..retriever import BaseRetriever
# from ..models import BaseModel
# from ..vector_stores import BaseVectorStore




class BasePipeline():

    
    def __init__(
                    self,
                    dataloader: BaseDataloader,
                    # retriever: BaseRetriever,
                    # model: BaseModel,
                    device: str | None = "cpu",
                    vector_store: bool | None = False
                ) -> None:
        
        self.device = device
        self.dataloader = dataloader,
        self.documents = dataloader.get_data()


        self.embeddingModel = HuggingFaceBgeEmbeddings(model_name="dunzhang/stella_en_400M_v5", model_kwargs={"device": self.device, "trust_remote_code":True}, encode_kwargs = {"normalize_embeddings": True})
        
        if not vector_store:
            self.vector_store = Chroma.from_documents(self.documents, embedding=self.embeddingModel, persist_directory="../data/chroma")
        
        else:
            self.vector_store = Chroma(persist_directory="../data/chroma", embedding_function=self.embeddingModel)

        # self.retriever = retriever,
        # self.vector_store = vector_store,
        # self.model = model





    
    def run(self, input: str) -> str:


        return "AHoy"
    
    def evaluate(self):
        pass