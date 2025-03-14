from dmcr.vector_stores import BaseVectorStore
from langchain.vectorstores import Chroma


class ChromaVectorStore(BaseVectorStore):

    def __init__(
                    self,
                    embedding_model,
                    folder: str | None = "chroma",
                ) -> None:
        
        self.folder = folder
        self.embedding_model = embedding_model
        self.vector_store = Chroma(embedding_function=self.embedding_model, persist_directory=f"../data/{self.folder}")



    def add_documents(self, batch: list) -> None:
        
        return self.vector_store.add_documents(batch)
