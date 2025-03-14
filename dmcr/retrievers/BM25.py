from dmcr.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

class BM25(BaseRetriever):

    def __init__(self,
                 
                ) -> None:
        
        self.retriever = None

    def _create_retriver(self, docs: list ,k: int) -> None:
        kwargs = {"k": k}
        self.retriever = BM25Retriever.from_documents(docs, **kwargs)

    def retrieve(self, docs: list ,k: int, input: str) -> list:
        

        
        self._create_retriver(docs, k)
        return self.retriever.invoke(input)