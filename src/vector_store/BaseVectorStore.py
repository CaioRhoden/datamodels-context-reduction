class BaseVectorStore():

    def __init__(self) -> None:
        pass

    def add_documents(self, documents: list) -> None:
        pass

    
    def chunk_data(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    