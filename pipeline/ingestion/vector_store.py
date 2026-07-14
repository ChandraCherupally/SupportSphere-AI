from pipeline.ingestion.pinecone_client import PineconeClient


class VectorStore:
    """
    Handles vector operations on Pinecone.
    """

    def __init__(self):

        self.client = PineconeClient()

        self.index = self.client.get_index()


    def upsert_vectors(self, vectors: list):

        self.index.upsert(vectors=vectors)


    def fetch_vectors(self, ids: list):

        return self.index.fetch(ids=ids)


    def delete_vectors(self, ids: list):

        self.index.delete(ids=ids)


    def delete_all(self):

        self.index.delete(delete_all=True)


    def describe_index(self):
        
        return self.index.describe_index_stats()