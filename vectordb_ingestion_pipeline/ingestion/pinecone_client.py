from pinecone import Pinecone, ServerlessSpec

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSION,
    PINECONE_METRIC
)


class PineconeClient:
    """
    Creates and manages the Pinecone index as a Singleton.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            instance = super(PineconeClient, cls).__new__(cls)
            instance.pc = Pinecone(api_key=PINECONE_API_KEY)
            instance.index = instance._get_index()
            cls._instance = instance
        return cls._instance

    def _get_index(self):
        indexes = self.pc.list_indexes().names()
        
        if PINECONE_INDEX_NAME not in indexes:
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
            )

        return self.pc.Index(PINECONE_INDEX_NAME)

    def get_index(self):
        return self.index