import os
import src.config as src_config
from pinecone import Pinecone, ServerlessSpec

# pyrefly: ignore [missing-import]
from src.config import (
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
        api_key = os.getenv("PINECONE_API_KEY") or getattr(src_config, "PINECONE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable is missing or empty. "
                "Please verify PINECONE_API_KEY is set in your .env file."
            )
        if cls._instance is None:
            instance = super(PineconeClient, cls).__new__(cls)
            instance.pc = Pinecone(api_key=api_key)
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