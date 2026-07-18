from google import genai
import src.config as config
from pipeline.ingestion.pinecone_client import PineconeClient
from google.genai import types

class VectorSearch:
    """
    Performs semantic search using Pinecone.
    """

    def __init__(self):
        self.index = PineconeClient().get_index()


    def search(self, query: str, top_k: int = 20, filters: dict | None = None):
        # Prefer the dedicated embed key; fall back to the main API key so the
        # genai.Client never receives None (which triggers OAuth and a 401 error).
        embed_api_key = config.GOOGLE_API_KEY_EMBED or config.GOOGLE_API_KEY
        client = genai.Client(api_key=embed_api_key)
        embedding = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(output_dimensionality=config.EMBEDDING_DIMENSION)
        )

        query_vector = embedding.embeddings[0].values
        response = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True,filter=filters)
        results = []

        for match in response.matches:

            metadata = dict(match.metadata)
            text = metadata.pop("text", "")
            
            results.append({"id": match.id,
                            "text": text,
                            "metadata": metadata,
                            "vector_score": float(match.score)})

        return results