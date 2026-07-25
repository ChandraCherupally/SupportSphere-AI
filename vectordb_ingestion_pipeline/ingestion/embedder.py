from google import genai
import src.config as config
from google.genai import types

class GeminiEmbedder:
    """
    Generates Gemini embeddings for chunks.
    """

    def __init__(self):
        pass


    def embed_chunks(self, chunks: list):
        embedded_chunks = []
        
        for start in range(0, len(chunks), config.EMBED_BATCH_SIZE):
            batch = chunks[start:start + config.EMBED_BATCH_SIZE]
            embeddings = self._embed_batch(batch)
            embedded_chunks.extend(embeddings)

        return embedded_chunks


    def _embed_batch(self, chunks: list):

        texts = [chunk["text"] for chunk in chunks]

        texts = [(
                    f"Title: {chunk['metadata']['title']}\n"
                    f"Product Area: {chunk['metadata']['product_area']}\n"
                    f"Section: {chunk['metadata']['section']}\n\n"
                    f"{chunk['text']}"
                    ) for chunk in chunks]


        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        response = client.models.embed_content(model=config.EMBEDDING_MODEL, contents=texts,
                                                    config=types.EmbedContentConfig(output_dimensionality=config.EMBEDDING_DIMENSION))

        embedded_chunks = []

        for chunk, embedding in zip(chunks, response.embeddings):

            embedded_chunks.append({"id": chunk["id"], 
                                    "values": embedding.values, 
                                    "metadata": {**chunk["metadata"],
                                                 "text": chunk["text"]}
                                    })

        return embedded_chunks
    
    



