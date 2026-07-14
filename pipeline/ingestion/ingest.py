from src.config import DATA_DIR
from pipeline.ingestion.document_processor import DocumentProcessor
from pipeline.ingestion.embedder import GeminiEmbedder
from pipeline.ingestion.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
import pickle

class IngestionPipeline:
    """
    Complete document ingestion pipeline.
    """

    def __init__(self):

        self.processor = DocumentProcessor()
        self.embedder = GeminiEmbedder()
        self.vector_store = VectorStore()


    def ingest(self, data_directory: str = DATA_DIR):
        print("Ingestion process started.....")
        
        chunks = self.processor.process_directory(data_directory)
        print("Chunking process completed.....")
        
        vectors = self.embedder.embed_chunks(chunks)
        print("Embedding process completed.....")
        
        with open("artifacts/vector_records.pkl", "wb") as f:
            pickle.dump(vectors, f)

        with open("artifacts/vector_records.pkl", "rb") as f:
            vectors = pickle.load(f)

        #self.vector_store.upsert_vectors(vectors)
        BATCH_SIZE = 100

        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            self.vector_store.upsert_vectors(vectors=batch)
            print(f"batch of {i+BATCH_SIZE} vectors are completed into pinecone vector db.....")


        print("Vector store process completed.....")

        bm25 = BM25Index()
        bm25.build(chunks)
        print("Bm25 Index process completed.....")
        
        return len(vectors)


def main():

    pipeline = IngestionPipeline()
    total_vectors = pipeline.ingest()
    print(f"\nSuccessfully indexed {total_vectors} vectors.")


if __name__ == "__main__":
    main()