import pickle
from rank_bm25 import BM25Okapi
from src.config import BM25_INDEX_PATH


class BM25Index:
    """
    Builds and searches a BM25 index.
    """

    def __init__(self):
        self.index = None
        self.documents = []


    def build(self, chunks: list):
        self.documents = chunks
        corpus = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.index = BM25Okapi(corpus)
        self.save()


    def search(self, query: str, top_k: int = 20, company: str | None = None):
        tokens = self._tokenize(query)
        scores = self.index.get_scores(tokens)
        ranked = sorted(zip(scores, self.documents), key=lambda x: x[0], reverse=True)
        results = []

        target_company = (company or "").strip().lower()

        for score, chunk in ranked:
            if len(results) >= top_k:
                break

            chunk_company = chunk.get("metadata", {}).get("company", "").strip().lower()
            if target_company and target_company != "none" and chunk_company != target_company:
                continue

            results.append({**chunk, "bm25_score": float(score)})

        return results



    def save(self):
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"index": self.index, "documents": self.documents}, f)


    def load(self):
        with open(BM25_INDEX_PATH, "rb") as f:

            data = pickle.load(f)

        self.index = data["index"]
        self.documents = data["documents"]


    def _tokenize(self, text: str):
        import re
        return re.findall(r'\b\w+\b', text.lower())

