from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim, semantic_search


class Embeddings:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        self.embeddings = self.model.encode(
            sentences, convert_to_tensor=True, show_progress_bar=True
        )

    def search(self, query, top_k=3):
        query_embedding = self.model.encode(
            query, convert_to_tensor=True, show_progress_bar=False
        )
        hits = semantic_search(query_embedding, self.embeddings, top_k=top_k)
        return hits[0]


class ReRank:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, queries):
        return self.model.predict(queries)
