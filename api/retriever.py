import numpy as np
from sentence_transformers import SentenceTransformer

TOP_K = 5

class ChunkRetriever:
    """Stage 1 Bi-Encoder: quickly narrows down hundreds of chunks
    to the few that are actually semantically relevant to the LLM output."""

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("Retriever (MiniLM-L6-v2) loaded.")

    def get_top_chunks(self, llm_output: str, chunks: list[str], top_k: int = TOP_K) -> list[str]:
        """Embeds everything, ranks by cosine similarity, returns the top_k chunks."""
        if len(chunks) <= top_k:
            return chunks

        query_embedding = self.model.encode(llm_output, normalize_embeddings=True)
        chunk_embeddings = self.model.encode(chunks, normalize_embeddings=True, batch_size=32)

        # cosine sim is just dot product when vectors are already L2-normalized
        similarities = np.dot(chunk_embeddings, query_embedding)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [chunks[i] for i in top_indices]
