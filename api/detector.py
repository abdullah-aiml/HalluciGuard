import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from api.retriever import ChunkRetriever

TEMPERATURE = 1.5
CONFIDENCE_THRESHOLD = 0.60
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def sliding_window_chunker(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits a large text into overlapping word-level chunks."""
    words = text.split()
    chunks = []

    if not words:
        return chunks

    step = chunk_size - overlap
    if step <= 0:
        step = 1

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))

        if i + chunk_size >= len(words):
            break

    return chunks


def split_into_claims(text: str) -> list[str]:
    """Breaks LLM output into individual sentences so each factual
    claim gets scored independently (avoids filler diluting scores)."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    valid_claims = []
    for s in raw_sentences:
        clean = s.strip()
        if len(clean.split()) >= 3:
            valid_claims.append(clean)

    if not valid_claims and text.strip():
        valid_claims = [text.strip()]

    return valid_claims


def normalize_scores(contradiction: float, entailment: float, neutral: float) -> tuple[float, float, float]:
    """Makes sure the three scores always add up to exactly 100%."""
    total = contradiction + entailment + neutral
    if total == 0:
        return (0.0, 0.0, 100.0)

    c = round((contradiction / total) * 100.0, 2)
    e = round((entailment / total) * 100.0, 2)
    n = round(100.0 - c - e, 2)
    return (c, e, n)


class HallucinationDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "cross-encoder/nli-deberta-v3-base"

        print(f"Initializing Detector on {self.device.type.upper()}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        print("Detector Ready!")

        # Stage 1 retriever — lightweight bi-encoder for pre-filtering chunks
        self.retriever = ChunkRetriever()

    def _infer_chunk(self, chunk: str, claim: str) -> dict:
        """Stage 2: runs the heavy cross-encoder on a single (chunk, claim) pair."""
        inputs = self.tokenizer(
            chunk, claim,
            return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            scaled_logits = outputs.logits / TEMPERATURE
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)

        c_raw = probs[0][0].item()
        e_raw = probs[0][1].item()
        n_raw = probs[0][2].item()

        # if the model isn't confident about anything, default to neutral
        max_score = max(c_raw, e_raw, n_raw)
        if max_score < CONFIDENCE_THRESHOLD:
            c_raw, e_raw, n_raw = 0.0, 0.0, 1.0

        return {
            "contradiction": c_raw,
            "entailment": e_raw,
            "neutral": n_raw,
            "spans": []  # placeholder for Captum attributions
        }

    def analyze(self, context: str, llm_response: str) -> dict:
        """Two-stage pipeline:
        1) Chunk the document → retrieve top-5 relevant chunks (bi-encoder)
        2) Score each claim against those top chunks (cross-encoder)
        3) Aggregate with priority resolution
        """
        all_chunks = sliding_window_chunker(context)
        if not all_chunks:
            all_chunks = [""]

        # Stage 1: narrow down to the most relevant chunks
        relevant_chunks = self.retriever.get_top_chunks(llm_response, all_chunks)

        claims = split_into_claims(llm_response)
        sentence_scores = []

        for claim in claims:
            # Stage 2: cross-encoder only runs on the pre-filtered chunks
            chunk_results = [self._infer_chunk(chunk, claim) for chunk in relevant_chunks]

            s_max_e = max(r["entailment"] for r in chunk_results)
            s_max_c = max(r["contradiction"] for r in chunk_results)
            s_max_n = max(r["neutral"] for r in chunk_results)

            # priority resolution — if the fact exists somewhere, entailment wins
            if s_max_e >= CONFIDENCE_THRESHOLD and s_max_e >= s_max_c:
                final_s_e = s_max_e
                final_s_c = s_max_c * 0.25
                final_s_n = max(0.0, 1.0 - final_s_e - final_s_c)
                winning_spans = max(chunk_results, key=lambda x: x["entailment"])["spans"]
            elif s_max_c >= CONFIDENCE_THRESHOLD and s_max_c > s_max_e:
                final_s_c = s_max_c
                final_s_e = s_max_e * 0.25
                final_s_n = max(0.0, 1.0 - final_s_c - final_s_e)
                winning_spans = max(chunk_results, key=lambda x: x["contradiction"])["spans"]
            else:
                final_s_c = s_max_c
                final_s_e = s_max_e
                final_s_n = s_max_n
                winning_spans = []

            sentence_scores.append({
                "c": final_s_c,
                "e": final_s_e,
                "n": final_s_n,
                "spans": winning_spans
            })

        # document-level aggregation
        # contradiction uses max (one-strike rule)
        doc_c = max(s["c"] for s in sentence_scores)
        # entailment and neutral use average across claims
        doc_e = sum(s["e"] for s in sentence_scores) / len(sentence_scores)
        doc_n = sum(s["n"] for s in sentence_scores) / len(sentence_scores)

        doc_c = max(doc_c, 0.0)
        doc_e = max(doc_e, 0.0)
        doc_n = max(doc_n, 0.0)

        c_pct, e_pct, n_pct = normalize_scores(doc_c, doc_e, doc_n)

        # grab attribution spans from the highest-severity claim
        if doc_c > doc_e:
            best_spans = max(sentence_scores, key=lambda x: x["c"])["spans"]
        else:
            best_spans = max(sentence_scores, key=lambda x: x["e"])["spans"]

        is_hallucination = (c_pct > e_pct) and (doc_c >= CONFIDENCE_THRESHOLD)

        return {
            "contradiction_score": c_pct,
            "entailment_score": e_pct,
            "neutral_score": n_pct,
            "is_hallucination": is_hallucination,
            "attribution_spans": best_spans
        }