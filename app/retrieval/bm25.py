import re
from collections import Counter

from app.schemas.models import Evidence, SourceMetadata


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """Small in-memory BM25 index over structured evidence chunks."""

    def __init__(self, documents: list[Evidence] | None = None) -> None:
        self.documents: list[Evidence] = []
        self._term_frequencies: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._average_length = 0.0
        if documents:
            self.index(documents)

    def index(self, documents: list[Evidence]) -> None:
        self.documents = list(documents)
        tokenized = [_tokens(item.text) for item in self.documents]
        self._term_frequencies = [Counter(tokens) for tokens in tokenized]
        self._average_length = sum(map(len, tokenized)) / len(tokenized) if tokenized else 0.0
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        size = len(tokenized)
        self._idf = {
            term: __import__("math").log(1 + (size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if not query or top_k <= 0 or not self.documents:
            return []
        query_terms = _tokens(query)
        if not query_terms:
            return []
        scores: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self._term_frequencies):
            length = sum(frequencies.values())
            score = 0.0
            for term in query_terms:
                if term not in frequencies:
                    continue
                numerator = frequencies[term] * 2.2
                denominator = frequencies[term] + 1.2 * (0.25 + 0.75 * length / (self._average_length or 1))
                score += self._idf.get(term, 0.0) * numerator / denominator
            if score:
                scores.append((score, index))
        scores.sort(reverse=True)
        return [self.documents[index].model_copy(update={"score": score, "retrieval_method": "bm25"}) for score, index in scores[:top_k]]


def evidence_from_records(records: list[dict]) -> list[Evidence]:
    return [Evidence(text=record["text"], metadata=SourceMetadata(**record["metadata"])) for record in records]