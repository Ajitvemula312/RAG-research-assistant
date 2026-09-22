from collections.abc import Iterable


def retrieval_recall(expected: Iterable[str], retrieved: Iterable[str], k: int = 5) -> float:
    expected_set, retrieved_set = set(expected), set(list(retrieved)[:k])
    return len(expected_set & retrieved_set) / len(expected_set) if expected_set else 0.0


def retrieval_precision(expected: Iterable[str], retrieved: Iterable[str], k: int = 5) -> float:
    expected_set, top = set(expected), list(retrieved)[:k]
    return sum(item in expected_set for item in top) / len(top) if top else 0.0


def keyword_score(answer: str, criteria: Iterable[str]) -> float:
    terms = list(criteria)
    return sum(term.lower() in answer.lower() for term in terms) / len(terms) if terms else 0.0


def citation_correctness(citations: Iterable[str], retrieved: Iterable[str]) -> float:
    citations_set, retrieved_set = set(citations), set(retrieved)
    return len(citations_set & retrieved_set) / len(citations_set) if citations_set else 0.0