# Evaluation

`python -m scripts.benchmark_retrieval` computes actual Recall@5 and Precision@5 for BM25, dense, and hybrid retrieval. `python -m scripts.run_eval` measures relevance, groundedness, citation correctness, failures, and latency, then compares the result with `data/evaluation/baseline.json`. No benchmark values are claimed until the commands are run against an ingested corpus.