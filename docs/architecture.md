# Architecture

The API owns request/response concerns. The LangGraph-compatible workflow owns planning, retrieval, synthesis, and verification. BM25 and dense retrieval are independent modules combined by reciprocal score normalization in the hybrid retriever. Chroma stores embeddings and metadata; SQLite stores only conversation messages.