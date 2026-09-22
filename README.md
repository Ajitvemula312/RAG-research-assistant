# AI Research Assistant for Evidence-Grounded QA

A multi-agent research assistant for answering technical research questions using a curated corpus of open-access research papers and web sources.

The system combines **agent orchestration, hybrid information retrieval, evidence-grounded generation, citation verification, and automated evaluation** into one end-to-end research workflow.

> **Project status:** Active research/engineering project. Benchmark and generation results are recorded only after running the actual evaluation pipeline.

## Overview

Research questions often require more than a single semantic search or an LLM response. Relevant information may be spread across multiple papers, terminology may vary between sources, and a fluent model can still produce claims that are not supported by the available evidence.

This project addresses that problem by combining:

- **LangGraph** for multi-agent workflow orchestration
- **LangChain** for LLM and retrieval integration
- **BM25** for lexical retrieval
- **Sentence Transformers + ChromaDB** for dense retrieval
- **Hybrid retrieval** to combine lexical and semantic evidence
- **Llama 3 via Ollama** for local generation
- **Citation-aware answer generation**
- **Verification** of generated claims against retrieved evidence
- **SQLite** for conversation/session state
- **MLflow** for experiment tracking
- **FastAPI** for the backend API
- **Docker** for reproducible execution
- **Pytest + Ruff + GitHub Actions** for testing and CI

The initial research corpus is focused primarily on **autonomous-vehicle perception, multimodal sensor fusion, adverse-weather robustness, uncertainty estimation, and vision-language/LLM-based autonomous driving**.

## Core Problem

Given a research question such as:

> What approaches have been proposed for uncertainty-aware multimodal sensor fusion under adverse weather?

The assistant should:

1. Understand the question.
2. Break it into useful research queries.
3. Search relevant paper and web sources.
4. Retrieve evidence using multiple retrieval methods.
5. Rank and combine the evidence.
6. Generate an answer grounded in the retrieved sources.
7. Attach valid citations.
8. Verify whether the answer is supported by the evidence.
9. Retry retrieval when the available evidence is insufficient.
10. Record the run for evaluation and experimentation.

The goal is not simply to generate a plausible answer. The goal is to make the **evidence retrieval and verification process explicit and measurable**.

## Architecture

```text
                         +-----------------------+
                         |      User / UI         |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         |       FastAPI          |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         |      LangGraph         |
                         |    Agent Workflow      |
                         +-----------+-----------+
                                     |
                  +------------------+------------------+
                  |                  |                  |
                  v                  v                  v
              Planner            Researcher         Verifier
                                     |
                                     v
                            +----------------+
                            | Hybrid Search  |
                            +-------+--------+
                                    |
                         +----------+----------+
                         |                     |
                         v                     v
                       BM25               Dense Search
                         |                     |
                         +----------+----------+
                                    |
                                    v
                                ChromaDB
                                    |
                                    v
                              Synthesizer
                               Llama 3
                                    |
                                    v
                            Evidence + Citations
```

## Agent Workflow

The core workflow is implemented using LangGraph.

```text
START
  |
  v
Planner
  |
  v
Researcher
  |
  v
Retriever
  |
  v
Synthesizer
  |
  v
Verifier
  |
  +---- Evidence insufficient ----> Researcher
  |
  +---- Verified -----------------> END
```

### Planner

Interprets the question and creates structured research queries. It determines the research intent and whether local papers, web sources, or both should be used.

### Researcher

Executes the planned searches and combines evidence from multiple sources.

### Retriever

Supports:

- BM25 lexical search
- Dense semantic search
- Hybrid BM25 + dense retrieval

### Synthesizer

Uses Llama 3 to generate an answer from retrieved evidence while preserving source attribution and avoiding unsupported claims.

### Verifier

Checks whether claims are supported, whether citations are valid, and whether another retrieval cycle is necessary.

The system does not expose or persist hidden chain-of-thought. Only structured application state is retained.

## Retrieval Pipeline

### Document Ingestion

```text
PDF / Web Source
      |
      v
Text Extraction
      |
      v
Cleaning
      |
      v
Chunking
      |
      v
Metadata
      |
      v
Sentence Transformers
      |
      v
ChromaDB
```

Each chunk preserves source metadata such as:

```text
document_id
chunk_id
title
authors
publication_year
source
URL
page_number
section
```

### Hybrid Retrieval

The system compares:

```text
BM25 only
Dense only
Hybrid BM25 + Dense
```

The hybrid retriever:

1. Executes BM25 retrieval.
2. Executes dense retrieval.
3. Normalizes scores.
4. Combines rankings using configurable weights.
5. Removes duplicate chunks.
6. Produces the final ranked evidence set.

Retrieval quality is measured experimentally rather than assumed.

## Evidence-Grounded Generation

The central rule is:

> **The model should answer from evidence available to the workflow, not from unsupported assumptions.**

A generated response contains:

- the answer
- supporting citations
- verification information
- source metadata
- model and retrieval version information

When sufficient evidence cannot be found, the assistant should explicitly state that the available corpus does not support a reliable answer.

## Citation System

Citations are generated from actual retrieved documents.

Each citation can contain:

```text
citation_id
document_id
title
page_number
section
URL
```

A citation cannot be generated for an arbitrary source that was not retrieved by the system.

## Memory and State

The system separates three forms of state.

### Workflow state

Maintained by LangGraph:

```text
session_id
user_question
research_plan
search_queries
retrieved_documents
retrieved_evidence
draft_answer
citations
verification_result
errors
model_version
prompt_version
retrieval_version
```

### Conversation memory

Stored using SQLite and linked to a `session_id`, allowing research conversations to continue without mixing conversational history into the document knowledge base.

### Knowledge memory

Research documents and embeddings are stored in ChromaDB.

## Evaluation Framework

Evaluation is treated as a first-class component.

The benchmark contains **200+ research questions** covering:

- direct retrieval
- multi-document reasoning
- paper + web research
- insufficient evidence
- citation validation
- conflicting evidence
- irrelevant retrieval
- unsupported claim detection

### Retrieval Metrics

- Recall@5
- Precision@5

### Generation Metrics

- Answer Relevance
- Groundedness
- Citation Correctness
- Failure Rate
- End-to-End Latency

All metrics are calculated from actual experiment runs. No benchmark value is hardcoded.

## Retrieval Experiment

The main retrieval experiment investigates whether combining lexical and semantic retrieval improves Top-5 retrieval quality.

```text
BM25
  vs
Dense
  vs
BM25 + Dense
```

The benchmark calculates the actual improvement from the selected baseline. Any ~18% improvement reported in project materials must come from a real run on the indexed corpus.

## Research Corpus

The initial corpus focuses on:

- multimodal autonomous-vehicle perception
- camera-LiDAR-radar fusion
- adverse weather
- uncertainty estimation
- sensor degradation
- robustness and domain generalization
- BEV perception
- transformer-based perception
- vision-language models
- LLM-based autonomous driving
- multimodal reasoning

The corpus is designed to support 25+ open-access papers and multiple public web sources.

## Example Research Questions

```text
What are the main approaches to uncertainty-aware sensor fusion?

Compare the methods used for multimodal perception under adverse weather.

Which papers investigate camera-LiDAR-radar fusion?

What are the limitations of fixed sensor fusion?

How is uncertainty estimated in multimodal autonomous-driving perception?

Which methods are designed to handle sensor degradation?

What role do vision-language models play in autonomous driving?

Compare the evidence for transformer-based sensor fusion methods.
```

## API

### Health

```http
GET /health
```

### Research

```http
POST /research
```

Example request:

```json
{
  "question": "What are the main approaches to uncertainty-aware sensor fusion?",
  "session_id": "demo-session"
}
```

Example response structure:

```json
{
  "answer": "...",
  "citations": [],
  "sources": [],
  "verification": {},
  "latency_ms": 0,
  "model_version": "llama3",
  "retrieval_version": "v1"
}
```

### Chat

```http
POST /chat
```

### Latest Evaluation

```http
GET /evaluation/latest
```

## Project Structure

```text
ai-research-assistant/
|
+-- app/
|   +-- main.py
|   +-- api/
|   +-- agents/
|   +-- retrieval/
|   +-- ingestion/
|   +-- llm/
|   +-- evaluation/
|   +-- config/
|   +-- schemas/
|
+-- data/
|   +-- papers/
|   +-- web/
|   +-- processed/
|   +-- evaluation/
|
+-- scripts/
|   +-- ingest_papers.py
|   +-- ingest_web.py
|   +-- run_eval.py
|   +-- benchmark_retrieval.py
|
+-- tests/
|   +-- unit/
|   +-- integration/
|
+-- deployment/
|   +-- Dockerfile
|   +-- docker-compose.yml
|
+-- docs/
+-- .github/
|   +-- workflows/
+-- requirements.txt
+-- .env.example
+-- README.md
+-- LICENSE
```

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd ai-research-assistant
```

Create the environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

## Llama 3 / Ollama

Install Ollama separately and start the service, then pull the configured Llama model:

```bash
ollama pull llama3
```

The model name should be configured through environment/configuration rather than hardcoded.

## Ingest Research Papers

Place PDF papers under:

```text
data/papers/
```

Then run:

```bash
python -m scripts.ingest_papers --path data/papers
```

## Ingest Web Sources

For a public web source:

```bash
python -m scripts.ingest_web <URL>
```

## Run the API

```bash
uvicorn app.main:app --reload
```

API:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## Run Retrieval Benchmark

```bash
python -m scripts.benchmark_retrieval
```

The benchmark should report measured Recall@5, Precision@5, and hybrid-retrieval improvement.

## Run Evaluation

```bash
python -m scripts.run_eval
```

The evaluation records retrieval quality, groundedness, citation correctness, relevance, failure rate, latency, and version metadata.

## MLflow

Start the MLflow UI when configured:

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

Tracked information includes:

- model version
- embedding version
- retrieval version
- BM25 weight
- dense weight
- top_k
- evaluation dataset version
- evaluation metrics
- latency
- failure rate

## Docker

Start the local stack:

```bash
docker compose -f deployment/docker-compose.yml up --build
```

The Docker setup is intended for reproducible local execution. Model weights remain external to the application image.

## Testing

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

Run both:

```bash
ruff check . && pytest
```

## CI/CD

The GitHub Actions workflow runs linting, tests, application checks and Docker builds. The evaluation workflow can additionally run the benchmark and fail on configured quality regressions.

Secrets must be provided through the CI environment and must never be committed to the repository.

## Failure Modes

The system explicitly handles:

- No relevant evidence
- Poor retrieval
- Conflicting sources
- Unsupported generated claims
- Missing citations
- Invalid citations
- Web retrieval failure
- Tool/retrieval timeout
- LLM timeout
- Prompt injection contained inside retrieved content
- Long conversation context
- Stale information
- Evaluation regression

Retrieved documents are treated as **data**, not trusted instructions.

## Responsible Use

This is a research and engineering system, not an authoritative scientific source. For important claims, users should inspect the cited original paper or source.

Errors can occur when evidence is sparse, sources conflict, metadata is incomplete, or the LLM misinterprets retrieved evidence.

## Current Research Context

The initial corpus supports an ongoing research direction on:

**Robust and Adaptive Multimodal Perception for Autonomous Vehicles Under Adverse Weather and Sensor Uncertainty**

The broader question is whether a perception system can estimate how reliable different sensing modalities are under changing environmental conditions and adapt the fusion process accordingly.

This connects the literature assistant to research areas including:

- adverse-weather perception
- sensor degradation
- uncertainty estimation
- adaptive sensor fusion
- robustness and generalization
- multimodal perception
- later communication-aware and edge extensions

## Future Work

- stronger reranking
- claim-level grounding evaluation
- citation completeness evaluation
- conflict detection
- versioned vector stores
- larger research corpora
- richer web-source handling
- evaluation dashboards
- improved agent planning
- stronger regression suites
- multimodal document support
- retrieval of research figures and tables

## Results Policy

Performance numbers such as retrieval improvement, groundedness, citation correctness, and latency must only be reported after running the benchmark on the actual indexed corpus.

No benchmark result is hardcoded or fabricated.

## Author

**Sai Ajit Vemula**  
MSc Software Engineering | Machine Learning Research Engineer

GitHub: https://github.com/Ajitvemula312
