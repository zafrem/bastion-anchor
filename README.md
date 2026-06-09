# Bastion-Anchor (Module E)

## Overview
**Bastion-Anchor** protects the embedding layer and ensures output integrity in the RAG pipeline. It ensures that numerical representations (vectors) are resistant to privacy attacks, free from systematic bias, and that LLM responses remain grounded and fair.

## Symmetric Security Modes
Anchor applies security at both ends of the embedding lifecycle:

### Phase 1: Input Security (Anchor-IN)
When documents are being indexed or queries are being searched:
- **Noise Injection:** Gaussian or Laplacian noise is added to embeddings.
- **Privacy Protection:** Prevents reconstruction of original text from vector representations.
- **Bias Measurement:** Uses WEAT (Word Embedding Association Test) to detect model bias.

### Phase 2: Output Security (Anchor-OUT)
When the LLM generates a response:
- **Response Verification:** Analyzes response text for bias and stereotypes.
- **Semantic Drift:** Detects if the response has drifted from the query or source context.
- **Grounding & Quality:** Verifies the response is grounded in the retrieved documents.

## Key Features
- **Configurable Noise Injection:** Supports Gaussian and Laplacian strategies with per-tenant overrides.
- **Norm Preservation:** Maintains vector unit length for compatibility with cosine similarity.
- **Bidirectional Bias Analysis:** Measures bias in both input embeddings and output response text.
- **Quality Monitoring:** Tracks recall@K preservation and end-to-end response quality.

## Documentation
- [Design Document](DESIGN.md)
- [Main SRS (Input Phase)](docs/bastion_anchor_srs_v1.0_en.md)
- [Output SRS (Output Phase)](docs/bastion_anchor_output_srs_v1.0_en.md)
- [Technical Guide](docs/TECHNICAL_GUIDE.md)

**Technical deep-dives (code-based):**
- [Embedding Noise Injection](docs/embedding-noise-injection.md)
- [Embedding Model Bias Verification](docs/embedding-bias-verification.md)

## Getting Started

Anchor is a **Python / FastAPI** service.

### Prerequisites
- Python 3.11+
- (Optional) NATS for event streaming — Anchor runs without it (events are disabled and logged as a warning)

### Install
```bash
make install          # = pip install -e .
```
Using a virtual environment is recommended (`python -m venv .venv && . .venv/bin/activate`).

### Run
```bash
make run              # = python -m anchor.main --config config/config.yaml
```
Anchor starts the REST API on **:8083** and gRPC on **:9093**.

### Development
- `make test`   — pytest suite
- `make lint`   — ruff check
- `make format` — ruff format

## API Interfaces
- **REST API:** `:8083` — `/v1/anchor/secure`, `/v1/anchor/verify`, `/v1/anchor/bias`, `/v1/anchor/quality`, `/v1/anchor/drift`
- **gRPC:** `:9093`
- **Health:** `GET /v1/health`
- **Metrics:** Prometheus at `GET /v1/metrics`

## Integration
Anchor follows the Bastion-RAG standard gRPC and REST patterns, typically sitting between Module C (Navigator) and the LLM, and feeding metrics to Module D (Tracker).
