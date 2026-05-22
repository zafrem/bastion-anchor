# Bastion-Anchor Technical Guide

## 1. System Overview

Bastion-Anchor is a bidirectional security module for RAG pipelines. It addresses three core risks:
1. **Privacy Leakage:** Reconstruction of original text from embeddings.
2. **Systemic Bias:** Propagating bias from models to search results and responses.
3. **Semantic Drift:** Responses becoming ungrounded or irrelevant to the context.

## 2. Core Algorithms

### 2.1 Noise Injection (Phase 1)
To protect against inversion attacks, Anchor injects random noise into embeddings:
- **Gaussian Noise:** $v' = \text{normalize}(v + \mathcal{N}(0, \sigma^2))$
- **Laplacian Noise:** $v' = \text{normalize}(v + \text{Laplace}(0, b))$
- **Normalization:** L2-normalization is performed after noise injection to ensure $\text{dist}_{cos}(a, b)$ remains stable.

### 2.2 Bias Measurement (WEAT)
Anchor implements a simplified Word Embedding Association Test:
- **Methodology:** Measures the difference in mean cosine similarity between target words (e.g., "doctor") and two sets of attribute words (e.g., "male" vs "female").
- **Metrics:** A bias score (0.0 - 1.0) is generated, with severity thresholds at 0.15 (Medium) and 0.25 (High).

### 2.3 Response Verification (Phase 2)
For LLM outputs, Anchor uses a multi-faceted approach:
- **Keyword & Pattern Matching:** Detects explicit stereotypes in response text.
- **Jaccard Similarity:** Used for lightweight semantic drift detection between Query, Context, and Response.
- **Quality Scoring:** A weighted score combining Grounding (source overlap), Coherence, and Relevance.

## 3. Project Structure

- `cmd/anchor-cli`: The CLI entry point for standalone testing and server mode.
- `internal/noise`: Implementation of Gaussian and Laplacian injection.
- `internal/bias`: WEAT analyzer logic and built-in test sets.
- `internal/verify`: Phase 2 verification logic (Anchor-OUT).
- `internal/quality`: Retrieval quality (recall@K) monitoring.
- `proto/anchor/v1`: gRPC service definitions.

## 4. Operational Modes

### Server Mode
Run as a microservice within the Bastion framework:
```bash
$ anchor-cli server
```

### Standalone Mode
Test embeddings or responses without full infrastructure:
```bash
$ anchor-cli secure --embedding-file vec.json
$ anchor-cli verify --response "LLM response text" --query "User query"
```

## 5. Configuration

Anchor is configured via `config/config.yaml`. Key parameters:
- `noise.default_sigma`: Intensity of noise (recommended: 0.01).
- `bias.thresholds`: Alert levels for bias detection.
- `tracker.enabled`: Whether to publish events to NATS.
