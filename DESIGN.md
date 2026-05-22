# Bastion-Anchor Design Document

## 1. Introduction

**Bastion-Anchor** (Module E) is the embedding security layer of the Bastion RAG Security Governance Framework. It provides protection against embedding-based attacks and bias issues in vector representations, ensuring that the retrieval-augmented generation pipeline remains secure, private, and fair.

### 1.1 Goals
- **Privacy:** Protect embeddings from inversion attacks using noise injection.
- **Fairness:** Detect and measure systematic biases in embedding models.
- **Reliability:** Monitor retrieval quality (recall) to ensure security doesn't break the system.
- **Auditability:** Report security events and metrics to Module D (Tracker).
- **Verification:** (Phase 2) Verify LLM responses for bias, grounding, and semantic drift.

For implementation details and algorithms, see the [Technical Guide](docs/TECHNICAL_GUIDE.md).

---

## 2. Architecture

Anchor is a Go-based microservice that sits between the search engine (Navigator) and the LLM. It exposes gRPC and REST APIs and includes a CLI for manual operations.

### 2.1 Component Diagram

```
┌──────────────────────────────────────────────────────┐
│                Anchor Service                         │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  gRPC API   │ │  REST API   │ │     CLI     │   │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘   │
│         └────────────────┴────────────────┘         │
│                          │                           │
│                          ▼                           │
│      ┌──────────────────────────────────┐           │
│      │       Request Handlers           │           │
│      └──────────────────┬───────────────┘           │
│                         │                           │
│      ┌──────────────────┼──────────────────┐        │
│      ▼                  ▼                  ▼        │
│  ┌─────────┐    ┌──────────────┐  ┌──────────────┐ │
│  │  Noise  │    │     Bias     │  │   Quality    │ │
│  │Injector │    │   Analyzer   │  │   Monitor    │ │
│  └─────────┘    └──────────────┘  └──────────────┘ │
│          │               │                 │         │
│          └───────────────┼─────────────────┘         │
│                          ▼                           │
│              ┌───────────────────────┐               │
│              │   Verifier (Phase 2)  │               │
│              └───────────┬───────────┘               │
│                          │                           │
│      ┌───────────────────┴─────────────────┐         │
│      ▼                                     ▼         │
│  ┌──────────────┐                  ┌──────────────┐  │
│  │Config Manager│                  │Event Publisher│  │
│  └──────────────┘                  └──────┬───────┘  │
│                                           │          │
└───────────────────────────────────────────┼──────────┘
                                            │
                                            ▼
                                      ┌──────────┐
                                      │  NATS    │ → To Tracker
                                      └──────────┘
```

---

## 3. Core Components

### 3.1 Noise Injector (`internal/noise`)
Applies random perturbation to embeddings to mitigate inversion attacks (reconstructing original text from vectors).
- **Algorithms:** Gaussian (default) and Laplacian noise.
- **Norm Preservation:** L2-normalization after noise injection to maintain compatibility with cosine similarity search.
- **Determinism:** Optional seeded RNG for reproducible testing.
- **Theoretical Model:** $v_{secured} = \text{normalize}(v_{original} + \mathcal{N}(0, \sigma^2))$.

### 3.2 Bias Analyzer (`internal/bias`)
Measures systematic skew in embedding representations using a simplified **Word Embedding Association Test (WEAT)**.
- **Methodology:** Compares cosine similarity between target words (e.g., "doctor") and attribute pairs (e.g., "male"/"female").
- **Built-in Sets:** Gender, ethnicity, and age.
- **Severity Scaling:** Categorizes bias as Low, Medium, or High based on configurable thresholds.

### 3.3 Quality Monitor (`internal/quality`)
Tracks the impact of noise injection on retrieval performance.
- **Recall@K:** Measures the retention of relevant results when noise is applied.
- **Similarity Preservation:** Monitors the cosine similarity between original and secured embeddings.
- **Alerting:** Notifies if recall drop exceeds defined thresholds (e.g., > 5%).

### 3.4 Verifier (`internal/verify`) - Phase 2
Verifies the final LLM response produced by the RAG pipeline.
- **Response Bias:** Detects bias in the generated text.
- **Drift Analysis:** Compares Phase 1 (input) bias with Phase 2 (output) bias to detect "bias amplification".
- **Semantic Drift:** Uses Jaccard similarity/embedding overlap to check if the response drifted from the query or context.
- **Grounding & Coherence:** Basic metrics for response quality.

---

## 4. Operational Modes

### 4.1 Indexing Time (Write Path)
- **Goal:** Protect stored data.
- **Flow:** `Document -> Embedding -> Anchor (Noise) -> Vector DB`.
- Noise level is usually higher for indexing to ensure long-term data protection.

### 4.2 Search Time (Read Path)
- **Goal:** Protect user queries.
- **Flow:** `Query -> Embedding -> Anchor (Noise) -> Search Engine`.
- Noise level is usually lower to maximize search accuracy.

### 4.3 Phase 2 Verification (Final Path)
- **Goal:** Ensure output safety.
- **Flow:** `LLM Response -> Anchor (Verify) -> User/Application`.

---

## 5. API Design

### 5.1 gRPC Interface (`proto/anchor/v1/anchor.proto`)
Standard interface for high-performance internal communication within the Bastion framework.
- `SecureEmbedding`: Single vector noise injection.
- `SecureBatch`: Batch vector noise injection.
- `MeasureBias`: WEAT analysis.
- `MeasureQuality`: Recall impact estimation.
- `VerifyResponse`: Phase 2 LLM output verification.

### 5.2 REST API
Exposed on `:8080` for external testing and integration.
- `POST /v1/anchor/secure`: Apply noise.
- `POST /v1/anchor/measure-bias`: Run bias analysis.
- `POST /v1/anchor/verify`: Run Phase 2 verification.
- `GET /v1/health`: Liveness and readiness checks.
- `GET /v1/metrics`: Prometheus metrics.

---

## 6. CLI (`cmd/anchor-cli`)

The CLI provides a manual interface for testing, demos, and standalone operations.
- `anchor-cli secure`: Inject noise into a vector.
- `anchor-cli measure-bias`: Run bias tests on a model.
- `anchor-cli test-quality`: Benchmark noise impact on recall.
- `anchor-cli server`: Start the Anchor microservice.
- `anchor-cli interactive`: REPL for exploring embedding security.

---

## 7. Integration

### 7.1 Navigator (Module C)
Navigator calls Anchor's gRPC service during its search and indexing flows to secure embeddings before they are stored or queried.

### 7.2 Tracker (Module D)
Anchor publishes events to NATS for auditing and visualization by Tracker:
- `anchor.embedding_secured`
- `anchor.bias_detected`
- `anchor.quality_degraded`
- `anchor.response_verified`

---

## 8. Configuration

Anchor uses a YAML-based configuration (`config/config.yaml`) with support for:
- **Tenant Overrides:** Different noise levels per tenant.
- **Hot Reload:** Updating noise sigma without restart.
- **Thresholds:** Configurable bias and quality alert levels.

```yaml
noise:
  default_sigma: 0.01
  preserve_norm: true
  tenant_overrides:
    tenant-acme: 0.015
bias:
  enabled: true
  thresholds: { medium: 0.15, high: 0.25 }
```

---

## 9. Performance & Scaling

- **Latency:** < 5ms for single embedding, < 50ms for batches.
- **Memory:** < 1GB RAM (standard Go runtime).
- **Concurrency:** Uses Go's concurrency primitives for high-throughput batch processing.
- **Stateless:** The service is horizontally scalable; configuration is shared via a database or volume mounts.

---

## 10. Security & Privacy

- **mTLS:** Secured gRPC communication.
- **RBAC:** Access control for configuration updates.
- **Entropy:** Uses `crypto/rand` for high-quality noise generation in production.
