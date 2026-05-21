# Bastion-Anchor (Module E)

## Overview
**Bastion-Anchor** protects the embedding layer of the RAG pipeline. It ensures that numerical representations (vectors) are resistant to privacy attacks and free from systematic bias.

## Symmetric Security Modes
Anchor applies security at both ends of the embedding lifecycle:

### Mode 1: Indexing Time (Write Path)
When documents are being added to the knowledge base:
- **Noise Injection:** Gaussian noise is added to document embeddings before storage.
- **Protection:** Prevents attackers from reconstructing original text from the vector database.

### Mode 2: Search Time (Read Path)
When a user query is being processed:
- **Query Perturbation:** Applies symmetric noise to the query embedding.
- **Integrity:** Maintains search compatibility while ensuring query privacy.

## Key Features
- **Gaussian Noise Injection:** Configurable $\sigma$ for privacy/quality trade-off.
- **Norm Preservation:** Re-normalizes vectors to L2 norm = 1.0.
- **Bias Measurement (WEAT):** Detects gender, ethnic, or age bias in the embedding model.
- **Quality Benchmarking:** Monitors recall@K degradation.

## Documentation
- [Design Document](DESIGN.md)
- [SRS Document](bastion_anchor_srs_v1.0_en.md)
