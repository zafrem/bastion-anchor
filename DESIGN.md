# Bastion-Anchor Design Document

## 1. Introduction
Anchor is Module E, providing security for the embedding space. It operates on a symmetric model, protecting both stored data and active queries.

## 2. Operating Modes

### 2.1 Write Path (Indexing)
```
[Document] -> [Embedder] -> [Anchor (Noise)] -> [Vector DB]
```
- Applied once per document.
- Permanent protection for stored vectors.

### 2.2 Read Path (Search)
```
[Query] -> [Embedder] -> [Anchor (Noise)] -> [Search Engine]
```
- Applied to every incoming query.
- Ensures the query itself doesn't leak sensitive information through its vector representation.

## 3. Core Algorithms

### 3.1 Noise Injection
{secured} = \text{normalize}(v_{original} + \mathcal{N}(0, \sigma^2))$
- **$\sigma$ Tuning:** Default 0.01.
- **Normalization:** Critical for maintaining cosine similarity search.

### 3.2 Bias Analysis
Uses the Word Embedding Association Test (WEAT) to measure the differential association between target and attribute sets.

## 4. Integration
- **Navigator:** Calls Anchor during search and indexing flows.
- **Tracker:** Receives metrics on noise magnitude and similarity preservation.
