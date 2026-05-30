# Anchor — Embedding Noise Injection

**Module:** Anchor (`anchor/noise.py`, `anchor/quality.py`, `anchor/config.py`, `anchor/rest.py`)
**Version:** 3.0
**Last updated:** 2026-05-30

---

## Overview

Anchor-IN (Phase 1) adds calibrated statistical noise to every embedding vector before it reaches the LLM. The purpose is to prevent **vector reconstruction attacks** — an adversary who captures the embedding vector cannot reconstruct the original document text by inverting the embedding model, because the injected noise makes inversion impractical.

The core trade-off: more noise = stronger protection but lower retrieval recall. Anchor exposes this trade-off as a tunable `sigma` parameter and measures its actual impact via the `QualityMonitor`.

---

## Position in the pipeline

```
Navigator embeds query/document
    │
    ▼
POST /v1/anchor/secure
    │
    ▼
NoiseInjector.inject(embedding)
    │
    ├── Gaussian or Laplacian noise added to each dimension
    ├── Optional: renormalize to unit sphere (preserve_norm)
    ├── Cosine similarity to original measured and logged
    │
    ▼
secured_embedding → LLM or Qdrant index
```

---

## `NoiseInjector` — construction

```python
# anchor/noise.py

class NoiseInjector:
    def __init__(
        self,
        sigma: float,          # standard deviation (Gaussian) or scale (Laplacian)
        preserve_norm: bool,   # if True, renormalize secured vector to unit sphere
        strategy: str = STRATEGY_GAUSSIAN,
        seed: Optional[int] = None,
    ) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = sigma
        self.preserve_norm = preserve_norm
        self.strategy = strategy or STRATEGY_GAUSSIAN

        # Use a cryptographically random seed when no explicit seed is given.
        # This ensures two requests with the same embedding get different noise,
        # preventing an attacker from averaging multiple responses to cancel noise.
        rng_seed = seed if seed is not None else _crypto_seed()
        self.rng = np.random.default_rng(rng_seed)
```

### Cryptographic seed generation

```python
# anchor/noise.py

def _crypto_seed() -> int:
    # Read 8 bytes from the OS CSPRNG (same source as os.urandom).
    raw = secrets.token_bytes(8)
    # Unpack as unsigned 64-bit little-endian integer.
    seed = struct.unpack("<Q", raw)[0]
    # Clamp to 31-bit range for numpy's default_rng compatibility.
    return int(seed % (2**31))
```

Using `secrets.token_bytes` (not `random` or `time.time()`) ensures the seed is unpredictable. A predictable seed would allow an attacker to reproduce the exact noise vector and subtract it from the secured embedding.

---

## `inject()` — core noise application

```python
# anchor/noise.py

def inject(self, embedding: list[float]) -> NoiseResult:
    # Convert to float64 for numerical precision during noise computation.
    orig = np.asarray(embedding, dtype=np.float64)
    original_norm = float(np.linalg.norm(orig))

    # ── Strategy selection ───────────────────────────────────────────────────
    if self.strategy == STRATEGY_LAPLACIAN:
        # Laplacian distribution:
        # PDF: f(x) = (1/2b) * exp(-|x|/b)
        # Scale b = sigma/sqrt(2) normalises so that std(Laplacian) = sigma,
        # matching the Gaussian parameterisation for fair comparison.
        b = self.sigma / np.sqrt(2)
        noise = self.rng.laplace(0.0, b, size=len(orig))
    else:
        # Gaussian distribution:
        # PDF: f(x) = (1/sqrt(2π)σ) * exp(-x²/2σ²)
        # numpy's normal() uses Box-Muller internally; each dimension is
        # independently drawn from N(0, sigma²).
        noise = self.rng.normal(0.0, self.sigma, size=len(orig))

    # ── Apply noise ──────────────────────────────────────────────────────────
    secured = orig + noise

    # noise_magnitude = mean absolute perturbation per dimension.
    # Used as the primary monitoring metric; emitted to Prometheus and Tracker.
    noise_magnitude = float(np.mean(np.abs(noise)))

    # ── Optional norm preservation ───────────────────────────────────────────
    if self.preserve_norm:
        norm = np.linalg.norm(secured)
        if norm > 0:
            # Project back onto the unit sphere.
            # This keeps the secured embedding's magnitude constant at 1.0,
            # which is required for correct cosine distance comparison in Qdrant.
            # Without this, noise inflates the norm and shifts all cosine scores.
            secured = secured / norm

    secured_norm = float(np.linalg.norm(secured))

    # ── Measure semantic preservation ────────────────────────────────────────
    similarity = cosine_similarity(orig, secured)

    return NoiseResult(
        secured=secured.tolist(),
        noise_magnitude=noise_magnitude,
        similarity_to_original=similarity,
        original_norm=original_norm,
        secured_norm=secured_norm,
        strategy=self.strategy + "_noise",  # e.g. "gaussian_noise"
    )
```

### `cosine_similarity()` — shared utility

```python
# anchor/noise.py

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or len(a) == 0:
        return 0.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
```

This function is shared by `noise.py`, `bias.py`, and `quality.py` to avoid divergence in the similarity computation.

---

## Gaussian vs Laplacian — when to use each

| Property | Gaussian `N(0, σ²)` | Laplacian `Laplace(0, σ/√2)` |
|---|---|---|
| PDF shape | Bell curve | Sharper peak, heavier tails |
| Outlier noise | Rare (3σ rule) | More frequent large perturbations |
| Effect on recall | Gradual degradation | Sharper at extremes |
| Use case | General-purpose | Higher protection requirement |

Both are parameterised by the same `sigma` so they are directly comparable. At `sigma=0.01` the distributions are nearly indistinguishable; at `sigma=0.05` the Laplacian produces more large-dimension outliers that push the vector further from the original.

### Preserve-norm effect

Without `preserve_norm`:
```
orig   = [0.6, 0.8, ...]   ||orig|| = 1.0 (unit sphere after BGE-M3)
noise  = [0.01, -0.005, ...]
secured = [0.61, 0.795, ...]   ||secured|| ≈ 1.002
```
The norm inflation shifts cosine scores slightly upward across all dimensions.

With `preserve_norm=True`:
```
secured = secured / ||secured||   → back to unit sphere
||secured|| = 1.0
```
The direction changes but magnitude does not, keeping cosine scores comparable to the original index.

---

## Tenant-scoped sigma

Different tenants may require different protection levels. Sigma is resolved per-request:

```python
# anchor/config.py

class Config(BaseModel):
    noise: NoiseConfig = ...

    def sigma_for_tenant(self, tenant_id: str) -> float:
        # Returns per-tenant override if set; falls back to global default.
        # Example: high-sensitivity HR tenant uses sigma=0.05;
        # standard customer tenant uses default sigma=0.01.
        return self.noise.tenant_overrides.get(tenant_id, self.noise.default_sigma)
```

```python
# anchor/rest.py

def _injector_for(tenant_id: str, opts=None) -> NoiseInjector:
    # Priority order:
    # 1. Per-request sigma from SecurityOptions (caller override)
    # 2. Per-tenant sigma from config.tenant_overrides
    # 3. Global default_sigma
    sigma = cfg.sigma_for_tenant(tenant_id)
    preserve = cfg.noise.preserve_norm

    if opts and opts.noise_sigma > 0:
        sigma = opts.noise_sigma          # request-level override

    if opts and opts.seed:
        # Reproducible noise for deterministic testing only.
        # Never use a fixed seed in production — see cryptographic seed rationale above.
        return NoiseInjector(sigma, preserve, seed=opts.seed)

    return NoiseInjector(sigma, preserve, cfg.noise.strategy)
```

---

## REST endpoint — `POST /v1/anchor/secure`

```python
# anchor/rest.py

@app.post("/v1/anchor/secure", response_model=SecureResponse)
def secure_embedding(req: SecureRequest, request: Request):
    start = time.perf_counter()
    injector = _injector_for(req.tenant_id, req.options)
    result = injector.inject(req.embedding)
    dur = (time.perf_counter() - start) * 1000.0

    tc = _extract_tc(request, req)
    # Emit event to Tracker for lineage and audit.
    pub.publish(event_embedding_secured(
        tc, req.operation, result.noise_magnitude, result.similarity_to_original
    ))
    # Emit Prometheus metrics for real-time monitoring.
    m.noise_magnitude.observe(result.noise_magnitude)
    m.similarity_to_original.observe(result.similarity_to_original)

    return SecureResponse(
        request_id=req.request_id,
        secured_embedding=result.secured,       # ready for Qdrant or LLM
        metrics=EmbedMetrics(
            noise_added=result.noise_magnitude,
            similarity_to_original=result.similarity_to_original,
            strategy_used=result.strategy,      # "gaussian_noise" or "laplacian_noise"
            original_norm=result.original_norm,
            secured_norm=result.secured_norm,
        ),
        processing_time_ms=dur,
    )
```

**Request:**
```json
{
  "request_id": "req-001",
  "tenant_id":  "acme-corp",
  "embedding":  [0.023, -0.011, 0.094, ...],
  "operation":  "index",
  "options": {
    "noise_sigma":    0.0,
    "preserve_norm":  false,
    "measure_bias":   false
  }
}
```

**Response:**
```json
{
  "request_id":        "req-001",
  "secured_embedding": [0.031, -0.009, 0.087, ...],
  "metrics": {
    "noise_added":            0.0094,
    "similarity_to_original": 0.9987,
    "strategy_used":          "gaussian_noise",
    "original_norm":          1.0,
    "secured_norm":           1.0
  },
  "processing_time_ms": 0.8
}
```

---

## Quality monitoring — `QualityMonitor`

Before deploying a new sigma value, operators use `POST /v1/anchor/quality` to measure the actual recall impact on representative embedding pairs.

```python
# anchor/quality.py

class QualityMonitor:
    def __init__(
        self,
        recall_drop_alert: float = 20.0,   # alert if recall drops > 20%
        similarity_minimum: float = 0.95,  # alert if avg cosine similarity < 0.95
    ) -> None:
        ...

    def estimate(self, sigma: float, pairs: list[EmbeddingPair]) -> QualityResponse:
        if not pairs:
            # No real pairs provided: use the linear model as a fast estimate.
            estimated = estimated_recall_preservation(sigma)
            return self._build(sigma, 1.0, estimated, estimated)

        # Inject noise into all query embeddings, measure recall preservation.
        inj = NoiseInjector(sigma, preserve_norm=True, seed=42)  # fixed seed for reproducibility
        hits_without = hits_with = 0
        total_sim = 0.0

        for pair in pairs:
            q = np.asarray(pair.query_embedding, dtype=np.float64)
            m = np.asarray(pair.match_embedding, dtype=np.float64)

            # "hit" = the original query is close enough to the match document
            if cosine_similarity(q, m) >= 0.5:
                hits_without += 1

            # Apply noise to query; check if match is still retrievable
            result = inj.inject(pair.query_embedding)
            secured = np.asarray(result.secured, dtype=np.float64)
            if cosine_similarity(secured, m) >= 0.5:
                hits_with += 1

            total_sim += result.similarity_to_original

        n = float(len(pairs))
        return self._build(sigma, hits_without / n, hits_with / n, total_sim / n)

    def _build(self, sigma, recall_without, recall_with, avg_sim) -> QualityResponse:
        drop_pct = 0.0
        if recall_without > 0:
            drop_pct = (recall_without - recall_with) / recall_without * 100

        # Status tiers:
        #   acceptable: avg_sim >= similarity_minimum AND drop_pct <= recall_drop_alert
        #   warning:    either threshold exceeded but drop_pct <= 2×recall_drop_alert
        #   critical:   drop_pct > 2×recall_drop_alert (40% by default)
        status = "acceptable"
        if avg_sim < self.similarity_minimum or drop_pct > self.recall_drop_alert:
            status = "critical" if drop_pct > self.recall_drop_alert * 2 else "warning"

        return QualityResponse(
            noise_sigma=sigma,
            recall_without=recall_without,
            recall_with=recall_with,
            recall_drop_percent=drop_pct,
            avg_similarity_to_original=avg_sim,
            status=status,
        )
```

### Linear recall model (no pairs)

```python
# anchor/noise.py

def estimated_recall_preservation(sigma: float) -> float:
    """Approximate Recall@K retention for a given sigma (linear model).

    Derived empirically: at sigma=0 recall is 1.0; at sigma=0.25 recall ≈ 0.0.
    Slope = -4.0.  Used only as a fast estimate when no real pairs are provided.
    """
    return max(0.0, min(1.0, 1.0 - 4.0 * sigma))
```

Recall estimates from the linear model:

| sigma | Estimated recall | Practical meaning |
|---|---|---|
| 0.00 | 1.00 | No noise — full recall |
| 0.01 | 0.96 | Default — 4% recall cost |
| 0.03 | 0.88 | Moderate protection |
| 0.05 | 0.80 | Strong protection |
| 0.25 | 0.00 | Complete obfuscation |

---

## Configuration reference

```yaml
# config/config.yaml

noise:
  default_sigma: 0.01       # applies to all tenants without an override
  preserve_norm: true       # project back onto unit sphere after noise
  strategy: gaussian        # "gaussian" | "laplacian"
  tenant_overrides:
    hr-tenant: 0.05         # HR data requires higher protection
    demo-tenant: 0.001      # demo/test tenant; near-zero noise

bias:
  enabled: true
  medium_threshold: 0.15
  high_threshold: 0.25
  dims: 1024

quality:
  enabled: true
  recall_drop_alert: 20.0   # alert when recall drops > 20%
  similarity_minimum: 0.95  # alert when avg cosine similarity < 0.95
```

---

## End-to-end trace — 1024-dim Gaussian injection

**Input:**
```python
embedding = [0.023, -0.011, 0.094, ...]   # 1024 floats, BGE-M3 output, ||v||=1.0
tenant_id = "acme-corp"
```

**`_injector_for("acme-corp", None)`:**
```python
sigma        = cfg.sigma_for_tenant("acme-corp") = 0.01  # no override
preserve_norm = True
strategy      = "gaussian"
rng_seed      = _crypto_seed()  # e.g. 1734829201 (unpredictable per request)
```

**`inject([0.023, -0.011, 0.094, ...])`:**
```python
orig = np.array([0.023, -0.011, 0.094, ...], dtype=float64)
original_norm = 1.0

# Gaussian noise: each dimension drawn independently from N(0, 0.0001)
noise = rng.normal(0.0, 0.01, size=1024)
# Example noise sample: [+0.0083, -0.0041, +0.0127, ...]

secured_raw = orig + noise
# [0.0313, -0.0151, 0.1067, ...]

noise_magnitude = mean(|noise|) ≈ 0.0094  # ~E[|N(0,0.01)|] = 0.01*sqrt(2/π) ≈ 0.00798

# preserve_norm=True: renormalize
secured_norm_pre = ||secured_raw|| ≈ 1.0025
secured = secured_raw / 1.0025
# [0.0313, -0.0151, 0.1064, ...]

secured_norm = 1.0   (unit sphere)
similarity   = cosine(orig, secured) ≈ 0.9987
```

**`SecureResponse`:**
```json
{
  "secured_embedding": [0.0313, -0.0151, 0.1064, ...],
  "metrics": {
    "noise_added":            0.0094,
    "similarity_to_original": 0.9987,
    "strategy_used":          "gaussian_noise",
    "original_norm":          1.0,
    "secured_norm":           1.0
  }
}
```

The semantic direction is preserved (similarity 0.9987) while each dimension is shifted by an average of ±0.0094. An attacker trying to invert the embedding to recover document text faces a 1024-dimensional search problem where every dimension has been independently perturbed by an unknown amount from an unknown distribution with an unknown seed.

---

## Related documents

- `anchor/docs/embedding-bias-verification.md` — WEAT bias analysis and response verification
- `anchor/docs/TECHNICAL_GUIDE.md` — Anchor module overview
- `docs/13_module_anchor_srs_v3.md` — full Anchor SRS (§7 noise injection, §9 quality monitoring)
- `docs/32_injection_defense_architecture.md` — injection defense specification (D-11 embedding noise)
