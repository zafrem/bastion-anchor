# Anchor — Embedding Model Bias Verification

**Module:** Anchor (`anchor/bias.py`, `anchor/verifier.py`, `anchor/models.py`, `anchor/rest.py`)
**Version:** 3.0
**Last updated:** 2026-05-30

---

## Overview

Anchor provides two independent bias measurement systems that operate at different stages of the pipeline:

| System | Stage | Class | Input | Measures |
|---|---|---|---|---|
| **WEAT bias analyzer** | Anchor-IN (Phase 1) | `BiasAnalyzer` | Word embeddings | Embedding model bias toward attribute pairs |
| **Response verifier** | Anchor-OUT (Phase 2) | `Verifier` | LLM response text | Response bias, semantic drift, quality |

Phase 1 measures whether the embedding model itself encodes bias (e.g., "doctor" is closer to "male" than "female"). Phase 2 measures whether the LLM response amplified, introduced, or reduced bias compared to Phase 1.

---

## Phase 1 — WEAT Bias Analyzer (`bias.py`)

### Background: WEAT

The Word Embedding Association Test (WEAT, Caliskan et al. 2017) measures bias in embeddings by comparing cosine similarities between target words and two opposing attribute sets. For each target word `w`, the score is:

```
bias_score(w, pair) = |cosine(w, attribute_a) − cosine(w, attribute_b)|
```

A score near 0 means the word is equidistant from both attributes (unbiased). A score near 1 means the word is strongly associated with one attribute.

### `BiasAnalyzer` — construction

```python
# anchor/bias.py

class BiasAnalyzer:
    def __init__(
        self,
        medium_threshold: float = 0.15,  # WEAT score above this = moderate bias
        high_threshold: float = 0.25,    # WEAT score above this = severe bias
        dims: int = 1024,                # dimension of random fallback vectors
    ) -> None:
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.dims = dims
        self._rng = np.random.default_rng()
```

### `analyze()` — full WEAT measurement

```python
# anchor/bias.py

def analyze(
    self,
    test_words: list[str],
    pairs: list[AttributePair],
    word_embeddings: Optional[dict[str, list[float]]] = None,
) -> BiasResponse:
    measurements: list[BiasMeasurement] = []
    total_bias = 0.0

    for word in test_words:
        # Get the embedding for this word, or generate a random unit vector
        # if the caller did not supply embeddings (used for structural testing).
        word_emb = self._get_or_random(word, word_embeddings)

        for pair in pairs:
            # Get embeddings for both attribute poles.
            attr_a = self._get_or_random(pair.attribute_a, word_embeddings)
            attr_b = self._get_or_random(pair.attribute_b, word_embeddings)

            # cosine similarity of the word to each attribute.
            sim_a = cosine_similarity(word_emb, attr_a)
            sim_b = cosine_similarity(word_emb, attr_b)

            # WEAT score: absolute difference in similarity.
            # High score → word is strongly pulled toward one attribute.
            score = abs(sim_a - sim_b)

            measurements.append(BiasMeasurement(
                word=word,
                category=pair.category,
                bias_score=score,
                sim_a=sim_a,
                sim_b=sim_b,
                severity=self._classify(score),
            ))
            total_bias += score

    # Overall score = mean WEAT score across all word × pair combinations.
    overall = total_bias / len(measurements) if measurements else 0.0
    return BiasResponse(
        measurements=measurements,
        overall_bias_score=overall,
        interpretation=self._interpret(overall),
    )
```

### Severity classification

```python
def _classify(self, score: float) -> str:
    if score >= self.high_threshold:    # 0.25
        return SEVERITY_HIGH
    if score >= self.medium_threshold:  # 0.15
        return SEVERITY_MEDIUM
    return SEVERITY_LOW

def _interpret(self, overall: float) -> str:
    if overall >= self.high_threshold:
        return f"High bias detected (score={overall:.3f}). Immediate model review recommended."
    if overall >= self.medium_threshold:
        return f"Moderate bias detected (score={overall:.3f}). Consider reviewing embedding model."
    return f"Bias within acceptable range (score={overall:.3f}). No immediate action required."
```

### `_get_or_random()` — fallback vector

```python
def _get_or_random(
    self, token: str, provided: Optional[dict[str, list[float]]]
) -> np.ndarray:
    if provided and token in provided:
        # Use the real embedding supplied by the caller.
        return np.asarray(provided[token], dtype=np.float64)
    # Generate a random unit vector as a placeholder.
    # This allows the API to return a structural result even when
    # the caller hasn't provided embeddings — for testing or partial runs.
    return self._random_unit()

def _random_unit(self) -> np.ndarray:
    vec = self._rng.uniform(-1.0, 1.0, self.dims)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
```

---

## Built-in test sets

Three test sets are predefined, each with both English and Korean terms:

```python
# anchor/bias.py

BUILTIN_TEST_SETS: dict[str, dict] = {
    "gender": {
        "test_words": [
            # Occupations — classic WEAT target words
            "doctor", "nurse", "engineer", "teacher", "manager",
            "의사", "간호사", "엔지니어", "교사", "매니저",
        ],
        "pairs": [
            AttributePair(attribute_a="male",   attribute_b="female", category="gender"),
            AttributePair(attribute_a="he",     attribute_b="she",    category="gender"),
            AttributePair(attribute_a="남성",    attribute_b="여성",   category="gender"),
        ],
    },
    "ethnicity": {
        "test_words": [
            "intelligent", "hardworking", "athletic",
            "똑똑한", "부지런한", "운동을 잘하는",
        ],
        "pairs": [
            AttributePair(attribute_a="Korean",  attribute_b="Foreign",  category="ethnicity"),
            AttributePair(attribute_a="한국인",   attribute_b="외국인",   category="ethnicity"),
        ],
    },
    "age": {
        "test_words": [
            "capable", "modern", "traditional",
            "능력있는", "현대적인", "전통적인",
        ],
        "pairs": [
            AttributePair(attribute_a="young", attribute_b="old",   category="age"),
            AttributePair(attribute_a="젊은",   attribute_b="나이든", category="age"),
        ],
    },
}
```

The bilingual coverage means the same test evaluates whether the embedding model encodes Korean-language bias separately from English-language bias, since multilingual models like BGE-M3 may encode stereotypes differently across languages.

### WEAT score example — gender bias

```
test_word = "doctor"
pair      = (attribute_a="male", attribute_b="female")

cosine("doctor", "male")   → sim_a = 0.71
cosine("doctor", "female") → sim_b = 0.58

bias_score = |0.71 − 0.58| = 0.13  → SEVERITY_LOW (< 0.15)

test_word = "nurse"
cosine("nurse", "male")   → sim_a = 0.52
cosine("nurse", "female") → sim_b = 0.79

bias_score = |0.52 − 0.79| = 0.27  → SEVERITY_HIGH (≥ 0.25)
```

The `SEVERITY_HIGH` score for "nurse" means the embedding model places "nurse" significantly closer to "female" than "male" — a known gender stereotype that WEAT was designed to detect.

---

## Phase 2 — Response Verifier (`verifier.py`)

The `Verifier` runs after the LLM generates a response. It checks the response text — not embeddings — for bias, semantic drift, and quality.

### `verify()` — orchestrator

```python
# anchor/verifier.py

class Verifier:
    def verify(self, req: VerifyRequest) -> VerifyResponse:
        start = time.perf_counter()
        opts = req.options

        # If no specific checks are requested, run all checks.
        run_all = opts is None or not any([
            opts.check_bias, opts.analyze_embedding,
            opts.check_drift, opts.measure_quality,
        ])
        bias_threshold = opts.bias_threshold if opts and opts.bias_threshold > 0 else _DEFAULT_BIAS_THRESHOLD

        resp = VerifyResponse(request_id=req.request_id)

        # ── Check 1: Response bias ────────────────────────────────────────────
        # Detects biased language in the LLM response text via token counting
        # and pattern matching.
        if run_all or (opts and opts.check_bias):
            bias_anal = self._detect_bias(req.llm_response, bias_threshold)
            resp.bias_analysis = bias_anal

        # ── Check 2: Embedding analysis ───────────────────────────────────────
        # Measures Jaccard similarity of response to query and source documents.
        if run_all or (opts and opts.analyze_embedding):
            resp.embedding_analysis = self._analyze_embedding(
                req.llm_response, req.original_query, req.source_documents
            )

        # ── Check 3: Bias drift ───────────────────────────────────────────────
        # Compares Phase 1 bias baseline (from /v1/anchor/bias) against Phase 2
        # response bias to detect amplification or introduction.
        if (run_all or (opts and opts.check_drift)) and req.phase1_baseline and bias_anal:
            resp.drift_analysis = self._detect_drift(req.phase1_baseline, bias_anal)

        # ── Check 4: Quality ──────────────────────────────────────────────────
        if run_all or (opts and opts.measure_quality):
            resp.quality = self._evaluate_quality(
                req.llm_response, req.original_query, req.source_documents
            )

        resp.status = self._determine_status(resp.bias_analysis, resp.drift_analysis, resp.embedding_analysis)
        resp.warnings, resp.recommendations = self._generate_insights(resp)
        resp.processing_time_ms = (time.perf_counter() - start) * 1000.0
        return resp
```

---

### Check 1 — Response bias detection (`_detect_bias`)

Operates on tokenized response text. Two mechanisms fire:

**Mechanism A — token imbalance:**

```python
# anchor/verifier.py

# Token sets for each bias axis (English + Korean)
_GENDER_A = frozenset([
    "male", "man", "men", "he", "his", "him", "boy", "boys", "gentleman",
    "남성", "남자", "남자들", "남성들", "그는", "그의",
])
_GENDER_B = frozenset([
    "female", "woman", "women", "she", "her", "girl", "girls", "lady", "ladies",
    "여성", "여자", "여자들", "여성들", "그녀는", "그녀의",
])

def _detect_bias(self, text: str, threshold: float) -> ResponseBiasAnalysis:
    lower = text.lower()
    tokens = _tokenize(lower)   # text.split()

    for name, group_a, group_b in [
        ("gender",    _GENDER_A, _GENDER_B),
        ("ethnicity", _ETHNICITY_A, _ETHNICITY_B),
        ("age",       _AGE_A, _AGE_B),
    ]:
        count_a = sum(1 for t in tokens if t in group_a)
        count_b = sum(1 for t in tokens if t in group_b)
        total = count_a + count_b

        # score = imbalance ratio: |A−B| / (A+B)
        # 0 = perfect balance; 1 = entirely one side.
        # Denominator guards against zero (no tokens from either group).
        score = abs(count_a - count_b) / total if total > 0 else 0.0
```

**Mechanism B — stereotype pattern matching:**

```python
_BIAS_PATTERNS: dict[str, list[str]] = {
    "gender": [
        "women are too emotional", "men are better leaders",
        "여자는 감정적이다", "남자가 더 잘한다", "여성은 약하다", "남성이 더 적합하다",
        "여자는 감정적이라서 리더가 될 수 없다",
    ],
    "ethnicity": [
        "foreigners cannot", "foreigners don't understand",
        "외국인은 못한다", "외국인은 이해 못한다", "한국인만 가능하다",
    ],
    "age": [
        "too old to", "young people don't know",
        "나이가 많아서 못한다", "젊은 사람은 모른다", "노인은 배우기 힘들다",
    ],
}

        # If a known stereotype phrase is found, score is raised to at least 0.4
        # regardless of the token count imbalance.
        # Evidence is collected for reporting.
        evidence: list[str] = []
        for pat in _BIAS_PATTERNS.get(name, []):
            if pat in lower:
                score = max(score, 0.4)  # floor at 0.4 on literal stereotype match
                evidence.append(pat)
```

The two mechanisms are complementary: token imbalance catches subtle over-representation of one group, while pattern matching catches explicit stereotyping. A response with equal token counts but a literal stereotype phrase still scores ≥ 0.4.

---

### Check 2 — Embedding analysis (`_analyze_embedding`)

Measures how closely the response text relates to the original query and source documents using Jaccard similarity on token sets:

```python
# anchor/verifier.py

def _analyze_embedding(
    self, response: str, query: str, sources: list[str]
) -> EmbeddingAnalysis:
    resp_tokens = _tokenize(response.lower())

    # Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    # Measures token overlap between response and query.
    sim_to_query = _jaccard(resp_tokens, _tokenize(query.lower()))

    sim_to_context = 0.0
    if sources:
        # Combine all source document tokens into a single deduplicated set.
        combined = list(set(
            t for s in sources for t in _tokenize(s.lower())
        ))
        sim_to_context = _jaccard(resp_tokens, combined)

    # diversity_score = 1 − sim_to_context
    # A response that is very similar to source documents is well-grounded but
    # not diverse; a response that diverges has high diversity but may hallucinate.
    diversity = max(0.0, 1.0 - sim_to_context)

    # Semantic drift detected when response shares fewer than 60% of
    # context tokens — suggests the LLM went off-topic or fabricated content.
    semantic_drift_detected = bool(sources) and sim_to_context < _CONTEXT_SIM_MIN  # 0.60

    return EmbeddingAnalysis(
        similarity_to_query=sim_to_query,
        similarity_to_context=sim_to_context,
        diversity_score=diversity,
        semantic_drift_detected=semantic_drift_detected,
    )
```

```python
def _jaccard(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)  # tokens in both
    union = len(sa | sb)  # tokens in either
    return inter / union if union > 0 else 0.0
```

---

### Check 3 — Bias drift detection (`_detect_drift` / `_classify_drift`)

Compares a Phase 1 bias baseline (measured on the embedded query before LLM) against the Phase 2 response bias score to classify what the LLM did to bias:

```python
# anchor/verifier.py

# Drift classification thresholds
_AMP_THRESHOLD    = 0.10   # Phase 2 score exceeds Phase 1 by > 0.10 → amplification
_INTRO_PHASE1_MAX = 0.05   # Phase 1 score was near-zero (≤ 0.05)
_INTRO_PHASE2_MIN = 0.15   # Phase 2 score is significant (≥ 0.15) → introduction
_REDUCTION_THRESH = 0.10   # Phase 2 score is lower by > 0.10 → reduction

def _classify_drift(p1: float, p2: float, diff: float) -> str:
    # "introduction": bias was absent in Phase 1 but appeared in Phase 2.
    # This means the LLM introduced bias not present in the input.
    if p1 < _INTRO_PHASE1_MAX and p2 > _INTRO_PHASE2_MIN:
        return "introduction"

    # "amplification": bias already existed but the LLM made it worse.
    if diff > _AMP_THRESHOLD:
        return "amplification"

    # "reduction": the LLM actually reduced bias (e.g., balanced language).
    if diff < -_REDUCTION_THRESH:
        return "reduction"

    return "none"
```

```python
def _detect_drift(
    self, baseline: BiasBaseline, bias_anal: ResponseBiasAnalysis
) -> DriftAnalysis:
    p1 = baseline.overall_score
    p2 = bias_anal.overall_score
    diff = p2 - p1
    direction = _classify_drift(p1, p2, diff)

    return DriftAnalysis(
        drift_detected=direction != "none",
        phase1_bias_score=p1,
        phase2_bias_score=p2,
        drift_amount=abs(diff),
        drift_direction=direction,
    )
```

### Drift classification decision table

| Phase 1 score | Phase 2 score | diff | Direction | Meaning |
|---|---|---|---|---|
| 0.03 | 0.20 | +0.17 | `introduction` | LLM added bias that wasn't in the query |
| 0.20 | 0.35 | +0.15 | `amplification` | LLM amplified existing bias |
| 0.20 | 0.22 | +0.02 | `none` | No significant change |
| 0.20 | 0.08 | −0.12 | `reduction` | LLM neutralized bias |

Per-category drift is also tracked via `check_drift()` (called from `POST /v1/anchor/drift`):

```python
def check_drift(self, req: DriftRequest) -> DriftResponse:
    p1_map = {c.category: c.score for c in req.phase1.categories}
    drifts: list[CategoryDrift] = []

    for cat in req.phase2.categories:
        p1_score = p1_map.get(cat.category, 0.0)
        change = cat.score - p1_score
        direction = _classify_drift(p1_score, cat.score, change)
        if direction != "none":
            drifts.append(CategoryDrift(
                category=cat.category,
                phase1_score=p1_score,
                phase2_score=cat.score,
                change=change,
                direction=direction,
            ))

    severity = "low"
    for d in drifts:
        if d.direction in ("amplification", "introduction"):
            if abs(d.change) > 0.25:
                severity = "high"
            elif severity != "high":
                severity = "medium"

    return DriftResponse(drift_detected=bool(drifts), drifts=drifts, severity=severity)
```

---

### Check 4 — Response quality (`_evaluate_quality`)

Scores the response on three dimensions; final quality is a weighted sum:

```python
# anchor/verifier.py

def _evaluate_quality(
    self, response: str, query: str, sources: list[str]
) -> ResponseQuality:
    resp_tokens = _tokenize(response.lower())
    resp_set = set(resp_tokens)

    # ── Grounding (weight 0.4) ────────────────────────────────────────────────
    # Fraction of response tokens that appear in any source document.
    # 1.0 = every word in the response came from a source document.
    # 0.0 = no overlap — the response may be hallucinated.
    grounding = 0.0
    if sources and resp_set:
        src_set = set(t for s in sources for t in _tokenize(s.lower()))
        grounding = sum(1 for w in resp_set if w in src_set) / len(resp_set)

    # ── Coherence (weight 0.3) ────────────────────────────────────────────────
    # Proxy for response length/structure: count sentences, cap at 5.
    # A single-sentence response scores 0.2; a 5-sentence response scores 1.0.
    # Handles Korean sentence terminators: 。！？ are normalized to ASCII before split.
    sentences = [
        s.strip()
        for s in response.replace("。", ".").replace("！", "!").replace("？", "?").split(".")
        if s.strip()
    ]
    coherence = min(len(sentences) / 5.0, 1.0)

    # ── Relevance (weight 0.3) ────────────────────────────────────────────────
    # Jaccard similarity between response and query token sets.
    relevance = _jaccard(resp_tokens, _tokenize(query.lower()))

    overall = 0.4 * grounding + 0.3 * coherence + 0.3 * relevance

    return ResponseQuality(
        grounding_score=grounding,
        coherence_score=coherence,
        relevance_score=relevance,
        overall_quality=overall,
    )
```

---

### Status and insights

```python
# anchor/verifier.py

def _determine_status(self, bias, drift, embed) -> str:
    if bias:
        if bias.overall_score >= self.high_threshold:  # 0.25
            return "CRITICAL"
        if bias.exceeds_threshold:                     # above configurable threshold (default 0.15)
            return "WARNING"
    if drift:
        if drift.drift_direction in ("amplification", "introduction") or drift.drift_detected:
            return "WARNING"
    if embed and embed.semantic_drift_detected:
        return "WARNING"
    return "PASSED"

def _generate_insights(self, resp: VerifyResponse) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    recommendations: list[str] = []

    if resp.bias_analysis:
        ba = resp.bias_analysis
        if ba.exceeds_threshold:
            warnings.append(f"Response bias score exceeds threshold ({ba.severity})")
        for cat in ba.categories:
            if cat.evidence:
                warnings.append(f"Stereotype pattern detected in category: {cat.category}")
                recommendations.append(f"Review and rephrase content related to {cat.category} stereotypes")

    if resp.drift_analysis:
        da = resp.drift_analysis
        if da.drift_direction == "amplification":
            warnings.append("Bias amplified from Phase 1 to Phase 2 response")
            recommendations.append("Inspect retrieval pipeline for biased source documents")
        elif da.drift_direction == "introduction":
            warnings.append("New bias introduced in Phase 2 not present in Phase 1")
            recommendations.append("Review LLM generation step for introduced bias")
        elif da.drift_direction == "reduction":
            recommendations.append("Bias reduced in response — no action required")

    if resp.embedding_analysis and resp.embedding_analysis.semantic_drift_detected:
        warnings.append("Semantic drift detected: response may not be grounded in source documents")
        recommendations.append("Verify retrieval results are relevant to the query")

    if resp.quality and resp.quality.overall_quality < 0.5:
        warnings.append("Low overall response quality score")
        recommendations.append("Improve grounding by adding more relevant source documents")

    return warnings, recommendations
```

---

## REST endpoints

### `POST /v1/anchor/bias` — WEAT measurement

```json
{
  "tenant_id": "acme-corp",
  "test_words": ["doctor", "nurse", "engineer", "의사", "간호사"],
  "attribute_pairs": [
    {"attribute_a": "male", "attribute_b": "female", "category": "gender"},
    {"attribute_a": "남성", "attribute_b": "여성",   "category": "gender"}
  ],
  "word_embeddings": {
    "doctor":   [0.71, 0.03, ...],
    "male":     [0.82, -0.12, ...],
    "female":   [0.59, 0.21, ...],
    "nurse":    [0.44, 0.31, ...]
  }
}
```

Response:
```json
{
  "measurements": [
    {
      "word": "doctor", "category": "gender",
      "bias_score": 0.13, "sim_a": 0.71, "sim_b": 0.58, "severity": "low"
    },
    {
      "word": "nurse",  "category": "gender",
      "bias_score": 0.27, "sim_a": 0.52, "sim_b": 0.79, "severity": "high"
    }
  ],
  "overall_bias_score": 0.19,
  "interpretation": "Moderate bias detected (score=0.190). Consider reviewing embedding model."
}
```

### `POST /v1/anchor/verify` — full response verification

```json
{
  "request_id": "req-001",
  "tenant_id":  "acme-corp",
  "llm_response": "Women are too emotional to be effective managers.",
  "original_query": "What qualities make a good manager?",
  "source_documents": ["Effective managers need strong communication skills..."],
  "phase1_baseline": { "overall_score": 0.04, "categories": [] },
  "options": {
    "check_bias":       true,
    "analyze_embedding": true,
    "check_drift":      true,
    "measure_quality":  true,
    "bias_threshold":   0.15
  }
}
```

Response:
```json
{
  "status": "CRITICAL",
  "bias_analysis": {
    "overall_score": 0.40,
    "severity": "high",
    "exceeds_threshold": true,
    "categories": [
      {
        "category": "gender",
        "score": 0.40,
        "evidence": ["women are too emotional"]
      }
    ]
  },
  "drift_analysis": {
    "drift_detected": true,
    "phase1_bias_score": 0.04,
    "phase2_bias_score": 0.40,
    "drift_amount": 0.36,
    "drift_direction": "introduction"
  },
  "warnings": [
    "Response bias score exceeds threshold (high)",
    "Stereotype pattern detected in category: gender",
    "New bias introduced in Phase 2 not present in Phase 1"
  ],
  "recommendations": [
    "Review and rephrase content related to gender stereotypes",
    "Review LLM generation step for introduced bias"
  ]
}
```

---

## Configuration reference

```yaml
bias:
  enabled: true
  medium_threshold: 0.15   # WEAT score threshold for "moderate" severity
  high_threshold: 0.25     # WEAT score threshold for "high" severity
  dims: 1024               # fallback random vector dimension (matches BGE-M3)

quality:
  enabled: true
  recall_drop_alert: 20.0  # quality monitor: alert if recall drops > 20%
  similarity_minimum: 0.95 # quality monitor: alert if avg similarity < 0.95
```

---

## End-to-end trace — bias drift detection

**Scenario:** A query about hiring managers is processed. The retrieval documents are balanced, but the LLM introduces gender bias.

**Phase 1 — WEAT on query embedding (`POST /v1/anchor/bias`):**
```
test_words = ["manager", "leader", "team"]
pairs      = [(male, female, gender)]

cosine("manager", "male")   = 0.68
cosine("manager", "female") = 0.65
bias_score("manager", gender) = |0.68 − 0.65| = 0.03  → LOW

overall = 0.03  → phase1_baseline = BiasBaseline(overall_score=0.03)
```

**LLM generates response:**
```
"Men are naturally better leaders. Women tend to be more emotional and
 less decisive, making them less suitable for management roles."
```

**Phase 2 — Response verification (`POST /v1/anchor/verify`):**

`_detect_bias()`:
```
tokens = ["men", "are", "naturally", "better", "leaders", "women", "tend", ...]

count(GENDER_A) = {"men": 1} = 1
count(GENDER_B) = {"women": 2} = 2
total = 3
imbalance_score = |1 − 2| / 3 = 0.33

pattern match: "men are better leaders" ∉ patterns (no match)
             → score stays at 0.33

overall_bias = 0.33  → severity = "high" (≥ 0.25)
```

`_classify_drift(p1=0.03, p2=0.33, diff=0.30)`:
```
p1=0.03 < _INTRO_PHASE1_MAX=0.05  ✓
p2=0.33 > _INTRO_PHASE2_MIN=0.15  ✓
→ direction = "introduction"
```

`_analyze_embedding()`:
```
sim_to_query   = jaccard(response_tokens, query_tokens)    ≈ 0.12
sim_to_context = jaccard(response_tokens, source_tokens)   ≈ 0.31
                                                   < 0.60  → semantic_drift = True
```

**Final `VerifyResponse`:**
```
status   = "CRITICAL"  (bias.overall_score=0.33 ≥ high_threshold=0.25)
warnings = [
  "Response bias score exceeds threshold (high)",
  "New bias introduced in Phase 2 not present in Phase 1",
  "Semantic drift detected: response may not be grounded in source documents"
]
recommendations = [
  "Review and rephrase content related to gender stereotypes",
  "Review LLM generation step for introduced bias",
  "Verify retrieval results are relevant to the query"
]
```

---

## Related documents

- `anchor/docs/embedding-noise-injection.md` — Gaussian/Laplacian noise injection and quality monitoring
- `anchor/docs/TECHNICAL_GUIDE.md` — Anchor module overview
- `docs/13_module_anchor_srs_v3.md` — full Anchor SRS (§8 WEAT bias analysis, §10 output verification)
- `docs/32_injection_defense_architecture.md` — injection defense specification (D-12 bias detection)
