"""Phase 2 (Anchor-OUT) response verifier."""
from __future__ import annotations

import math
import re
import string
import time
import unicodedata
from typing import Callable, Optional

from .exceptions import Client as ExceptionsClient
from .models import (
    BiasBaseline,
    CategoryDrift,
    DriftAnalysis,
    DriftRequest,
    DriftResponse,
    EmbeddingAnalysis,
    EmbeddingAnalysisRequest,
    EmbeddingAnalysisResponse,
    ResponseBiasAnalysis,
    ResponseCategoryBias,
    ResponseQuality,
    VerifyRequest,
    VerifyResponse,
)

_DEFAULT_BIAS_THRESHOLD = 0.15
_HIGH_BIAS_THRESHOLD = 0.25
_AMP_THRESHOLD = 0.10
_INTRO_PHASE1_MAX = 0.05
_INTRO_PHASE2_MIN = 0.15
_REDUCTION_THRESH = 0.10
_CONTEXT_SIM_MIN = 0.60

_GENDER_A = frozenset([
    "male", "man", "men", "he", "his", "him", "boy", "boys", "gentleman",
    "남성", "남자", "남자들", "남성들", "그는", "그의",
])
_GENDER_B = frozenset([
    "female", "woman", "women", "she", "her", "girl", "girls", "lady", "ladies",
    "여성", "여자", "여자들", "여성들", "그녀는", "그녀의",
])
_ETHNICITY_A = frozenset(["korean", "한국인", "한국인들"])
_ETHNICITY_B = frozenset(["foreign", "foreigner", "foreigners", "외국인", "외국인들"])
_AGE_A = frozenset(["young", "youth", "junior", "젊은", "청년", "어린"])
_AGE_B = frozenset(["old", "elder", "elderly", "senior", "나이든", "노인"])

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


# Stereotype shapes: a group, a generalising verb, then a negative trait. Each needs all three,
# so neutral text that merely names a group does not match. They complement the fixed phrases above.
_BIAS_REGEX: dict[str, list["re.Pattern[str]"]] = {
    "gender": [re.compile(p, re.I) for p in (
        r"\b(women|girls|females?|men|boys|males?)\s+(are|is|can'?t|cannot|aren'?t)\s+(just\s+|naturally\s+|simply\s+|too\s+|not\s+)?(emotional|irrational|weak|bossy|hysterical|less\s+capable|more\s+capable|better\s+leaders|worse\s+leaders|suited|fit)\b",
        r"\b(women|girls|females?)\s+(should\s+(not|never|stay)|belong\s+in)\b",
    )],
    "age": [re.compile(p, re.I) for p in (
        r"\b(older|old|elderly|aged|senior)\s+(workers|people|employees|staff|adults|persons)\s+(cannot|can'?t|can\s+not|are\s+(too\s+|all\s+)?(slow|incapable|unable|useless|obsolete|set\s+in))",
        r"\b(young|younger)\s+(workers|people|employees|staff|adults)\s+(are|don'?t|cannot|can'?t)\s+(all\s+)?(lazy|irresponsible|entitled|know|understand)",
        r"\btoo\s+(old|young)\s+to\b",
    )],
    "ethnicity": [re.compile(p, re.I) for p in (
        r"\b(people\s+from\s+(that|those|this|these)\s+\w+|immigrants|foreigners|those\s+people|these\s+people)\s+(are|can'?t|cannot|don'?t)\s+(all\s+|just\s+)?(lazy|untrustworthy|dishonest|criminals?|stupid|dirty|dangerous|violent|understand|work)\b",
    )],
}

# A group needs at least this many mentions before an imbalance between two sides means anything.
# One mention of "women" is a fact about the text, not bias.
_MIN_MENTIONS_FOR_IMBALANCE = 4

_WORD = re.compile(r"[\w']+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    # Words without surrounding punctuation ("women." is "women"). Korean particles attach to the
    # noun (남성은), so a Korean token also counts when it starts with a listed term.
    return _WORD.findall(text)


def _count(tokens: list[str], group: frozenset) -> int:
    n = 0
    for t in tokens:
        if t in group or (not t.isascii() and any(t.startswith(g) for g in group if not g.isascii())):
            n += 1
    return n


def _word_set(tokens: list[str]) -> set[str]:
    return set(tokens)


def _jaccard(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


# Process-wide scoped-exception store (set at startup). None = nothing is exempted.
_exceptions: Optional[ExceptionsClient] = None


def set_exceptions(client: Optional[ExceptionsClient]) -> None:
    global _exceptions
    _exceptions = client


class Verifier:
    def __init__(self, medium_threshold: float = 0.15, high_threshold: float = 0.25) -> None:
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold

    def verify(self, req: VerifyRequest) -> VerifyResponse:
        start = time.perf_counter()
        opts = req.options
        run_all = opts is None or not any([
            opts.check_bias, opts.analyze_embedding, opts.check_drift, opts.measure_quality,
        ])
        bias_threshold = opts.bias_threshold if opts and opts.bias_threshold > 0 else _DEFAULT_BIAS_THRESHOLD

        resp = VerifyResponse(request_id=req.request_id)
        bias_anal: Optional[ResponseBiasAnalysis] = None

        if run_all or (opts and opts.check_bias):
            tenant = req.tenant_id or (req.user.tenant_id if req.user else "")
            user = req.user.user_id if req.user else ""
            client = _exceptions

            def exempt(category: str) -> Optional[str]:
                # Rule "anchor.bias:<category>" exempts one bias category for one tenant.
                return client.allowed(f"anchor.bias:{category}", tenant, user) if client else None

            bias_anal = self._detect_bias(req.llm_response, bias_threshold, exempt, resp.exceptions_applied)
            resp.bias_analysis = bias_anal

        if run_all or (opts and opts.analyze_embedding):
            resp.embedding_analysis = self._analyze_embedding(
                req.llm_response, req.original_query, req.source_documents
            )

        if (run_all or (opts and opts.check_drift)) and req.phase1_baseline and bias_anal:
            resp.drift_analysis = self._detect_drift(req.phase1_baseline, bias_anal)

        if run_all or (opts and opts.measure_quality):
            resp.quality = self._evaluate_quality(
                req.llm_response, req.original_query, req.source_documents
            )

        resp.status = self._determine_status(resp.bias_analysis, resp.drift_analysis, resp.embedding_analysis)
        resp.warnings, resp.recommendations = self._generate_insights(resp)
        resp.processing_time_ms = (time.perf_counter() - start) * 1000.0
        return resp

    def analyze_embedding(self, req: EmbeddingAnalysisRequest) -> EmbeddingAnalysisResponse:
        tokens = _tokenize(req.text.lower())
        sims = [_jaccard(tokens, _tokenize(ct.lower())) for ct in req.comparison_texts]
        diversity = max(0.0, 1.0 - (sum(sims) / len(sims))) if sims else 0.0
        return EmbeddingAnalysisResponse(embedding=[], similarities=sims, diversity=diversity)

    def check_drift(self, req: DriftRequest) -> DriftResponse:
        if not req.phase1 or not req.phase2:
            return DriftResponse(drift_detected=False, severity="low")

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

    # ── bias detection ────────────────────────────────────────────────────────

    def _detect_bias(
        self,
        text: str,
        threshold: float,
        exempt: Optional[Callable[[str], Optional[str]]] = None,
        applied: Optional[list[str]] = None,
    ) -> ResponseBiasAnalysis:
        lower = text.lower()
        tokens = _tokenize(lower)

        categories = [
            ("gender", _GENDER_A, _GENDER_B),
            ("ethnicity", _ETHNICITY_A, _ETHNICITY_B),
            ("age", _AGE_A, _AGE_B),
        ]
        cat_bias: list[ResponseCategoryBias] = []
        overall_max = 0.0

        for name, group_a, group_b in categories:
            count_a = _count(tokens, group_a)
            count_b = _count(tokens, group_b)
            total = count_a + count_b
            score = abs(count_a - count_b) / total if total >= _MIN_MENTIONS_FOR_IMBALANCE else 0.0

            evidence: list[str] = []
            for pat in _BIAS_PATTERNS.get(name, []):
                if pat in lower:
                    score = max(score, 0.4)
                    evidence.append(pat)
            for rx in _BIAS_REGEX.get(name, []):
                m = rx.search(text)
                if m:
                    score = max(score, 0.4)
                    evidence.append(m.group(0))

            # A scoped exception drops this category from the analysis. It is only
            # recorded as used when the category actually showed bias.
            if (score > 0 or evidence) and exempt is not None:
                exc_id = exempt(name)
                if exc_id:
                    if applied is not None and exc_id not in applied:
                        applied.append(exc_id)
                    continue

            if score > 0 or evidence:
                cat_bias.append(ResponseCategoryBias(category=name, score=score, evidence=evidence))
            overall_max = max(overall_max, score)

        severity = "low"
        if overall_max >= self.high_threshold:
            severity = "high"
        elif overall_max >= self.medium_threshold:
            severity = "medium"

        return ResponseBiasAnalysis(
            overall_score=overall_max,
            categories=cat_bias,
            severity=severity,
            exceeds_threshold=overall_max >= threshold,
        )

    # ── embedding analysis ────────────────────────────────────────────────────

    def _analyze_embedding(
        self, response: str, query: str, sources: list[str]
    ) -> EmbeddingAnalysis:
        resp_tokens = _tokenize(response.lower())
        sim_to_query = _jaccard(resp_tokens, _tokenize(query.lower()))

        sim_to_context = 0.0
        if sources:
            combined: list[str] = []
            for s in sources:
                combined.extend(_tokenize(s.lower()))
            combined = list(set(combined))
            sim_to_context = _jaccard(resp_tokens, combined)

        diversity = max(0.0, 1.0 - sim_to_context)
        return EmbeddingAnalysis(
            similarity_to_query=sim_to_query,
            similarity_to_context=sim_to_context,
            diversity_score=diversity,
            semantic_drift_detected=bool(sources) and sim_to_context < _CONTEXT_SIM_MIN,
        )

    # ── quality evaluation ────────────────────────────────────────────────────

    def _evaluate_quality(
        self, response: str, query: str, sources: list[str]
    ) -> ResponseQuality:
        resp_tokens = _tokenize(response.lower())
        resp_set = set(resp_tokens)

        grounding = 0.0
        if sources and resp_set:
            src_tokens: list[str] = []
            for s in sources:
                src_tokens.extend(_tokenize(s.lower()))
            src_set = set(src_tokens)
            grounding = sum(1 for w in resp_set if w in src_set) / len(resp_set)

        sentences = [s.strip() for s in response.replace("。", ".").replace("！", "!").replace("？", "?").split(".") if s.strip()]
        coherence = min(len(sentences) / 5.0, 1.0)
        relevance = _jaccard(resp_tokens, _tokenize(query.lower()))
        overall = 0.4 * grounding + 0.3 * coherence + 0.3 * relevance

        return ResponseQuality(
            grounding_score=grounding,
            coherence_score=coherence,
            relevance_score=relevance,
            overall_quality=overall,
        )

    # ── drift detection ───────────────────────────────────────────────────────

    def _detect_drift(
        self, baseline: BiasBaseline, bias_anal: ResponseBiasAnalysis
    ) -> DriftAnalysis:
        p1, p2 = baseline.overall_score, bias_anal.overall_score
        diff = p2 - p1
        direction = _classify_drift(p1, p2, diff)
        return DriftAnalysis(
            drift_detected=direction != "none",
            phase1_bias_score=p1,
            phase2_bias_score=p2,
            drift_amount=abs(diff),
            drift_direction=direction,
        )

    # ── status & insights ─────────────────────────────────────────────────────

    def _determine_status(
        self,
        bias: Optional[ResponseBiasAnalysis],
        drift: Optional[DriftAnalysis],
        embed: Optional[EmbeddingAnalysis],
    ) -> str:
        if bias:
            if bias.overall_score >= self.high_threshold:
                return "CRITICAL"
            if bias.exceeds_threshold:
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


def _classify_drift(p1: float, p2: float, diff: float) -> str:
    if p1 < _INTRO_PHASE1_MAX and p2 > _INTRO_PHASE2_MIN:
        return "introduction"
    if diff > _AMP_THRESHOLD:
        return "amplification"
    if diff < -_REDUCTION_THRESH:
        return "reduction"
    return "none"
