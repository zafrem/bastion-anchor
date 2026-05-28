"""FastAPI REST server for Anchor."""
from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .config import Config
from .noise import NoiseInjector
from .bias import BiasAnalyzer
from .quality import QualityMonitor
from .verifier import Verifier
from .events import (
    Publisher, TraceContext,
    event_embedding_secured, event_bias_detected, event_response_verified, event_anomaly_detected,
    event_batch_secured, event_quality_estimated, event_embedding_analyzed, event_drift_checked,
)
from .hooks import (
    HookManager, HookEvent,
    EVENT_EMBEDDING_SECURED, EVENT_BIAS_DETECTED, EVENT_RESPONSE_VERIFIED, EVENT_ANOMALY_DETECTED,
    EVENT_BATCH_SECURED, EVENT_QUALITY_ESTIMATED, EVENT_EMBEDDING_ANALYZED, EVENT_DRIFT_CHECKED,
)
from .models import (
    AnchorConfig, BatchSecureRequest, BatchSecureResponse, BiasRequest, BiasResponse,
    DriftRequest, DriftResponse, EmbeddingAnalysisRequest, EmbeddingAnalysisResponse,
    EmbedMetrics, HealthStatus, QualityRequest, QualityResponse, SecureRequest,
    SecureResponse, VerifyRequest, VerifyResponse,
)
from . import metrics as m

import logging

log = logging.getLogger(__name__)

_CONTEXT_SIM_MIN = 0.40


def _extract_tc(request: Request, req_body=None) -> TraceContext:
    h = request.headers
    tc = TraceContext(
        trace_id=h.get("x-trace-id", ""),
        span_id=h.get("x-span-id", ""),
        parent_span_id=h.get("x-parent-span-id", ""),
        tenant_id=h.get("x-tenant-id", ""),
        user_id=h.get("x-user-id", ""),
        request_id=h.get("x-request-id", ""),
    )
    if req_body:
        if not tc.tenant_id:
            tc.tenant_id = getattr(req_body, "tenant_id", "") or ""
        if not tc.request_id:
            tc.request_id = getattr(req_body, "request_id", "") or ""
    return tc


def build_app(cfg: Config, pub: Publisher, hm: HookManager) -> FastAPI:
    app = FastAPI(title="Bastion Anchor", version="2.0.0")

    inj = NoiseInjector(cfg.noise.default_sigma, cfg.noise.preserve_norm, cfg.noise.strategy)
    ana = BiasAnalyzer(cfg.bias.medium_threshold, cfg.bias.high_threshold, cfg.bias.dims)
    mon = QualityMonitor(cfg.quality.recall_drop_alert, cfg.quality.similarity_minimum)
    ver = Verifier(cfg.bias.medium_threshold, cfg.bias.high_threshold)

    def _injector_for(tenant_id: str, opts=None) -> NoiseInjector:
        sigma = cfg.sigma_for_tenant(tenant_id)
        preserve = cfg.noise.preserve_norm
        if opts and opts.noise_sigma > 0:
            sigma = opts.noise_sigma
        if opts and opts.seed:
            return NoiseInjector(sigma, preserve, seed=opts.seed)
        return NoiseInjector(sigma, preserve, cfg.noise.strategy)

    # ── health ─────────────────────────────────────────────────────────────

    @app.get("/v1/health")
    def health():
        return HealthStatus(status="ok", checks={"service": "up"})

    @app.get("/v1/metrics")
    def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ── Phase 1: Embedding security ────────────────────────────────────────

    @app.post("/v1/anchor/secure", response_model=SecureResponse)
    def secure_embedding(req: SecureRequest, request: Request):
        start = time.perf_counter()
        injector = _injector_for(req.tenant_id, req.options)
        result = injector.inject(req.embedding)
        dur = (time.perf_counter() - start) * 1000.0

        tc = _extract_tc(request, req)
        pub.publish(event_embedding_secured(tc, req.operation, result.noise_magnitude, result.similarity_to_original))
        hm.fire(HookEvent(
            type=EVENT_EMBEDDING_SECURED, tenant_id=req.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id, request_id=req.request_id,
            data={"noise": result.noise_magnitude, "similarity": result.similarity_to_original},
        ))
        m.noise_magnitude.observe(result.noise_magnitude)
        m.similarity_to_original.observe(result.similarity_to_original)

        return SecureResponse(
            request_id=req.request_id,
            secured_embedding=result.secured,
            metrics=EmbedMetrics(
                noise_added=result.noise_magnitude,
                similarity_to_original=result.similarity_to_original,
                strategy_used=result.strategy,
                original_norm=result.original_norm,
                secured_norm=result.secured_norm,
            ),
            processing_time_ms=dur,
        )

    @app.post("/v1/anchor/secure/batch", response_model=BatchSecureResponse)
    def secure_batch(req: BatchSecureRequest, request: Request):
        tc = _extract_tc(request, req)
        items = []
        total_noise = 0.0
        for item in req.items:
            injector = _injector_for(item.tenant_id, item.options)
            result = injector.inject(item.embedding)
            total_noise += result.noise_magnitude
            items.append(SecureResponse(
                request_id=item.request_id,
                secured_embedding=result.secured,
                metrics=EmbedMetrics(
                    noise_added=result.noise_magnitude,
                    similarity_to_original=result.similarity_to_original,
                    strategy_used=result.strategy,
                ),
            ))
        avg_noise = total_noise / len(items) if items else 0.0
        pub.publish(event_batch_secured(tc, len(items), avg_noise))
        hm.fire(HookEvent(
            type=EVENT_BATCH_SECURED, tenant_id=tc.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id, request_id=req.request_id,
            data={"item_count": len(items), "avg_noise": avg_noise},
        ))
        return BatchSecureResponse(request_id=req.request_id, items=items)

    @app.post("/v1/anchor/bias", response_model=BiasResponse)
    def measure_bias(req: BiasRequest, request: Request):
        resp = ana.analyze(req.test_words, req.attribute_pairs, req.word_embeddings)
        tc = _extract_tc(request, req)
        for measurement in resp.measurements:
            if measurement.severity == "high":
                pub.publish(event_bias_detected(tc, measurement.category, resp.overall_bias_score))
                hm.fire(HookEvent(
                    type=EVENT_BIAS_DETECTED, tenant_id=req.tenant_id,
                    trace_id=tc.trace_id, span_id=tc.span_id,
                    data={"category": measurement.category, "bias_score": resp.overall_bias_score},
                ))
                break
        m.bias_score.observe(resp.overall_bias_score)
        return resp

    @app.post("/v1/anchor/quality", response_model=QualityResponse)
    def measure_quality(req: QualityRequest, request: Request):
        tc = _extract_tc(request, req)
        sigma = req.noise_sigma if req.noise_sigma > 0 else cfg.noise.default_sigma
        resp = mon.estimate(sigma, req.embedding_pairs)
        pub.publish(event_quality_estimated(tc, sigma, resp.recall_with, resp.status))
        hm.fire(HookEvent(
            type=EVENT_QUALITY_ESTIMATED, tenant_id=tc.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id,
            data={"sigma": sigma, "recall": resp.recall_with, "status": resp.status},
        ))
        return resp

    @app.get("/v1/anchor/config", response_model=AnchorConfig)
    def get_config():
        return AnchorConfig(
            default_noise_sigma=cfg.noise.default_sigma,
            preserve_norm=cfg.noise.preserve_norm,
            strategy=cfg.noise.strategy,
            tenant_overrides=cfg.noise.tenant_overrides,
            bias_enabled=cfg.bias.enabled,
            quality_enabled=cfg.quality.enabled,
        )

    @app.put("/v1/anchor/config", response_model=AnchorConfig)
    def update_config(update: AnchorConfig):
        if update.default_noise_sigma > 0:
            cfg.noise.default_sigma = update.default_noise_sigma
            inj.set_sigma(update.default_noise_sigma)
        if update.tenant_overrides:
            cfg.noise.tenant_overrides = update.tenant_overrides
        return get_config()

    # ── Phase 2: Output verification ───────────────────────────────────────

    @app.post("/v1/anchor/verify", response_model=VerifyResponse)
    def verify_response(req: VerifyRequest, request: Request):
        resp = ver.verify(req)
        tc = _extract_tc(request, req)
        bias_score_val = resp.bias_analysis.overall_score if resp.bias_analysis else 0.0
        pub.publish(event_response_verified(tc, resp.status, bias_score_val))
        hm.fire(HookEvent(
            type=EVENT_RESPONSE_VERIFIED, tenant_id=req.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id, request_id=req.request_id,
            data={"status": resp.status, "bias_score": bias_score_val},
        ))
        if resp.embedding_analysis and resp.embedding_analysis.semantic_drift_detected:
            drift_score = resp.embedding_analysis.similarity_to_context
            pub.publish(event_anomaly_detected(tc, "semantic_drift", drift_score))
            hm.fire(HookEvent(
                type=EVENT_ANOMALY_DETECTED, tenant_id=req.tenant_id,
                trace_id=tc.trace_id, span_id=tc.span_id, request_id=req.request_id,
                data={"anomaly_type": "semantic_drift", "score": drift_score},
            ))
        return resp

    @app.post("/v1/anchor/analyze", response_model=VerifyResponse)
    def analyze_response(req: VerifyRequest, request: Request):
        return verify_response(req, request)

    @app.post("/v1/anchor/analyze/embedding", response_model=EmbeddingAnalysisResponse)
    def analyze_embedding(req: EmbeddingAnalysisRequest, request: Request):
        tc = _extract_tc(request, req)
        resp = ver.analyze_embedding(req)
        avg_sim = sum(resp.similarities) / len(resp.similarities) if resp.similarities else 0.0
        pub.publish(event_embedding_analyzed(tc, avg_sim, resp.diversity))
        hm.fire(HookEvent(
            type=EVENT_EMBEDDING_ANALYZED, tenant_id=tc.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id,
            data={"avg_similarity": avg_sim, "diversity": resp.diversity},
        ))
        return resp

    @app.post("/v1/anchor/drift", response_model=DriftResponse)
    def check_drift(req: DriftRequest, request: Request):
        tc = _extract_tc(request, req)
        resp = ver.check_drift(req)
        pub.publish(event_drift_checked(tc, resp.drift_detected, float(len(resp.drifts))))
        hm.fire(HookEvent(
            type=EVENT_DRIFT_CHECKED, tenant_id=tc.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id,
            data={"drift_detected": resp.drift_detected, "severity": resp.severity},
        ))
        return resp

    return app
