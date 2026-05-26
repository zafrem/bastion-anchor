"""gRPC server for Anchor — JSON codec, same wire contract as the Go version."""
from __future__ import annotations

import json
import logging
import uuid
from concurrent import futures
from typing import Any

import grpc

from .config import Config
from .noise import NoiseInjector
from .bias import BiasAnalyzer
from .quality import QualityMonitor
from .verifier import Verifier
from .events import Publisher, TraceContext, event_embedding_secured, event_bias_detected, event_response_verified
from .hooks import HookManager, HookEvent, EVENT_EMBEDDING_SECURED, EVENT_BIAS_DETECTED, EVENT_RESPONSE_VERIFIED
from .models import (
    AnchorConfig, BatchSecureRequest, BatchSecureResponse, BiasRequest, DriftRequest,
    EmbeddingAnalysisRequest, EmbedMetrics, QualityRequest, SecureRequest, SecureResponse,
    VerifyRequest,
)

log = logging.getLogger(__name__)


def _deserialize(data: bytes) -> dict:
    return json.loads(data)


def _serialize(obj: Any) -> bytes:
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump()).encode()
    if isinstance(obj, dict):
        return json.dumps(obj).encode()
    return json.dumps(obj).encode()


def _handler(fn, request_model):
    def _wrapped(request_bytes, context):
        try:
            req = request_model.model_validate(_deserialize(request_bytes))
            return fn(req, context)
        except Exception as exc:
            log.exception("[grpc] handler error: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    return grpc.unary_unary_rpc_method_handler(
        _wrapped,
        request_deserializer=lambda b: b,
        response_serializer=_serialize,
    )


def _tc_from_context(context: grpc.ServicerContext) -> TraceContext:
    md = dict(context.invocation_metadata())
    return TraceContext(
        trace_id=md.get("x-trace-id", "") or str(uuid.uuid4()),
        span_id=md.get("x-span-id", "") or str(uuid.uuid4())[:16],
        parent_span_id=md.get("x-parent-span-id", ""),
        tenant_id=md.get("x-tenant-id", ""),
        user_id=md.get("x-user-id", ""),
        request_id=md.get("x-request-id", ""),
    )


class _AnchorServiceHandler(grpc.GenericRpcHandler):
    SERVICE_NAME = "bastion.anchor.v1.AnchorService"

    def __init__(self, cfg: Config, pub: Publisher, hm: HookManager) -> None:
        self._cfg = cfg
        self._pub = pub
        self._hm = hm
        self._ana = BiasAnalyzer(cfg.bias.medium_threshold, cfg.bias.high_threshold, cfg.bias.dims)
        self._mon = QualityMonitor(cfg.quality.recall_drop_alert, cfg.quality.similarity_minimum)
        self._ver = Verifier(cfg.bias.medium_threshold, cfg.bias.high_threshold)
        self._methods = self._build_methods()

    def _injector_for(self, tenant_id: str, opts=None) -> NoiseInjector:
        sigma = self._cfg.sigma_for_tenant(tenant_id)
        preserve = self._cfg.noise.preserve_norm
        if opts and opts.noise_sigma > 0:
            sigma = opts.noise_sigma
        if opts and opts.seed:
            return NoiseInjector(sigma, preserve, seed=opts.seed)
        return NoiseInjector(sigma, preserve, self._cfg.noise.strategy)

    def _build_methods(self) -> dict:
        return {
            "SecureEmbedding": _handler(self._secure_embedding, SecureRequest),
            "SecureBatch": _handler(self._secure_batch, BatchSecureRequest),
            "MeasureBias": _handler(self._measure_bias, BiasRequest),
            "MeasureQuality": _handler(self._measure_quality, QualityRequest),
            "GetConfig": _handler(self._get_config, _Empty),
            "UpdateConfig": _handler(self._update_config, AnchorConfig),
            "Health": _handler(self._health, _Empty),
            "VerifyResponse": _handler(self._verify_response, VerifyRequest),
            "AnalyzeResponse": _handler(self._verify_response, VerifyRequest),
            "AnalyzeResponseEmbedding": _handler(self._analyze_embedding, EmbeddingAnalysisRequest),
            "CheckDrift": _handler(self._check_drift, DriftRequest),
        }

    def service_name(self) -> str:
        return self.SERVICE_NAME

    def service(self, handler_call_details):
        method = handler_call_details.method.split("/")[-1]
        return self._methods.get(method)

    def _secure_embedding(self, req: SecureRequest, ctx):
        injector = self._injector_for(req.tenant_id, req.options)
        result = injector.inject(req.embedding)
        tc = _tc_from_context(ctx)
        tc.tenant_id = tc.tenant_id or req.tenant_id
        tc.request_id = tc.request_id or req.request_id
        self._pub.publish(event_embedding_secured(tc, req.operation, result.noise_magnitude, result.similarity_to_original))
        self._hm.fire(HookEvent(
            type=EVENT_EMBEDDING_SECURED, tenant_id=req.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id, request_id=req.request_id,
            data={"noise": result.noise_magnitude, "similarity": result.similarity_to_original},
        ))
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
        )

    def _secure_batch(self, req: BatchSecureRequest, ctx):
        items = []
        for item in req.items:
            injector = self._injector_for(item.tenant_id, item.options)
            result = injector.inject(item.embedding)
            items.append(SecureResponse(
                request_id=item.request_id,
                secured_embedding=result.secured,
                metrics=EmbedMetrics(
                    noise_added=result.noise_magnitude,
                    similarity_to_original=result.similarity_to_original,
                    strategy_used=result.strategy,
                ),
            ))
        return BatchSecureResponse(request_id=req.request_id, items=items)

    def _measure_bias(self, req: BiasRequest, ctx):
        resp = self._ana.analyze(req.test_words, req.attribute_pairs, req.word_embeddings)
        tc = _tc_from_context(ctx)
        tc.tenant_id = tc.tenant_id or req.tenant_id
        for m in resp.measurements:
            if m.severity == "high":
                self._pub.publish(event_bias_detected(tc, m.category, resp.overall_bias_score))
                self._hm.fire(HookEvent(
                    type=EVENT_BIAS_DETECTED, tenant_id=req.tenant_id,
                    trace_id=tc.trace_id, span_id=tc.span_id,
                    data={"category": m.category, "bias_score": resp.overall_bias_score},
                ))
                break
        return resp

    def _measure_quality(self, req: QualityRequest, ctx):
        sigma = req.noise_sigma if req.noise_sigma > 0 else self._cfg.noise.default_sigma
        return self._mon.estimate(sigma, req.embedding_pairs)

    def _get_config(self, _req, ctx):
        return AnchorConfig(
            default_noise_sigma=self._cfg.noise.default_sigma,
            preserve_norm=self._cfg.noise.preserve_norm,
            strategy=self._cfg.noise.strategy,
            tenant_overrides=self._cfg.noise.tenant_overrides,
            bias_enabled=self._cfg.bias.enabled,
            quality_enabled=self._cfg.quality.enabled,
        )

    def _update_config(self, update: AnchorConfig, ctx):
        if update.default_noise_sigma > 0:
            self._cfg.noise.default_sigma = update.default_noise_sigma
        if update.tenant_overrides:
            self._cfg.noise.tenant_overrides = update.tenant_overrides
        return self._get_config(None, ctx)

    def _health(self, _req, ctx):
        return {"status": "ok", "version": "2.0.0", "checks": {"grpc": "up"}}

    def _verify_response(self, req: VerifyRequest, ctx):
        resp = self._ver.verify(req)
        tc = _tc_from_context(ctx)
        tc.tenant_id = tc.tenant_id or req.tenant_id
        tc.request_id = tc.request_id or req.request_id
        bias_score = resp.bias_analysis.overall_score if resp.bias_analysis else 0.0
        self._pub.publish(event_response_verified(tc, resp.status, bias_score))
        self._hm.fire(HookEvent(
            type=EVENT_RESPONSE_VERIFIED, tenant_id=req.tenant_id,
            trace_id=tc.trace_id, span_id=tc.span_id,
            data={"status": resp.status, "bias_score": bias_score},
        ))
        return resp

    def _analyze_embedding(self, req: EmbeddingAnalysisRequest, ctx):
        return self._ver.analyze_embedding(req)

    def _check_drift(self, req: DriftRequest, ctx):
        return self._ver.check_drift(req)


class _Empty:
    @classmethod
    def model_validate(cls, data):
        return cls()

    def model_dump(self):
        return {}


class GRPCServer:
    def __init__(self, cfg: Config, pub: Publisher, hm: HookManager) -> None:
        self._handler = _AnchorServiceHandler(cfg, pub, hm)

    def hooks(self) -> HookManager:
        return self._handler._hm

    def serve(self, port: int) -> None:
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_receive_message_length", 16 * 1024 * 1024),
                ("grpc.max_send_message_length", 16 * 1024 * 1024),
            ],
        )
        server.add_generic_rpc_handlers([self._handler])
        server.add_insecure_port(f"[::]:{port}")
        log.info("[grpc] anchor listening on :%d", port)
        server.start()
        server.wait_for_termination()
