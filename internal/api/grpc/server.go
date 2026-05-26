package grpc

import (
	"context"
	"fmt"
	"log"
	"net"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"

	"github.com/bastion/anchor/internal/bias"
	"github.com/bastion/anchor/internal/config"
	"github.com/bastion/anchor/internal/hooks"
	"github.com/bastion/anchor/internal/models"
	"github.com/bastion/anchor/internal/noise"
	"github.com/bastion/anchor/internal/publisher"
	"github.com/bastion/anchor/internal/quality"
	"github.com/bastion/anchor/internal/verify"
)

// traceCtxKey is the context key for trace metadata injected by the interceptor.
type traceCtxKey struct{}

// traceFromCtx retrieves the TraceContext stored by traceInterceptor.
func traceFromCtx(ctx context.Context) publisher.TraceContext {
	if tc, ok := ctx.Value(traceCtxKey{}).(publisher.TraceContext); ok {
		return tc
	}
	return publisher.TraceContext{
		TraceID: uuid.New().String(),
		SpanID:  uuid.New().String()[:16],
	}
}

// traceInterceptor extracts W3C / Bastion trace headers from gRPC metadata
// and stores a TraceContext in the context for downstream handlers.
func traceInterceptor(ctx context.Context, req interface{}, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	tc := publisher.TraceContext{
		TraceID: uuid.New().String(),
		SpanID:  uuid.New().String()[:16],
	}
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if v := md.Get("x-trace-id"); len(v) > 0 {
			tc.TraceID = v[0]
		}
		if v := md.Get("x-span-id"); len(v) > 0 {
			tc.SpanID = v[0]
		}
		if v := md.Get("x-parent-span-id"); len(v) > 0 {
			tc.ParentSpanID = v[0]
		}
		if v := md.Get("x-tenant-id"); len(v) > 0 {
			tc.TenantID = v[0]
		}
		if v := md.Get("x-user-id"); len(v) > 0 {
			tc.UserID = v[0]
		}
		if v := md.Get("x-request-id"); len(v) > 0 {
			tc.RequestID = v[0]
		}
	}
	ctx = context.WithValue(ctx, traceCtxKey{}, tc)
	return handler(ctx, req)
}

// Server implements AnchorServiceServer.
type Server struct {
	UnimplementedAnchorServiceServer
	cfg        *config.Config
	inj        *noise.Injector
	ana        *bias.Analyzer
	mon        *quality.Monitor
	pub        publisher.Publisher
	ver        *verify.Verifier
	hm         *hooks.Manager
	grpcServer *grpc.Server
	port       int
}

// Hooks returns the hook manager so external coordinators can register listeners.
func (s *Server) Hooks() *hooks.Manager { return s.hm }

func New(
	cfg *config.Config,
	inj *noise.Injector,
	ana *bias.Analyzer,
	mon *quality.Monitor,
	pub publisher.Publisher,
	ver *verify.Verifier,
	port int,
) *Server {
	s := &Server{cfg: cfg, inj: inj, ana: ana, mon: mon, pub: pub, ver: ver, port: port, hm: hooks.New()}
	s.grpcServer = grpc.NewServer(
		grpc.MaxRecvMsgSize(16*1024*1024),
		grpc.MaxSendMsgSize(16*1024*1024),
		grpc.KeepaliveParams(keepalive.ServerParameters{
			MaxConnectionIdle: 5 * time.Minute,
			Time:              1 * time.Minute,
			Timeout:           20 * time.Second,
		}),
		grpc.ChainUnaryInterceptor(
			traceInterceptor,
			loggingInterceptor,
			recoveryInterceptor,
		),
	)
	RegisterAnchorServiceServer(s.grpcServer, s)
	reflection.Register(s.grpcServer)
	return s
}

func (s *Server) Start(ctx context.Context) error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		return fmt.Errorf("anchor grpc listen :%d: %w", s.port, err)
	}
	log.Printf("[grpc] anchor listening on :%d", s.port)

	errCh := make(chan error, 1)
	go func() {
		if err := s.grpcServer.Serve(lis); err != nil {
			errCh <- err
		}
	}()
	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		s.grpcServer.GracefulStop()
		return nil
	}
}

// ─── AnchorServiceServer implementation ──────────────────────────────────────

func (s *Server) SecureEmbedding(ctx context.Context, req *models.SecureRequest) (*models.SecureResponse, error) {
	if len(req.Embedding) == 0 {
		return nil, status.Error(codes.InvalidArgument, "embedding must not be empty")
	}
	inj := s.injectorFor(req.TenantID, req.Options)
	result := inj.Inject(req.Embedding)

	tc := traceFromCtx(ctx)
	tc.TenantID = req.TenantID
	tc.RequestID = req.RequestID
	go s.pub.Publish(publisher.EmbeddingSecured(tc, req.Operation, result.NoiseMagnitude, result.SimilarityToOriginal))
	s.hm.Fire(hooks.Event{
		Type:     hooks.EventEmbeddingSecured,
		TenantID: req.TenantID,
		TraceID:  tc.TraceID,
		SpanID:   tc.SpanID,
		Data: map[string]interface{}{
			"noise":      result.NoiseMagnitude,
			"similarity": result.SimilarityToOriginal,
			"operation":  req.Operation,
		},
	})

	return &models.SecureResponse{
		RequestID:        req.RequestID,
		SecuredEmbedding: result.Secured,
		Metrics: models.EmbedMetrics{
			NoiseAdded:           result.NoiseMagnitude,
			SimilarityToOriginal: result.SimilarityToOriginal,
			StrategyUsed:         result.Strategy,
			OriginalNorm:         result.OriginalNorm,
			SecuredNorm:          result.SecuredNorm,
		},
	}, nil
}

func (s *Server) SecureBatch(_ context.Context, req *models.BatchSecureRequest) (*models.BatchSecureResponse, error) {
	items := make([]models.SecureResponse, 0, len(req.Items))
	for _, item := range req.Items {
		inj := s.injectorFor(item.TenantID, item.Options)
		result := inj.Inject(item.Embedding)
		items = append(items, models.SecureResponse{
			RequestID:        item.RequestID,
			SecuredEmbedding: result.Secured,
			Metrics: models.EmbedMetrics{
				NoiseAdded:           result.NoiseMagnitude,
				SimilarityToOriginal: result.SimilarityToOriginal,
				StrategyUsed:         result.Strategy,
			},
		})
	}
	return &models.BatchSecureResponse{RequestID: req.RequestID, Items: items}, nil
}

func (s *Server) MeasureBias(ctx context.Context, req *models.BiasRequest) (*models.BiasResponse, error) {
	resp := s.ana.Analyze(req.TestWords, req.AttributePairs, req.WordEmbeddings)
	tc := traceFromCtx(ctx)
	tc.TenantID = req.TenantID
	for _, m := range resp.Measurements {
		if m.Severity == bias.SeverityHigh {
			go s.pub.Publish(publisher.BiasDetected(tc, m.Category, resp.OverallBiasScore, []string{m.Word}))
			s.hm.Fire(hooks.Event{
				Type:     hooks.EventBiasDetected,
				TenantID: req.TenantID,
				TraceID:  tc.TraceID,
				SpanID:   tc.SpanID,
				Data: map[string]interface{}{
					"category":   m.Category,
					"bias_score": resp.OverallBiasScore,
				},
			})
		}
	}
	return &resp, nil
}

func (s *Server) MeasureQuality(_ context.Context, req *models.QualityRequest) (*models.QualityResponse, error) {
	if req.NoiseSigma == 0 {
		req.NoiseSigma = s.cfg.Noise.DefaultSigma
	}
	resp := s.mon.Estimate(req.NoiseSigma, req.EmbeddingPairs)
	return &resp, nil
}

func (s *Server) GetConfig(_ context.Context, _ *models.ConfigRequest) (*models.AnchorConfig, error) {
	return &models.AnchorConfig{
		DefaultNoiseSigma: s.cfg.Noise.DefaultSigma,
		PreserveNorm:      s.cfg.Noise.PreserveNorm,
		Strategy:          s.cfg.Noise.Strategy,
		TenantOverrides:   s.cfg.Noise.TenantOverrides,
		BiasEnabled:       s.cfg.Bias.Enabled,
		QualityEnabled:    s.cfg.Quality.Enabled,
	}, nil
}

func (s *Server) UpdateConfig(_ context.Context, update *models.AnchorConfig) (*models.AnchorConfig, error) {
	if update.DefaultNoiseSigma > 0 {
		s.cfg.Noise.DefaultSigma = update.DefaultNoiseSigma
		s.inj.SetSigma(update.DefaultNoiseSigma)
	}
	if update.TenantOverrides != nil {
		s.cfg.Noise.TenantOverrides = update.TenantOverrides
	}
	return s.GetConfig(context.Background(), nil)
}

func (s *Server) Health(_ context.Context, _ *models.HealthRequest) (*models.HealthStatus, error) {
	return &models.HealthStatus{
		Status:  "ok",
		Version: "1.0.0",
		Checks:  map[string]string{"grpc": "up"},
	}, nil
}

// ─── Phase 2: Output verification ────────────────────────────────────────────

func (s *Server) VerifyResponse(ctx context.Context, req *models.VerifyRequest) (*models.VerifyResponse, error) {
	if req.LLMResponse == "" {
		return nil, status.Error(codes.InvalidArgument, "llm_response must not be empty")
	}
	resp := s.ver.Verify(req)
	tc := traceFromCtx(ctx)
	tc.TenantID = req.TenantID
	biasScore := 0.0
	if resp.BiasAnalysis != nil {
		biasScore = resp.BiasAnalysis.OverallScore
	}
	go s.pub.Publish(publisher.ResponseVerified(tc, string(resp.Status), biasScore))
	s.hm.Fire(hooks.Event{
		Type:     hooks.EventResponseVerified,
		TenantID: req.TenantID,
		TraceID:  tc.TraceID,
		SpanID:   tc.SpanID,
		Data:     map[string]interface{}{"status": string(resp.Status), "bias_score": biasScore},
	})
	return resp, nil
}

func (s *Server) AnalyzeResponse(ctx context.Context, req *models.VerifyRequest) (*models.VerifyResponse, error) {
	return s.VerifyResponse(ctx, req) // alias — same pipeline, same events
}

func (s *Server) AnalyzeResponseEmbedding(_ context.Context, req *models.EmbeddingAnalysisRequest) (*models.EmbeddingAnalysisResponse, error) {
	if req.Text == "" {
		return nil, status.Error(codes.InvalidArgument, "text must not be empty")
	}
	return s.ver.AnalyzeEmbedding(req), nil
}

func (s *Server) CheckDrift(_ context.Context, req *models.DriftRequest) (*models.DriftResponse, error) {
	return s.ver.CheckDrift(req), nil
}

func (s *Server) injectorFor(tenantID string, opts *models.SecurityOptions) *noise.Injector {
	sigma := s.cfg.SigmaForTenant(tenantID)
	preserveNorm := s.cfg.Noise.PreserveNorm
	if opts != nil && opts.NoiseSigma > 0 {
		sigma = opts.NoiseSigma
	}
	if opts != nil && opts.Seed != 0 {
		return noise.NewDeterministic(sigma, preserveNorm, opts.Seed)
	}
	inj, err := noise.New(sigma, preserveNorm, s.cfg.Noise.Strategy)
	if err != nil {
		return s.inj
	}
	return inj
}

// ─── Interceptors ─────────────────────────────────────────────────────────────

func loggingInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	start := time.Now()
	resp, err := handler(ctx, req)
	code := codes.OK
	if err != nil {
		code = status.Code(err)
	}
	log.Printf("[grpc] %s %s (%v)", info.FullMethod, code, time.Since(start))
	return resp, err
}

func recoveryInterceptor(ctx context.Context, req interface{}, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("[grpc] panic: %v", r)
			err = status.Errorf(codes.Internal, "internal server error")
		}
	}()
	return handler(ctx, req)
}
