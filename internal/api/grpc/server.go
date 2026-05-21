package grpc

import (
	"context"
	"fmt"
	"log"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"

	"github.com/bastion/anchor/internal/bias"
	"github.com/bastion/anchor/internal/config"
	"github.com/bastion/anchor/internal/models"
	"github.com/bastion/anchor/internal/noise"
	"github.com/bastion/anchor/internal/publisher"
	"github.com/bastion/anchor/internal/quality"
	"github.com/bastion/anchor/internal/verify"
)

// Server implements AnchorServiceServer.
type Server struct {
	UnimplementedAnchorServiceServer
	cfg        *config.Config
	inj        *noise.Injector
	ana        *bias.Analyzer
	mon        *quality.Monitor
	pub        publisher.Publisher
	ver        *verify.Verifier
	grpcServer *grpc.Server
	port       int
}

func New(
	cfg *config.Config,
	inj *noise.Injector,
	ana *bias.Analyzer,
	mon *quality.Monitor,
	pub publisher.Publisher,
	ver *verify.Verifier,
	port int,
) *Server {
	s := &Server{cfg: cfg, inj: inj, ana: ana, mon: mon, pub: pub, ver: ver, port: port}
	s.grpcServer = grpc.NewServer(
		grpc.MaxRecvMsgSize(16*1024*1024),
		grpc.MaxSendMsgSize(16*1024*1024),
		grpc.KeepaliveParams(keepalive.ServerParameters{
			MaxConnectionIdle: 5 * time.Minute,
			Time:              1 * time.Minute,
			Timeout:           20 * time.Second,
		}),
		grpc.ChainUnaryInterceptor(
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

func (s *Server) SecureEmbedding(_ context.Context, req *models.SecureRequest) (*models.SecureResponse, error) {
	if len(req.Embedding) == 0 {
		return nil, status.Error(codes.InvalidArgument, "embedding must not be empty")
	}
	inj := s.injectorFor(req.TenantID, req.Options)
	result := inj.Inject(req.Embedding)
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

func (s *Server) MeasureBias(_ context.Context, req *models.BiasRequest) (*models.BiasResponse, error) {
	resp := s.ana.Analyze(req.TestWords, req.AttributePairs, req.WordEmbeddings)
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

func (s *Server) VerifyResponse(_ context.Context, req *models.VerifyRequest) (*models.VerifyResponse, error) {
	if req.LLMResponse == "" {
		return nil, status.Error(codes.InvalidArgument, "llm_response must not be empty")
	}
	return s.ver.Verify(req), nil
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
