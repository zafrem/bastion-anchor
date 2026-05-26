package rest

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/bastion/anchor/internal/bias"
	"github.com/bastion/anchor/internal/config"
	"github.com/bastion/anchor/internal/hooks"
	"github.com/bastion/anchor/internal/noise"
	"github.com/bastion/anchor/internal/publisher"
	"github.com/bastion/anchor/internal/quality"
	"github.com/bastion/anchor/internal/verify"
)

// Server is the HTTP REST server for Anchor.
type Server struct {
	httpServer *http.Server
	port       int
	hm         *hooks.Manager
}

// Hooks returns the HookManager so external coordinators can register listeners.
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
	hm := hooks.New()
	h := &handlers{cfg: cfg, inj: inj, ana: ana, mon: mon, pub: pub, ver: ver, hm: hm}
	s := &Server{port: port, hm: hm}
	s.httpServer = &http.Server{
		Addr:         fmt.Sprintf(":%d", port),
		Handler:      s.routes(h),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}
	return s
}

func (s *Server) routes(h *handlers) http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(10 * time.Second))

	r.Get("/v1/health", h.Health)
	r.Get("/v1/health/live", h.Live)
	r.Get("/v1/health/ready", h.Ready)
	r.Handle("/v1/metrics", promhttp.Handler())

	r.Post("/v1/anchor/secure", h.Secure)
	r.Post("/v1/anchor/secure/batch", h.SecureBatch)
	r.Post("/v1/anchor/measure-bias", h.MeasureBias)
	r.Post("/v1/anchor/measure-quality", h.MeasureQuality)
	r.Get("/v1/anchor/config", h.GetConfig)
	r.Post("/v1/anchor/config", h.UpdateConfig)

	// Phase 2: Output verification
	r.Post("/v1/anchor/verify", h.Verify)
	r.Post("/v1/anchor/analyze", h.Verify) // SRS alias: POST /v1/anchor/analyze → Verify
	r.Post("/v1/anchor/analyze-embedding", h.AnalyzeEmbedding)
	r.Post("/v1/anchor/check-drift", h.CheckDrift)

	return r
}

func (s *Server) Start(ctx context.Context) error {
	log.Printf("[rest] anchor listening on :%d", s.port)
	errCh := make(chan error, 1)
	go func() {
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()
	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		shutCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		return s.httpServer.Shutdown(shutCtx)
	}
}
