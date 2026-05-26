package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	grpcserver "github.com/bastion/anchor/internal/api/grpc"
	"github.com/bastion/anchor/internal/api/rest"
	"github.com/bastion/anchor/internal/bias"
	"github.com/bastion/anchor/internal/cli"
	"github.com/bastion/anchor/internal/config"
	"github.com/bastion/anchor/internal/noise"
	"github.com/bastion/anchor/internal/publisher"
	"github.com/bastion/anchor/internal/quality"
	"github.com/bastion/anchor/internal/verify"
)

var cfgPath string

func main() {
	root := &cobra.Command{
		Use:   "anchor-cli",
		Short: "Bastion-Anchor — embedding security layer",
	}
	root.PersistentFlags().StringVar(&cfgPath, "config", "/etc/bastion-anchor/config.yaml", "Config file path")

	// server sub-command
	var restPort, grpcPort int
	serverCmd := &cobra.Command{
		Use:   "server",
		Short: "Start the Anchor API server (REST + gRPC)",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runServer(restPort, grpcPort)
		},
	}
	serverCmd.Flags().IntVar(&restPort, "port", 8080, "REST API port")
	serverCmd.Flags().IntVar(&grpcPort, "grpc-port", 9090, "gRPC port")
	root.AddCommand(serverCmd)

	cfg, inj, ana, mon, ver := buildComponents()

	for _, sub := range cli.Build(cfg, inj, ana, mon, ver).Commands() {
		root.AddCommand(sub)
	}

	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

func runServer(restPort, grpcPort int) error {
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Printf("config load failed (%v), using defaults", err)
		cfg = config.Defaults()
	}
	if restPort != 0 {
		cfg.Server.RESTPort = restPort
	}
	if grpcPort != 0 {
		cfg.Server.GRPCPort = grpcPort
	}

	fmt.Println("Bastion-Anchor v1.0 starting...")

	inj, err := noise.New(cfg.Noise.DefaultSigma, cfg.Noise.PreserveNorm, cfg.Noise.Strategy)
	if err != nil {
		return fmt.Errorf("noise injector: %w", err)
	}
	ana := bias.New(cfg.Bias.AlertMedium, cfg.Bias.AlertHigh, cfg.Embedding.ExpectedDimensions)
	mon := quality.New(cfg.Quality.RecallDropAlert, cfg.Quality.SimilarityMinimum)
	ver := verify.New(cfg.Bias.AlertMedium, cfg.Bias.AlertHigh)

	var pub publisher.Publisher
	if cfg.Tracker.Enabled && cfg.Tracker.NATSUrl != "" {
		natsPub, err := publisher.NewNATSWithSubject(cfg.Tracker.NATSUrl, cfg.Tracker.EventSubject)
		if err != nil {
			log.Printf("NATS unavailable (%v), using log publisher", err)
			pub = publisher.NewLog()
		} else {
			pub = natsPub
			defer natsPub.Close()
		}
	} else {
		pub = publisher.NewLog()
	}

	fmt.Printf("Noise injector ready (sigma=%.4f, strategy=%s)\n", cfg.Noise.DefaultSigma, cfg.Noise.Strategy)
	fmt.Printf("REST API on :%d\n", cfg.Server.RESTPort)
	fmt.Printf("gRPC API on :%d\n", cfg.Server.GRPCPort)
	fmt.Println("Ready")

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	restSrv := rest.New(cfg, inj, ana, mon, pub, ver, cfg.Server.RESTPort)
	grpcSrv := grpcserver.New(cfg, inj, ana, mon, pub, ver, cfg.Server.GRPCPort)

	errCh := make(chan error, 2)
	go func() { errCh <- restSrv.Start(ctx) }()
	go func() { errCh <- grpcSrv.Start(ctx) }()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		return nil
	}
}

func buildComponents() (*config.Config, *noise.Injector, *bias.Analyzer, *quality.Monitor, *verify.Verifier) {
	cfg, err := config.Load(cfgPath)
	if err != nil {
		cfg = config.Defaults()
	}
	inj, err := noise.New(cfg.Noise.DefaultSigma, cfg.Noise.PreserveNorm, cfg.Noise.Strategy)
	if err != nil {
		inj = noise.NewDeterministic(0.01, true, 0)
	}
	ana := bias.New(cfg.Bias.AlertMedium, cfg.Bias.AlertHigh, cfg.Embedding.ExpectedDimensions)
	mon := quality.New(cfg.Quality.RecallDropAlert, cfg.Quality.SimilarityMinimum)
	ver := verify.New(cfg.Bias.AlertMedium, cfg.Bias.AlertHigh)
	return cfg, inj, ana, mon, ver
}
