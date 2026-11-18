package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"kernel-sandbox/pkg/config"
	"kernel-sandbox/pkg/docker"
	"kernel-sandbox/pkg/kubernetes"
	"kernel-sandbox/pkg/master"
	"kernel-sandbox/pkg/server"
)

func main() {
	if err := run(); err != nil {
		slog.Error("kernel sandbox master failed", "error", err)
		os.Exit(1)
	}
}

func run() error {
	configPath := flag.String("config", "", "path to the master config file")
	flag.Parse()
	if *configPath == "" {
		return fmt.Errorf("--config is required")
	}
	cfg, err := config.Load(*configPath)
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	var kubernetesClient master.KubernetesClient
	if cfg.Backend == "docker" {
		client, e := docker.New(cfg)
		if e != nil {
			return fmt.Errorf("create Docker backend: %w", e)
		}
		defer client.Close()
		kubernetesClient = client
	} else {
		client, e := kubernetes.New(cfg)
		if e != nil {
			return fmt.Errorf("create Kubernetes clients: %w", e)
		}
		kubernetesClient = client
	}
	gpuSessionClients := map[string]master.GPUSessionKubernetesClient{}
	for clusterName, clusterConfig := range cfg.GPUSessionClusters {
		client, err := kubernetes.NewGPUSessionClient(cfg, clusterConfig)
		if err != nil {
			return fmt.Errorf("create GPU Session Kubernetes client for cluster %q: %w", clusterName, err)
		}
		gpuSessionClients[clusterName] = client
	}
	gpuPoolClients := map[string]master.GPUPoolKubernetesClient{}
	for _, clusterConfig := range cfg.GPUPoolClusters {
		client, err := kubernetes.NewGPUPoolClient(cfg, clusterConfig)
		if err != nil {
			return fmt.Errorf("create GPU Pool Kubernetes client for cluster %q: %w", clusterConfig.Name, err)
		}
		gpuPoolClients[clusterConfig.Name] = client
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	sandboxMaster := master.New(cfg, kubernetesClient, gpuSessionClients, gpuPoolClients)
	sandboxMaster.Start(ctx)
	httpServer := server.New(cfg, sandboxMaster)
	go func() {
		<-ctx.Done()
		shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(shutdownContext); err != nil {
			slog.Warn("shutdown HTTP server", "error", err)
		}
	}()
	if err := httpServer.Start(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
