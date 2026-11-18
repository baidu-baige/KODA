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
	"strings"
	"syscall"
	"time"

	"kernel-sandbox/pkg/monitor"
)

type targetFlags []string

func (values *targetFlags) String() string {
	return strings.Join(*values, ",")
}

func (values *targetFlags) Set(value string) error {
	*values = append(*values, value)
	return nil
}

func main() {
	if err := run(); err != nil {
		slog.Error("GPU pool monitor failed", "error", err)
		os.Exit(1)
	}
}

func run() error {
	listen := flag.String("listen", ":18780", "HTTP listen address")
	interval := flag.Duration("interval", 5*time.Second, "pool polling interval")
	history := flag.Duration("history", 72*time.Hour, "history window retained per pool")
	statePath := flag.String("state", "gpu-pool-monitor-state.json", "persistent state file")
	var rawTargets targetFlags
	flag.Var(&rawTargets, "target", "Master endpoint as name=url; repeat for multiple targets")
	flag.Parse()
	if len(rawTargets) == 0 {
		if value := os.Getenv("GPU_POOL_MONITOR_TARGETS"); value != "" {
			rawTargets = strings.Split(value, ",")
		} else {
			rawTargets = []string{"local=http://127.0.0.1:8767"}
		}
	}
	if *interval <= 0 || *history <= 0 {
		return fmt.Errorf("--interval and --history must be positive")
	}
	targets := make([]monitor.Target, 0, len(rawTargets))
	for _, value := range rawTargets {
		target, err := monitor.ParseTarget(value)
		if err != nil {
			return err
		}
		targets = append(targets, target)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	collector, err := monitor.NewCollectorWithHistory(targets, *interval, *history, *statePath)
	if err != nil {
		return err
	}
	go collector.Run(ctx)
	server := &http.Server{
		Addr:              *listen,
		Handler:           monitor.Handler(collector),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownContext); err != nil {
			slog.Warn("shutdown monitor HTTP server", "error", err)
		}
	}()
	slog.Info("GPU pool monitor listening", "address", *listen, "targets", targets)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
