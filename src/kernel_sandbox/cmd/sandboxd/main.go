package main

import (
	"log/slog"
	"os"

	"kernel-sandbox/pkg/sandboxd"
)

func main() {
	if err := sandboxd.Run(os.Args[1:]); err != nil {
		slog.Error("sandboxd failed", "error", err)
		os.Exit(1)
	}
}
