package monitor

import (
	"embed"
	"encoding/json"
	"io/fs"
	"net/http"
)

//go:embed web/*
var webFiles embed.FS

func Handler(collector *Collector) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/v1/snapshot", func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Cache-Control", "no-store")
		writer.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(writer).Encode(collector.Snapshot())
	})
	mux.HandleFunc("GET /healthz", func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"status":"ok"}`))
	})
	root, _ := fs.Sub(webFiles, "web")
	mux.Handle("/", http.FileServer(http.FS(root)))
	return mux
}
