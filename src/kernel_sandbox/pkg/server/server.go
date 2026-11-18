package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"time"

	"kernel-sandbox/pkg/command"
	"kernel-sandbox/pkg/config"
	"kernel-sandbox/pkg/master"
)

type Server struct {
	httpServer *http.Server
	master     *master.Master
	maxBody    int64
}

func New(cfg *config.Config, sandboxMaster *master.Master) *Server {
	server := &Server{master: sandboxMaster, maxBody: int64(cfg.Execution.MaxStdinMB<<20)*6 + 1<<20}
	server.httpServer = &http.Server{
		Addr:              cfg.Server.ListenAddress,
		Handler:           server.routes(),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	return server
}

func (s *Server) Start() error {
	slog.Info("kernel sandbox master listening", "address", s.httpServer.Addr)
	return s.httpServer.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/gpu_pool", func(writer http.ResponseWriter, request *http.Request) {
		var input master.CreateGPUPoolRequest
		if !s.decodeJSON(writer, request, &input) {
			return
		}
		response, serviceError := s.master.CreateGPUPool(request.Context(), input)
		s.writeResponse(writer, http.StatusAccepted, response, serviceError)
	})
	mux.HandleFunc("GET /v1/gpu_pool", func(writer http.ResponseWriter, _ *http.Request) {
		writeJSON(writer, http.StatusOK, s.master.ListGPUPools())
	})
	mux.HandleFunc("GET /v1/gpu_pool/{gpu_pool_id}", func(writer http.ResponseWriter, request *http.Request) {
		response, serviceError := s.master.GetGPUPool(request.PathValue("gpu_pool_id"))
		s.writeResponse(writer, http.StatusOK, response, serviceError)
	})
	mux.HandleFunc("PATCH /v1/gpu_pool/{gpu_pool_id}", func(writer http.ResponseWriter, request *http.Request) {
		var input master.UpdateGPUPoolRequest
		if !s.decodeJSON(writer, request, &input) {
			return
		}
		response, serviceError := s.master.UpdateGPUPool(request.Context(), request.PathValue("gpu_pool_id"), input)
		s.writeResponse(writer, http.StatusOK, response, serviceError)
	})
	mux.HandleFunc("DELETE /v1/gpu_pool/{gpu_pool_id}", func(writer http.ResponseWriter, request *http.Request) {
		s.writeResponse(writer, http.StatusAccepted, map[string]string{"status": "deletion_requested"}, s.master.DeleteGPUPool(request.Context(), request.PathValue("gpu_pool_id")))
	})
	mux.HandleFunc("GET /v1/gpu_pool/{gpu_pool_id}/sessions", func(writer http.ResponseWriter, request *http.Request) {
		writeJSON(writer, http.StatusOK, s.master.ListSessionsByGPUPool(request.PathValue("gpu_pool_id")))
	})
	mux.HandleFunc("DELETE /v1/gpu_pool/{gpu_pool_id}/sessions", func(writer http.ResponseWriter, request *http.Request) {
		s.writeResponse(writer, http.StatusAccepted, map[string]string{"status": "deletion_requested"}, s.master.DeleteSessionsByGPUPool(request.Context(), request.PathValue("gpu_pool_id")))
	})
	mux.HandleFunc("POST /v1/session", func(writer http.ResponseWriter, request *http.Request) {
		var input master.CreateSessionRequest
		if !s.decodeJSON(writer, request, &input) {
			return
		}
		response, serviceError := s.master.CreateSession(request.Context(), input)
		s.writeResponse(writer, http.StatusAccepted, response, serviceError)
	})
	mux.HandleFunc("GET /v1/session", func(writer http.ResponseWriter, _ *http.Request) {
		writeJSON(writer, http.StatusOK, s.master.ListSessions())
	})
	mux.HandleFunc("GET /v1/session/{session_id}", func(writer http.ResponseWriter, request *http.Request) {
		response, serviceError := s.master.GetSession(request.PathValue("session_id"))
		s.writeResponse(writer, http.StatusOK, response, serviceError)
	})
	mux.HandleFunc("DELETE /v1/session/{session_id}", func(writer http.ResponseWriter, request *http.Request) {
		s.writeResponse(writer, http.StatusAccepted, map[string]string{"status": "deletion_requested"}, s.master.DeleteSession(request.Context(), request.PathValue("session_id")))
	})
	mux.HandleFunc("DELETE /v1/session", func(writer http.ResponseWriter, request *http.Request) {
		query, err := url.ParseQuery(request.URL.RawQuery)
		if err != nil {
			writeJSON(writer, http.StatusBadRequest, errorResponse{ErrorCode: "BAD_REQUEST", Message: "invalid query parameters"})
			return
		}
		if query.Has("prefix") {
			s.writeResponse(writer, http.StatusAccepted, map[string]string{"status": "deletion_requested"}, s.master.DeleteSessionsByPrefix(request.Context(), query.Get("prefix")))
			return
		}
		s.writeResponse(writer, http.StatusAccepted, map[string]string{"status": "deletion_requested"}, s.master.DeleteAllSessions(request.Context()))
	})
	mux.HandleFunc("POST /v1/session/{session_id}/command_exec", func(writer http.ResponseWriter, request *http.Request) {
		var input command.Request
		if !s.decodeJSON(writer, request, &input) {
			return
		}
		startedAt := time.Now()
		sessionID := request.PathValue("session_id")
		response, serviceError := s.master.ExecuteCommand(request.Context(), sessionID, input)
		if serviceError != nil {
			status := statusCode(serviceError.Kind)
			level := slog.LevelInfo
			if status >= http.StatusInternalServerError {
				level = slog.LevelWarn
			}
			slog.Log(
				request.Context(),
				level,
				"Command request failed",
				"session_id", sessionID,
				"execution_id", serviceError.ExecutionID,
				"http_status", status,
				"error_code", serviceError.Code,
				"error", serviceError.Message,
				"duration", time.Since(startedAt),
			)
		}
		s.writeResponse(writer, http.StatusOK, response, serviceError)
	})
	mux.HandleFunc("GET /healthz", func(writer http.ResponseWriter, _ *http.Request) {
		writeJSON(writer, http.StatusOK, map[string]string{"status": "ok"})
	})
	return mux
}

func (s *Server) decodeJSON(writer http.ResponseWriter, request *http.Request, target any) bool {
	decoder := json.NewDecoder(http.MaxBytesReader(writer, request.Body, s.maxBody))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		writeJSON(writer, http.StatusBadRequest, errorResponse{ErrorCode: "INVALID_JSON", Message: err.Error()})
		return false
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		writeJSON(writer, http.StatusBadRequest, errorResponse{ErrorCode: "INVALID_JSON", Message: "request body must contain one JSON object"})
		return false
	}
	return true
}

func (s *Server) writeResponse(writer http.ResponseWriter, status int, response any, serviceError *master.ServiceError) {
	if serviceError != nil {
		writeJSON(writer, statusCode(serviceError.Kind), errorResponse{
			ErrorCode:   serviceError.Code,
			Message:     serviceError.Message,
			ExecutionID: serviceError.ExecutionID,
			Extra:       serviceError.Extra,
		})
		return
	}
	writeJSON(writer, status, response)
}

type errorResponse struct {
	ErrorCode   string                `json:"error_code"`
	Message     string                `json:"message"`
	ExecutionID string                `json:"execution_id,omitempty"`
	Extra       *master.ResponseExtra `json:"extra,omitempty"`
}

func statusCode(kind master.ErrorKind) int {
	switch kind {
	case master.ErrorInvalid:
		return http.StatusBadRequest
	case master.ErrorNotFound:
		return http.StatusNotFound
	case master.ErrorConflict:
		return http.StatusConflict
	case master.ErrorUnavailable:
		return http.StatusServiceUnavailable
	case master.ErrorTimeout:
		return http.StatusGatewayTimeout
	default:
		return http.StatusInternalServerError
	}
}

func writeJSON(writer http.ResponseWriter, status int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	if err := json.NewEncoder(writer).Encode(value); err != nil && !errors.Is(err, http.ErrHandlerTimeout) {
		slog.Warn("write HTTP response", "error", err)
	}
}
