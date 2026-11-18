package server

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"kernel-sandbox/pkg/command"
	"kernel-sandbox/pkg/config"
	"kernel-sandbox/pkg/master"
)

type fakeKubernetes struct {
	mu         sync.Mutex
	id         string
	image      string
	replicas   int32
	cpuPod     master.PodInfo
	cpuResult  command.Result
	scheduling *master.Scheduling
}

func (f *fakeKubernetes) CreateCPUPod(_ context.Context, sessionID, gpuPoolID, image, _ string, resources *master.ResourceRequirements, prefix string) (master.PodInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cpuPod = master.PodInfo{Name: "cpu-" + sessionID, UID: "cpu-uid", SessionID: sessionID, GPUPoolID: gpuPoolID, Image: image, Resources: resources, Ready: true}
	if prefix != "" {
		f.cpuPod.Name = prefix + "-" + sessionID
	}
	return f.cpuPod, nil
}
func (f *fakeKubernetes) DeleteCPUPod(context.Context, master.PodInfo) error {
	f.mu.Lock()
	f.cpuPod = master.PodInfo{}
	f.mu.Unlock()
	return nil
}
func (f *fakeKubernetes) ListCPUPods(context.Context) ([]master.PodInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.cpuPod.Name == "" {
		return nil, nil
	}
	return []master.PodInfo{f.cpuPod}, nil
}
func (f *fakeKubernetes) CreateGPUPool(_ context.Context, gpuPoolID, image string, replicas int32, _ *master.ResourceRequirements, scheduling *master.Scheduling) error {
	f.id, f.image, f.replicas = gpuPoolID, image, replicas
	f.scheduling = scheduling
	return nil
}
func (f *fakeKubernetes) ListGPUPools(context.Context) ([]master.GPUPoolInfo, error) {
	if f.id == "" {
		return nil, nil
	}
	return []master.GPUPoolInfo{{GPUPoolID: f.id, Image: f.image, Replicas: f.replicas}}, nil
}
func (f *fakeKubernetes) UpdateGPUPool(_ context.Context, _ string, image string, replicas *int32, scheduling *master.Scheduling) error {
	if image != "" {
		f.image = image
	}
	if replicas != nil {
		f.replicas = *replicas
	}
	f.scheduling = scheduling
	return nil
}
func (f *fakeKubernetes) DeleteGPUPool(context.Context, string) error { return nil }
func (f *fakeKubernetes) ListGPUPods(context.Context) ([]master.PodInfo, error) {
	return nil, nil
}
func (f *fakeKubernetes) DeleteGPUWorkerPod(context.Context, master.PodInfo) error  { return nil }
func (f *fakeKubernetes) DeleteGPUSessionPod(context.Context, master.PodInfo) error { return nil }
func (f *fakeKubernetes) CreateGPUSessionPod(_ context.Context, sessionID, image string, resources *master.ResourceRequirements, _ string) (master.PodInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return master.PodInfo{Name: "gpu-pool-" + sessionID, UID: "gpu-uid-" + sessionID, SessionID: sessionID, Image: image, Resources: resources, Ready: true}, nil
}
func (f *fakeKubernetes) ListGPUSessionPods(context.Context) ([]master.PodInfo, error) {
	return nil, nil
}
func (f *fakeKubernetes) ExecuteCPU(_ context.Context, _ master.PodInfo, request command.Request, _ int) command.Result {
	if len(request.Command) == 3 && request.Command[0] == "python3" && strings.Contains(request.Command[2], "KERNEL_SANDBOX_URL") {
		exitCode := 0
		return command.Result{Kind: command.ResultCompleted, ExitCode: &exitCode}
	}
	if f.cpuResult.Kind != "" {
		return f.cpuResult
	}
	return command.Result{}
}
func (f *fakeKubernetes) ExecuteGPU(context.Context, master.PodInfo, command.Request, int) command.Result {
	return command.Result{}
}
func (f *fakeKubernetes) ExecuteGPUSession(context.Context, master.PodInfo, command.Request, int) command.Result {
	return command.Result{}
}
func (f *fakeKubernetes) ReadGPUPodState(context.Context, master.PodInfo) (master.GPUPodState, error) {
	return master.GPUPodState{}, nil
}

func testServer(kubernetes *fakeKubernetes) *Server {
	cfg := &config.Config{MasterID: "test", Server: config.ServerConfig{ListenAddress: ":0"}, GPUPoolCluster: config.GPUPoolClusterConfig{PoolNamePrefix: "gpu-pool"}, Execution: config.ExecutionConfig{MaxStdinMB: 1}}
	return New(cfg, master.New(cfg, kubernetes, nil, nil))
}

func TestPoolSchedulingAndSessionPrefixDeletion(t *testing.T) {
	kube := &fakeKubernetes{}
	server := testServer(kube)
	call := func(method, path, body string) *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		server.httpServer.Handler.ServeHTTP(recorder, httptest.NewRequest(method, path, strings.NewReader(body)))
		return recorder
	}
	pool := call(http.MethodPost, "/v1/gpu_pool", `{"image":"image","replicas":1,"node_selector":{"tenant":"test"},"tolerations":[{"key":"tenant","operator":"Exists","effect":"NoExecute","toleration_seconds":300}]}`)
	if pool.Code != http.StatusAccepted || !strings.HasPrefix(kube.id, "gpu-pool-") || kube.scheduling.NodeSelector["tenant"] != "test" || *kube.scheduling.Tolerations[0].TolerationSeconds != 300 {
		t.Fatalf("pool: %d %s", pool.Code, pool.Body.String())
	}
	response := call(http.MethodPost, "/v1/gpu_pool", `{"image":"image","replicas":1,"prefix":"tenant"}`)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "INVALID_JSON") {
		t.Fatal("GPU Pool accepted a request prefix")
	}
	response = call(http.MethodPatch, "/v1/gpu_pool/"+kube.id, `{"node_selector":{},"tolerations":[]}`)
	if response.Code != http.StatusOK || kube.scheduling.NodeSelector == nil || kube.scheduling.Tolerations == nil {
		t.Fatalf("pool scheduling clear: %d %s", response.Code, response.Body.String())
	}
	for _, fields := range []string{`"node_selector":{}`, `"tolerations":[]`} {
		body := `{"session_type":"cpu_gpu","gpu_pool_id":"` + kube.id + `",` + fields + `}`
		response := call(http.MethodPost, "/v1/session", body)
		if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "INVALID_JSON") {
			t.Fatalf("Session accepted scheduling overrides: %s", response.Body.String())
		}
	}
	response = call(http.MethodPost, "/v1/session", `{"session_type":"cpu_gpu","gpu_pool_id":"`+kube.id+`","prefix":"BAD_PREFIX"}`)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "BAD_REQUEST") {
		t.Fatal("Session accepted an invalid prefix")
	}
	response = call(http.MethodPost, "/v1/session", `{"session_type":"cpu_gpu","gpu_pool_id":"`+kube.id+`","prefix":"tenant"}`)
	if response.Code != http.StatusAccepted || !strings.HasPrefix(kube.cpuPod.Name, "tenant-") {
		t.Fatalf("session: %d %s", response.Code, response.Body.String())
	}
	for _, test := range []struct {
		query             string
		status, remaining int
	}{{"?prefix=", 400, 1}, {"?prefix=%zz", 400, 1}, {"?prefix=BAD_PREFIX", 400, 1}, {"?prefix=missing", 202, 1}, {"?prefix=tenant", 202, 0}, {"?prefix=tenant", 202, 0}} {
		response := call(http.MethodDelete, "/v1/session"+test.query, "")
		if response.Code != test.status || len(server.master.ListSessions()) != test.remaining {
			t.Fatalf("delete %s: %d %s", test.query, response.Code, response.Body.String())
		}
	}
}

func captureLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var output bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&output, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return &output
}

func TestCreateGPUPool(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	server := testServer(kubernetes)
	request := httptest.NewRequest(http.MethodPost, "/v1/gpu_pool", strings.NewReader(`{"image":"example","replicas":2}`))
	recorder := httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusAccepted || !strings.HasPrefix(kubernetes.id, "gpu-pool-") || kubernetes.image != "example" || kubernetes.replicas != 2 || !strings.Contains(recorder.Body.String(), `"timing":{"create_ms":`) || strings.Contains(recorder.Body.String(), `"ready_ms"`) {
		t.Fatalf("status=%d image=%q replicas=%d body=%s", recorder.Code, kubernetes.image, kubernetes.replicas, recorder.Body.String())
	}
	for _, path := range []string{"/v1/gpu_pool", "/v1/gpu_pool/" + kubernetes.id} {
		request = httptest.NewRequest(http.MethodGet, path, nil)
		recorder = httptest.NewRecorder()
		server.httpServer.Handler.ServeHTTP(recorder, request)
		if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), kubernetes.id) {
			t.Fatalf("path=%s status=%d body=%s", path, recorder.Code, recorder.Body.String())
		}
	}
}

func TestUpdateGPUPool(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	server := testServer(kubernetes)
	request := httptest.NewRequest(http.MethodPost, "/v1/gpu_pool", strings.NewReader(`{"image":"example","replicas":1}`))
	recorder := httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(recorder, request)
	if recorder.Code >= http.StatusBadRequest {
		t.Fatalf("create status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	request = httptest.NewRequest(http.MethodPatch, "/v1/gpu_pool/"+kubernetes.id, strings.NewReader(`{"image":"updated","replicas":4}`))
	recorder = httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(recorder, request)
	if recorder.Code >= http.StatusBadRequest {
		t.Fatalf("update status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if kubernetes.image != "updated" || kubernetes.replicas != 4 {
		t.Fatalf("image=%q replicas=%d", kubernetes.image, kubernetes.replicas)
	}
}

func TestRejectsUnknownField(t *testing.T) {
	server := testServer(&fakeKubernetes{})
	request := httptest.NewRequest(http.MethodPost, "/v1/gpu_pool", strings.NewReader(`{"image":"example","replicas":1,"unknown":true}`))
	recorder := httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status=%d", recorder.Code)
	}
}

func TestRejectsMultipleJSONValues(t *testing.T) {
	server := testServer(&fakeKubernetes{})
	request := httptest.NewRequest(http.MethodPost, "/v1/gpu_pool", strings.NewReader(`{"image":"example","replicas":1} {}`))
	recorder := httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status=%d", recorder.Code)
	}
}

func TestErrorResponseIncludesSandboxTimings(t *testing.T) {
	queueMS := int64(7)
	server := testServer(&fakeKubernetes{})
	recorder := httptest.NewRecorder()
	server.writeResponse(recorder, http.StatusOK, nil, &master.ServiceError{
		Kind:        master.ErrorUnavailable,
		Code:        "INFRASTRUCTURE_ERROR",
		Message:     "GPU infrastructure retries exhausted",
		ExecutionID: "execution-test",
		Extra:       &master.ResponseExtra{SandboxTimings: &master.SandboxTimings{GPUExecutions: []*master.ExecutionTimingResponse{{TotalMS: 10, QueueMS: &queueMS, ExecMS: 3, Attempts: 2}}}},
	})
	if recorder.Code != http.StatusServiceUnavailable || !strings.Contains(recorder.Body.String(), `"extra":{"sandbox_timings":{"gpu_executions":[{"total_ms":10,"queue_ms":7,"exec_ms":3,"attempts":2}]}}`) {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
}

func TestCommandExecLogsServiceUnavailable(t *testing.T) {
	logs := captureLogs(t)
	server := testServer(&fakeKubernetes{cpuResult: command.Result{Kind: command.ResultInfrastructure, Error: "exec failed"}})
	pool, serviceError := server.master.CreateGPUPool(context.Background(), master.CreateGPUPoolRequest{Image: "image", Replicas: 1})
	if serviceError != nil {
		t.Fatal(serviceError)
	}
	session, serviceError := server.master.CreateSession(context.Background(), master.CreateSessionRequest{SessionType: master.SessionTypeCPUGPU, GPUPoolID: pool.GPUPoolID, Image: "cpu-image"})
	if serviceError != nil {
		t.Fatal(serviceError)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	server.master.Start(ctx)
	for deadline := time.Now().Add(time.Second); ; time.Sleep(time.Millisecond) {
		current, serviceError := server.master.GetSession(session.SessionID)
		if serviceError != nil {
			t.Fatal(serviceError)
		}
		if current.Status == master.SessionStatusReady {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("session did not become ready: %+v", current)
		}
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/session/"+session.SessionID+"/command_exec", strings.NewReader(`{"device":"cpu","command":["true"],"timeout_seconds":1}`))
	recorder := httptest.NewRecorder()
	server.httpServer.Handler.ServeHTTP(recorder, request)
	output := logs.String()
	for _, expected := range []string{
		`msg="Command accepted"`,
		`msg="Command finished"`,
		"error_code=INFRASTRUCTURE_ERROR",
		`error="exec failed"`,
		`msg="Command request failed"`,
		"http_status=503",
	} {
		if !strings.Contains(output, expected) {
			t.Fatalf("missing %q in logs=%s", expected, output)
		}
	}
	if recorder.Code != http.StatusServiceUnavailable || !strings.Contains(recorder.Body.String(), `"message":"exec failed"`) {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
}

func TestCPUExecutionTimingOmitsGPUFields(t *testing.T) {
	server := testServer(&fakeKubernetes{})
	recorder := httptest.NewRecorder()
	server.writeResponse(recorder, http.StatusOK, master.ExecuteCommandResponse{
		ExecutionID: "execution-test",
		Status:      master.ExecutionStatusCompleted,
		Extra:       &master.ResponseExtra{SandboxTimings: &master.SandboxTimings{CPUExecution: &master.ExecutionTimingResponse{TotalMS: 10, ExecMS: 8, Attempts: 1}}},
	}, nil)
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), `"extra":{"sandbox_timings":{"cpu_execution":{"total_ms":10,"exec_ms":8,"attempts":1}}}`) || strings.Contains(recorder.Body.String(), `"queue_ms"`) || strings.Contains(recorder.Body.String(), `"precheck_ms"`) || strings.Contains(recorder.Body.String(), `"gpu_executions":`) {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
}

func TestCPUExecutionIncludesTriggeredGPUTimings(t *testing.T) {
	queueMS := int64(7)
	server := testServer(&fakeKubernetes{})
	recorder := httptest.NewRecorder()
	server.writeResponse(recorder, http.StatusOK, master.ExecuteCommandResponse{
		ExecutionID: "execution-cpu",
		Status:      master.ExecutionStatusCompleted,
		Extra: &master.ResponseExtra{SandboxTimings: &master.SandboxTimings{
			CPUExecution:  &master.ExecutionTimingResponse{TotalMS: 20, ExecMS: 15, Attempts: 1},
			GPUExecutions: []*master.ExecutionTimingResponse{{TotalMS: 10, QueueMS: &queueMS, ExecMS: 3, Attempts: 1}},
		}},
	}, nil)
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), `"extra":{"sandbox_timings":{"cpu_execution":{"total_ms":20,"exec_ms":15,"attempts":1},"gpu_executions":[{"total_ms":10,"queue_ms":7,"exec_ms":3,"attempts":1}]}}`) {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
}
