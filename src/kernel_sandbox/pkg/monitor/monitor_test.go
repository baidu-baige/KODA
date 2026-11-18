package monitor

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func intPointer(value int) *int {
	return &value
}

func TestCollectorTracksPoolUtilizationAndCounterReset(t *testing.T) {
	var mu sync.Mutex
	pools := []poolResponse{{
		GPUPoolID: "pool-a", Cluster: "h20", Image: "image", Replicas: 2,
		Workers: WorkerCounts{Idle: 2},
		Usage: &poolUsage{BusyMS: 100, CapacityMS: 200, Workers: map[string]workerUsage{
			"worker-a": {BusyMS: 50, CapacityMS: 100},
			"worker-b": {BusyMS: 50, CapacityMS: 100},
		}, Queue: &queueUsage{Count: 4, TotalMS: 400, MaxMS: 900}, QueueLength: intPointer(1)},
	}}
	collector := NewCollector([]Target{{Name: "dev", URL: "http://master"}}, time.Second, 0)
	collector.client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path == "/v1/session" {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body: io.NopCloser(strings.NewReader(`[
					{"session_id":"session-1","gpu_pool_id":"pool-a","status":"ready","cpu":{"pod":"team-a-session-1"}},
					{"session_id":"session-2","gpu_pool_id":"pool-a","status":"ready","cpu":{"pod":"team-a-session-2"}},
					{"session_id":"session-3","gpu_pool_id":"pool-a","status":"creating","cpu":{"pod":"team-b-session-3"}}
				]`)),
			}, nil
		}
		if request.URL.Path != "/v1/gpu_pool" {
			return &http.Response{StatusCode: http.StatusNotFound, Body: io.NopCloser(strings.NewReader(""))}, nil
		}
		mu.Lock()
		defer mu.Unlock()
		var body bytes.Buffer
		_ = json.NewEncoder(&body).Encode(pools)
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(&body)}, nil
	})}
	collector.Poll(context.Background())
	first := collector.Snapshot()
	if len(first.Pools) != 1 || first.Pools[0].Cluster != "h20" || first.Pools[0].Utilization != nil {
		t.Fatalf("unexpected first snapshot: %+v", first)
	}
	if first.Pools[0].LifetimeUtilization == nil || *first.Pools[0].LifetimeUtilization != 0.5 {
		t.Fatalf("unexpected lifetime utilization: %+v", first.Pools[0].LifetimeUtilization)
	}
	if first.Pools[0].Sessions.Total != 3 || first.Pools[0].Sessions.Ready != 2 || first.Pools[0].Sessions.Creating != 1 {
		t.Fatalf("unexpected session counts: %+v", first.Pools[0].Sessions)
	}
	if first.Summary.Sessions != 3 || first.Summary.SessionsReady != 2 || first.Summary.SessionsCreating != 1 {
		t.Fatalf("unexpected summary session counts: %+v", first.Summary)
	}
	if first.Pools[0].SessionPrefixes["team-a"] != 2 || first.Pools[0].SessionPrefixes["team-b"] != 1 {
		t.Fatalf("unexpected session prefix counts: %+v", first.Pools[0].SessionPrefixes)
	}

	mu.Lock()
	pools[0].Workers = WorkerCounts{Busy: 1, Idle: 1}
	pools[0].Usage = &poolUsage{BusyMS: 200, CapacityMS: 400, Workers: map[string]workerUsage{
		"worker-a": {BusyMS: 150, CapacityMS: 200},
		"worker-b": {BusyMS: 50, CapacityMS: 200},
	}, Queue: &queueUsage{Count: 7, TotalMS: 1000, MaxMS: 1200}, QueueLength: intPointer(2)}
	mu.Unlock()
	collector.Poll(context.Background())
	second := collector.Snapshot()
	if second.Pools[0].Utilization == nil || *second.Pools[0].Utilization != 0.5 {
		t.Fatalf("unexpected interval utilization: %+v", second.Pools[0].Utilization)
	}
	if second.Summary.Utilization == nil || *second.Summary.Utilization != 0.5 || len(second.Pools[0].History) != 1 {
		t.Fatalf("unexpected summary/history: %+v", second)
	}
	if !second.Pools[0].QueueSupported || second.Pools[0].QueueCount != 3 || second.Pools[0].QueueAvgMS == nil || *second.Pools[0].QueueAvgMS != 200 || second.Pools[0].QueueMaxMS != 1200 || second.Pools[0].QueueLength != 2 {
		t.Fatalf("unexpected pool queue metrics: %+v", second.Pools[0])
	}
	if !second.Summary.QueueSupported || second.Summary.Queued != 2 || second.Summary.QueueAvgMS == nil || *second.Summary.QueueAvgMS != 200 || second.Summary.QueueMaxMS != 1200 || second.Summary.QueueRate == nil || *second.Summary.QueueRate != 180 {
		t.Fatalf("unexpected summary queue metrics: %+v", second.Summary)
	}

	mu.Lock()
	pools[0].Usage = &poolUsage{BusyMS: 300, CapacityMS: 600, Workers: map[string]workerUsage{
		"worker-a": {BusyMS: 250, CapacityMS: 300},
		"worker-b": {BusyMS: 50, CapacityMS: 300},
	}, Queue: &queueUsage{Count: 8, TotalMS: 1300, MaxMS: 1400}, QueueLength: intPointer(1)}
	mu.Unlock()
	collector.Poll(context.Background())
	third := collector.Snapshot()
	if len(third.Pools[0].History) != 2 {
		t.Fatalf("unlimited history was truncated: %+v", third.Pools[0].History)
	}

	mu.Lock()
	pools[0].Usage = &poolUsage{BusyMS: 10, CapacityMS: 20, Workers: map[string]workerUsage{
		"worker-a": {BusyMS: 10, CapacityMS: 20},
	}}
	mu.Unlock()
	collector.Poll(context.Background())
	reset := collector.Snapshot()
	if reset.Pools[0].Utilization != nil || reset.Summary.Utilization != nil {
		t.Fatalf("counter reset should suppress interval utilization: %+v", reset)
	}
}

func TestCollectorMarksMissingQueueMetricsUnsupported(t *testing.T) {
	collector := NewCollector([]Target{{Name: "legacy", URL: "http://master"}}, time.Second, 0)
	collector.client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body := `[]`
		if request.URL.Path == "/v1/gpu_pool" {
			body = `[{"gpu_pool_id":"pool-a","replicas":1,"workers":{"idle":1},"usage":{"busy_ms":0,"capacity_ms":100,"workers":{}}}]`
		}
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body))}, nil
	})}
	collector.Poll(context.Background())
	snapshot := collector.Snapshot()
	if len(snapshot.Pools) != 1 || snapshot.Pools[0].QueueSupported || snapshot.Summary.QueueSupported {
		t.Fatalf("legacy master queue metrics should be unsupported: %+v", snapshot)
	}
}

func TestCollectorReportsTargetFailureWithoutDroppingHealthyTargets(t *testing.T) {
	collector := NewCollector([]Target{
		{Name: "healthy", URL: "http://healthy"},
		{Name: "broken", URL: "http://broken"},
	}, time.Second, 10)
	collector.client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Host == "broken" {
			return nil, context.DeadlineExceeded
		}
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`[]`))}, nil
	})}
	collector.Poll(context.Background())
	snapshot := collector.Snapshot()
	if len(snapshot.Targets) != 2 || snapshot.Summary.Unhealthy != 1 || !snapshot.Targets[0].Healthy || snapshot.Targets[1].Healthy {
		t.Fatalf("unexpected target status: %+v", snapshot.Targets)
	}
}

func TestCollectorPersistsHistoryAndCounterBaseline(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "state.json")
	first, err := NewCollectorWithState(nil, time.Second, 0, statePath)
	if err != nil {
		t.Fatal(err)
	}
	first.previous["dev/pool"] = poolUsage{
		BusyMS:     100,
		CapacityMS: 200,
		Workers:    map[string]workerUsage{"worker": {BusyMS: 100, CapacityMS: 200}},
	}
	first.history["dev/pool"] = []Point{{Timestamp: time.Unix(1, 0), Utilization: 0.5}}
	if err := first.persistState(); err != nil {
		t.Fatal(err)
	}

	second, err := NewCollectorWithState(nil, time.Second, 0, statePath)
	if err != nil {
		t.Fatal(err)
	}
	if second.previous["dev/pool"].Workers["worker"].BusyMS != 100 {
		t.Fatalf("counter baseline was not restored: %+v", second.previous)
	}
	if len(second.history["dev/pool"]) != 1 || second.history["dev/pool"][0].Utilization != 0.5 {
		t.Fatalf("history was not restored: %+v", second.history)
	}
}

func TestCollectorTrimsHistoryByWindow(t *testing.T) {
	collector, err := NewCollectorWithHistory(nil, time.Second, 72*time.Hour, "")
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	collector.history["pool"] = []Point{
		{Timestamp: now.Add(-73 * time.Hour), Utilization: 0.1},
		{Timestamp: now.Add(-71 * time.Hour), Utilization: 0.2},
		{Timestamp: now.Add(-time.Hour), Utilization: 0.3},
	}
	collector.trimHistory(now)
	if len(collector.history["pool"]) != 2 || collector.history["pool"][0].Utilization != 0.2 {
		t.Fatalf("history window was not applied: %+v", collector.history["pool"])
	}
}

func TestCollectorCleansUpRemovedPoolStateAndRetainsOnTargetFailure(t *testing.T) {
	var mu sync.Mutex
	unhealthy := false
	pools := []poolResponse{{
		GPUPoolID: "pool-a", Cluster: "h20", Image: "image", Replicas: 1,
		Workers: WorkerCounts{Idle: 1},
		Usage: &poolUsage{BusyMS: 100, CapacityMS: 200, Workers: map[string]workerUsage{
			"worker-a": {BusyMS: 100, CapacityMS: 200},
		}, Queue: &queueUsage{}, QueueLength: intPointer(0)},
	}}
	collector := NewCollector([]Target{{Name: "dev", URL: "http://master"}}, time.Second, 0)
	collector.client = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path == "/v1/session" {
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`[]`))}, nil
		}
		mu.Lock()
		defer mu.Unlock()
		if unhealthy {
			return nil, context.DeadlineExceeded
		}
		var body bytes.Buffer
		_ = json.NewEncoder(&body).Encode(pools)
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(&body)}, nil
	})}

	collector.Poll(context.Background())
	mu.Lock()
	pools[0].Usage = &poolUsage{BusyMS: 200, CapacityMS: 400, Workers: map[string]workerUsage{
		"worker-a": {BusyMS: 200, CapacityMS: 400},
	}, Queue: &queueUsage{}, QueueLength: intPointer(0)}
	mu.Unlock()
	collector.Poll(context.Background())
	if _, ok := collector.previous["dev/pool-a"]; !ok {
		t.Fatalf("expected previous entry after polling")
	}
	if _, ok := collector.history["dev/pool-a"]; !ok {
		t.Fatalf("expected history entry after polling")
	}

	mu.Lock()
	unhealthy = true
	mu.Unlock()
	collector.Poll(context.Background())
	if _, ok := collector.previous["dev/pool-a"]; !ok {
		t.Fatalf("previous entry should be retained while target is unavailable: %+v", collector.previous)
	}
	if _, ok := collector.history["dev/pool-a"]; !ok {
		t.Fatalf("history entry should be retained while target is unavailable: %+v", collector.history)
	}

	mu.Lock()
	unhealthy = false
	pools = []poolResponse{}
	mu.Unlock()
	collector.Poll(context.Background())
	if _, ok := collector.previous["dev/pool-a"]; ok {
		t.Fatalf("previous entry should be removed when pool disappears: %+v", collector.previous)
	}
	if _, ok := collector.history["dev/pool-a"]; ok {
		t.Fatalf("history entry should be removed when pool disappears: %+v", collector.history)
	}
}

func TestHandlerServesSnapshotAndFrontend(t *testing.T) {
	collector := NewCollector(nil, time.Second, 10)
	handler := Handler(collector)
	for path, content := range map[string]string{
		"/api/v1/snapshot": `"poll_interval_seconds":1`,
		"/":                "GPU Pool Monitor",
		"/app.js":          "function render",
		"/styles.css":      ".summary-band",
	} {
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, path, nil))
		if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), content) {
			t.Fatalf("%s status=%d body=%q", path, recorder.Code, recorder.Body.String())
		}
	}
}

func TestParseTarget(t *testing.T) {
	target, err := ParseTarget("dev=http://127.0.0.1:8767/")
	if err != nil || target.Name != "dev" || target.URL != "http://127.0.0.1:8767" {
		t.Fatalf("target=%+v err=%v", target, err)
	}
	for _, value := range []string{"dev", "=http://127.0.0.1", "dev=localhost"} {
		if _, err := ParseTarget(value); err == nil {
			t.Fatalf("expected invalid target %q", value)
		}
	}
}
