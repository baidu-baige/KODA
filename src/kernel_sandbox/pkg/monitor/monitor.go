package monitor

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type Target struct {
	Name string `json:"name"`
	URL  string `json:"url"`
}

type WorkerCounts struct {
	Pending int `json:"pending"`
	Idle    int `json:"idle"`
	Busy    int `json:"busy"`
}

type workerUsage struct {
	BusyMS     int64 `json:"busy_ms"`
	CapacityMS int64 `json:"capacity_ms"`
}

type queueUsage struct {
	Count   int64 `json:"count"`
	TotalMS int64 `json:"total_ms"`
	MaxMS   int64 `json:"max_ms"`
}

type poolUsage struct {
	BusyMS      int64                  `json:"busy_ms"`
	CapacityMS  int64                  `json:"capacity_ms"`
	Workers     map[string]workerUsage `json:"workers"`
	Queue       *queueUsage            `json:"queue"`
	QueueLength *int                   `json:"queue_length"`
}

type poolResponse struct {
	GPUPoolID string       `json:"gpu_pool_id"`
	Cluster   string       `json:"cluster"`
	Image     string       `json:"image"`
	Replicas  int32        `json:"replicas"`
	Workers   WorkerCounts `json:"workers"`
	Usage     *poolUsage   `json:"usage"`
}

type sessionResponse struct {
	SessionID string `json:"session_id"`
	GPUPoolID string `json:"gpu_pool_id"`
	Status    string `json:"status"`
	CPU       *struct {
		Pod string `json:"pod"`
	} `json:"cpu"`
}

type SessionCounts struct {
	Total    int `json:"total"`
	Creating int `json:"creating"`
	Ready    int `json:"ready"`
	Failed   int `json:"failed"`
}

type Point struct {
	Timestamp   time.Time `json:"timestamp"`
	Utilization float64   `json:"utilization"`
}

type Pool struct {
	Key                 string         `json:"key"`
	Target              string         `json:"target"`
	Cluster             string         `json:"cluster"`
	GPUPoolID           string         `json:"gpu_pool_id"`
	Image               string         `json:"image"`
	Replicas            int32          `json:"replicas"`
	Workers             WorkerCounts   `json:"workers"`
	Utilization         *float64       `json:"utilization"`
	LifetimeUtilization *float64       `json:"lifetime_utilization"`
	Balance             *float64       `json:"balance"`
	BusyMS              int64          `json:"busy_ms"`
	CapacityMS          int64          `json:"capacity_ms"`
	QueueSupported      bool           `json:"queue_supported"`
	QueueLength         int            `json:"queue_length"`
	QueueCount          int64          `json:"queue_count"`
	QueueAvgMS          *float64       `json:"queue_avg_ms"`
	QueueMaxMS          int64          `json:"queue_max_ms"`
	QueueRatePerMinute  *float64       `json:"queue_rate_per_minute"`
	Sessions            SessionCounts  `json:"sessions"`
	SessionPrefixes     map[string]int `json:"session_prefixes"`
	History             []Point        `json:"history"`
}

type TargetStatus struct {
	Name      string    `json:"name"`
	URL       string    `json:"url"`
	Healthy   bool      `json:"healthy"`
	PoolCount int       `json:"pool_count"`
	Error     string    `json:"error,omitempty"`
	CheckedAt time.Time `json:"checked_at"`
}

type Summary struct {
	Utilization      *float64 `json:"utilization"`
	Pools            int      `json:"pools"`
	Replicas         int32    `json:"replicas"`
	WorkersBusy      int      `json:"workers_busy"`
	WorkersIdle      int      `json:"workers_idle"`
	Pending          int      `json:"workers_pending"`
	QueueSupported   bool     `json:"queue_supported"`
	Queued           int      `json:"queued_commands"`
	QueueAvgMS       *float64 `json:"queue_avg_ms"`
	QueueMaxMS       int64    `json:"queue_max_ms"`
	QueueRate        *float64 `json:"queue_rate_per_minute"`
	Sessions         int      `json:"sessions"`
	SessionsReady    int      `json:"sessions_ready"`
	SessionsCreating int      `json:"sessions_creating"`
	SessionsFailed   int      `json:"sessions_failed"`
	Unhealthy        int      `json:"unhealthy_targets"`
}

type Snapshot struct {
	GeneratedAt         time.Time      `json:"generated_at"`
	PollIntervalSeconds float64        `json:"poll_interval_seconds"`
	Summary             Summary        `json:"summary"`
	Targets             []TargetStatus `json:"targets"`
	Pools               []Pool         `json:"pools"`
}

type persistedState struct {
	Previous map[string]poolUsage `json:"previous"`
	History  map[string][]Point   `json:"history"`
}

type Collector struct {
	mu            sync.RWMutex
	client        *http.Client
	targets       []Target
	interval      time.Duration
	historyLimit  int
	historyWindow time.Duration
	statePath     string
	previous      map[string]poolUsage
	history       map[string][]Point
	snapshot      Snapshot
}

func NewCollector(targets []Target, interval time.Duration, historyLimit int) *Collector {
	collector, _ := newCollector(targets, interval, historyLimit, 0, "")
	return collector
}

func NewCollectorWithState(targets []Target, interval time.Duration, historyLimit int, statePath string) (*Collector, error) {
	return newCollector(targets, interval, historyLimit, 0, statePath)
}

func NewCollectorWithHistory(targets []Target, interval, historyWindow time.Duration, statePath string) (*Collector, error) {
	return newCollector(targets, interval, 0, historyWindow, statePath)
}

func newCollector(targets []Target, interval time.Duration, historyLimit int, historyWindow time.Duration, statePath string) (*Collector, error) {
	collector := &Collector{
		client:        &http.Client{Timeout: 10 * time.Second},
		targets:       targets,
		interval:      interval,
		historyLimit:  historyLimit,
		historyWindow: historyWindow,
		statePath:     statePath,
		previous:      map[string]poolUsage{},
		history:       map[string][]Point{},
		snapshot:      Snapshot{PollIntervalSeconds: interval.Seconds(), Targets: []TargetStatus{}, Pools: []Pool{}},
	}
	if err := collector.loadState(); err != nil {
		return nil, err
	}
	return collector, nil
}

func ParseTarget(value string) (Target, error) {
	name, rawURL, ok := strings.Cut(value, "=")
	if !ok || strings.TrimSpace(name) == "" || strings.TrimSpace(rawURL) == "" {
		return Target{}, fmt.Errorf("target must use name=url")
	}
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return Target{}, fmt.Errorf("target %q has invalid URL", value)
	}
	return Target{Name: strings.TrimSpace(name), URL: strings.TrimRight(parsed.String(), "/")}, nil
}

func (c *Collector) Run(ctx context.Context) {
	c.Poll(ctx)
	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.Poll(ctx)
		}
	}
}

func (c *Collector) Poll(ctx context.Context) {
	now := time.Now()
	var pools []Pool
	var statuses []TargetStatus
	var totalBusy, totalCapacity int64
	var totalQueueCount, totalQueueMS int64
	var totalQueueRate float64
	var queueRateSamples int
	var queueSupportedPools int
	summary := Summary{}
	healthyTargets := make(map[string]struct{}, len(c.targets))
	alive := make(map[string]struct{})

	for _, target := range c.targets {
		targetPools, err := c.fetchPools(ctx, target)
		status := TargetStatus{Name: target.Name, URL: target.URL, Healthy: err == nil, CheckedAt: now}
		if err != nil {
			status.Error = err.Error()
			summary.Unhealthy++
			statuses = append(statuses, status)
			continue
		}
		healthyTargets[target.Name] = struct{}{}
		sessions, sessionErr := c.fetchSessions(ctx, target)
		if sessionErr != nil {
			slog.Warn("fetch GPU pool sessions", "target", target.Name, "error", sessionErr)
		}
		sessionCounts, sessionPrefixes := countSessions(sessions)
		status.PoolCount = len(targetPools)
		statuses = append(statuses, status)
		for _, current := range targetPools {
			key := target.Name + "/" + current.GPUPoolID
			alive[key] = struct{}{}
			cluster := current.Cluster
			if cluster == "" {
				cluster = target.Name
			}
			var utilization, balance *float64
			var queueCount, queueTotalMS int64
			var queueRate *float64
			var queueLength int
			var queueMaxMS int64
			queueSupported := false
			if current.Usage != nil {
				if previous, ok := c.previous[key]; ok {
					utilization, balance, totalBusy, totalCapacity = addInterval(
						current.Usage.Workers, previous.Workers, totalBusy, totalCapacity,
					)
					if current.Usage.Queue != nil && previous.Queue != nil {
						queueCount, queueTotalMS, queueRate = addQueueInterval(*current.Usage.Queue, *previous.Queue, c.interval)
					}
				}
				c.previous[key] = cloneUsage(*current.Usage)
				if current.Usage.Queue != nil && current.Usage.QueueLength != nil {
					queueSupported = true
					queueLength = *current.Usage.QueueLength
					queueMaxMS = current.Usage.Queue.MaxMS
				}
			}
			lifetime := ratio(current.Usage)
			if utilization != nil {
				c.history[key] = append(c.history[key], Point{Timestamp: now, Utilization: *utilization})
				if c.historyLimit > 0 && len(c.history[key]) > c.historyLimit {
					c.history[key] = c.history[key][len(c.history[key])-c.historyLimit:]
				}
				c.trimHistory(now)
			}
			pool := Pool{
				Key: key, Target: target.Name, Cluster: cluster, GPUPoolID: current.GPUPoolID,
				Image: current.Image, Replicas: current.Replicas, Workers: current.Workers,
				Utilization: utilization, LifetimeUtilization: lifetime, Balance: balance,
				QueueSupported: queueSupported, QueueLength: queueLength,
				QueueCount: queueCount, QueueAvgMS: average(queueTotalMS, queueCount),
				QueueMaxMS:         queueMaxMS,
				QueueRatePerMinute: queueRate,
				History:            append([]Point(nil), c.history[key]...),
			}
			if current.Usage != nil {
				pool.BusyMS = current.Usage.BusyMS
				pool.CapacityMS = current.Usage.CapacityMS
				if queueSupported {
					queueSupportedPools++
					totalQueueCount += queueCount
					totalQueueMS += queueTotalMS
					if queueRate != nil {
						totalQueueRate += *queueRate
						queueRateSamples++
					}
				}
			}
			pool.Sessions = sessionCounts[current.GPUPoolID]
			pool.SessionPrefixes = sessionPrefixes[current.GPUPoolID]
			pools = append(pools, pool)
			summary.Pools++
			summary.Replicas += current.Replicas
			summary.WorkersBusy += current.Workers.Busy
			summary.WorkersIdle += current.Workers.Idle
			summary.Pending += current.Workers.Pending
			summary.Sessions += pool.Sessions.Total
			summary.SessionsReady += pool.Sessions.Ready
			summary.SessionsCreating += pool.Sessions.Creating
			summary.SessionsFailed += pool.Sessions.Failed
		}
	}
	for key := range c.previous {
		target, _, _ := strings.Cut(key, "/")
		if _, ok := healthyTargets[target]; !ok {
			continue
		}
		if _, ok := alive[key]; !ok {
			delete(c.previous, key)
		}
	}
	for key := range c.history {
		target, _, _ := strings.Cut(key, "/")
		if _, ok := healthyTargets[target]; !ok {
			continue
		}
		if _, ok := alive[key]; !ok {
			delete(c.history, key)
		}
	}
	if totalCapacity > 0 {
		value := math.Min(1, float64(totalBusy)/float64(totalCapacity))
		summary.Utilization = &value
	}
	summary.Queued = 0
	for _, pool := range pools {
		if pool.QueueSupported {
			summary.Queued += pool.QueueLength
		}
	}
	summary.QueueSupported = summary.Pools > 0 && queueSupportedPools == summary.Pools
	if summary.QueueSupported {
		summary.QueueAvgMS = average(totalQueueMS, totalQueueCount)
		if queueRateSamples > 0 {
			summary.QueueRate = &totalQueueRate
		}
		for _, pool := range pools {
			if pool.QueueMaxMS > summary.QueueMaxMS {
				summary.QueueMaxMS = pool.QueueMaxMS
			}
		}
	}
	sort.Slice(pools, func(i, j int) bool {
		if pools[i].Cluster != pools[j].Cluster {
			return pools[i].Cluster < pools[j].Cluster
		}
		return pools[i].GPUPoolID < pools[j].GPUPoolID
	})
	c.mu.Lock()
	c.snapshot = Snapshot{
		GeneratedAt: now, PollIntervalSeconds: c.interval.Seconds(),
		Summary: summary, Targets: statuses, Pools: pools,
	}
	c.mu.Unlock()
	if err := c.persistState(); err != nil {
		slog.Warn("persist GPU pool monitor state", "path", c.statePath, "error", err)
	}
}

func (c *Collector) Snapshot() Snapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()
	snapshot := c.snapshot
	snapshot.Targets = append([]TargetStatus(nil), c.snapshot.Targets...)
	snapshot.Pools = append([]Pool(nil), c.snapshot.Pools...)
	for index := range snapshot.Pools {
		snapshot.Pools[index].History = append([]Point(nil), snapshot.Pools[index].History...)
	}
	return snapshot
}

func (c *Collector) fetchPools(ctx context.Context, target Target) ([]poolResponse, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target.URL+"/v1/gpu_pool", nil)
	if err != nil {
		return nil, err
	}
	response, err := c.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return nil, fmt.Errorf("GET /v1/gpu_pool: %s: %s", response.Status, strings.TrimSpace(string(body)))
	}
	var pools []poolResponse
	if err := json.NewDecoder(response.Body).Decode(&pools); err != nil {
		return nil, fmt.Errorf("decode pools: %w", err)
	}
	return pools, nil
}

func (c *Collector) fetchSessions(ctx context.Context, target Target) ([]sessionResponse, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target.URL+"/v1/session", nil)
	if err != nil {
		return nil, err
	}
	response, err := c.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return nil, fmt.Errorf("GET /v1/session: %s: %s", response.Status, strings.TrimSpace(string(body)))
	}
	var sessions []sessionResponse
	if err := json.NewDecoder(response.Body).Decode(&sessions); err != nil {
		return nil, fmt.Errorf("decode sessions: %w", err)
	}
	return sessions, nil
}

func countSessions(sessions []sessionResponse) (map[string]SessionCounts, map[string]map[string]int) {
	counts := make(map[string]SessionCounts)
	prefixes := make(map[string]map[string]int)
	for _, session := range sessions {
		if session.GPUPoolID == "" {
			continue
		}
		current := counts[session.GPUPoolID]
		current.Total++
		switch session.Status {
		case "creating":
			current.Creating++
		case "ready":
			current.Ready++
		case "failed":
			current.Failed++
		}
		counts[session.GPUPoolID] = current
		if prefixes[session.GPUPoolID] == nil {
			prefixes[session.GPUPoolID] = make(map[string]int)
		}
		prefixes[session.GPUPoolID][sessionPrefix(session)]++
	}
	return counts, prefixes
}

func sessionPrefix(session sessionResponse) string {
	if session.CPU == nil || session.CPU.Pod == "" || session.SessionID == "" {
		return "(unknown)"
	}
	suffix := "-" + session.SessionID
	if !strings.HasSuffix(session.CPU.Pod, suffix) {
		return "(unknown)"
	}
	prefix := strings.TrimSuffix(session.CPU.Pod, suffix)
	if prefix == "" {
		return "(unknown)"
	}
	return prefix
}

func addInterval(current, previous map[string]workerUsage, totalBusy, totalCapacity int64) (*float64, *float64, int64, int64) {
	var busy, capacity int64
	var fairnessDenominator float64
	for name, worker := range current {
		before := previous[name]
		busyDelta := worker.BusyMS - before.BusyMS
		capacityDelta := worker.CapacityMS - before.CapacityMS
		if busyDelta < 0 || capacityDelta < 0 || (capacityDelta == 0 && busyDelta > 0) {
			return nil, nil, totalBusy, totalCapacity
		}
		if capacityDelta == 0 {
			continue
		}
		busy += busyDelta
		capacity += capacityDelta
		fairnessDenominator += float64(busyDelta) * float64(busyDelta) / float64(capacityDelta)
	}
	if capacity == 0 {
		return nil, nil, totalBusy, totalCapacity
	}
	utilization := math.Min(1, float64(busy)/float64(capacity))
	var balance *float64
	if busy > 0 && fairnessDenominator > 0 {
		value := math.Min(1, float64(busy)*float64(busy)/(float64(capacity)*fairnessDenominator))
		balance = &value
	}
	return &utilization, balance, totalBusy + busy, totalCapacity + capacity
}

func addQueueInterval(current, previous queueUsage, interval time.Duration) (int64, int64, *float64) {
	if current.Count < previous.Count || current.TotalMS < previous.TotalMS || interval <= 0 {
		return 0, 0, nil
	}
	count := current.Count - previous.Count
	totalMS := current.TotalMS - previous.TotalMS
	rate := float64(count) * float64(time.Minute) / float64(interval)
	return count, totalMS, &rate
}

func average(totalMS, count int64) *float64 {
	if count <= 0 {
		return nil
	}
	value := float64(totalMS) / float64(count)
	return &value
}

func ratio(usage *poolUsage) *float64 {
	if usage == nil || usage.CapacityMS <= 0 || usage.BusyMS < 0 {
		return nil
	}
	value := math.Min(1, float64(usage.BusyMS)/float64(usage.CapacityMS))
	return &value
}

func cloneUsage(usage poolUsage) poolUsage {
	workers := make(map[string]workerUsage, len(usage.Workers))
	for name, worker := range usage.Workers {
		workers[name] = worker
	}
	usage.Workers = workers
	if usage.Queue != nil {
		queue := *usage.Queue
		usage.Queue = &queue
	}
	if usage.QueueLength != nil {
		queueLength := *usage.QueueLength
		usage.QueueLength = &queueLength
	}
	return usage
}

func (c *Collector) loadState() error {
	if c.statePath == "" {
		return nil
	}
	data, err := os.ReadFile(c.statePath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read monitor state: %w", err)
	}
	var state persistedState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("decode monitor state: %w", err)
	}
	if state.Previous != nil {
		c.previous = state.Previous
	}
	if state.History != nil {
		c.history = state.History
	}
	c.trimHistory(time.Now())
	return nil
}

func (c *Collector) persistState() error {
	if c.statePath == "" {
		return nil
	}
	state := persistedState{
		Previous: cloneUsageMap(c.previous),
		History:  cloneHistory(c.history),
	}
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("encode monitor state: %w", err)
	}
	dir := filepath.Dir(c.statePath)
	temp, err := os.CreateTemp(dir, ".gpu-pool-monitor-state-*")
	if err != nil {
		return fmt.Errorf("create monitor state temp file: %w", err)
	}
	tempName := temp.Name()
	defer os.Remove(tempName)
	if _, err := temp.Write(data); err != nil {
		_ = temp.Close()
		return fmt.Errorf("write monitor state: %w", err)
	}
	if err := temp.Close(); err != nil {
		return fmt.Errorf("close monitor state: %w", err)
	}
	if err := os.Rename(tempName, c.statePath); err != nil {
		return fmt.Errorf("replace monitor state: %w", err)
	}
	return nil
}

func cloneUsageMap(usages map[string]poolUsage) map[string]poolUsage {
	cloned := make(map[string]poolUsage, len(usages))
	for key, usage := range usages {
		cloned[key] = cloneUsage(usage)
	}
	return cloned
}

func cloneHistory(history map[string][]Point) map[string][]Point {
	cloned := make(map[string][]Point, len(history))
	for key, points := range history {
		cloned[key] = append([]Point(nil), points...)
	}
	return cloned
}

func (c *Collector) trimHistory(now time.Time) {
	if c.historyWindow <= 0 {
		return
	}
	cutoff := now.Add(-c.historyWindow)
	for key, points := range c.history {
		index := 0
		for index < len(points) && points[index].Timestamp.Before(cutoff) {
			index++
		}
		if index > 0 {
			c.history[key] = append([]Point(nil), points[index:]...)
		}
	}
}
