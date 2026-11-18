package master

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"kernel-sandbox/pkg/command"
	"kernel-sandbox/pkg/config"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

type fakeKubernetes struct {
	mu sync.Mutex

	cpuPods               []PodInfo
	gpuPods               []PodInfo
	cpuImage              string
	cpuResources          *ResourceRequirements
	cpuPrefix             string
	cpuSandboxURL         string
	injectedEnv           map[string]string
	cpuResult             command.Result
	cpuProbeResults       []command.Result
	cpuProbeCommand       []string
	cpuProbeCalls         int
	cpuProbeStarted       chan struct{}
	cpuProbeRelease       chan struct{}
	cpuCreateCount        int
	gpuResult             command.Result
	gpuStates             []GPUPodState
	gpuStateErrors        []error
	gpuState              GPUPodState
	stateCalls            int
	gpuExecCalls          int
	gpuExecutedPools      []string
	gpuPools              map[string]GPUPoolInfo
	gpuPoolImage          string
	gpuPoolID             string
	gpuPoolReplicas       int32
	gpuPoolResources      *ResourceRequirements
	gpuPoolScheduling     *Scheduling
	gpuPoolCreateCount    int
	gpuPoolCreateErr      error
	gpuPoolCreateStarted  chan struct{}
	gpuPoolCreateRelease  chan struct{}
	gpuPoolDeleteCount    int
	gpuPoolListErr        error
	deletedCPU            int
	deletedGPU            int
	gpuListErr            error
	deleteGPUErr          error
	gpuSessionPods        []PodInfo
	gpuSessionImage       string
	gpuSessionResources   *ResourceRequirements
	gpuSessionCreateCount int
	gpuSessionCreateErr   error
	gpuSessionReady       bool
	gpuSessionExecCalls   int
	gpuSessionExecPods    []string
	gpuSessionResult      command.Result
	gpuSessionExecStarted chan struct{}
	gpuSessionExecRelease chan struct{}
	cpuStarted            chan struct{}
	cpuRelease            chan struct{}
	cpuCreateStarted      chan struct{}
	cpuCreateRelease      chan struct{}
	cpuPodNotReady        bool
}

type lockedBuffer struct {
	mu     sync.Mutex
	buffer bytes.Buffer
}

func (b *lockedBuffer) Write(data []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buffer.Write(data)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buffer.String()
}

func captureLogs(t *testing.T) *lockedBuffer {
	t.Helper()
	output := &lockedBuffer{}
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(output, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return output
}

func TestGPUUsageSurvivesWorkerReplacement(t *testing.T) {
	now := time.Now()
	pool := &gpuPool{id: "pool", replicas: 1}
	sandboxMaster := &Master{
		gpuPoolsByID: map[string]*gpuPool{"pool": pool},
		gpuWorkersByUID: map[string]*gpuWorker{"old": {
			pod:       PodInfo{Name: "pool-0", UID: "old", GPUPoolID: "pool"},
			gpuPoolID: "pool", status: GPUWorkerBusy,
			busySince: now.Add(-10 * time.Second), availableSince: now.Add(-20 * time.Second),
		}},
	}
	before := sandboxMaster.gpuPoolResponseLocked(pool).Usage
	sandboxMaster.syncGPUWorkers(context.Background(), "", nil)
	after := sandboxMaster.gpuPoolResponseLocked(pool).Usage
	if after.BusyMS < before.BusyMS || after.CapacityMS < before.CapacityMS || len(after.Workers) != 1 {
		t.Fatalf("worker removal lost usage: before=%+v after=%+v", before, after)
	}
	sandboxMaster.gpuWorkersByUID["new"] = &gpuWorker{
		pod:       PodInfo{Name: "pool-0", UID: "new", GPUPoolID: "pool"},
		gpuPoolID: "pool", status: GPUWorkerIdle, availableSince: time.Now().Add(-time.Second),
	}
	replaced := sandboxMaster.gpuPoolResponseLocked(pool)
	if replaced.Usage.BusyMS != after.BusyMS || replaced.Usage.CapacityMS < after.CapacityMS+1000 {
		t.Fatalf("replacement reset counters: after=%+v replaced=%+v", after, replaced.Usage)
	}
	if len(replaced.Usage.Workers) != 1 || replaced.Workers.Idle != 1 {
		t.Fatalf("usage history leaked into current worker counts: %+v", replaced)
	}
}

func TestGPUPoolResponseIncludesCluster(t *testing.T) {
	pool := &gpuPool{id: "pool", clusterName: "h20", replicas: 1}
	sandboxMaster := &Master{gpuWorkersByUID: map[string]*gpuWorker{}}
	if response := sandboxMaster.gpuPoolResponseLocked(pool); response.Cluster != "h20" {
		t.Fatalf("cluster=%q", response.Cluster)
	}
}

func TestGPUQueueUsageCountsCommandOnceAcrossRetries(t *testing.T) {
	pool := &gpuPool{id: "pool", replicas: 1}
	now := time.Now()
	execution := &execution{
		id:        "execution",
		gpuPoolID: "pool",
		request:   command.Request{Device: command.DeviceGPU},
		status:    ExecutionStatusQueued,
		createdAt: now.Add(-300 * time.Millisecond),
		queuedAt:  now.Add(-150 * time.Millisecond),
		done:      make(chan struct{}),
	}
	sandboxMaster := &Master{
		gpuPoolsByID: map[string]*gpuPool{"pool": pool},
	}

	sandboxMaster.finishExecutionLocked(execution, ExecuteCommandResponse{
		ExecutionID: execution.id,
		Status:      ExecutionStatusFailed,
	})
	sandboxMaster.finishExecutionLocked(execution, ExecuteCommandResponse{
		ExecutionID: execution.id,
		Status:      ExecutionStatusFailed,
	})

	usage := sandboxMaster.gpuPoolResponseLocked(pool).Usage
	if usage.Queue.Count != 1 || usage.Queue.TotalMS < 100 || usage.Queue.MaxMS < 100 {
		t.Fatalf("unexpected queue usage: %+v", usage.Queue)
	}
}

func TestGPUUsageIntegratesWorkerCapacityAndDoesNotCountBusyTwice(t *testing.T) {
	now := time.Now()
	pool := &gpuPool{id: "pool", replicas: 2}
	worker := &gpuWorker{
		pod: PodInfo{Name: "pool-0"}, gpuPoolID: "pool", status: GPUWorkerBusy,
		busySince: now.Add(-10 * time.Second), availableSince: now.Add(-10 * time.Second),
	}
	sandboxMaster := &Master{
		gpuPoolsByID: map[string]*gpuPool{"pool": pool},
		gpuWorkersByUID: map[string]*gpuWorker{"first": worker, "second": {
			pod: PodInfo{Name: "pool-1"}, gpuPoolID: "pool", status: GPUWorkerBusy,
			busySince: now.Add(-5 * time.Second), availableSince: now.Add(-5 * time.Second),
		}},
	}
	usage := sandboxMaster.gpuPoolResponseLocked(pool).Usage
	if usage.BusyMS < 15000 || usage.CapacityMS != usage.BusyMS {
		t.Fatalf("worker capacity was not integrated: %+v", usage)
	}
	sandboxMaster.recordGPUWorkerUsageLocked(worker, false)
	first := pool.workerUsage["pool-0"].busy
	sandboxMaster.recordGPUWorkerUsageLocked(worker, false)
	if pool.workerUsage["pool-0"].busy != first {
		t.Fatal("released worker busy interval was counted twice")
	}
}

func (f *fakeKubernetes) CreateCPUPod(_ context.Context, sessionID, gpuPoolID, image, sandboxURL string, resources *ResourceRequirements, prefix string) (PodInfo, error) {
	if f.cpuCreateStarted != nil {
		f.cpuCreateStarted <- struct{}{}
	}
	if f.cpuCreateRelease != nil {
		<-f.cpuCreateRelease
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cpuImage = image
	f.cpuResources = resources
	f.cpuPrefix = prefix
	f.cpuSandboxURL = sandboxURL
	f.cpuCreateCount++
	pod := PodInfo{Name: "cpu-" + sessionID, UID: fmt.Sprintf("cpu-uid-%d", f.cpuCreateCount), SessionID: sessionID, GPUPoolID: gpuPoolID, Image: image, Resources: resources, CreatedAt: time.Now(), Ready: !f.cpuPodNotReady}
	if prefix != "" {
		pod.Name = prefix + "-" + sessionID
	}
	pod.Prefix = prefix
	f.cpuPods = append(f.cpuPods, pod)
	return pod, nil
}

func TestCPUSessionsAreCreatedConcurrently(t *testing.T) {
	kubernetes := &fakeKubernetes{
		cpuCreateStarted: make(chan struct{}, 2),
		cpuCreateRelease: make(chan struct{}),
	}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "image")
	done := make(chan *ServiceError, 2)
	for range 2 {
		go func() {
			_, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeCPUGPU, GPUPoolID: gpuPoolID})
			done <- serviceError
		}()
	}
	for range 2 {
		select {
		case <-kubernetes.cpuCreateStarted:
		case <-time.After(time.Second):
			t.Fatal("CPU Session creation was serialized")
		}
	}
	close(kubernetes.cpuCreateRelease)
	for range 2 {
		if serviceError := <-done; serviceError != nil {
			t.Fatal(serviceError)
		}
	}
}

func (f *fakeKubernetes) DeleteCPUPod(_ context.Context, pod PodInfo) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deletedCPU++
	for index, current := range f.cpuPods {
		if current.UID == pod.UID {
			f.cpuPods = append(f.cpuPods[:index], f.cpuPods[index+1:]...)
			break
		}
	}
	return nil
}

func (f *fakeKubernetes) ListCPUPods(context.Context) ([]PodInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]PodInfo(nil), f.cpuPods...), nil
}

func (f *fakeKubernetes) CreateGPUPool(ctx context.Context, gpuPoolID, image string, replicas int32, resources *ResourceRequirements, scheduling *Scheduling) error {
	if f.gpuPoolCreateStarted != nil {
		f.gpuPoolCreateStarted <- struct{}{}
	}
	if f.gpuPoolCreateRelease != nil {
		select {
		case <-f.gpuPoolCreateRelease:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.gpuPoolCreateCount++
	if f.gpuPoolCreateErr != nil {
		return f.gpuPoolCreateErr
	}
	if f.gpuPools == nil {
		f.gpuPools = map[string]GPUPoolInfo{}
	}
	f.gpuPools[gpuPoolID] = GPUPoolInfo{GPUPoolID: gpuPoolID, Image: image, Replicas: replicas, Resources: resources, Scheduling: scheduling}
	f.gpuPoolResources, f.gpuPoolScheduling = resources, scheduling
	f.gpuPoolID, f.gpuPoolImage, f.gpuPoolReplicas = gpuPoolID, image, replicas
	uid := "gpu-uid"
	if len(f.gpuPods) != 0 {
		uid = gpuPoolID + "-uid"
	}
	f.gpuPods = append(f.gpuPods, PodInfo{Name: gpuPoolID + "-0", UID: uid, GPUPoolID: gpuPoolID, Ready: true})
	return nil
}

func (f *fakeKubernetes) ListGPUPools(context.Context) ([]GPUPoolInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.gpuPoolListErr != nil {
		return nil, f.gpuPoolListErr
	}
	if len(f.gpuPools) == 0 && f.gpuPoolImage != "" {
		id := f.gpuPoolID
		if id == "" {
			id = "gpu-pool-restored"
		}
		return []GPUPoolInfo{{GPUPoolID: id, Image: f.gpuPoolImage, Replicas: f.gpuPoolReplicas}}, nil
	}
	result := make([]GPUPoolInfo, 0, len(f.gpuPools))
	for _, pool := range f.gpuPools {
		result = append(result, pool)
	}
	return result, nil
}

func (f *fakeKubernetes) UpdateGPUPool(_ context.Context, gpuPoolID, image string, replicas *int32, scheduling *Scheduling) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	pool := f.gpuPools[gpuPoolID]
	if image != "" {
		pool.Image = image
	}
	if replicas != nil {
		pool.Replicas = *replicas
	}
	updated := Scheduling{}
	if pool.Scheduling != nil {
		updated = *pool.Scheduling
	}
	if scheduling.NodeSelector != nil {
		updated.NodeSelector = scheduling.NodeSelector
	}
	if scheduling.Tolerations != nil {
		updated.Tolerations = scheduling.Tolerations
	}
	pool.Scheduling = &updated
	f.gpuPools[gpuPoolID] = pool
	return nil
}

func TestStartRestoresGPUPool(t *testing.T) {
	kubernetes := &fakeKubernetes{gpuPoolImage: "gpu-image", gpuPoolReplicas: 3}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sandboxMaster.Start(ctx)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if response, serviceError := sandboxMaster.GetGPUPool("gpu-pool-restored"); serviceError == nil {
			if response.GPUPoolID != "gpu-pool-restored" || response.Image != "gpu-image" || response.Replicas != 3 || response.Timing != nil {
				t.Fatalf("response=%+v", response)
			}
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("GPU Pool was not restored")
}

func TestStartDoesNotRecoverGPUSessionsFromGPUPoolCluster(t *testing.T) {
	kubernetes := &fakeKubernetes{
		gpuSessionPods: []PodInfo{{Name: "production-gpu-session-1234", SessionID: "production-gpu-session-1234", Image: "production-image", Ready: true}},
	}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sandboxMaster.Start(ctx)
	time.Sleep(50 * time.Millisecond)
	if sessions := sandboxMaster.ListSessions(); len(sessions) != 0 {
		t.Fatalf("GPU Session was restored from the GPU Pool cluster: %+v", sessions)
	}
}

func (f *fakeKubernetes) DeleteGPUPool(_ context.Context, gpuPoolID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.gpuPoolDeleteCount++
	delete(f.gpuPools, gpuPoolID)
	return nil
}

func (f *fakeKubernetes) ListGPUPods(context.Context) ([]PodInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.gpuListErr != nil {
		return nil, f.gpuListErr
	}
	return append([]PodInfo(nil), f.gpuPods...), nil
}

func (f *fakeKubernetes) deleteGPUPod(pod PodInfo) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deletedGPU++
	if f.deleteGPUErr != nil {
		err := f.deleteGPUErr
		f.deleteGPUErr = nil
		return err
	}
	for index, current := range f.gpuPods {
		if current.UID == pod.UID {
			f.gpuPods = append(f.gpuPods[:index], f.gpuPods[index+1:]...)
			break
		}
	}
	for index, current := range f.gpuSessionPods {
		if current.UID == pod.UID {
			f.gpuSessionPods = append(f.gpuSessionPods[:index], f.gpuSessionPods[index+1:]...)
			break
		}
	}

	return nil
}

func (f *fakeKubernetes) DeleteGPUWorkerPod(_ context.Context, pod PodInfo) error {
	return f.deleteGPUPod(pod)
}

func (f *fakeKubernetes) DeleteGPUSessionPod(_ context.Context, pod PodInfo) error {
	return f.deleteGPUPod(pod)
}

func (f *fakeKubernetes) CreateGPUSessionPod(_ context.Context, sessionID, image string, resources *ResourceRequirements, prefix string) (PodInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.gpuSessionCreateCount++
	f.gpuSessionImage = image
	f.gpuSessionResources = resources
	if f.gpuSessionCreateErr != nil {
		return PodInfo{}, f.gpuSessionCreateErr
	}
	pod := PodInfo{Name: "gpu-pool-" + sessionID, UID: "gpu-uid-" + sessionID, SessionID: sessionID, Image: image, Resources: resources, Ready: f.gpuSessionReady}
	if prefix != "" {
		pod.Name = prefix + "-" + sessionID
	}
	pod.Prefix = prefix
	f.gpuSessionPods = append(f.gpuSessionPods, pod)
	return pod, nil
}

func (f *fakeKubernetes) ListGPUSessionPods(context.Context) ([]PodInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.gpuListErr != nil {
		return nil, f.gpuListErr
	}
	pods := make([]PodInfo, len(f.gpuSessionPods))
	for index, pod := range f.gpuSessionPods {
		if f.gpuSessionReady {
			pod.Ready = true
		}
		pods[index] = pod
	}
	return pods, nil
}

func (f *fakeKubernetes) ExecuteCPU(_ context.Context, _ PodInfo, request command.Request, _ int) command.Result {
	f.mu.Lock()
	f.injectedEnv = request.Env
	isProbe := len(request.Command) == 3 && request.Command[0] == "python3" && request.Command[1] == "-c" && strings.Contains(request.Command[2], "KERNEL_SANDBOX_URL")
	if isProbe {
		index := f.cpuProbeCalls
		f.cpuProbeCalls++
		f.cpuProbeCommand = append([]string(nil), request.Command...)
		started, release := f.cpuProbeStarted, f.cpuProbeRelease
		var result command.Result
		if index < len(f.cpuProbeResults) {
			result = f.cpuProbeResults[index]
		}
		f.mu.Unlock()
		if started != nil {
			started <- struct{}{}
		}
		if release != nil {
			<-release
		}
		if result.Kind != "" {
			return result
		}
		exitCode := 0
		return command.Result{Kind: command.ResultCompleted, ExitCode: &exitCode}
	}
	started, release, result := f.cpuStarted, f.cpuRelease, f.cpuResult
	f.mu.Unlock()
	if started != nil {
		close(started)
	}
	if release != nil {
		<-release
	}
	if result.Kind == "" {
		exitCode := 0
		return command.Result{Kind: command.ResultCompleted, ExitCode: &exitCode, Stdout: "cpu-ok"}
	}
	return result
}

func (f *fakeKubernetes) ExecuteGPU(_ context.Context, pod PodInfo, _ command.Request, _ int) command.Result {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.gpuExecCalls++
	f.gpuExecutedPools = append(f.gpuExecutedPools, pod.GPUPoolID)
	if f.gpuResult.Kind != "" {
		return f.gpuResult
	}
	exitCode := 0
	return command.Result{Kind: command.ResultCompleted, ExitCode: &exitCode, Stdout: "gpu-ok"}
}

func (f *fakeKubernetes) ExecuteGPUSession(_ context.Context, pod PodInfo, _ command.Request, _ int) command.Result {
	f.mu.Lock()
	f.gpuSessionExecCalls++
	f.gpuSessionExecPods = append(f.gpuSessionExecPods, pod.Name)
	started, release, result := f.gpuSessionExecStarted, f.gpuSessionExecRelease, f.gpuSessionResult
	f.mu.Unlock()
	if started != nil {
		started <- struct{}{}
	}
	if release != nil {
		<-release
	}
	if result.Kind == "" {
		exitCode := 0
		result = command.Result{Kind: command.ResultCompleted, ExitCode: &exitCode, Stdout: "gpu-session-ok"}
	}
	return result
}

func (f *fakeKubernetes) ReadGPUPodState(context.Context, PodInfo) (GPUPodState, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	index := f.stateCalls
	f.stateCalls++
	if index < len(f.gpuStateErrors) && f.gpuStateErrors[index] != nil {
		return GPUPodState{}, f.gpuStateErrors[index]
	}
	if index < len(f.gpuStates) {
		return f.gpuStates[index], nil
	}
	return f.gpuState, nil
}

func testConfig() *config.Config {
	return &config.Config{
		Server:         config.ServerConfig{AdvertisedEndpoint: "http://master:8765"},
		CPUCluster:     config.CPUClusterConfig{PodNamePrefix: "kernel-sandbox"},
		MasterID:       "test",
		GPUPoolCluster: config.GPUPoolClusterConfig{PoolNamePrefix: "gpu-pool"},
		Execution: config.ExecutionConfig{
			MaxAttempts: 1,
			MaxStdinMB:  1,
			MaxOutputMB: 1,
		},
	}
}

func testConfigWithGPUSessionClusters(clusterResources map[string][]string) *config.Config {
	cfg := testConfig()
	cfg.GPUSessionClusters = make(map[string]config.GPUSessionClusterConfig, len(clusterResources))
	for clusterName, resourceNames := range clusterResources {
		cfg.GPUSessionClusters[clusterName] = config.GPUSessionClusterConfig{ResourceNames: resourceNames}
	}
	return cfg
}

func testGPUSessionConfig() *config.Config {
	return testConfigWithGPUSessionClusters(map[string][]string{"h20": {"example.org/gpu_count_best_effort"}})
}

func testGPUSessionClients(client GPUSessionKubernetesClient) map[string]GPUSessionKubernetesClient {
	return map[string]GPUSessionKubernetesClient{"h20": client}
}

func commandTiming(response ExecuteCommandResponse, device command.Device) *ExecutionTimingResponse {
	if response.Extra == nil || response.Extra.SandboxTimings == nil {
		return nil
	}
	if device == command.DeviceCPU {
		return response.Extra.SandboxTimings.CPUExecution
	}
	if len(response.Extra.SandboxTimings.GPUExecutions) != 1 {
		return nil
	}
	return response.Extra.SandboxTimings.GPUExecutions[0]
}

func prepareSession(t *testing.T, master *Master, resources *ResourceRequirements) SessionResponse {
	t.Helper()
	startGPUQueue(t, master)
	gpuPoolID := createPool(t, master, "business-image")
	session, serviceError := master.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeCPUGPU, GPUPoolID: gpuPoolID, Resources: resources})
	if serviceError != nil {
		t.Fatal(serviceError)
	}
	syncMaster(t, master)
	return waitForSessionStatus(t, master, session.SessionID, SessionStatusReady)
}

func waitForSessionStatus(t *testing.T, master *Master, sessionID string, status SessionStatus) SessionResponse {
	t.Helper()
	for deadline := time.Now().Add(2 * time.Second); time.Now().Before(deadline); time.Sleep(time.Millisecond) {
		pods, err := master.kubernetes.ListCPUPods(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		master.syncCPUPods(pods)
		session, serviceError := master.GetSession(sessionID)
		if serviceError != nil {
			t.Fatal(serviceError)
		}
		if session.Status == status {
			return session
		}
		if session.Status == SessionStatusFailed {
			t.Fatalf("session failed: %+v", session)
		}
	}
	t.Fatalf("session %s did not reach %s", sessionID, status)
	return SessionResponse{}
}

func createPool(t *testing.T, master *Master, image string) string {
	t.Helper()
	pool, serviceError := master.CreateGPUPool(context.Background(), CreateGPUPoolRequest{Image: image, Replicas: 1})
	if serviceError != nil {
		t.Fatal(serviceError)
	}
	return pool.GPUPoolID
}

func TestCPUGPUSessionDropsNonCPUMemoryResources(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "business-image")
	resources := &ResourceRequirements{
		Requests: map[string]string{"cpu": "6", "memory": "24Gi", "example.org/gpu_count_best_effort": "1"},
		Limits:   map[string]string{"example.org/gpu_count_best_effort": "1"},
	}
	session, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeCPUGPU, GPUPoolID: gpuPoolID, Resources: resources})
	if serviceError != nil {
		t.Fatal(serviceError)
	}
	kubernetes.mu.Lock()
	captured := kubernetes.cpuResources
	kubernetes.mu.Unlock()
	if captured == nil || len(captured.Requests) != 2 || captured.Requests["cpu"] != "6" || captured.Requests["memory"] != "24Gi" || captured.Limits != nil {
		t.Fatalf("cpu pod resources=%+v", captured)
	}
	if session.CPU == nil || session.CPU.Resources == nil || len(session.CPU.Resources.Requests) != 2 || session.CPU.Resources.Limits != nil {
		t.Fatalf("session cpu=%+v", session.CPU)
	}
}

func TestCPUGPUSessionKeepsCPUMemoryLimits(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "business-image")
	resources := &ResourceRequirements{
		Requests: map[string]string{"cpu": "6"},
		Limits:   map[string]string{"cpu": "8", "example.org/gpu_count_best_effort": "1"},
	}
	session, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeCPUGPU, GPUPoolID: gpuPoolID, Resources: resources})
	if serviceError != nil {
		t.Fatal(serviceError)
	}
	kubernetes.mu.Lock()
	captured := kubernetes.cpuResources
	kubernetes.mu.Unlock()
	if captured == nil || len(captured.Limits) != 1 || captured.Limits["cpu"] != "8" {
		t.Fatalf("cpu pod limits=%+v", captured)
	}
	if session.CPU == nil || session.CPU.Resources == nil || len(session.CPU.Resources.Limits) != 1 || session.CPU.Resources.Limits["cpu"] != "8" {
		t.Fatalf("session cpu=%+v", session.CPU)
	}
}

// readyGPUSession injects a ready session bound to gpuPoolID so GPU command
// dispatch tests have a session that can enqueue GPU commands. gpu_only sessions
// no longer bind pools, so tests that exercise the pool+queue+worker path use a
// directly-constructed ready session instead.
func readyGPUSession(t *testing.T, master *Master, gpuPoolID string) string {
	t.Helper()
	id := newID("session")
	master.mu.Lock()
	master.sessionsByID[id] = &session{id: id, sessionType: SessionTypeCPUGPU, status: SessionStatusReady, cpuGPU: &cpuGPUSession{gpuPoolID: gpuPoolID}, createdAt: time.Now(), lastActivityAt: time.Now()}
	master.mu.Unlock()
	return id
}

func TestResourceTimingLifecycle(t *testing.T) {
	kubernetes := &fakeKubernetes{cpuPodNotReady: true, cpuProbeStarted: make(chan struct{}, 1), cpuProbeRelease: make(chan struct{})}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	pool, serviceError := sandboxMaster.CreateGPUPool(context.Background(), CreateGPUPoolRequest{Image: "image", Replicas: 1})
	if serviceError != nil || pool.Timing == nil || pool.Timing.CreateMS == nil || pool.Timing.ReadyMS != nil {
		t.Fatalf("pool=%+v error=%+v", pool, serviceError)
	}
	syncMaster(t, sandboxMaster)
	pool, serviceError = sandboxMaster.GetGPUPool(pool.GPUPoolID)
	if serviceError != nil || pool.Timing == nil || pool.Timing.ReadyMS == nil {
		t.Fatalf("pool=%+v error=%+v", pool, serviceError)
	}

	session, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeCPUGPU, GPUPoolID: pool.GPUPoolID})
	if serviceError != nil || session.Status != SessionStatusCreating || session.Timing == nil || session.Timing.CreateMS == nil || session.Timing.ReadyMS != nil {
		t.Fatalf("session=%+v error=%+v", session, serviceError)
	}
	pods, err := kubernetes.ListCPUPods(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	pods[0].Ready = true
	sandboxMaster.syncCPUPods(pods)
	select {
	case <-kubernetes.cpuProbeStarted:
	case <-time.After(time.Second):
		t.Fatal("CPU pod connectivity probe did not start")
	}
	if current, serviceError := sandboxMaster.GetSession(session.SessionID); serviceError != nil || current.Status != SessionStatusCreating {
		t.Fatalf("session was ready before connectivity probe: session=%+v error=%+v", current, serviceError)
	}
	kubernetes.mu.Lock()
	kubernetes.cpuPods[0].Ready = true
	kubernetes.mu.Unlock()
	close(kubernetes.cpuProbeRelease)
	session = waitForSessionStatus(t, sandboxMaster, session.SessionID, SessionStatusReady)
	session, serviceError = sandboxMaster.GetSession(session.SessionID)
	if serviceError != nil || session.Status != SessionStatusReady || session.Timing == nil || session.Timing.ReadyMS == nil {
		t.Fatalf("session=%+v error=%+v", session, serviceError)
	}
	pods, _ = kubernetes.ListCPUPods(context.Background())
	sandboxMaster.syncCPUPods(pods)
	kubernetes.mu.Lock()
	defer kubernetes.mu.Unlock()
	if kubernetes.cpuProbeCalls != 1 {
		t.Fatalf("ready session was probed again: probes=%d", kubernetes.cpuProbeCalls)
	}
}

func TestGPUOnlySessionProvisionsDedicatedPod(t *testing.T) {
	kubernetes := &fakeKubernetes{gpuSessionReady: true}
	sandboxMaster := New(testGPUSessionConfig(), kubernetes, testGPUSessionClients(kubernetes), nil)
	resources := &ResourceRequirements{Requests: map[string]string{"cpu": "4", "memory": "16Gi", "example.org/gpu_count_best_effort": "1"}, Limits: map[string]string{"example.org/gpu_count_best_effort": "1"}}
	session, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeGPUOnly, Image: "gpu-image", Resources: resources})
	if serviceError != nil {
		t.Fatalf("session=%+v error=%+v", session, serviceError)
	}
	if session.SessionType != SessionTypeGPUOnly || session.GPUPoolID != "" || session.CPU != nil || session.GPU == nil {
		t.Fatalf("session=%+v", session)
	}
	if session.GPU.Image != "gpu-image" || session.GPU.Resources == nil || session.GPU.Resources.Requests["example.org/gpu_count_best_effort"] != "1" {
		t.Fatalf("gpu=%+v", session.GPU)
	}
	if session.Status != SessionStatusReady || session.Timing == nil || session.Timing.CreateMS == nil || session.Timing.ReadyMS == nil {
		t.Fatalf("session=%+v", session)
	}
	kubernetes.mu.Lock()
	defer kubernetes.mu.Unlock()
	if kubernetes.gpuSessionCreateCount != 1 || kubernetes.gpuSessionImage != "gpu-image" || len(kubernetes.gpuSessionPods) != 1 {
		t.Fatalf("create_count=%d image=%q pods=%d", kubernetes.gpuSessionCreateCount, kubernetes.gpuSessionImage, len(kubernetes.gpuSessionPods))
	}
}

func TestGPUOnlySessionBecomesReadyAfterProbe(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testGPUSessionConfig(), kubernetes, testGPUSessionClients(kubernetes), nil)
	session, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeGPUOnly, Image: "gpu-image", Resources: &ResourceRequirements{Requests: map[string]string{"example.org/gpu_count_best_effort": "1"}}})
	if serviceError != nil || session.Status != SessionStatusCreating || session.GPU == nil || session.GPU.Ready {
		t.Fatalf("session=%+v error=%+v", session, serviceError)
	}
	kubernetes.mu.Lock()
	kubernetes.gpuSessionReady = true
	kubernetes.mu.Unlock()
	for deadline := time.Now().Add(2 * time.Second); time.Now().Before(deadline); time.Sleep(time.Millisecond) {
		pods, err := kubernetes.ListGPUSessionPods(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		sandboxMaster.syncGPUSessionPods("h20", pods)
		current, serviceError := sandboxMaster.GetSession(session.SessionID)
		if serviceError != nil {
			t.Fatal(serviceError)
		}
		if current.Status == SessionStatusReady {
			if current.GPU == nil || !current.GPU.Ready || current.Timing == nil || current.Timing.ReadyMS == nil {
				t.Fatalf("session=%+v", current)
			}
			if serviceError := sandboxMaster.DeleteSession(context.Background(), session.SessionID); serviceError != nil {
				t.Fatal(serviceError)
			}
			kubernetes.mu.Lock()
			defer kubernetes.mu.Unlock()
			if kubernetes.deletedGPU != 1 {
				t.Fatalf("deleted_gpu=%d", kubernetes.deletedGPU)
			}
			return
		}
		if current.Status == SessionStatusFailed {
			t.Fatalf("session failed: %+v", current)
		}
	}
	t.Fatal("gpu_only session did not become ready")
}

func TestGPUOnlySessionIsRestoredFromModelCluster(t *testing.T) {
	createdAt := time.Now().Add(-time.Minute)
	resources := &ResourceRequirements{Requests: map[string]string{"cpu": "4", "example.org/a100": "1"}, Limits: map[string]string{"example.org/a100": "1"}}
	defaultKubernetes := &fakeKubernetes{}
	modelKubernetes := &fakeKubernetes{gpuSessionPods: []PodInfo{{Name: "gpu-pool-gpu-session-restored", UID: "gpu-uid", SessionID: "session-restored", Image: "gpu-image", Resources: resources, CreatedAt: createdAt, Ready: true}}}
	sandboxMaster := New(testConfigWithGPUSessionClusters(map[string][]string{"h20": {"example.org/a100"}}), defaultKubernetes, map[string]GPUSessionKubernetesClient{"h20": modelKubernetes}, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sandboxMaster.Start(ctx)

	for deadline := time.Now().Add(time.Second); time.Now().Before(deadline); time.Sleep(time.Millisecond) {
		session, serviceError := sandboxMaster.GetSession("session-restored")
		if serviceError != nil {
			continue
		}
		if session.SessionType != SessionTypeGPUOnly || session.Status != SessionStatusReady || session.CreatedAt != createdAt || session.GPU == nil || session.GPU.Pod != "gpu-pool-gpu-session-restored" || session.GPU.Image != "gpu-image" || session.GPU.Resources.Requests["example.org/a100"] != "1" || session.Timing != nil {
			t.Fatalf("session=%+v", session)
		}
		response, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 1})
		if serviceError != nil || response.Stdout != "gpu-session-ok" {
			t.Fatalf("response=%+v error=%v", response, serviceError)
		}
		modelKubernetes.mu.Lock()
		modelCalls := modelKubernetes.gpuSessionExecCalls
		modelKubernetes.mu.Unlock()
		defaultKubernetes.mu.Lock()
		defaultCalls := defaultKubernetes.gpuSessionExecCalls
		defaultKubernetes.mu.Unlock()
		if modelCalls != 1 || defaultCalls != 0 {
			t.Fatalf("execution calls: model=%d default=%d", modelCalls, defaultCalls)
		}
		return
	}
	t.Fatal("GPU-only Session was not restored")
}

func TestGPUOnlySessionFailsWhenPodDisappears(t *testing.T) {
	kubernetes := &fakeKubernetes{gpuSessionPods: []PodInfo{{Name: "gpu-session-restored", UID: "gpu-uid", SessionID: "session-restored", Image: "gpu-image", Resources: &ResourceRequirements{Requests: map[string]string{"example.org/a100": "1"}}, Ready: true}}}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	pods, err := kubernetes.ListGPUSessionPods(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	sandboxMaster.syncGPUSessionPods("", pods)
	sandboxMaster.syncGPUSessionPods("", nil)
	session, serviceError := sandboxMaster.GetSession("session-restored")
	if serviceError != nil || session.Status != SessionStatusFailed || session.Error != "GPU Session Pod disappeared" || session.GPU == nil || session.GPU.Ready {
		t.Fatalf("session=%+v error=%v", session, serviceError)
	}
}

func TestDeletedGPUOnlySessionIsNotRestoredFromStalePodList(t *testing.T) {
	pod := PodInfo{Name: "gpu-session-deleted", UID: "gpu-uid", SessionID: "session-deleted", Image: "gpu-image", Resources: &ResourceRequirements{Requests: map[string]string{"example.org/a100": "1"}}, Ready: true}
	kubernetes := &fakeKubernetes{gpuSessionPods: []PodInfo{pod}}
	sandboxMaster := New(testGPUSessionConfig(), kubernetes, testGPUSessionClients(kubernetes), nil)
	sandboxMaster.syncGPUSessionPods("h20", []PodInfo{pod})
	stalePods, err := kubernetes.ListGPUSessionPods(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if serviceError := sandboxMaster.DeleteSession(context.Background(), pod.SessionID); serviceError != nil {
		t.Fatal(serviceError)
	}

	sandboxMaster.syncGPUSessionPods("h20", stalePods)
	if sessions := sandboxMaster.ListSessions(); len(sessions) != 0 {
		t.Fatalf("deleted Session was restored from stale Pod list: %+v", sessions)
	}
	if current := sandboxMaster.sessionsByID[pod.SessionID]; current == nil || current.status != SessionStatusDeleting {
		t.Fatal("GPU Session deletion tombstone was cleared while the stale Pod was still listed")
	}
	sandboxMaster.syncGPUSessionPods("h20", nil)
	if _, exists := sandboxMaster.sessionsByID[pod.SessionID]; exists {
		t.Fatal("GPU Session deletion tombstone was not cleared after the Pod disappeared")
	}
}

func TestGPUOnlySessionRoutesByResourceModel(t *testing.T) {
	defaultKube := &fakeKubernetes{gpuSessionReady: true}
	modelKube := &fakeKubernetes{gpuSessionReady: true}
	sandboxMaster := New(testConfigWithGPUSessionClusters(map[string][]string{"h20": {"example.org/a100"}}), defaultKube, map[string]GPUSessionKubernetesClient{"h20": modelKube}, nil)

	routed, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeGPUOnly, Image: "gpu-image", Resources: &ResourceRequirements{Requests: map[string]string{"cpu": "4", "example.org/a100": "1"}}})
	if serviceError != nil || routed.Status != SessionStatusReady {
		t.Fatalf("routed=%+v error=%+v", routed, serviceError)
	}
	_, serviceError = sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeGPUOnly, Image: "gpu-image", Resources: &ResourceRequirements{Requests: map[string]string{"example.org/unknown": "1"}}})
	if serviceError == nil || serviceError.Kind != ErrorInvalid {
		t.Fatalf("error=%+v", serviceError)
	}
	_, serviceError = sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeGPUOnly, Image: "gpu-image", Resources: &ResourceRequirements{Requests: map[string]string{"example.org/a100": "0"}}})
	if serviceError == nil || serviceError.Kind != ErrorInvalid {
		t.Fatalf("zero GPU error=%+v", serviceError)
	}
	if sessions := sandboxMaster.ListSessions(); len(sessions) != 1 || sessions[0].SessionID != routed.SessionID {
		t.Fatalf("invalid routes registered Sessions: %+v", sessions)
	}

	modelKube.mu.Lock()
	defer modelKube.mu.Unlock()
	defaultKube.mu.Lock()
	defer defaultKube.mu.Unlock()
	if modelKube.gpuSessionCreateCount != 1 || defaultKube.gpuSessionCreateCount != 0 {
		t.Fatalf("routing counts: model=%d default=%d", modelKube.gpuSessionCreateCount, defaultKube.gpuSessionCreateCount)
	}
}

func TestGPUOnlySessionRoutingRejectsMultipleClusters(t *testing.T) {
	first := &fakeKubernetes{}
	second := &fakeKubernetes{}
	sandboxMaster := New(
		testConfigWithGPUSessionClusters(map[string][]string{"h20": {"example.org/h20"}, "a100": {"example.org/a100"}}),
		first,
		map[string]GPUSessionKubernetesClient{"h20": first, "a100": second},
		nil,
	)
	_, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{
		SessionType: SessionTypeGPUOnly,
		Image:       "gpu-image",
		Resources:   &ResourceRequirements{Requests: map[string]string{"example.org/h20": "1", "example.org/a100": "1"}},
	})
	if serviceError == nil || serviceError.Kind != ErrorInvalid || !strings.Contains(serviceError.Message, "multiple") {
		t.Fatalf("error=%+v", serviceError)
	}
}

func TestGPUOnlySessionExecution(t *testing.T) {
	kubernetes := &fakeKubernetes{gpuSessionReady: true}
	sandboxMaster := New(testGPUSessionConfig(), kubernetes, testGPUSessionClients(kubernetes), nil)
	session, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{
		SessionType: SessionTypeGPUOnly,
		Image:       "gpu-image",
		Resources:   &ResourceRequirements{Requests: map[string]string{"example.org/gpu_count_best_effort": "1"}},
	})
	if serviceError != nil {
		t.Fatal(serviceError)
	}
	result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Command: []string{"true"}, TimeoutSeconds: 1})
	timing := commandTiming(result, command.DeviceGPU)
	if serviceError != nil || result.Stdout != "gpu-session-ok" || timing == nil || timing.QueueMS != nil || timing.PrecheckMS != nil {
		t.Fatalf("result=%+v error=%+v", result, serviceError)
	}
}

func TestGPUOnlySessionAllowsConcurrentCommands(t *testing.T) {
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	kubernetes := &fakeKubernetes{gpuSessionReady: true, gpuSessionExecStarted: started, gpuSessionExecRelease: release}
	sandboxMaster := New(testGPUSessionConfig(), kubernetes, testGPUSessionClients(kubernetes), nil)
	session, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{
		SessionType: SessionTypeGPUOnly,
		Image:       "gpu-image",
		Resources:   &ResourceRequirements{Requests: map[string]string{"example.org/gpu_count_best_effort": "1"}},
	})
	if serviceError != nil {
		t.Fatal(serviceError)
	}
	type outcome struct {
		result ExecuteCommandResponse
		err    *ServiceError
	}
	outcomes := make(chan outcome, 2)
	run := func() {
		result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Command: []string{"true"}, TimeoutSeconds: 5})
		outcomes <- outcome{result: result, err: serviceError}
	}
	go run()
	select {
	case <-started:
	case <-time.After(time.Second):
		close(release)
		t.Fatal("first GPU-only command did not start")
	}
	go run()
	select {
	case <-started:
	case <-time.After(time.Second):
		close(release)
		t.Fatal("second GPU-only command was not started concurrently")
	}
	close(release)
	for range 2 {
		outcome := <-outcomes
		if outcome.err != nil || outcome.result.Status != ExecutionStatusCompleted {
			t.Fatalf("outcome=%+v", outcome)
		}
	}
}

func TestGPUSessionMonitorTransitions(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfigWithGPUSessionClusters(map[string][]string{"h20": {"example.org/h20"}}), kubernetes, map[string]GPUSessionKubernetesClient{"h20": kubernetes}, nil)
	now := time.Now()
	sandboxMaster.sessionsByID["failed"] = &session{id: "failed", sessionType: SessionTypeGPUOnly, status: SessionStatusCreating, createdAt: now, gpuOnly: &gpuOnlySession{clusterName: "h20", gpuPod: &PodInfo{Name: "failed", UID: "failed"}}}
	sandboxMaster.sessionsByID["deleting"] = &session{id: "deleting", sessionType: SessionTypeGPUOnly, status: SessionStatusCreating, createdAt: now, gpuOnly: &gpuOnlySession{clusterName: "h20", gpuPod: &PodInfo{Name: "deleting", UID: "deleting"}}}
	sandboxMaster.sessionsByID["timeout"] = &session{id: "timeout", sessionType: SessionTypeGPUOnly, status: SessionStatusCreating, createdAt: now.Add(-gpuSessionReadyTimeout), gpuOnly: &gpuOnlySession{clusterName: "h20"}}
	sandboxMaster.syncGPUSessionPods("h20", []PodInfo{
		{Name: "failed", UID: "failed", SessionID: "failed", Failed: true},
		{Name: "deleting", UID: "deleting", SessionID: "deleting", Deleting: true},
	})
	for id, message := range map[string]string{
		"failed":   "GPU Session Pod failed",
		"deleting": "GPU Session Pod is deleting",
		"timeout":  "GPU Session Pod did not become ready in time",
	} {
		current, serviceError := sandboxMaster.GetSession(id)
		if serviceError != nil || current.Status != SessionStatusFailed || current.Error != message {
			t.Fatalf("session %s=%+v error=%+v", id, current, serviceError)
		}
	}
}

func TestMultipleGPUPoolsAndSessionBindings(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	firstID := createPool(t, sandboxMaster, "image-a")
	secondID := createPool(t, sandboxMaster, "image-b")
	if firstID == secondID || !strings.HasPrefix(firstID, "gpu-pool-") || !strings.HasPrefix(secondID, "gpu-pool-") {
		t.Fatalf("pool IDs=%q %q", firstID, secondID)
	}
	if pools := sandboxMaster.ListGPUPools(); len(pools) != 2 || pools[0].GPUPoolID > pools[1].GPUPoolID {
		t.Fatalf("pools=%+v", pools)
	}
	firstSession, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeCPUGPU, GPUPoolID: firstID})
	if serviceError != nil || firstSession.GPUPoolID != firstID || firstSession.CPU.Image != "image-a" {
		t.Fatalf("session=%+v error=%+v", firstSession, serviceError)
	}
	if serviceError := sandboxMaster.DeleteGPUPool(context.Background(), firstID); serviceError == nil || serviceError.Code != "GPU_POOL_IN_USE" {
		t.Fatalf("error=%+v", serviceError)
	}
	if serviceError := sandboxMaster.DeleteSession(context.Background(), firstSession.SessionID); serviceError != nil {
		t.Fatal(serviceError)
	}
	if serviceError := sandboxMaster.DeleteGPUPool(context.Background(), firstID); serviceError != nil {
		t.Fatal(serviceError)
	}
	replicas := int32(4)
	updated, serviceError := sandboxMaster.UpdateGPUPool(context.Background(), secondID, UpdateGPUPoolRequest{Image: "image-b2", Replicas: &replicas})
	if serviceError != nil || updated.Image != "image-b2" || updated.Replicas != 4 || len(sandboxMaster.ListGPUPools()) != 1 {
		t.Fatalf("pool=%+v error=%+v", updated, serviceError)
	}
}

func TestGPUPoolUsesConfiguredPrefix(t *testing.T) {
	cfg := testConfig()
	cfg.GPUPoolCluster.PoolNamePrefix = "custom-pool"
	sandboxMaster := New(cfg, &fakeKubernetes{}, nil, nil)
	if gpuPoolID := createPool(t, sandboxMaster, "image"); !strings.HasPrefix(gpuPoolID, "custom-pool-") || len(gpuPoolID) != len("custom-pool-")+16 {
		t.Fatalf("gpu_pool_id=%q", gpuPoolID)
	}
}

func TestCreateSessionRequiresExistingGPUPool(t *testing.T) {
	sandboxMaster := New(testConfig(), &fakeKubernetes{}, nil, nil)
	for _, testCase := range []struct {
		request CreateSessionRequest
		code    string
		kind    ErrorKind
	}{
		{CreateSessionRequest{SessionType: SessionTypeGPUOnly}, "", ErrorInvalid},
		{CreateSessionRequest{SessionType: SessionTypeCPUGPU}, "", ErrorInvalid},
		{CreateSessionRequest{SessionType: SessionTypeCPUGPU, GPUPoolID: "gpu-pool-missing"}, "GPU_POOL_NOT_FOUND", ErrorNotFound},
	} {
		_, serviceError := sandboxMaster.CreateSession(context.Background(), testCase.request)
		if serviceError == nil || serviceError.Kind != testCase.kind || (testCase.code != "" && serviceError.Code != testCase.code) {
			t.Fatalf("request=%+v error=%+v", testCase.request, serviceError)
		}
	}
}

func TestGPUQueuesAreIndependentPerPool(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	startGPUQueue(t, sandboxMaster)
	firstID := createPool(t, sandboxMaster, "image-a")
	secondID := createPool(t, sandboxMaster, "image-b")
	firstSessionID := readyGPUSession(t, sandboxMaster, firstID)
	secondSessionID := readyGPUSession(t, sandboxMaster, secondID)
	kubernetes.mu.Lock()
	firstPod, secondPod := kubernetes.gpuPods[0], kubernetes.gpuPods[1]
	kubernetes.gpuPods = []PodInfo{secondPod}
	kubernetes.mu.Unlock()
	syncMaster(t, sandboxMaster)

	firstDone := make(chan ExecuteCommandResponse, 1)
	go func() {
		response, _ := sandboxMaster.ExecuteCommand(context.Background(), firstSessionID, command.Request{Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 5})
		firstDone <- response
	}()
	for deadline := time.Now().Add(time.Second); time.Now().Before(deadline); {
		sandboxMaster.mu.Lock()
		pool := sandboxMaster.gpuPoolsByID[firstID]
		queued := pool != nil && len(pool.queue) == 1
		sandboxMaster.mu.Unlock()
		if queued {
			break
		}
		time.Sleep(time.Millisecond)
	}
	second, serviceError := sandboxMaster.ExecuteCommand(context.Background(), secondSessionID, command.Request{Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 5})
	if serviceError != nil || second.Status != ExecutionStatusCompleted {
		t.Fatalf("second=%+v error=%+v", second, serviceError)
	}
	kubernetes.mu.Lock()
	kubernetes.gpuPods = append(kubernetes.gpuPods, firstPod)
	kubernetes.mu.Unlock()
	syncMaster(t, sandboxMaster)
	if first := <-firstDone; first.Status != ExecutionStatusCompleted {
		t.Fatalf("first=%+v", first)
	}
	kubernetes.mu.Lock()
	defer kubernetes.mu.Unlock()
	if len(kubernetes.gpuExecutedPools) != 2 || kubernetes.gpuExecutedPools[0] != secondID || kubernetes.gpuExecutedPools[1] != firstID {
		t.Fatalf("executed pools=%v", kubernetes.gpuExecutedPools)
	}
}

func TestRestoresMultipleGPUPoolsAndWorkers(t *testing.T) {
	sandboxMaster := New(testConfig(), &fakeKubernetes{}, nil, nil)
	sandboxMaster.syncGPUPools("", []GPUPoolInfo{{GPUPoolID: "gpu-pool-a", Image: "image-a", Replicas: 1}, {GPUPoolID: "gpu-pool-b", Image: "image-b", Replicas: 2}})
	sandboxMaster.syncGPUWorkers(context.Background(), "", []PodInfo{{Name: "worker-a", UID: "uid-a", GPUPoolID: "gpu-pool-a", Ready: true}, {Name: "worker-b", UID: "uid-b", GPUPoolID: "gpu-pool-b", Ready: true}})
	if pools := sandboxMaster.ListGPUPools(); len(pools) != 2 || pools[0].Workers.Idle != 1 || pools[1].Workers.Idle != 1 {
		t.Fatalf("pools=%+v", pools)
	}
}

func TestMissingGPUPoolIsRecreatedWithOriginalState(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "gpu-image")
	createdAt := time.Now().Add(-time.Hour)
	lastActivityAt := time.Now().Add(-time.Minute)
	createDuration := 3 * time.Second
	readyDuration := 7 * time.Second
	sandboxMaster.mu.Lock()
	pool := sandboxMaster.gpuPoolsByID[gpuPoolID]
	pool.queue = []string{"execution-queued"}
	pool.createdAt = createdAt
	pool.lastActivityAt = lastActivityAt
	pool.createDuration = &createDuration
	pool.readyDuration = &readyDuration
	sandboxMaster.sessionsByID["session-existing"] = &session{id: "session-existing", sessionType: SessionTypeCPUGPU, cpuGPU: &cpuGPUSession{gpuPoolID: gpuPoolID}}
	sandboxMaster.mu.Unlock()
	kubernetes.mu.Lock()
	delete(kubernetes.gpuPools, gpuPoolID)
	kubernetes.gpuPoolImage = ""
	kubernetes.mu.Unlock()

	recreations := sandboxMaster.syncGPUPools("", nil)
	if len(recreations) != 1 || recreations[0] != (gpuPoolRecreation{id: gpuPoolID, image: "gpu-image", replicas: 1, scheduling: pool.scheduling}) {
		t.Fatalf("recreations=%+v", recreations)
	}
	if duplicates := sandboxMaster.syncGPUPools("", nil); len(duplicates) != 0 {
		t.Fatalf("duplicate recreations=%+v", duplicates)
	}
	sandboxMaster.recreateGPUPool(context.Background(), recreations[0])

	kubernetes.mu.Lock()
	recreated, exists := kubernetes.gpuPools[gpuPoolID]
	createCount := kubernetes.gpuPoolCreateCount
	kubernetes.mu.Unlock()
	if !exists || recreated.Image != "gpu-image" || recreated.Replicas != 1 || createCount != 2 {
		t.Fatalf("recreated=%+v exists=%t create_count=%d", recreated, exists, createCount)
	}
	sandboxMaster.mu.Lock()
	defer sandboxMaster.mu.Unlock()
	current := sandboxMaster.gpuPoolsByID[gpuPoolID]
	if current != pool || current.recreating || len(current.queue) != 1 || current.queue[0] != "execution-queued" || !current.createdAt.Equal(createdAt) || !current.lastActivityAt.Equal(lastActivityAt) || current.createDuration != &createDuration || current.readyDuration != &readyDuration {
		t.Fatalf("pool=%+v", current)
	}
	if sandboxMaster.sessionsByID["session-existing"] == nil {
		t.Fatal("existing Session was removed during GPU Pool recreation")
	}
}

func TestGPUPoolListFailureDoesNotTriggerRecreation(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "gpu-image")
	kubernetes.mu.Lock()
	delete(kubernetes.gpuPools, gpuPoolID)
	kubernetes.gpuPoolImage = ""
	kubernetes.gpuPoolListErr = errors.New("list failed")
	kubernetes.mu.Unlock()
	ctx, cancel := context.WithCancel(context.Background())
	go sandboxMaster.gpuPodSyncLoop(ctx)
	time.Sleep(50 * time.Millisecond)
	cancel()

	kubernetes.mu.Lock()
	createCount := kubernetes.gpuPoolCreateCount
	kubernetes.mu.Unlock()
	sandboxMaster.mu.Lock()
	recreating := sandboxMaster.gpuPoolsByID[gpuPoolID].recreating
	sandboxMaster.mu.Unlock()
	if createCount != 1 || recreating {
		t.Fatalf("create_count=%d recreating=%t", createCount, recreating)
	}
}

func TestFailedGPUPoolRecreationCanBeRetried(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "gpu-image")
	kubernetes.mu.Lock()
	delete(kubernetes.gpuPools, gpuPoolID)
	kubernetes.gpuPoolImage = ""
	kubernetes.gpuPoolCreateErr = errors.New("create failed")
	kubernetes.mu.Unlock()
	recreation := sandboxMaster.syncGPUPools("", nil)[0]
	sandboxMaster.recreateGPUPool(context.Background(), recreation)
	if retry := sandboxMaster.syncGPUPools("", nil); len(retry) != 1 || retry[0].id != gpuPoolID {
		t.Fatalf("retry=%+v", retry)
	}
}

func TestDeletingGPUPoolWinsRaceWithRecreation(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "gpu-image")
	kubernetes.mu.Lock()
	delete(kubernetes.gpuPools, gpuPoolID)
	kubernetes.gpuPoolImage = ""
	kubernetes.gpuPoolCreateStarted = make(chan struct{}, 1)
	kubernetes.gpuPoolCreateRelease = make(chan struct{})
	kubernetes.mu.Unlock()
	recreation := sandboxMaster.syncGPUPools("", nil)[0]
	done := make(chan struct{})
	go func() {
		sandboxMaster.recreateGPUPool(context.Background(), recreation)
		close(done)
	}()
	select {
	case <-kubernetes.gpuPoolCreateStarted:
	case <-time.After(time.Second):
		t.Fatal("GPU Pool recreation did not start")
	}
	if serviceError := sandboxMaster.DeleteGPUPool(context.Background(), gpuPoolID); serviceError != nil {
		t.Fatal(serviceError)
	}
	close(kubernetes.gpuPoolCreateRelease)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("GPU Pool recreation did not finish")
	}

	kubernetes.mu.Lock()
	_, exists := kubernetes.gpuPools[gpuPoolID]
	deleteCount := kubernetes.gpuPoolDeleteCount
	kubernetes.mu.Unlock()
	if exists || deleteCount != 2 {
		t.Fatalf("deployment_exists=%t delete_count=%d", exists, deleteCount)
	}
	sandboxMaster.syncGPUPools("", nil)
	if _, exists := sandboxMaster.gpuPoolsByID[gpuPoolID]; exists {
		t.Fatal("GPU Pool deletion tombstone was not cleared")
	}
}

func TestDeletedGPUPoolIsNotRestoredFromStaleList(t *testing.T) {
	sandboxMaster := New(testConfig(), &fakeKubernetes{}, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "image")
	stale := []GPUPoolInfo{{GPUPoolID: gpuPoolID, Image: "image", Replicas: 1}}
	if serviceError := sandboxMaster.DeleteGPUPool(context.Background(), gpuPoolID); serviceError != nil {
		t.Fatal(serviceError)
	}
	sandboxMaster.syncGPUPools("", stale)
	if _, serviceError := sandboxMaster.GetGPUPool(gpuPoolID); serviceError == nil {
		t.Fatal("deleted GPU Pool was restored from a stale list")
	}
	if pool := sandboxMaster.gpuPoolsByID[gpuPoolID]; pool == nil || !pool.deleting {
		t.Fatal("GPU Pool deletion tombstone was cleared while the stale Deployment was still listed")
	}
	sandboxMaster.syncGPUPools("", nil)
	if _, exists := sandboxMaster.gpuPoolsByID[gpuPoolID]; exists {
		t.Fatal("GPU Pool deletion tombstone was not cleared after the Deployment disappeared")
	}
}

func TestSessionUsesCustomCPUImage(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "gpu-image")
	if _, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeCPUGPU, GPUPoolID: gpuPoolID, Image: "cpu-image"}); serviceError != nil {
		t.Fatal(serviceError)
	}
	if kubernetes.cpuImage != "cpu-image" {
		t.Fatalf("CPU image=%q", kubernetes.cpuImage)
	}
}

func TestCPUSessionIsRestoredFromPod(t *testing.T) {
	createdAt := time.Now().Add(-time.Minute)
	kubernetes := &fakeKubernetes{cpuPods: []PodInfo{{Name: "kernel-sandbox-session-restored", UID: "cpu-uid", SessionID: "session-restored", GPUPoolID: "gpu-pool-restored", Image: "cpu-image", Resources: &ResourceRequirements{Requests: map[string]string{"cpu": "6", "memory": "24Gi"}}, CreatedAt: createdAt, Ready: true}}}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go sandboxMaster.cpuPodSyncLoop(ctx)
	for deadline := time.Now().Add(time.Second); time.Now().Before(deadline); {
		if sessions := sandboxMaster.ListSessions(); len(sessions) == 1 {
			if sessions[0].SessionID != "session-restored" || sessions[0].GPUPoolID != "gpu-pool-restored" || sessions[0].Status != SessionStatusReady || sessions[0].CreatedAt != createdAt || sessions[0].CPU.Image != "cpu-image" || sessions[0].CPU.Resources.Requests["cpu"] != "6" || sessions[0].Timing != nil {
				t.Fatalf("session=%+v", sessions[0])
			}
			sandboxMaster.mu.Lock()
			lastActivityAt := sandboxMaster.sessionsByID[sessions[0].SessionID].lastActivityAt
			sandboxMaster.mu.Unlock()
			if lastActivityAt.IsZero() || time.Since(lastActivityAt) > time.Second {
				t.Fatalf("last activity=%s", lastActivityAt)
			}
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("CPU Session was not restored")
}

func TestDeletedCPUSessionIsNotRestoredFromStalePodList(t *testing.T) {
	pod := PodInfo{Name: "cpu-session-deleted", UID: "cpu-uid", SessionID: "session-deleted", GPUPoolID: "gpu-pool-test", Ready: true}
	kubernetes := &fakeKubernetes{cpuPods: []PodInfo{pod}}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	sandboxMaster.sessionsByID[pod.SessionID] = &session{id: pod.SessionID, sessionType: SessionTypeCPUGPU, status: SessionStatusReady, cpuGPU: &cpuGPUSession{gpuPoolID: pod.GPUPoolID, cpuPod: &pod}}

	stalePods, err := kubernetes.ListCPUPods(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if serviceError := sandboxMaster.DeleteSession(context.Background(), pod.SessionID); serviceError != nil {
		t.Fatal(serviceError)
	}

	sandboxMaster.syncCPUPods(stalePods)
	if sessions := sandboxMaster.ListSessions(); len(sessions) != 0 {
		t.Fatalf("deleted session was restored from stale pod list: %+v", sessions)
	}
	if session := sandboxMaster.sessionsByID[pod.SessionID]; session == nil || session.status != SessionStatusDeleting {
		t.Fatal("deletion tombstone was cleared while the stale pod was still listed")
	}

	sandboxMaster.syncCPUPods(nil)
	if _, exists := sandboxMaster.sessionsByID[pod.SessionID]; exists {
		t.Fatal("deletion tombstone was not cleared after the pod disappeared")
	}
}

func TestDeleteSessionWhileCPUPodIsCreating(t *testing.T) {
	kubernetes := &fakeKubernetes{cpuCreateStarted: make(chan struct{}, 1), cpuCreateRelease: make(chan struct{})}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "image")
	type createResult struct {
		response SessionResponse
		err      *ServiceError
	}
	done := make(chan createResult, 1)
	go func() {
		response, serviceError := sandboxMaster.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeCPUGPU, GPUPoolID: gpuPoolID})
		done <- createResult{response: response, err: serviceError}
	}()
	select {
	case <-kubernetes.cpuCreateStarted:
	case <-time.After(time.Second):
		t.Fatal("CPU Pod creation did not start")
	}
	sandboxMaster.mu.Lock()
	var sessionID string
	for id := range sandboxMaster.sessionsByID {
		sessionID = id
	}
	sandboxMaster.mu.Unlock()
	if serviceError := sandboxMaster.DeleteSession(context.Background(), sessionID); serviceError != nil {
		t.Fatal(serviceError)
	}
	sandboxMaster.syncCPUPods(nil)
	if len(sandboxMaster.ListSessions()) != 0 || sandboxMaster.sessionsByID[sessionID] == nil {
		t.Fatal("creating Session deletion tombstone was not retained and hidden")
	}
	close(kubernetes.cpuCreateRelease)
	result := <-done
	if result.err == nil || result.err.Code != "SESSION_DELETED" {
		t.Fatalf("response=%+v error=%+v", result.response, result.err)
	}
	sandboxMaster.mu.Lock()
	stalePod := *sandboxMaster.sessionsByID[sessionID].cpuGPU.cpuPod
	sandboxMaster.mu.Unlock()
	sandboxMaster.syncCPUPods([]PodInfo{stalePod})
	if len(sandboxMaster.ListSessions()) != 0 || sandboxMaster.sessionsByID[sessionID] == nil {
		t.Fatal("Session was restored from a stale Pod list")
	}
	sandboxMaster.syncCPUPods(nil)
	if _, exists := sandboxMaster.sessionsByID[sessionID]; exists {
		t.Fatal("Session deletion tombstone was not cleared")
	}
}

func TestDeleteAllSessionsKeepsIdleGPUWorkers(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "image")
	readyGPUSession(t, sandboxMaster, gpuPoolID)
	pods, _ := kubernetes.ListGPUPods(context.Background())
	sandboxMaster.syncGPUWorkers(context.Background(), "", pods)
	if serviceError := sandboxMaster.DeleteAllSessions(context.Background()); serviceError != nil {
		t.Fatal(serviceError)
	}
	kubernetes.mu.Lock()
	deletedGPU := kubernetes.deletedGPU
	kubernetes.mu.Unlock()
	if len(sandboxMaster.ListSessions()) != 0 || len(sandboxMaster.gpuWorkersByUID) != 1 || deletedGPU != 0 {
		t.Fatalf("sessions=%d workers=%d deleted_gpu=%d", len(sandboxMaster.ListSessions()), len(sandboxMaster.gpuWorkersByUID), deletedGPU)
	}
}

func startGPUQueue(t *testing.T, master *Master) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go master.gpuCommandDispatchLoop(ctx)
}

func syncMaster(t *testing.T, master *Master) {
	t.Helper()
	gpuPods, err := master.kubernetes.ListGPUPods(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	master.syncGPUWorkers(context.Background(), "", gpuPods)
	master.dispatchGPUCommands()
}

func TestCPUAndGPUExecution(t *testing.T) {
	logs := captureLogs(t)
	kubernetes := &fakeKubernetes{}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	session := prepareSession(t, sandboxMaster, &ResourceRequirements{Requests: map[string]string{"cpu": "6", "memory": "24Gi"}})

	cpu, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{
		Device: command.DeviceCPU, Command: []string{"true"}, TimeoutSeconds: 1,
	})
	cpuTiming := commandTiming(cpu, command.DeviceCPU)
	if serviceError != nil || cpu.Stdout != "cpu-ok" || cpuTiming == nil || cpuTiming.Attempts != 1 || cpuTiming.QueueMS != nil || cpuTiming.PrecheckMS != nil {
		t.Fatalf("cpu=%+v error=%v", cpu, serviceError)
	}
	gpu, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{
		Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 1,
	})
	gpuTiming := commandTiming(gpu, command.DeviceGPU)
	if serviceError != nil || gpu.Stdout != "gpu-ok" || gpuTiming == nil || gpuTiming.Attempts != 1 || gpuTiming.QueueMS == nil || gpuTiming.PrecheckMS == nil {
		t.Fatalf("gpu=%+v error=%v", gpu, serviceError)
	}

	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		kubernetes.mu.Lock()
		deleted, checks := kubernetes.deletedGPU, kubernetes.stateCalls
		resources := kubernetes.cpuResources
		sandboxURL := kubernetes.cpuSandboxURL
		kubernetes.mu.Unlock()
		if checks >= 2 {
			if deleted != 0 || resources.Requests["cpu"] != "6" || resources.Requests["memory"] != "24Gi" || sandboxURL != "http://master:8765/v1/session/"+session.SessionID {
				t.Fatalf("checks=%d resources=%+v sandboxURL=%s", checks, resources, sandboxURL)
			}
			if output := logs.String(); strings.Count(output, `msg="Command accepted"`) != 2 || strings.Count(output, `msg="Command finished"`) != 2 || strings.Count(output, "exit_code=0") != 2 || strings.Contains(output, "http_status=200") {
				t.Fatalf("logs=%s", output)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("GPU Worker Pod was not deleted after execution")
}

func TestCPUResponseIncludesGPUTimings(t *testing.T) {
	for _, test := range []struct {
		name     string
		gpuCount int
	}{{"none", 0}, {"one", 1}, {"multiple", 2}} {
		t.Run(test.name, func(t *testing.T) {
			kubernetes := &fakeKubernetes{cpuStarted: make(chan struct{}), cpuRelease: make(chan struct{})}
			sandboxMaster := New(testConfig(), kubernetes, nil, nil)
			session := prepareSession(t, sandboxMaster, nil)
			if test.gpuCount > 1 {
				kubernetes.mu.Lock()
				kubernetes.gpuPods = append(kubernetes.gpuPods, PodInfo{Name: session.GPUPoolID + "-1", UID: "gpu-uid-1", GPUPoolID: session.GPUPoolID, Ready: true})
				kubernetes.mu.Unlock()
				syncMaster(t, sandboxMaster)
			}
			type cpuResult struct {
				response ExecuteCommandResponse
				err      *ServiceError
			}
			done := make(chan cpuResult, 1)
			go func() {
				response, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceCPU, Command: []string{"true"}, TimeoutSeconds: 5})
				done <- cpuResult{response: response, err: serviceError}
			}()
			<-kubernetes.cpuStarted
			for range test.gpuCount {
				response, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 1})
				if serviceError != nil || commandTiming(response, command.DeviceGPU) == nil {
					t.Fatalf("gpu=%+v error=%v", response, serviceError)
				}
			}
			close(kubernetes.cpuRelease)
			result := <-done
			if result.err != nil || result.response.Extra == nil || result.response.Extra.SandboxTimings == nil || result.response.Extra.SandboxTimings.CPUExecution == nil || len(result.response.Extra.SandboxTimings.GPUExecutions) != test.gpuCount {
				t.Fatalf("cpu=%+v error=%v", result.response, result.err)
			}
			for _, timing := range result.response.Extra.SandboxTimings.GPUExecutions {
				if timing.QueueMS == nil {
					t.Fatalf("cpu=%+v gpu_timing=%+v", result.response, timing)
				}
			}
		})
	}
}

func TestDeletingGPUWorkerIsNotSchedulable(t *testing.T) {
	sandboxMaster := New(testConfig(), &fakeKubernetes{}, nil, nil)
	sandboxMaster.syncGPUWorkers(context.Background(), "", []PodInfo{{Name: "gpu-0", UID: "gpu-uid", Ready: true, Deleting: true}})
	sandboxMaster.mu.Lock()
	defer sandboxMaster.mu.Unlock()
	if len(sandboxMaster.gpuWorkersByUID) != 0 {
		t.Fatalf("workers=%+v", sandboxMaster.gpuWorkersByUID)
	}
}

func TestCPUCommandsAreSerialAndDoNotDeletePod(t *testing.T) {
	kubernetes := &fakeKubernetes{cpuStarted: make(chan struct{}), cpuRelease: make(chan struct{})}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	session := prepareSession(t, sandboxMaster, nil)
	firstDone := make(chan struct{})
	go func() {
		_, _ = sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceCPU, Command: []string{"sleep"}, TimeoutSeconds: 5})
		close(firstDone)
	}()
	<-kubernetes.cpuStarted
	_, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceCPU, Command: []string{"true"}, TimeoutSeconds: 1})
	if serviceError == nil || serviceError.Code != "CPU_BUSY" {
		t.Fatalf("error=%+v", serviceError)
	}
	close(kubernetes.cpuRelease)
	<-firstDone
	kubernetes.mu.Lock()
	defer kubernetes.mu.Unlock()
	if kubernetes.deletedCPU != 0 {
		t.Fatalf("CPU Pod was deleted %d times", kubernetes.deletedCPU)
	}
}

func TestGPUPreCheckFailureSkipsCommand(t *testing.T) {
	kubernetes := &fakeKubernetes{gpuStates: []GPUPodState{{}, {ProcessCount: 1}}}
	cfg := testConfig()
	sandboxMaster := New(cfg, kubernetes, nil, nil)
	startGPUQueue(t, sandboxMaster)
	gpuPoolID := createPool(t, sandboxMaster, "image")
	sessionID := readyGPUSession(t, sandboxMaster, gpuPoolID)
	syncMaster(t, sandboxMaster)
	result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), sessionID, command.Request{Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 1})
	timing := commandTiming(result, command.DeviceGPU)
	if serviceError == nil || result.ErrorCode != "INFRASTRUCTURE_ERROR" || timing == nil || timing.QueueMS == nil || timing.PrecheckMS == nil || timing.Attempts != 1 || serviceError.Extra != result.Extra {
		t.Fatalf("result=%+v error=%+v", result, serviceError)
	}
	kubernetes.mu.Lock()
	defer kubernetes.mu.Unlock()
	if kubernetes.gpuExecCalls != 0 {
		t.Fatalf("GPU command executed %d times", kubernetes.gpuExecCalls)
	}
}

func TestGPUPreCheckLogsEachFailureCause(t *testing.T) {
	logs := captureLogs(t)
	kubernetes := &fakeKubernetes{}
	cfg := testConfig()
	cfg.Execution.MaxAttempts = 3
	sandboxMaster := New(cfg, kubernetes, nil, nil)
	startGPUQueue(t, sandboxMaster)
	gpuPoolID := createPool(t, sandboxMaster, "image")
	kubernetes.mu.Lock()
	kubernetes.gpuPods = append(
		kubernetes.gpuPods,
		PodInfo{Name: gpuPoolID + "-1", UID: "gpu-uid-1", GPUPoolID: gpuPoolID, Ready: true},
		PodInfo{Name: gpuPoolID + "-2", UID: "gpu-uid-2", GPUPoolID: gpuPoolID, Ready: true},
	)
	kubernetes.mu.Unlock()
	syncMaster(t, sandboxMaster)
	sessionID := readyGPUSession(t, sandboxMaster, gpuPoolID)
	kubernetes.mu.Lock()
	kubernetes.stateCalls = 0
	kubernetes.gpuStateErrors = []error{errors.New("sandboxd unavailable")}
	kubernetes.gpuStates = []GPUPodState{{}, {Busy: true}, {ProcessCount: 1, Processes: []GPUProcess{{PID: "42", ProcessName: "python"}}}}
	kubernetes.mu.Unlock()
	result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), sessionID, command.Request{Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 1})
	timing := commandTiming(result, command.DeviceGPU)
	if serviceError == nil || result.ErrorCode != "INFRASTRUCTURE_ERROR" || timing == nil || timing.Attempts != 3 || !strings.Contains(serviceError.Message, "retries exhausted after 3/3 attempts") || !strings.Contains(serviceError.Message, "1 residual GPU processes") {
		t.Fatalf("result=%+v error=%+v", result, serviceError)
	}
	kubernetes.mu.Lock()
	execCalls := kubernetes.gpuExecCalls
	kubernetes.mu.Unlock()
	output := logs.String()
	for _, expected := range []string{
		`attempt=1 error="GPU Worker state check failed: sandboxd unavailable"`,
		`attempt=2 error="GPU Worker is busy"`,
		`attempt=3 error="GPU Worker has 1 residual GPU processes:`,
		"error_code=INFRASTRUCTURE_ERROR",
	} {
		if !strings.Contains(output, expected) {
			t.Fatalf("missing %q in logs=%s", expected, output)
		}
	}
	if execCalls != 0 || strings.Count(output, `msg="GPU execution failed"`) != 3 {
		t.Fatalf("gpu_exec_calls=%d logs=%s", execCalls, output)
	}
}

func TestGPUTimingAccumulatesAcrossRetries(t *testing.T) {
	kubernetes := &fakeKubernetes{gpuStates: []GPUPodState{{}, {}, {ProcessCount: 1}, {}}}
	cfg := testConfig()
	cfg.Execution.MaxAttempts = 2
	sandboxMaster := New(cfg, kubernetes, nil, nil)
	startGPUQueue(t, sandboxMaster)
	gpuPoolID := createPool(t, sandboxMaster, "image")
	kubernetes.mu.Lock()
	kubernetes.gpuPods = append(kubernetes.gpuPods, PodInfo{Name: gpuPoolID + "-1", UID: "gpu-uid-1", GPUPoolID: gpuPoolID, Ready: true})
	kubernetes.mu.Unlock()
	sessionID := readyGPUSession(t, sandboxMaster, gpuPoolID)
	syncMaster(t, sandboxMaster)
	result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), sessionID, command.Request{Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 1})
	timing := commandTiming(result, command.DeviceGPU)
	if serviceError != nil || result.Status != ExecutionStatusCompleted || timing == nil || timing.Attempts != 2 || timing.QueueMS == nil || timing.PrecheckMS == nil {
		t.Fatalf("result=%+v error=%+v", result, serviceError)
	}
	kubernetes.mu.Lock()
	defer kubernetes.mu.Unlock()
	if kubernetes.gpuExecCalls != 1 {
		t.Fatalf("GPU command executed %d times", kubernetes.gpuExecCalls)
	}
}

func TestCPUInfrastructureFailureKeepsSessionReady(t *testing.T) {
	kubernetes := &fakeKubernetes{cpuResult: command.Result{Kind: command.ResultInfrastructure, Error: "exec failed"}}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	session := prepareSession(t, sandboxMaster, nil)
	result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceCPU, Command: []string{"true"}, TimeoutSeconds: 1})
	if serviceError == nil || serviceError.Kind != ErrorUnavailable || serviceError.Code != "INFRASTRUCTURE_ERROR" || result.Status != ExecutionStatusFailed {
		t.Fatalf("result=%+v error=%+v", result, serviceError)
	}
	current, serviceError := sandboxMaster.GetSession(session.SessionID)
	if serviceError != nil || current.Status != SessionStatusReady || current.Error != "" {
		t.Fatalf("session=%+v error=%+v", current, serviceError)
	}
	result, serviceError = sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceCPU, Command: []string{"true"}, TimeoutSeconds: 1})
	if serviceError == nil || serviceError.Kind != ErrorUnavailable || serviceError.Code != "INFRASTRUCTURE_ERROR" || result.Status != ExecutionStatusFailed {
		t.Fatalf("second result=%+v error=%+v", result, serviceError)
	}
}

func TestCPUTimeoutKeepsSessionReady(t *testing.T) {
	kubernetes := &fakeKubernetes{cpuResult: command.Result{Kind: command.ResultTimeout}}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	session := prepareSession(t, sandboxMaster, nil)
	result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceCPU, Command: []string{"sleep"}, TimeoutSeconds: 1})
	if timing := commandTiming(result, command.DeviceCPU); serviceError != nil || result.Status != ExecutionStatusTimedOut || timing == nil || timing.Attempts != 1 {
		t.Fatalf("result=%+v error=%+v", result, serviceError)
	}
	if current, serviceError := sandboxMaster.GetSession(session.SessionID); serviceError != nil || current.Status != SessionStatusReady {
		t.Fatalf("session=%+v error=%+v", current, serviceError)
	}
}

func TestCPUExit127IsCompleted(t *testing.T) {
	exitCode := 127
	kubernetes := &fakeKubernetes{cpuResult: command.Result{Kind: command.ResultCompleted, ExitCode: &exitCode, Stderr: "command not found"}}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	session := prepareSession(t, sandboxMaster, nil)
	result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceCPU, Command: []string{"missing-command"}, TimeoutSeconds: 1})
	if timing := commandTiming(result, command.DeviceCPU); serviceError != nil || result.Status != ExecutionStatusCompleted || result.ExitCode == nil || *result.ExitCode != exitCode || result.ErrorCode != "" || timing == nil || timing.Attempts != 1 {
		t.Fatalf("result=%+v error=%+v", result, serviceError)
	}
}

func TestGPUQueueTimeoutWhenWorkerSyncFails(t *testing.T) {
	kubernetes := &fakeKubernetes{gpuListErr: errors.New("GPU API unavailable")}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sandboxMaster.Start(ctx)
	gpuPoolID := createPool(t, sandboxMaster, "image")
	sessionID := readyGPUSession(t, sandboxMaster, gpuPoolID)
	result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), sessionID, command.Request{Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 1})
	if timing := commandTiming(result, command.DeviceGPU); serviceError != nil || result.Status != ExecutionStatusTimedOut || result.ErrorCode != "COMMAND_TIMEOUT" || timing == nil || timing.QueueMS == nil || timing.Attempts != 0 {
		t.Fatalf("result=%+v error=%+v", result, serviceError)
	}
}

func TestGPUWorkerDeleteIsRetried(t *testing.T) {
	kubernetes := &fakeKubernetes{deleteGPUErr: errors.New("temporary delete failure")}
	sandboxMaster := New(testConfig(), kubernetes, nil, nil)
	session := prepareSession(t, sandboxMaster, nil)
	go sandboxMaster.deleteGPUWorker(PodInfo{Name: "gpu-0", UID: "gpu-uid", GPUPoolID: session.GPUPoolID}, "")
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		kubernetes.mu.Lock()
		calls := kubernetes.deletedGPU
		pods := append([]PodInfo(nil), kubernetes.gpuPods...)
		kubernetes.mu.Unlock()
		if calls >= 1 {
			sandboxMaster.syncGPUWorkers(context.Background(), "", pods)
		}
		kubernetes.mu.Lock()
		calls = kubernetes.deletedGPU
		kubernetes.mu.Unlock()
		if calls == 2 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("GPU Worker deletion was not retried")
}

func TestReuseGPUWorker(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	cfg := testConfig()
	sandboxMaster := New(cfg, kubernetes, nil, nil)
	session := prepareSession(t, sandboxMaster, nil)
	for range 2 {
		result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 1})
		if timing := commandTiming(result, command.DeviceGPU); serviceError != nil || timing == nil || timing.QueueMS == nil || timing.PrecheckMS == nil || timing.Attempts != 1 {
			t.Fatalf("result=%+v error=%+v", result, serviceError)
		}
		deadline := time.Now().Add(time.Second)
		for time.Now().Before(deadline) {
			sandboxMaster.mu.Lock()
			worker := sandboxMaster.gpuWorkersByUID["gpu-uid"]
			idle := worker != nil && worker.status == GPUWorkerIdle
			sandboxMaster.mu.Unlock()
			if idle {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	kubernetes.mu.Lock()
	defer kubernetes.mu.Unlock()
	if kubernetes.gpuExecCalls != 2 || kubernetes.stateCalls != 5 || kubernetes.deletedGPU != 0 {
		t.Fatalf("exec=%d checks=%d deleted=%d", kubernetes.gpuExecCalls, kubernetes.stateCalls, kubernetes.deletedGPU)
	}
}

func TestReuseDeletesGPUWorkerAfterTimeout(t *testing.T) {
	kubernetes := &fakeKubernetes{gpuResult: command.Result{Kind: command.ResultTimeout}}
	cfg := testConfig()
	sandboxMaster := New(cfg, kubernetes, nil, nil)
	session := prepareSession(t, sandboxMaster, nil)
	result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceGPU, Command: []string{"sleep"}, TimeoutSeconds: 1})
	if serviceError != nil || result.Status != ExecutionStatusTimedOut || result.ErrorCode != "COMMAND_TIMEOUT" {
		t.Fatalf("result=%+v error=%+v", result, serviceError)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		kubernetes.mu.Lock()
		deleted := kubernetes.deletedGPU
		kubernetes.mu.Unlock()
		if deleted == 1 {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("timed out GPU Worker was not deleted")
}

func TestReuseKeepsGPUWorkerAfterStartFailure(t *testing.T) {
	kubernetes := &fakeKubernetes{gpuResult: command.Result{Kind: command.ResultStartFailure, Error: "executable file not found"}}
	cfg := testConfig()
	sandboxMaster := New(cfg, kubernetes, nil, nil)
	session := prepareSession(t, sandboxMaster, nil)
	result, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceGPU, Command: []string{"missing-command"}, TimeoutSeconds: 1})
	if serviceError != nil || result.Status != ExecutionStatusFailed || result.ErrorCode != "COMMAND_START_FAILED" {
		t.Fatalf("result=%+v error=%+v", result, serviceError)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		sandboxMaster.mu.Lock()
		worker := sandboxMaster.gpuWorkersByUID["gpu-uid"]
		idle := worker != nil && worker.status == GPUWorkerIdle
		sandboxMaster.mu.Unlock()
		kubernetes.mu.Lock()
		deleted := kubernetes.deletedGPU
		kubernetes.mu.Unlock()
		if idle {
			if deleted != 0 {
				t.Fatalf("GPU Worker was deleted %d times", deleted)
			}
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("GPU Worker was not released after command start failure")
}

func TestReuseDeletesGPUWorkerWithResidualProcess(t *testing.T) {
	oldTimeout, oldInterval := reuseCheckTimeout, reuseCheckInterval
	reuseCheckTimeout, reuseCheckInterval = 30*time.Millisecond, time.Millisecond
	t.Cleanup(func() { reuseCheckTimeout, reuseCheckInterval = oldTimeout, oldInterval })
	kubernetes := &fakeKubernetes{gpuStates: []GPUPodState{{}, {}}, gpuState: GPUPodState{ProcessCount: 1}}
	cfg := testConfig()
	sandboxMaster := New(cfg, kubernetes, nil, nil)
	session := prepareSession(t, sandboxMaster, nil)
	if _, serviceError := sandboxMaster.ExecuteCommand(context.Background(), session.SessionID, command.Request{Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 1}); serviceError != nil {
		t.Fatal(serviceError)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		kubernetes.mu.Lock()
		deleted := kubernetes.deletedGPU
		kubernetes.mu.Unlock()
		if deleted == 1 {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("GPU Worker with residual process was not deleted")
}

func TestGPUPoolSyncKeepsOtherClusters(t *testing.T) {
	be := &gpuPool{clusterName: "be", image: "be-image", replicas: 20}
	deleting := &gpuPool{clusterName: "be", deleting: true}
	m := &Master{gpuPoolsByID: map[string]*gpuPool{
		"be-pool": be, "be-deleting": deleting, "h20-pool": {clusterName: "h20"},
	}}
	if got := m.syncGPUPools("h20", []GPUPoolInfo{{GPUPoolID: "h20-pool"}}); len(got) != 0 {
		t.Fatalf("syncing h20 scheduled another cluster's pool for recreation: %v", got)
	}
	m.syncGPUPools("", []GPUPoolInfo{{GPUPoolID: "be-pool", Image: "wrong-image", Replicas: 1}})
	if be.recreating || be.image != "be-image" || be.replicas != 20 || m.gpuPoolsByID["be-deleting"] != deleting {
		t.Fatal("another cluster's sync changed BE pools")
	}
	got := m.syncGPUPools("be", nil)
	if len(got) != 1 || got[0].id != "be-pool" || got[0].clusterName != "be" || !be.recreating {
		t.Fatalf("syncing BE must recreate its missing pool: %v", got)
	}
	if m.gpuPoolsByID["be-deleting"] != nil || m.gpuPoolsByID["h20-pool"] == nil {
		t.Fatal("syncing BE must finish its own deletion and preserve h20")
	}
}

func TestGPUWorkerSyncKeepsOtherClusters(t *testing.T) {
	be := &gpuWorker{gpuPoolID: "be-pool", status: GPUWorkerBusy, executionID: "active"}
	h20 := &gpuWorker{gpuPoolID: "h20-pool", status: GPUWorkerIdle}
	m := &Master{
		gpuPoolsByID: map[string]*gpuPool{
			"be-pool":  {clusterName: "be"},
			"h20-pool": {clusterName: "h20"},
		},
		gpuWorkersByUID: map[string]*gpuWorker{"be-uid": be, "h20-uid": h20},
	}
	m.syncGPUWorkers(context.Background(), "h20", []PodInfo{{UID: "h20-uid", GPUPoolID: "h20-pool"}})
	if m.gpuWorkersByUID["be-uid"] != be || be.status != GPUWorkerBusy || be.executionID != "active" {
		t.Fatal("syncing h20 changed the active BE worker")
	}
	m.syncGPUWorkers(context.Background(), "", []PodInfo{{UID: "new-be-uid", GPUPoolID: "be-pool"}})
	if len(m.gpuWorkersByUID) != 2 || m.gpuWorkersByUID["be-uid"] != be {
		t.Fatal("the default client changed another cluster's workers")
	}
	m.syncGPUWorkers(context.Background(), "be", nil)
	if len(m.gpuWorkersByUID) != 1 || m.gpuWorkersByUID["h20-uid"] != h20 {
		t.Fatal("syncing BE must remove its missing worker and preserve h20")
	}
}

func TestPrefixSurvivesSessionReplacementRestorationAndDeletion(t *testing.T) {
	ctx := context.Background()
	kube := &fakeKubernetes{gpuSessionReady: true}
	cfg := testGPUSessionConfig()
	m := New(cfg, kube, testGPUSessionClients(kube), nil)
	resources := &ResourceRequirements{Requests: map[string]string{"example.org/gpu_count_best_effort": "1"}}
	poolID := createPool(t, m, "image")
	for _, prefix := range []string{"tenant", "tenantish"} {
		if _, serviceError := m.CreateSession(ctx, CreateSessionRequest{SessionType: SessionTypeCPUGPU, GPUPoolID: poolID, Prefix: prefix}); serviceError != nil {
			t.Fatal(serviceError)
		}
	}
	if _, serviceError := m.CreateSession(ctx, CreateSessionRequest{SessionType: SessionTypeGPUOnly, Image: "image", Resources: resources, Prefix: "tenant"}); serviceError != nil {
		t.Fatal(serviceError)
	}
	restored := New(cfg, kube, testGPUSessionClients(kube), nil)
	poolInfos, _ := kube.ListGPUPools(ctx)
	restored.syncGPUPools("", poolInfos)
	restored.syncCPUPods(kube.cpuPods)
	restored.syncGPUSessionPods("h20", kube.gpuSessionPods)
	current := restored.sessionsByID[kube.cpuPods[0].SessionID]
	if current.prefix != "tenant" {
		t.Fatal("CPU Pod restoration lost prefix")
	}
	current.status, current.cpuGPU.probeRunning = SessionStatusCreating, true
	exitCode := 1
	kube.cpuProbeResults = []command.Result{{Kind: command.ResultCompleted, ExitCode: &exitCode}}
	restored.probeCPUPod(current, *current.cpuGPU.cpuPod, 0)
	if kube.cpuCreateCount != 3 || kube.cpuPrefix != "tenant" {
		t.Fatal("CPU Pod replacement lost prefix")
	}
	for _, prefix := range []string{"", "BAD_PREFIX"} {
		if serviceError := restored.DeleteSessionsByPrefix(ctx, prefix); serviceError == nil || serviceError.Kind != ErrorInvalid {
			t.Fatalf("invalid prefix %q: %v", prefix, serviceError)
		}
	}
	if serviceError := restored.DeleteSessionsByPrefix(ctx, "missing"); serviceError != nil || len(restored.ListSessions()) != 3 {
		t.Fatal("unmatched prefix changed sessions")
	}
	if serviceError := restored.DeleteSessionsByPrefix(ctx, "tenant"); serviceError != nil {
		t.Fatal(serviceError)
	}
	remaining := restored.ListSessions()
	if len(remaining) != 1 || !strings.HasPrefix(remaining[0].CPU.Pod, "tenantish-") || len(restored.ListGPUPools()) != 1 || len(kube.cpuPods) != 1 || len(kube.gpuSessionPods) != 0 {
		t.Fatalf("prefix deletion affected the wrong resources: %+v", remaining)
	}
}

func TestPrefixDeletionDuringCreation(t *testing.T) {
	kube := &fakeKubernetes{cpuCreateStarted: make(chan struct{}, 1), cpuCreateRelease: make(chan struct{})}
	m := New(testConfig(), kube, nil, nil)
	poolID := createPool(t, m, "image")
	done := make(chan *ServiceError, 1)
	go func() {
		_, serviceError := m.CreateSession(context.Background(), CreateSessionRequest{SessionType: SessionTypeCPUGPU, GPUPoolID: poolID, Prefix: "tenant"})
		done <- serviceError
	}()
	<-kube.cpuCreateStarted
	if serviceError := m.DeleteSessionsByPrefix(context.Background(), "tenant"); serviceError != nil {
		t.Fatal(serviceError)
	}
	close(kube.cpuCreateRelease)
	if serviceError := <-done; serviceError == nil || serviceError.Code != "SESSION_DELETED" || len(kube.cpuPods) != 0 {
		t.Fatalf("creation escaped prefix deletion: %v", serviceError)
	}
}

func TestGPUPoolSchedulingSurvivesRecreationAndRestoration(t *testing.T) {
	ctx := context.Background()
	kube := &fakeKubernetes{}
	cfg := testConfig()
	m := New(cfg, kube, nil, nil)
	scheduling := Scheduling{NodeSelector: map[string]string{"tenant": "test"}, Tolerations: []Toleration{{Key: "tenant", Operator: "Exists"}}}
	resources := &ResourceRequirements{Requests: map[string]string{"cpu": "2"}}
	pool, serviceError := m.CreateGPUPool(ctx, CreateGPUPoolRequest{Image: "image", Replicas: 1, Resources: resources, Scheduling: scheduling})
	if serviceError != nil {
		t.Fatal(serviceError)
	}
	if _, serviceError := m.UpdateGPUPool(ctx, pool.GPUPoolID, UpdateGPUPoolRequest{Scheduling: Scheduling{Tolerations: []Toleration{}}}); serviceError != nil {
		t.Fatal(serviceError)
	}
	scheduling.Tolerations = []Toleration{}
	m.recreateGPUPool(ctx, m.syncGPUPools("", nil)[0])
	if !reflect.DeepEqual(kube.gpuPoolScheduling, &scheduling) || !reflect.DeepEqual(kube.gpuPoolResources, resources) {
		t.Fatal("GPU Pool recreation lost scheduling or resources")
	}
	restored := New(cfg, kube, nil, nil)
	pools, _ := kube.ListGPUPools(ctx)
	restored.syncGPUPools("", pools)
	restored.recreateGPUPool(ctx, restored.syncGPUPools("", nil)[0])
	if !reflect.DeepEqual(kube.gpuPoolScheduling, &scheduling) || !reflect.DeepEqual(kube.gpuPoolResources, resources) {
		t.Fatal("Restored GPU Pool lost scheduling or resources")
	}
}

func TestKubernetesValidationRejectsPoolCreation(t *testing.T) {
	for _, err := range []error{
		apierrors.NewInvalid(schema.GroupKind{Group: "apps", Kind: "Deployment"}, "pool", field.ErrorList{field.Invalid(field.NewPath("spec", "template", "spec", "nodeSelector"), "invalid-key", "invalid label")}),
		apierrors.NewBadRequest("invalid scheduling"),
	} {
		kube := &fakeKubernetes{gpuPoolCreateErr: err}
		m := New(testConfig(), kube, nil, nil)
		_, serviceError := m.CreateGPUPool(context.Background(), CreateGPUPoolRequest{Image: "image", Replicas: 1})
		if serviceError == nil || serviceError.Kind != ErrorInvalid || serviceError.Code != "BAD_REQUEST" || len(m.ListGPUPools()) != 0 {
			t.Fatalf("Kubernetes validation was not reported as a bad request: %v", serviceError)
		}
	}
}
