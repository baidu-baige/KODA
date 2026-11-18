package master

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"sync"
	"time"

	"kernel-sandbox/pkg/command"
	"kernel-sandbox/pkg/config"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
)

const (
	resourceSyncInterval  = 2 * time.Second
	resourceSyncTimeout   = 10 * time.Second
	gpuQueueCheckInterval = time.Second
)

type KubernetesClient interface {
	CreateCPUPod(context.Context, string, string, string, string, *ResourceRequirements, string) (PodInfo, error)
	DeleteCPUPod(context.Context, PodInfo) error
	ListCPUPods(context.Context) ([]PodInfo, error)
	CreateGPUPool(context.Context, string, string, int32, *ResourceRequirements, *Scheduling) error
	ListGPUPools(context.Context) ([]GPUPoolInfo, error)
	UpdateGPUPool(context.Context, string, string, *int32, *Scheduling) error
	DeleteGPUPool(context.Context, string) error
	ListGPUPods(context.Context) ([]PodInfo, error)
	DeleteGPUWorkerPod(context.Context, PodInfo) error
	ExecuteCPU(context.Context, PodInfo, command.Request, int) command.Result
	ExecuteGPU(context.Context, PodInfo, command.Request, int) command.Result
	ReadGPUPodState(context.Context, PodInfo) (GPUPodState, error)
}

type GPUSessionKubernetesClient interface {
	CreateGPUSessionPod(context.Context, string, string, *ResourceRequirements, string) (PodInfo, error)
	ListGPUSessionPods(context.Context) ([]PodInfo, error)
	DeleteGPUSessionPod(context.Context, PodInfo) error
	ExecuteGPUSession(context.Context, PodInfo, command.Request, int) command.Result
}

type GPUPoolKubernetesClient interface {
	CreateGPUPool(context.Context, string, string, int32, *ResourceRequirements, *Scheduling) error
	ListGPUPools(context.Context) ([]GPUPoolInfo, error)
	UpdateGPUPool(context.Context, string, string, *int32, *Scheduling) error
	DeleteGPUPool(context.Context, string) error
	ListGPUPods(context.Context) ([]PodInfo, error)
	DeleteGPUWorkerPod(context.Context, PodInfo) error
	ExecuteGPU(context.Context, PodInfo, command.Request, int) command.Result
	ReadGPUPodState(context.Context, PodInfo) (GPUPodState, error)
}

type Master struct {
	config     *config.Config
	kubernetes KubernetesClient

	gpuSessionClients        map[string]GPUSessionKubernetesClient
	gpuSessionResourceRoutes map[string]string
	gpuPoolClients           map[string]GPUPoolKubernetesClient
	gpuPoolResourceRoutes    map[string]string
	gpuPoolClusterConfigs    map[string]config.GPUPoolClusterConfig

	mu sync.Mutex

	gpuPoolsByID      map[string]*gpuPool
	sessionsByID      map[string]*session
	executionsByID    map[string]*execution
	gpuWorkersByUID   map[string]*gpuWorker
	gpuDispatchWakeup chan struct{}
}

const gpuSessionReadyTimeout = 10 * time.Minute

func New(cfg *config.Config, kubernetes KubernetesClient, gpuSessionClients map[string]GPUSessionKubernetesClient, gpuPoolClients map[string]GPUPoolKubernetesClient) *Master {
	if len(cfg.GPUSessionClusters) != len(gpuSessionClients) {
		panic("GPU Session cluster config and client count differ")
	}
	resourceRoutes := map[string]string{}
	for clusterName, clusterConfig := range cfg.GPUSessionClusters {
		if gpuSessionClients[clusterName] == nil {
			panic("GPU Session cluster client is missing for " + clusterName)
		}
		for _, resourceName := range clusterConfig.ResourceNames {
			resourceRoutes[resourceName] = clusterName
		}
	}
	poolResourceRoutes := map[string]string{}
	poolClusterConfigs := map[string]config.GPUPoolClusterConfig{}
	for _, clusterConfig := range cfg.GPUPoolClusters {
		clusterName := clusterConfig.Name
		if gpuPoolClients[clusterName] == nil {
			panic("GPU Pool cluster client is missing for " + clusterName)
		}
		for _, resourceName := range clusterConfig.ResourceNames {
			poolResourceRoutes[resourceName] = clusterName
		}
		poolClusterConfigs[clusterName] = clusterConfig
	}
	if len(cfg.GPUPoolClusters) != len(gpuPoolClients) {
		panic("GPU Pool cluster config and client count differ")
	}
	return &Master{
		config:                   cfg,
		kubernetes:               kubernetes,
		gpuSessionClients:        gpuSessionClients,
		gpuSessionResourceRoutes: resourceRoutes,
		gpuPoolClients:           gpuPoolClients,
		gpuPoolResourceRoutes:    poolResourceRoutes,
		gpuPoolClusterConfigs:    poolClusterConfigs,
		gpuPoolsByID:             map[string]*gpuPool{},
		sessionsByID:             map[string]*session{},
		executionsByID:           map[string]*execution{},
		gpuWorkersByUID:          map[string]*gpuWorker{},
		gpuDispatchWakeup:        make(chan struct{}, 1),
	}
}

func (m *Master) Start(ctx context.Context) {
	go m.cpuPodSyncLoop(ctx)
	go m.gpuPodSyncLoop(ctx)
	go m.gpuSessionPodSyncLoop(ctx)
	go m.gpuCommandDispatchLoop(ctx)
	go m.gcLoop(ctx)
}

func (m *Master) cpuPodSyncLoop(ctx context.Context) {
	ticker := time.NewTicker(resourceSyncInterval)
	defer ticker.Stop()
	for {
		syncContext, cancel := context.WithTimeout(ctx, resourceSyncTimeout)
		if pods, err := m.kubernetes.ListCPUPods(syncContext); err == nil {
			m.syncCPUPods(pods)
		} else {
			slog.Warn("list CPU Pods", "error", err)
		}
		cancel()
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (m *Master) syncCPUPods(pods []PodInfo) {
	byName := make(map[string]PodInfo, len(pods))
	activeBySessionID := make(map[string]struct{}, len(pods))
	for _, pod := range pods {
		byName[pod.Name] = pod
		if pod.SessionID != "" && !pod.Deleting {
			activeBySessionID[pod.SessionID] = struct{}{}
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, pod := range pods {
		if pod.SessionID == "" || pod.GPUPoolID == "" || pod.Deleting || m.sessionsByID[pod.SessionID] != nil {
			continue
		}
		status, errorMessage := SessionStatusCreating, ""
		if pod.Ready {
			status = SessionStatusReady
		} else if pod.Failed {
			status, errorMessage = SessionStatusFailed, "CPU pod failed"
		}
		createdAt := pod.CreatedAt
		if createdAt.IsZero() {
			createdAt = time.Now()
		}
		podCopy := pod
		m.sessionsByID[pod.SessionID] = &session{
			id:          pod.SessionID,
			prefix:      pod.Prefix,
			sessionType: SessionTypeCPUGPU,
			status:      status,
			cpuGPU: &cpuGPUSession{
				gpuPoolID:    pod.GPUPoolID,
				cpuPod:       &podCopy,
				cpuImage:     pod.Image,
				cpuResources: pod.Resources,
			},
			errorMessage:   errorMessage,
			createdAt:      createdAt,
			lastActivityAt: time.Now(),
		}
		slog.Info("Session restored", "session_id", pod.SessionID, "gpu_pool_id", pod.GPUPoolID, "session_type", SessionTypeCPUGPU, "status", status, "pod", pod.Name, "image", pod.Image)
	}
	for _, session := range m.sessionsByID {
		if session.sessionType != SessionTypeCPUGPU || session.cpuGPU == nil || session.cpuGPU.cpuPod == nil || session.status != SessionStatusCreating {
			continue
		}
		pod, exists := byName[session.cpuGPU.cpuPod.Name]
		if !exists {
			continue
		}
		if session.cpuGPU.cpuPod.UID != "" && pod.UID != "" && session.cpuGPU.cpuPod.UID != pod.UID {
			continue
		}
		*session.cpuGPU.cpuPod = pod
		if pod.Ready {
			if session.cpuGPU.probeRunning {
				continue
			}
			session.cpuGPU.probeRunning = true
			session.cpuGPU.probeAttempts++
			go m.probeCPUPod(session, pod, session.cpuGPU.probeAttempts)
		} else if pod.Failed {
			session.cpuGPU.probeRunning = false
			session.status = SessionStatusFailed
			session.errorMessage = "CPU pod failed"
			slog.Warn("Session failed", "session_id", session.id, "gpu_pool_id", session.cpuGPU.gpuPoolID, "session_type", session.sessionType, "pod", pod.Name, "error", session.errorMessage, "duration", time.Since(session.createdAt))
		}
	}
	for sessionID, session := range m.sessionsByID {
		if session.status != SessionStatusDeleting || session.cpuGPU == nil || session.cpuGPU.cpuPod == nil {
			continue
		}
		if _, exists := activeBySessionID[sessionID]; !exists {
			delete(m.sessionsByID, sessionID)
		}
	}
}

func (m *Master) gpuPodSyncLoop(ctx context.Context) {
	ticker := time.NewTicker(resourceSyncInterval)
	defer ticker.Stop()
	syncPools := func(clusterName string, client GPUPoolKubernetesClient) {
		syncContext, cancel := context.WithTimeout(ctx, resourceSyncTimeout)
		defer cancel()
		if pools, err := client.ListGPUPools(syncContext); err == nil {
			for _, recreation := range m.syncGPUPools(clusterName, pools) {
				go func() {
					recreateContext, recreateCancel := context.WithTimeout(ctx, resourceSyncTimeout)
					defer recreateCancel()
					m.recreateGPUPool(recreateContext, recreation)
				}()
			}
		} else {
			slog.Warn("list GPU Pools", "cluster", clusterName, "error", err)
		}
		if pods, err := client.ListGPUPods(syncContext); err == nil {
			m.syncGPUWorkers(syncContext, clusterName, pods)
			m.wakeGPUDispatcher()
		} else {
			slog.Warn("list GPU Pods", "cluster", clusterName, "error", err)
		}
	}
	for {
		syncPools("", m.kubernetes)
		for clusterName, client := range m.gpuPoolClients {
			syncPools(clusterName, client)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (m *Master) gpuSessionPodSyncLoop(ctx context.Context) {
	syncOnce := func(clusterName string, client GPUSessionKubernetesClient) {
		syncContext, cancel := context.WithTimeout(ctx, resourceSyncTimeout)
		pods, err := client.ListGPUSessionPods(syncContext)
		cancel()
		if err != nil {
			slog.Warn("failed to list GPU Session Pods", "error", err)
			return
		}
		m.syncGPUSessionPods(clusterName, pods)
	}
	syncAll := func() {
		for clusterName, client := range m.gpuSessionClients {
			syncOnce(clusterName, client)
		}
	}
	syncAll()
	ticker := time.NewTicker(resourceSyncInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			syncAll()
		}
	}
}

func (m *Master) syncGPUSessionPods(clusterName string, pods []PodInfo) {
	present := make(map[string]PodInfo, len(pods))
	now := time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, pod := range pods {
		if pod.SessionID == "" {
			continue
		}
		present[pod.SessionID] = pod
		existing := m.sessionsByID[pod.SessionID]
		if existing == nil {
			if pod.Deleting {
				continue
			}
			createdAt := pod.CreatedAt
			if createdAt.IsZero() {
				createdAt = now
			}
			restored := &session{
				id:             pod.SessionID,
				prefix:         pod.Prefix,
				sessionType:    SessionTypeGPUOnly,
				status:         SessionStatusCreating,
				createdAt:      createdAt,
				lastActivityAt: now,
				gpuOnly: &gpuOnlySession{
					clusterName:  clusterName,
					gpuPod:       &pod,
					gpuImage:     pod.Image,
					gpuResources: pod.Resources,
				},
			}
			m.updateGPUSessionStatusLocked(restored, pod)
			m.sessionsByID[pod.SessionID] = restored
			slog.Info("restored GPU Session from Pod", "session_id", pod.SessionID, "pod", pod.Name, "status", restored.status)
			continue
		}
		if existing.sessionType != SessionTypeGPUOnly || existing.gpuOnly == nil || existing.status == SessionStatusDeleting || existing.gpuOnly.gpuPod != nil && existing.gpuOnly.gpuPod.UID != "" && pod.UID != "" && existing.gpuOnly.gpuPod.UID != pod.UID {
			continue
		}
		existing.gpuOnly.gpuPod = &pod
		existing.gpuOnly.clusterName = clusterName
		existing.gpuOnly.gpuImage = pod.Image
		existing.gpuOnly.gpuResources = pod.Resources
		m.updateGPUSessionStatusLocked(existing, pod)
	}
	for id, current := range m.sessionsByID {
		if current.sessionType != SessionTypeGPUOnly || current.gpuOnly == nil || current.gpuOnly.clusterName != clusterName {
			continue
		}
		if current.gpuOnly.gpuPod == nil {
			if current.status == SessionStatusCreating && time.Since(current.createdAt) >= gpuSessionReadyTimeout {
				current.status = SessionStatusFailed
				current.errorMessage = "GPU Session Pod did not become ready in time"
			}
			continue
		}
		if _, exists := present[id]; exists {
			continue
		}
		if current.status == SessionStatusDeleting {
			delete(m.sessionsByID, id)
			continue
		}
		current.status = SessionStatusFailed
		current.errorMessage = "GPU Session Pod disappeared"
		current.gpuOnly.gpuPod.Ready = false
	}
}

func (m *Master) updateGPUSessionStatusLocked(current *session, pod PodInfo) {
	if pod.Failed {
		current.status = SessionStatusFailed
		current.errorMessage = "GPU Session Pod failed"
		return
	}
	if pod.Deleting {
		current.status = SessionStatusFailed
		current.errorMessage = "GPU Session Pod is deleting"
		return
	}
	if pod.Ready {
		current.status = SessionStatusReady
		current.errorMessage = ""
		if current.createDuration != nil && current.readyDuration == nil {
			readyDuration := time.Since(current.createdAt)
			current.readyDuration = &readyDuration
		}
		return
	}
	if current.status != SessionStatusFailed {
		if time.Since(current.createdAt) >= gpuSessionReadyTimeout {
			current.status = SessionStatusFailed
			current.errorMessage = "GPU Session Pod did not become ready in time"
			return
		}
		current.status = SessionStatusCreating
		current.errorMessage = ""
	}
}

type gpuPoolRecreation struct {
	id          string
	image       string
	replicas    int32
	clusterName string
	resources   *ResourceRequirements
	scheduling  *Scheduling
}

func (m *Master) syncGPUPools(clusterName string, pools []GPUPoolInfo) []gpuPoolRecreation {
	present := make(map[string]struct{}, len(pools))
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, info := range pools {
		present[info.GPUPoolID] = struct{}{}
		pool := m.gpuPoolsByID[info.GPUPoolID]
		if pool == nil {
			m.gpuPoolsByID[info.GPUPoolID] = &gpuPool{id: info.GPUPoolID, image: info.Image, replicas: info.Replicas, clusterName: clusterName, resources: info.Resources, scheduling: info.Scheduling, lastActivityAt: time.Now()}
			slog.Info("GPU Pool restored", "gpu_pool_id", info.GPUPoolID, "image", info.Image, "replicas", info.Replicas, "cluster", clusterName)
			continue
		}
		if pool.deleting || pool.clusterName != clusterName {
			continue
		}
		pool.image = info.Image
		pool.replicas = info.Replicas
		pool.resources = info.Resources
		pool.scheduling = info.Scheduling
	}
	recreations := make([]gpuPoolRecreation, 0)
	for gpuPoolID, pool := range m.gpuPoolsByID {
		if pool.clusterName != clusterName {
			continue
		}
		if pool.deleting {
			if _, exists := present[gpuPoolID]; !exists {
				delete(m.gpuPoolsByID, gpuPoolID)
			}
			continue
		}
		if _, exists := present[gpuPoolID]; !exists && !pool.recreating {
			pool.recreating = true
			recreations = append(recreations, gpuPoolRecreation{id: gpuPoolID, image: pool.image, replicas: pool.replicas, clusterName: pool.clusterName, resources: pool.resources, scheduling: pool.scheduling})
		}
	}
	return recreations
}

func (m *Master) gpuPoolClientForCluster(clusterName string) GPUPoolKubernetesClient {
	if clusterName == "" {
		return m.kubernetes
	}
	if client, ok := m.gpuPoolClients[clusterName]; ok {
		return client
	}
	return m.kubernetes
}

func (m *Master) recreateGPUPool(ctx context.Context, recreation gpuPoolRecreation) {
	if err := m.gpuPoolClientForCluster(recreation.clusterName).CreateGPUPool(ctx, recreation.id, recreation.image, recreation.replicas, recreation.resources, recreation.scheduling); err != nil {
		m.finishGPUPoolRecreation(recreation.id)
		slog.Warn("recreate GPU Pool", "gpu_pool_id", recreation.id, "image", recreation.image, "replicas", recreation.replicas, "error", err)
		return
	}
	deletedDuringRecreation := m.finishGPUPoolRecreation(recreation.id)
	if !deletedDuringRecreation {
		slog.Info("GPU Pool recreated", "gpu_pool_id", recreation.id, "image", recreation.image, "replicas", recreation.replicas)
		return
	}
	m.deleteRecreatedGPUPool(recreation.id, recreation.clusterName)
}

func (m *Master) deleteRecreatedGPUPool(gpuPoolID string, clusterName string) {
	ctx, cancel := context.WithTimeout(context.Background(), resourceSyncTimeout)
	defer cancel()
	if err := m.gpuPoolClientForCluster(clusterName).DeleteGPUPool(ctx, gpuPoolID); err != nil {
		slog.Warn("delete concurrently recreated GPU Pool", "gpu_pool_id", gpuPoolID, "error", err)
	}
}

func (m *Master) finishGPUPoolRecreation(gpuPoolID string) (deletedDuringRecreation bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	pool := m.gpuPoolsByID[gpuPoolID]
	if pool == nil {
		return true
	}
	pool.recreating = false
	return pool.deleting
}

func (m *Master) gpuCommandDispatchLoop(ctx context.Context) {
	ticker := time.NewTicker(gpuQueueCheckInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-m.gpuDispatchWakeup:
		case <-ticker.C:
		}
		m.dispatchGPUCommands()
	}
}

func newID(prefix string) string {
	data := make([]byte, 8)
	if _, err := rand.Read(data); err != nil {
		panic(err)
	}
	return prefix + "-" + hex.EncodeToString(data)
}

func invalid(message string) *ServiceError {
	return &ServiceError{Kind: ErrorInvalid, Code: "BAD_REQUEST", Message: message}
}

func notFound(code, message string) *ServiceError {
	return &ServiceError{Kind: ErrorNotFound, Code: code, Message: message}
}

func conflict(code, message string) *ServiceError {
	return &ServiceError{Kind: ErrorConflict, Code: code, Message: message}
}

func internal(code string, err error) *ServiceError {
	if apierrors.IsInvalid(err) || apierrors.IsBadRequest(err) {
		return invalid(err.Error())
	}
	return &ServiceError{Kind: ErrorInternal, Code: code, Message: err.Error()}
}

func executionError(kind ErrorKind, result ExecuteCommandResponse) *ServiceError {
	return &ServiceError{Kind: kind, Code: result.ErrorCode, Message: result.Message, ExecutionID: result.ExecutionID, Extra: result.Extra}
}

func errorMessage(result command.Result, fallback string) string {
	if result.Error != "" {
		return result.Error
	}
	return fallback
}
