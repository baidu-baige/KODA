package master

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"time"

	"kernel-sandbox/pkg/command"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation"
)

func (m *Master) CreateGPUPool(ctx context.Context, request CreateGPUPoolRequest) (GPUPoolResponse, *ServiceError) {
	started := time.Now()
	if request.Image == "" || request.Replicas <= 0 {
		return GPUPoolResponse{}, invalid("image is required and replicas must be positive")
	}
	clusterName := ""
	var poolClient GPUPoolKubernetesClient = m.kubernetes
	prefix := m.config.GPUPoolCluster.PoolNamePrefix
	if request.Resources != nil && len(m.gpuPoolResourceRoutes) > 0 {
		selected, client, serviceError := m.gpuPoolClientForResources(request.Resources)
		if serviceError != nil {
			return GPUPoolResponse{}, serviceError
		}
		clusterName = selected
		poolClient = client
		prefix = m.gpuPoolClusterConfigs[selected].PoolNamePrefix
	}
	m.mu.Lock()
	id := newID(prefix)
	for m.gpuPoolsByID[id] != nil {
		id = newID(prefix)
	}
	m.mu.Unlock()

	if err := poolClient.CreateGPUPool(ctx, id, request.Image, request.Replicas, request.Resources, &request.Scheduling); err != nil {
		return GPUPoolResponse{}, internal("GPU_POOL_CREATE_FAILED", err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	createDuration := time.Since(started)
	now := time.Now()
	m.gpuPoolsByID[id] = &gpuPool{id: id, image: request.Image, replicas: request.Replicas, clusterName: clusterName, resources: request.Resources, scheduling: &request.Scheduling, createdAt: started, lastActivityAt: now, createDuration: &createDuration}
	slog.Info("GPU Pool created", "gpu_pool_id", id, "image", request.Image, "replicas", request.Replicas, "cluster", clusterName)
	return m.gpuPoolResponseLocked(m.gpuPoolsByID[id]), nil
}

func (m *Master) GetGPUPool(gpuPoolID string) (GPUPoolResponse, *ServiceError) {
	m.mu.Lock()
	defer m.mu.Unlock()
	pool := m.gpuPoolsByID[gpuPoolID]
	if pool == nil || pool.deleting {
		return GPUPoolResponse{}, notFound("GPU_POOL_NOT_FOUND", "GPU pool does not exist")
	}
	return m.gpuPoolResponseLocked(pool), nil
}

func (m *Master) ListGPUPools() []GPUPoolResponse {
	m.mu.Lock()
	defer m.mu.Unlock()
	response := make([]GPUPoolResponse, 0, len(m.gpuPoolsByID))
	for _, pool := range m.gpuPoolsByID {
		if pool.deleting {
			continue
		}
		response = append(response, m.gpuPoolResponseLocked(pool))
	}
	sort.Slice(response, func(i, j int) bool { return response[i].GPUPoolID < response[j].GPUPoolID })
	return response
}

func (m *Master) UpdateGPUPool(ctx context.Context, gpuPoolID string, request UpdateGPUPoolRequest) (GPUPoolResponse, *ServiceError) {
	if request.Image == "" && request.Replicas == nil && request.NodeSelector == nil && request.Tolerations == nil {
		return GPUPoolResponse{}, invalid("image, replicas, node_selector or tolerations is required")
	}
	if request.Replicas != nil && *request.Replicas <= 0 {
		return GPUPoolResponse{}, invalid("replicas must be positive")
	}
	m.mu.Lock()
	pool := m.gpuPoolsByID[gpuPoolID]
	if pool == nil || pool.deleting {
		m.mu.Unlock()
		return GPUPoolResponse{}, notFound("GPU_POOL_NOT_FOUND", "GPU pool does not exist")
	}
	clusterName := pool.clusterName
	m.mu.Unlock()

	if err := m.gpuPoolClientForCluster(clusterName).UpdateGPUPool(ctx, gpuPoolID, request.Image, request.Replicas, &request.Scheduling); err != nil {
		return GPUPoolResponse{}, internal("GPU_POOL_UPDATE_FAILED", err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	pool = m.gpuPoolsByID[gpuPoolID]
	if pool == nil || pool.deleting {
		return GPUPoolResponse{}, notFound("GPU_POOL_NOT_FOUND", "GPU pool does not exist")
	}
	if request.Image != "" {
		pool.image = request.Image
	}
	if request.Replicas != nil {
		pool.replicas = *request.Replicas
	}
	scheduling := Scheduling{}
	if pool.scheduling != nil {
		scheduling = *pool.scheduling
	}
	if request.NodeSelector != nil {
		scheduling.NodeSelector = request.NodeSelector
	}
	if request.Tolerations != nil {
		scheduling.Tolerations = request.Tolerations
	}
	pool.scheduling = &scheduling
	pool.lastActivityAt = time.Now()
	slog.Info("GPU Pool updated", "gpu_pool_id", gpuPoolID, "image", pool.image, "replicas", pool.replicas)
	return m.gpuPoolResponseLocked(pool), nil
}

func (m *Master) DeleteGPUPool(ctx context.Context, gpuPoolID string) *ServiceError {
	m.mu.Lock()
	pool := m.gpuPoolsByID[gpuPoolID]
	if pool == nil || pool.deleting {
		m.mu.Unlock()
		return notFound("GPU_POOL_NOT_FOUND", "GPU pool does not exist")
	}
	for _, session := range m.sessionsByID {
		if session.cpuGPU != nil && session.cpuGPU.gpuPoolID == gpuPoolID && session.status != SessionStatusDeleting {
			m.mu.Unlock()
			return conflict("GPU_POOL_IN_USE", "delete sessions bound to the GPU pool before deleting it")
		}
	}
	pool.deleting = true
	clusterName := pool.clusterName
	m.mu.Unlock()

	if err := m.gpuPoolClientForCluster(clusterName).DeleteGPUPool(ctx, gpuPoolID); err != nil {
		m.mu.Lock()
		if p := m.gpuPoolsByID[gpuPoolID]; p != nil {
			p.deleting = false
		}
		m.mu.Unlock()
		return internal("GPU_POOL_DELETE_FAILED", err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if p := m.gpuPoolsByID[gpuPoolID]; p != nil {
		p.queue = nil
	}
	for uid, worker := range m.gpuWorkersByUID {
		if worker.gpuPoolID == gpuPoolID {
			delete(m.gpuWorkersByUID, uid)
		}
	}
	slog.Info("GPU Pool deleted", "gpu_pool_id", gpuPoolID)
	return nil
}

func (m *Master) CreateSession(ctx context.Context, request CreateSessionRequest) (SessionResponse, *ServiceError) {
	started := time.Now()
	if request.SessionType != SessionTypeGPUOnly && request.SessionType != SessionTypeCPUGPU {
		return SessionResponse{}, invalid("session_type must be gpu_only or cpu_gpu")
	}
	if request.Prefix != "" {
		suffix := "session-"
		if request.SessionType == SessionTypeGPUOnly {
			suffix = "gpu-session-"
		}
		if problems := validation.IsDNS1123Subdomain(request.Prefix); len(problems) != 0 || len(request.Prefix)+1+len(suffix)+16 > validation.DNS1123SubdomainMaxLength {
			return SessionResponse{}, invalid("prefix must be a DNS subdomain that fits the generated Pod name")
		}
	}
	if request.SessionType == SessionTypeGPUOnly {
		return m.createGPUSession(ctx, request, started)
	}
	return m.createCPUGPUSession(ctx, request, started)
}

func (m *Master) createCPUGPUSession(ctx context.Context, request CreateSessionRequest, started time.Time) (SessionResponse, *ServiceError) {
	if request.Prefix == "" {
		request.Prefix = m.config.CPUCluster.PodNamePrefix
	}
	if request.GPUPoolID == "" {
		return SessionResponse{}, invalid("gpu_pool_id is required")
	}
	m.mu.Lock()
	pool := m.gpuPoolsByID[request.GPUPoolID]
	if pool == nil || pool.deleting {
		m.mu.Unlock()
		return SessionResponse{}, notFound("GPU_POOL_NOT_FOUND", "GPU pool does not exist")
	}
	pool.lastActivityAt = started
	id, image := newID("session"), pool.image
	if request.Image != "" {
		image = request.Image
	}
	resources := cpuMemoryResources(request.Resources)
	session := &session{
		id:          id,
		prefix:      request.Prefix,
		sessionType: SessionTypeCPUGPU,
		status:      SessionStatusCreating,
		cpuGPU: &cpuGPUSession{
			gpuPoolID:    request.GPUPoolID,
			cpuImage:     image,
			cpuResources: resources,
		},
		createdAt:      started,
		lastActivityAt: started,
	}
	m.sessionsByID[id] = session
	m.mu.Unlock()
	slog.Info("Session created", "session_id", id, "gpu_pool_id", request.GPUPoolID, "session_type", SessionTypeCPUGPU, "status", session.status, "cpu_image", image)
	return m.createCPUSession(ctx, id, request.GPUPoolID, image, resources, session, started)
}

func (m *Master) createGPUSession(ctx context.Context, request CreateSessionRequest, started time.Time) (SessionResponse, *ServiceError) {
	if request.Image == "" {
		return SessionResponse{}, invalid("image is required")
	}
	if request.GPUPoolID != "" {
		return SessionResponse{}, invalid("gpu_pool_id is not valid for gpu_only sessions")
	}
	if request.Resources == nil || len(request.Resources.Requests) == 0 {
		return SessionResponse{}, invalid("resources.requests is required to select a GPU cluster")
	}
	clusterName, client, serviceError := m.gpuSessionClientForResources(request.Resources)
	if serviceError != nil {
		return SessionResponse{}, serviceError
	}
	id := newID("gpu-session")
	if request.Prefix == "" {
		request.Prefix = m.config.GPUSessionClusters[clusterName].PodNamePrefix
	}
	session := &session{
		id:          id,
		prefix:      request.Prefix,
		sessionType: SessionTypeGPUOnly,
		status:      SessionStatusCreating,
		gpuOnly: &gpuOnlySession{
			clusterName:  clusterName,
			gpuImage:     request.Image,
			gpuResources: request.Resources,
		},
		createdAt:      started,
		lastActivityAt: started,
	}
	m.mu.Lock()
	m.sessionsByID[id] = session
	m.mu.Unlock()
	slog.Info("Session created", "session_id", id, "session_type", SessionTypeGPUOnly, "status", session.status, "image", request.Image)
	pod, err := client.CreateGPUSessionPod(ctx, id, request.Image, request.Resources, request.Prefix)
	if err != nil {
		m.mu.Lock()
		deleted := m.sessionsByID[id] == session && session.status == SessionStatusDeleting
		delete(m.sessionsByID, id)
		m.mu.Unlock()
		if deleted {
			return SessionResponse{}, conflict("SESSION_DELETED", "session was deleted while its GPU Pod was being created")
		}
		slog.Warn("Session creation failed", "session_id", id, "session_type", SessionTypeGPUOnly, "error", err)
		return SessionResponse{}, internal("GPU_POD_CREATE_FAILED", err)
	}
	slog.Info("GPU Pod created", "session_id", id, "pod", pod.Name, "image", request.Image, "duration", time.Since(started))

	m.mu.Lock()
	if m.sessionsByID[id] != session {
		m.mu.Unlock()
		_ = client.DeleteGPUSessionPod(ctx, pod)
		return SessionResponse{}, conflict("SESSION_DELETED", "session was deleted while its GPU Pod was being created")
	}
	if session.status == SessionStatusDeleting {
		session.gpuOnly.gpuPod = &pod
		m.mu.Unlock()
		if err := client.DeleteGPUSessionPod(ctx, pod); err != nil {
			m.mu.Lock()
			if m.sessionsByID[id] == session {
				session.status = SessionStatusFailed
				session.errorMessage = err.Error()
			}
			m.mu.Unlock()
			return SessionResponse{}, internal("SESSION_DELETE_FAILED", err)
		}
		return SessionResponse{}, conflict("SESSION_DELETED", "session was deleted while its GPU Pod was being created")
	}
	session.gpuOnly.gpuPod = &pod
	createDuration := time.Since(started)
	session.createDuration = &createDuration
	if pod.Failed {
		session.status = SessionStatusFailed
		session.errorMessage = "GPU pod failed"
		slog.Warn("Session failed", "session_id", id, "session_type", SessionTypeGPUOnly, "pod", pod.Name, "error", session.errorMessage, "duration", time.Since(session.createdAt))
	} else if pod.Ready {
		session.status = SessionStatusReady
		readyDuration := time.Since(session.createdAt)
		session.readyDuration = &readyDuration
	}
	response := sessionResponse(session)
	m.mu.Unlock()
	return response, nil
}

func (m *Master) gpuSessionClientForResources(resources *ResourceRequirements) (string, GPUSessionKubernetesClient, *ServiceError) {
	selectedCluster := ""
	for resourceName := range resources.Requests {
		clusterName, exists := m.gpuSessionResourceRoutes[resourceName]
		if !exists {
			continue
		}
		requestQuantity, err := resource.ParseQuantity(resources.Requests[resourceName])
		if err != nil || requestQuantity.Sign() <= 0 {
			return "", nil, invalid("resources.requests." + resourceName + " must be a positive quantity")
		}
		if limitValue, ok := resources.Limits[resourceName]; ok {
			limitQuantity, err := resource.ParseQuantity(limitValue)
			if err != nil || limitQuantity.Sign() <= 0 || limitQuantity.Cmp(requestQuantity) != 0 {
				return "", nil, invalid("resources.limits." + resourceName + " must equal its request")
			}
		}
		if selectedCluster != "" && selectedCluster != clusterName {
			return "", nil, invalid("resources.requests selects multiple GPU Session clusters")
		}
		selectedCluster = clusterName
	}
	if selectedCluster == "" {
		return "", nil, invalid("resources.requests does not match a GPU Session cluster")
	}
	client := m.gpuSessionClients[selectedCluster]
	if client == nil {
		return "", nil, internal("GPU_SESSION_CLUSTER_CLIENT_MISSING", fmt.Errorf("GPU Session cluster %q has no client", selectedCluster))
	}
	return selectedCluster, client, nil
}

func (m *Master) gpuPoolClientForResources(resources *ResourceRequirements) (string, GPUPoolKubernetesClient, *ServiceError) {
	selectedCluster := ""
	for resourceName := range resources.Requests {
		clusterName, exists := m.gpuPoolResourceRoutes[resourceName]
		if !exists {
			continue
		}
		requestQuantity, err := resource.ParseQuantity(resources.Requests[resourceName])
		if err != nil || requestQuantity.Sign() <= 0 {
			return "", nil, invalid("resources.requests." + resourceName + " must be a positive quantity")
		}
		if limitValue, ok := resources.Limits[resourceName]; ok {
			limitQuantity, err := resource.ParseQuantity(limitValue)
			if err != nil || limitQuantity.Sign() <= 0 || limitQuantity.Cmp(requestQuantity) != 0 {
				return "", nil, invalid("resources.limits." + resourceName + " must equal its request")
			}
		}
		if selectedCluster != "" && selectedCluster != clusterName {
			return "", nil, invalid("resources.requests selects multiple GPU Pool clusters")
		}
		selectedCluster = clusterName
	}
	if selectedCluster == "" {
		return "", nil, invalid("resources.requests does not match a GPU Pool cluster")
	}
	client := m.gpuPoolClients[selectedCluster]
	if client == nil {
		return "", nil, internal("GPU_POOL_CLUSTER_CLIENT_MISSING", fmt.Errorf("GPU Pool cluster %q has no client", selectedCluster))
	}
	return selectedCluster, client, nil
}

func (m *Master) createCPUSession(ctx context.Context, id, gpuPoolID, image string, resources *ResourceRequirements, session *session, started time.Time) (SessionResponse, *ServiceError) {
	podCreateStarted := time.Now()
	sandboxURL := strings.TrimRight(m.config.Server.AdvertisedEndpoint, "/") + "/v1/session/" + id
	pod, err := m.kubernetes.CreateCPUPod(ctx, id, gpuPoolID, image, sandboxURL, resources, session.prefix)
	if err != nil {
		m.mu.Lock()
		deleted := m.sessionsByID[id] == session && session.status == SessionStatusDeleting
		delete(m.sessionsByID, id)
		m.mu.Unlock()
		if deleted {
			return SessionResponse{}, conflict("SESSION_DELETED", "session was deleted while its CPU Pod was being created")
		}
		slog.Warn("Session creation failed", "session_id", id, "gpu_pool_id", gpuPoolID, "session_type", SessionTypeCPUGPU, "error", err)
		return SessionResponse{}, internal("CPU_POD_CREATE_FAILED", err)
	}
	slog.Info("CPU Pod created", "session_id", id, "gpu_pool_id", gpuPoolID, "pod", pod.Name, "image", image, "duration", time.Since(podCreateStarted))
	m.mu.Lock()
	if m.sessionsByID[id] != session {
		m.mu.Unlock()
		_ = m.kubernetes.DeleteCPUPod(ctx, pod)
		return SessionResponse{}, conflict("SESSION_DELETED", "session was deleted while its CPU Pod was being created")
	}
	if session.status == SessionStatusDeleting {
		session.cpuGPU.cpuPod = &pod
		m.mu.Unlock()
		if err := m.kubernetes.DeleteCPUPod(ctx, pod); err != nil {
			m.mu.Lock()
			if m.sessionsByID[id] == session {
				session.status = SessionStatusFailed
				session.errorMessage = err.Error()
			}
			m.mu.Unlock()
			return SessionResponse{}, internal("SESSION_DELETE_FAILED", err)
		}
		return SessionResponse{}, conflict("SESSION_DELETED", "session was deleted while its CPU Pod was being created")
	}
	session.cpuGPU.cpuPod = &pod
	createDuration := time.Since(started)
	session.createDuration = &createDuration
	if pod.Failed {
		session.status = SessionStatusFailed
		session.errorMessage = "CPU pod failed"
		slog.Warn("Session failed", "session_id", id, "gpu_pool_id", gpuPoolID, "session_type", SessionTypeCPUGPU, "pod", pod.Name, "error", session.errorMessage, "duration", time.Since(session.createdAt))
	}
	response := sessionResponse(session)
	m.mu.Unlock()
	return response, nil
}

func (m *Master) probeCPUPodConnectivity(ctx context.Context, pod PodInfo) error {
	probeCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	result := m.kubernetes.ExecuteCPU(probeCtx, pod, command.Request{Command: []string{
		"python3", "-c", `import os, urllib.request; urllib.request.urlopen(os.environ["KERNEL_SANDBOX_URL"], timeout=5).read()`,
	}, TimeoutSeconds: 6}, 1<<10)
	if result.Kind != command.ResultCompleted {
		return fmt.Errorf("CPU pod probe failed (%s): %s", result.Kind, strings.TrimSpace(strings.Join([]string{result.Error, result.Stdout, result.Stderr}, " ")))
	}
	if result.ExitCode == nil {
		return fmt.Errorf("CPU pod probe completed without an exit code")
	}
	if *result.ExitCode != 0 {
		return fmt.Errorf("CPU pod probe exited with code %d: %s", *result.ExitCode, strings.TrimSpace(strings.Join([]string{result.Stdout, result.Stderr}, " ")))
	}
	return nil
}

func (m *Master) probeCPUPod(session *session, pod PodInfo, attempt int) {
	probeErr := m.probeCPUPodConnectivity(context.Background(), pod)
	m.mu.Lock()
	if !m.cpuProbeIsCurrentLocked(session, pod) {
		m.mu.Unlock()
		return
	}
	if probeErr == nil {
		session.cpuGPU.probeRunning = false
		session.status = SessionStatusReady
		readyDuration := time.Since(session.createdAt)
		if session.createDuration != nil && session.readyDuration == nil {
			session.readyDuration = &readyDuration
		}
		m.mu.Unlock()
		slog.Info("Session ready", "session_id", session.id, "gpu_pool_id", session.cpuGPU.gpuPoolID, "session_type", SessionTypeCPUGPU, "pod", pod.Name, "duration", readyDuration)
		return
	}
	m.mu.Unlock()
	slog.Warn("CPU pod connectivity probe failed", "session_id", session.id, "pod", pod.Name, "attempt", attempt, "error", probeErr)

	if err := m.deleteCPUPod(pod); err != nil {
		m.failCPUPodProbe(session, pod, fmt.Errorf("replace unreachable CPU pod: %w", err))
		return
	}

	createCtx, cancel := context.WithTimeout(context.Background(), resourceSyncTimeout)
	replacement, err := m.kubernetes.CreateCPUPod(createCtx, session.id, session.cpuGPU.gpuPoolID, session.cpuGPU.cpuImage,
		strings.TrimRight(m.config.Server.AdvertisedEndpoint, "/")+"/v1/session/"+session.id, session.cpuGPU.cpuResources, session.prefix)
	cancel()
	if err != nil {
		m.failCPUPodProbe(session, pod, fmt.Errorf("create replacement CPU pod: %w", err))
		return
	}

	replacementName := replacement.Name
	m.mu.Lock()
	if !m.cpuProbeIsCurrentLocked(session, pod) {
		m.mu.Unlock()
		deleteCtx, cancel := context.WithTimeout(context.Background(), resourceSyncTimeout)
		_ = m.kubernetes.DeleteCPUPod(deleteCtx, replacement)
		cancel()
		return
	}
	session.cpuGPU.cpuPod = &replacement
	session.cpuGPU.probeRunning = false
	m.mu.Unlock()
	slog.Info("CPU Pod recreated", "session_id", session.id, "gpu_pool_id", session.cpuGPU.gpuPoolID, "pod", replacementName, "attempt", attempt+1)
}

func (m *Master) deleteCPUPod(pod PodInfo) error {
	ctx, cancel := context.WithTimeout(context.Background(), resourceSyncTimeout)
	defer cancel()
	if err := m.kubernetes.DeleteCPUPod(ctx, pod); err != nil {
		return fmt.Errorf("delete unreachable CPU pod: %w", err)
	}
	ticker := time.NewTicker(resourceSyncInterval)
	defer ticker.Stop()
	for {
		pods, err := m.kubernetes.ListCPUPods(ctx)
		if err == nil {
			found := false
			for _, current := range pods {
				if current.Name == pod.Name {
					found = true
					break
				}
			}
			if !found {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func (m *Master) cpuProbeIsCurrentLocked(session *session, pod PodInfo) bool {
	return m.sessionsByID[session.id] == session && session.status == SessionStatusCreating && session.cpuGPU != nil && session.cpuGPU.probeRunning &&
		session.cpuGPU.cpuPod != nil && session.cpuGPU.cpuPod.Name == pod.Name && (session.cpuGPU.cpuPod.UID == "" || pod.UID == "" || session.cpuGPU.cpuPod.UID == pod.UID)
}

func (m *Master) failCPUPodProbe(session *session, pod PodInfo, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.cpuProbeIsCurrentLocked(session, pod) {
		return
	}
	session.cpuGPU.probeRunning = false
	session.cpuGPU.cpuPod.Ready = false
	session.status = SessionStatusFailed
	session.errorMessage = err.Error()
	slog.Warn("Session failed", "session_id", session.id, "gpu_pool_id", session.cpuGPU.gpuPoolID, "session_type", SessionTypeCPUGPU, "pod", pod.Name, "error", err, "duration", time.Since(session.createdAt))
}

func (m *Master) GetSession(sessionID string) (SessionResponse, *ServiceError) {
	m.mu.Lock()
	defer m.mu.Unlock()
	session := m.sessionsByID[sessionID]
	if session == nil || session.status == SessionStatusDeleting {
		return SessionResponse{}, notFound("SESSION_NOT_FOUND", "session does not exist")
	}
	return sessionResponse(session), nil
}

func (m *Master) ListSessions() []SessionResponse {
	m.mu.Lock()
	defer m.mu.Unlock()
	response := make([]SessionResponse, 0, len(m.sessionsByID))
	for _, session := range m.sessionsByID {
		if session.status == SessionStatusDeleting {
			continue
		}
		response = append(response, sessionResponse(session))
	}
	sort.Slice(response, func(i, j int) bool { return response[i].CreatedAt.Before(response[j].CreatedAt) })
	return response
}

func (m *Master) DeleteSession(ctx context.Context, sessionID string) *ServiceError {
	started := time.Now()
	m.mu.Lock()
	session := m.sessionsByID[sessionID]
	if session == nil || session.status == SessionStatusDeleting {
		m.mu.Unlock()
		return notFound("SESSION_NOT_FOUND", "session does not exist")
	}
	sessionType := session.sessionType
	session.status = SessionStatusDeleting
	var cpuPod *PodInfo
	if session.cpuGPU != nil && session.cpuGPU.cpuPod != nil {
		copy := *session.cpuGPU.cpuPod
		cpuPod = &copy
	}
	cpuPodName := ""
	if cpuPod != nil {
		cpuPodName = cpuPod.Name
	}
	logPoolID := ""
	if session.cpuGPU != nil {
		logPoolID = session.cpuGPU.gpuPoolID
	}
	var gpuSessionPod *PodInfo
	var gpuSessionClient GPUSessionKubernetesClient
	if session.gpuOnly != nil && session.gpuOnly.gpuPod != nil {
		copy := *session.gpuOnly.gpuPod
		gpuSessionPod = &copy
		if client, ok := m.gpuSessionClients[session.gpuOnly.clusterName]; ok {
			gpuSessionClient = client
		}
	}
	gpuPods := m.failSessionExecutionsLocked(sessionID)
	m.mu.Unlock()
	slog.Info("Session deleting", "session_id", sessionID, "gpu_pool_id", logPoolID, "session_type", sessionType, "cpu_pod", cpuPodName, "gpu_pods", len(gpuPods))

	var deleteErr error
	if cpuPod != nil {
		deleteErr = errors.Join(deleteErr, m.kubernetes.DeleteCPUPod(ctx, *cpuPod))
	}
	for _, pod := range gpuPods {
		deleteErr = errors.Join(deleteErr, m.gpuPoolClientForPod(pod.GPUPoolID).DeleteGPUWorkerPod(ctx, pod))
	}

	if gpuSessionPod != nil {
		if gpuSessionClient == nil {
			deleteErr = errors.Join(deleteErr, fmt.Errorf("GPU Session cluster client %q is unavailable", session.gpuOnly.clusterName))
		} else {
			deleteErr = errors.Join(deleteErr, gpuSessionClient.DeleteGPUSessionPod(ctx, *gpuSessionPod))
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if deleteErr != nil {
		session.status = SessionStatusFailed
		session.errorMessage = deleteErr.Error()
		slog.Warn("Session deletion failed", "session_id", sessionID, "gpu_pool_id", logPoolID, "session_type", sessionType, "error", deleteErr, "duration", time.Since(started))
		return internal("SESSION_DELETE_FAILED", deleteErr)
	}
	for _, pod := range gpuPods {
		delete(m.gpuWorkersByUID, pod.UID)
	}
	slog.Info("Session deletion requested", "session_id", sessionID, "gpu_pool_id", logPoolID, "session_type", sessionType, "duration", time.Since(started))
	return nil
}

func (m *Master) ListSessionsByGPUPool(gpuPoolID string) []SessionResponse {
	m.mu.Lock()
	defer m.mu.Unlock()
	response := make([]SessionResponse, 0)
	for _, session := range m.sessionsByID {
		if session.cpuGPU != nil && session.cpuGPU.gpuPoolID == gpuPoolID && session.status != SessionStatusDeleting {
			response = append(response, sessionResponse(session))
		}
	}
	sort.Slice(response, func(i, j int) bool { return response[i].CreatedAt.Before(response[j].CreatedAt) })
	return response
}

func (m *Master) DeleteSessionsByGPUPool(ctx context.Context, gpuPoolID string) *ServiceError {
	m.mu.Lock()
	ids := make([]string, 0)
	for id, session := range m.sessionsByID {
		if session.cpuGPU != nil && session.cpuGPU.gpuPoolID == gpuPoolID && session.status != SessionStatusDeleting {
			ids = append(ids, id)
		}
	}
	m.mu.Unlock()
	var deleteErr error
	for _, id := range ids {
		if serviceError := m.DeleteSession(ctx, id); serviceError != nil && serviceError.Kind != ErrorNotFound {
			deleteErr = errors.Join(deleteErr, serviceError)
		}
	}
	if deleteErr != nil {
		return internal("CLEANUP_FAILED", deleteErr)
	}
	return nil
}

func (m *Master) DeleteAllSessions(ctx context.Context) *ServiceError {
	return m.deleteSessions(ctx, "")
}

func (m *Master) DeleteSessionsByPrefix(ctx context.Context, prefix string) *ServiceError {
	if problems := validation.IsDNS1123Subdomain(prefix); len(problems) != 0 {
		return invalid("prefix must be a non-empty DNS subdomain")
	}
	return m.deleteSessions(ctx, prefix)
}

func (m *Master) deleteSessions(ctx context.Context, prefix string) *ServiceError {
	m.mu.Lock()
	ids := make([]string, 0, len(m.sessionsByID))
	for id, session := range m.sessionsByID {
		name := session.prefix + "-" + id
		if session.cpuGPU != nil && session.cpuGPU.cpuPod != nil {
			name = session.cpuGPU.cpuPod.Name
		} else if session.gpuOnly != nil && session.gpuOnly.gpuPod != nil {
			name = session.gpuOnly.gpuPod.Name
		}
		if session.status != SessionStatusDeleting && (prefix == "" || strings.HasPrefix(name, prefix+"-")) {
			ids = append(ids, id)
		}
	}
	m.mu.Unlock()
	var deleteErr error
	for _, id := range ids {
		if serviceError := m.DeleteSession(ctx, id); serviceError != nil && serviceError.Kind != ErrorNotFound {
			deleteErr = errors.Join(deleteErr, serviceError)
		}
	}
	if deleteErr != nil {
		return internal("CLEANUP_FAILED", deleteErr)
	}
	return nil
}

func (m *Master) gpuPoolResponseLocked(pool *gpuPool) GPUPoolResponse {
	response := GPUPoolResponse{GPUPoolID: pool.id, Cluster: pool.clusterName, Image: pool.image, Replicas: pool.replicas, Pods: []GPUWorkerResponse{}, Timing: resourceTimingResponse(pool.createDuration, pool.readyDuration)}
	now := time.Now()
	usage := &GPUPoolUsageResponse{
		Workers: map[string]GPUWorkerUsageResponse{},
		Queue: GPUPoolQueueUsageResponse{
			Count:   pool.queueCount,
			TotalMS: pool.queueTotal.Milliseconds(),
			MaxMS:   pool.queueMax.Milliseconds(),
		},
		QueueLength: len(pool.queue),
	}
	for name, measured := range pool.workerUsage {
		usage.Workers[name] = GPUWorkerUsageResponse{BusyMS: measured.busy.Milliseconds(), CapacityMS: measured.capacity.Milliseconds()}
	}
	for _, worker := range m.gpuWorkersByUID {
		if worker.gpuPoolID != pool.id {
			continue
		}
		response.Pods = append(response.Pods, GPUWorkerResponse{Pod: worker.pod.Name, Status: worker.status, Ready: worker.pod.Ready, RestartCount: worker.pod.RestartCount})
		measured := pool.workerUsage[worker.pod.Name]
		if !worker.busySince.IsZero() {
			measured.busy += now.Sub(worker.busySince)
		}
		if !worker.availableSince.IsZero() {
			measured.capacity += now.Sub(worker.availableSince)
		}
		usage.Workers[worker.pod.Name] = GPUWorkerUsageResponse{BusyMS: measured.busy.Milliseconds(), CapacityMS: measured.capacity.Milliseconds()}
		switch worker.status {
		case GPUWorkerIdle:
			response.Workers.Idle++
		case GPUWorkerBusy:
			response.Workers.Busy++
		default:
			response.Workers.Pending++
		}
	}
	for _, measured := range usage.Workers {
		usage.BusyMS += measured.BusyMS
		usage.CapacityMS += measured.CapacityMS
	}
	response.Usage = usage
	sort.Slice(response.Pods, func(i, j int) bool { return response.Pods[i].Pod < response.Pods[j].Pod })
	return response
}

func (m *Master) failSessionExecutionsLocked(sessionID string) []PodInfo {
	assigned := map[string]PodInfo{}
	for _, execution := range m.executionsByID {
		if execution.sessionID != sessionID || execution.terminal() {
			continue
		}
		for _, worker := range m.gpuWorkersByUID {
			if worker.executionID == execution.id {
				assigned[worker.pod.UID] = worker.pod
			}
		}
		if pool := m.gpuPoolsByID[execution.gpuPoolID]; pool != nil {
			for index, executionID := range pool.queue {
				if executionID == execution.id {
					pool.queue = append(pool.queue[:index], pool.queue[index+1:]...)
					break
				}
			}
		}
		m.finishExecutionLocked(execution, failedExecution(execution.id, "SESSION_DELETED", "session was deleted"))
	}
	result := make([]PodInfo, 0, len(assigned))
	for _, pod := range assigned {
		result = append(result, pod)
	}
	return result
}

func sessionResponse(session *session) SessionResponse {
	response := SessionResponse{SessionID: session.id, SessionType: session.sessionType, Status: session.status, CreatedAt: session.createdAt, Error: session.errorMessage, Timing: resourceTimingResponse(session.createDuration, session.readyDuration)}
	if session.sessionType == SessionTypeCPUGPU && session.cpuGPU != nil {
		response.GPUPoolID = session.cpuGPU.gpuPoolID
		response.CPU = &CPUResponse{Image: session.cpuGPU.cpuImage, Resources: session.cpuGPU.cpuResources}
		if session.cpuGPU.cpuPod != nil {
			response.CPU.Pod = session.cpuGPU.cpuPod.Name
			response.CPU.Ready = session.cpuGPU.cpuPod.Ready
		}
	}
	if session.sessionType == SessionTypeGPUOnly && session.gpuOnly != nil {
		response.GPU = &GPUSessionResponse{Image: session.gpuOnly.gpuImage, Resources: session.gpuOnly.gpuResources}
		if session.gpuOnly.gpuPod != nil {
			response.GPU.Pod = session.gpuOnly.gpuPod.Name
			response.GPU.Ready = session.gpuOnly.gpuPod.Ready
		}
	}
	return response
}

func resourceTimingResponse(createDuration, readyDuration *time.Duration) *ResourceTimingResponse {
	if createDuration == nil && readyDuration == nil {
		return nil
	}
	response := &ResourceTimingResponse{}
	if createDuration != nil {
		value := createDuration.Milliseconds()
		response.CreateMS = &value
	}
	if readyDuration != nil {
		value := readyDuration.Milliseconds()
		response.ReadyMS = &value
	}
	return response
}

func cpuMemoryResources(resources *ResourceRequirements) *ResourceRequirements {
	if resources == nil {
		return nil
	}
	filtered := &ResourceRequirements{Requests: cpuMemoryOnly(resources.Requests), Limits: cpuMemoryOnly(resources.Limits)}
	if filtered.Requests == nil && filtered.Limits == nil {
		return nil
	}
	return filtered
}

func cpuMemoryOnly(values map[string]string) map[string]string {
	if values == nil {
		return nil
	}
	result := map[string]string{}
	for _, name := range []string{"cpu", "memory"} {
		if value, ok := values[name]; ok {
			result[name] = value
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}
