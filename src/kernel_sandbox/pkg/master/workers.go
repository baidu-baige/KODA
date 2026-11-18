package master

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

const gpuOperationTimeout = 10 * time.Second

var (
	reuseCheckTimeout  = 10 * time.Second
	reuseCheckInterval = 200 * time.Millisecond
)

func (m *Master) gpuPoolClientForPod(gpuPoolID string) GPUPoolKubernetesClient {
	m.mu.Lock()
	pool := m.gpuPoolsByID[gpuPoolID]
	clusterName := ""
	if pool != nil {
		clusterName = pool.clusterName
	}
	m.mu.Unlock()
	return m.gpuPoolClientForCluster(clusterName)
}

func (m *Master) deleteGPUWorker(pod PodInfo, executionID string) {
	ctx, cancel := context.WithTimeout(context.Background(), gpuOperationTimeout)
	defer cancel()
	if err := m.gpuPoolClientForPod(pod.GPUPoolID).DeleteGPUWorkerPod(ctx, pod); err != nil {
		slog.Warn("delete GPU Worker", "gpu_pool_id", pod.GPUPoolID, "pod", pod.Name, "error", err)
		m.mu.Lock()
		if worker := m.gpuWorkersByUID[pod.UID]; worker != nil && worker.executionID == executionID {
			m.recordGPUWorkerUsageLocked(worker, true)
			worker.status = GPUWorkerDeleting
			worker.executionID = ""
		}
		m.mu.Unlock()
		return
	}
	m.mu.Lock()
	if worker := m.gpuWorkersByUID[pod.UID]; worker != nil && worker.executionID == executionID {
		m.recordGPUWorkerUsageLocked(worker, true)
		delete(m.gpuWorkersByUID, pod.UID)
	}
	m.mu.Unlock()
}

func (m *Master) recordGPUWorkerUsageLocked(worker *gpuWorker, retired bool) {
	pool := m.gpuPoolsByID[worker.gpuPoolID]
	if pool == nil {
		return
	}
	if pool.workerUsage == nil {
		pool.workerUsage = map[string]gpuWorkerUsage{}
	}
	now := time.Now()
	usage := pool.workerUsage[worker.pod.Name]
	if !worker.busySince.IsZero() {
		usage.busy += now.Sub(worker.busySince)
		worker.busySince = time.Time{}
	}
	if retired && !worker.availableSince.IsZero() {
		usage.capacity += now.Sub(worker.availableSince)
		worker.availableSince = time.Time{}
	}
	pool.workerUsage[worker.pod.Name] = usage
}

func (m *Master) releaseGPUWorker(pod PodInfo, executionID string) {
	started := time.Now()
	deadline := time.Now().Add(reuseCheckTimeout)
	var idleError error
	for time.Now().Before(deadline) {
		idleError = m.checkGPUWorkerIdle(context.Background(), pod)
		if idleError == nil {
			m.mu.Lock()
			if worker := m.gpuWorkersByUID[pod.UID]; worker != nil && worker.executionID == executionID {
				m.recordGPUWorkerUsageLocked(worker, false)
				worker.executionID = ""
				worker.status = GPUWorkerIdle
			}
			m.mu.Unlock()
			slog.Info("GPU Worker released", "gpu_pool_id", pod.GPUPoolID, "pod", pod.Name, "postcheck_ms", time.Since(started).Milliseconds())
			m.wakeGPUDispatcher()
			return
		}
		time.Sleep(reuseCheckInterval)
	}
	slog.Warn("GPU Worker postcheck failed", "gpu_pool_id", pod.GPUPoolID, "pod", pod.Name, "postcheck_ms", time.Since(started).Milliseconds(), "error", idleError)
	m.deleteGPUWorker(pod, executionID)
}

func (m *Master) checkGPUWorkerIdle(ctx context.Context, pod PodInfo) error {
	checkContext, cancel := context.WithTimeout(ctx, gpuOperationTimeout)
	defer cancel()
	state, err := m.gpuPoolClientForPod(pod.GPUPoolID).ReadGPUPodState(checkContext, pod)
	if err != nil {
		return fmt.Errorf("GPU Worker state check failed: %w", err)
	}
	if state.Busy {
		return fmt.Errorf("GPU Worker is busy")
	}
	if state.ProcessCount != 0 {
		return fmt.Errorf("GPU Worker has %d residual GPU processes: %v", state.ProcessCount, state.Processes)
	}
	return nil
}

func (m *Master) syncGPUWorkers(ctx context.Context, clusterName string, pods []PodInfo) {
	present := make(map[string]PodInfo, len(pods))
	for _, pod := range pods {
		if pod.Deleting || pod.GPUPoolID == "" {
			continue
		}
		present[pod.UID] = pod
	}
	type assignment struct {
		executionID string
		pod         PodInfo
	}
	m.mu.Lock()
	var lost, deleting []assignment
	for uid, worker := range m.gpuWorkersByUID {
		if pool := m.gpuPoolsByID[worker.gpuPoolID]; pool != nil && pool.clusterName != clusterName {
			continue
		}
		pod, exists := present[uid]
		if exists {
			worker.pod = pod
			worker.gpuPoolID = pod.GPUPoolID
			if worker.status == GPUWorkerDeleting {
				deleting = append(deleting, assignment{pod: pod})
			}
			continue
		}
		if worker.executionID != "" {
			if current := m.executionsByID[worker.executionID]; current != nil && current.gpuWorkerUID == uid && current.status == ExecutionStatusRunning {
				lost = append(lost, assignment{executionID: worker.executionID, pod: worker.pod})
			}
		}
		m.recordGPUWorkerUsageLocked(worker, true)
		delete(m.gpuWorkersByUID, uid)
	}
	for uid, pod := range present {
		if _, exists := m.gpuWorkersByUID[uid]; !exists {
			if pool := m.gpuPoolsByID[pod.GPUPoolID]; pool == nil || pool.deleting || pool.clusterName != clusterName {
				continue
			}
			m.gpuWorkersByUID[uid] = &gpuWorker{pod: pod, gpuPoolID: pod.GPUPoolID, status: GPUWorkerPending, discoveredAt: time.Now()}
		}
	}
	m.mu.Unlock()

	for _, item := range lost {
		m.retryGPUCommand(item.executionID, item.pod, "GPU Worker disappeared")
	}
	for _, item := range deleting {
		go m.deleteGPUWorker(item.pod, "")
	}
	for uid, pod := range present {
		if !pod.Ready {
			continue
		}
		m.mu.Lock()
		worker := m.gpuWorkersByUID[uid]
		pool := m.gpuPoolsByID[pod.GPUPoolID]
		pending := worker != nil && pool != nil && pool.clusterName == clusterName && worker.status == GPUWorkerPending
		m.mu.Unlock()
		if !pending {
			continue
		}
		checkContext, cancel := context.WithTimeout(ctx, gpuOperationTimeout)
		state, err := m.gpuPoolClientForPod(pod.GPUPoolID).ReadGPUPodState(checkContext, pod)
		cancel()
		if err != nil || state.Busy || state.ProcessCount != 0 {
			slog.Warn("GPU Worker rejected", "gpu_pool_id", pod.GPUPoolID, "pod", pod.Name, "busy", state.Busy, "process_count", state.ProcessCount, "processes", state.Processes, "error", err)
			go m.deleteGPUWorker(pod, "")
			continue
		}
		m.mu.Lock()
		if worker = m.gpuWorkersByUID[uid]; worker != nil && worker.status == GPUWorkerPending {
			worker.status = GPUWorkerIdle
			worker.availableSince = time.Now()
			m.recordGPUPoolReadyLocked(worker.gpuPoolID)
			slog.Info("GPU Worker ready", "gpu_pool_id", worker.gpuPoolID, "pod", pod.Name, "duration", time.Since(worker.discoveredAt))
		}
		m.mu.Unlock()
	}
}

func (m *Master) recordGPUPoolReadyLocked(gpuPoolID string) {
	pool := m.gpuPoolsByID[gpuPoolID]
	if pool == nil || pool.createDuration == nil || pool.readyDuration != nil {
		return
	}
	readyDuration := time.Since(pool.createdAt)
	pool.readyDuration = &readyDuration
}
