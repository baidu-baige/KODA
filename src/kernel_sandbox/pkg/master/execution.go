package master

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"kernel-sandbox/pkg/command"
)

func (m *Master) ExecuteCommand(_ context.Context, sessionID string, request command.Request) (ExecuteCommandResponse, *ServiceError) {
	requestStartedAt := time.Now()
	var executionState *execution
	var serviceError *ServiceError
	m.mu.Lock()
	gpuOnly := false
	if session := m.sessionsByID[sessionID]; session != nil {
		gpuOnly = session.sessionType == SessionTypeGPUOnly
	}
	m.mu.Unlock()
	if gpuOnly {
		request.Device = command.DeviceGPU
	}
	if err := request.Validate(m.config.Execution.MaxStdinMB << 20); err != nil {
		return ExecuteCommandResponse{}, invalid(err.Error())
	}
	switch {
	case gpuOnly:
		executionState, serviceError = m.handleGPUSessionCommand(sessionID, request, requestStartedAt)
	case request.Device == command.DeviceCPU:
		executionState, serviceError = m.handleCPUCommand(sessionID, request, requestStartedAt)
	default:
		executionState, serviceError = m.handleGPUCommand(sessionID, request, requestStartedAt)
	}
	if serviceError != nil {
		return ExecuteCommandResponse{}, serviceError
	}
	<-executionState.done
	m.mu.Lock()
	result := *executionState.result
	delete(m.executionsByID, executionState.id)
	m.mu.Unlock()
	if result.ErrorCode == "INFRASTRUCTURE_ERROR" {
		return result, executionError(ErrorUnavailable, result)
	}
	return result, nil
}

func (m *Master) handleCPUCommand(sessionID string, request command.Request, requestStartedAt time.Time) (*execution, *ServiceError) {
	m.mu.Lock()
	session := m.sessionsByID[sessionID]
	if session == nil {
		m.mu.Unlock()
		return nil, notFound("SESSION_NOT_FOUND", "session does not exist")
	}
	if session.status != SessionStatusReady {
		m.mu.Unlock()
		return nil, conflict("SESSION_NOT_READY", "session is not ready")
	}
	if session.sessionType != SessionTypeCPUGPU || session.cpuGPU == nil || session.cpuGPU.cpuPod == nil {
		m.mu.Unlock()
		return nil, conflict("CPU_NOT_AVAILABLE", "session has no CPU pod")
	}
	if session.cpuGPU.activeExecution != "" {
		m.mu.Unlock()
		return nil, conflict("CPU_BUSY", "session already has a running CPU command")
	}
	m.touchSessionLocked(session)
	execution := &execution{id: newID("execution"), sessionID: sessionID, gpuPoolID: session.cpuGPU.gpuPoolID, request: request, done: make(chan struct{}), createdAt: requestStartedAt, attemptCount: 1}
	m.executionsByID[execution.id] = execution
	execution.status = ExecutionStatusRunning
	session.cpuGPU.activeExecution = execution.id
	pod := *session.cpuGPU.cpuPod
	logCommandAccepted(execution)
	m.mu.Unlock()
	go m.executeCPUCommand(execution.id, pod)
	return execution, nil
}

func (m *Master) handleGPUSessionCommand(sessionID string, request command.Request, requestStartedAt time.Time) (*execution, *ServiceError) {
	m.mu.Lock()
	session := m.sessionsByID[sessionID]
	if session == nil {
		m.mu.Unlock()
		return nil, notFound("SESSION_NOT_FOUND", "session does not exist")
	}
	if session.status != SessionStatusReady {
		m.mu.Unlock()
		return nil, conflict("SESSION_NOT_READY", "session is not ready")
	}
	if session.sessionType != SessionTypeGPUOnly || session.gpuOnly == nil || session.gpuOnly.gpuPod == nil {
		m.mu.Unlock()
		return nil, conflict("GPU_NOT_AVAILABLE", "session has no GPU pod")
	}
	m.touchSessionLocked(session)
	execution := &execution{id: newID("execution"), sessionID: sessionID, request: request, done: make(chan struct{}), createdAt: requestStartedAt, attemptCount: 1}
	m.executionsByID[execution.id] = execution
	execution.status = ExecutionStatusRunning
	pod := *session.gpuOnly.gpuPod
	logCommandAccepted(execution)
	m.mu.Unlock()
	go m.executeGPUSessionCommand(execution.id, pod)
	return execution, nil
}

func (m *Master) handleGPUCommand(sessionID string, request command.Request, requestStartedAt time.Time) (*execution, *ServiceError) {
	m.mu.Lock()
	session := m.sessionsByID[sessionID]
	if session == nil {
		m.mu.Unlock()
		return nil, notFound("SESSION_NOT_FOUND", "session does not exist")
	}
	if session.status != SessionStatusReady {
		m.mu.Unlock()
		return nil, conflict("SESSION_NOT_READY", "session is not ready")
	}
	if session.cpuGPU == nil || session.cpuGPU.gpuPoolID == "" {
		m.mu.Unlock()
		return nil, conflict("GPU_POOL_NOT_AVAILABLE", "session has no GPU pool")
	}
	if pool := m.gpuPoolsByID[session.cpuGPU.gpuPoolID]; pool == nil || pool.deleting {
		m.mu.Unlock()
		return nil, notFound("GPU_POOL_NOT_FOUND", "GPU pool does not exist")
	}
	m.touchSessionLocked(session)
	execution := &execution{id: newID("execution"), sessionID: sessionID, gpuPoolID: session.cpuGPU.gpuPoolID, parentExecutionID: session.cpuGPU.activeExecution, request: request, done: make(chan struct{}), createdAt: requestStartedAt}
	m.executionsByID[execution.id] = execution
	m.enqueueGPUCommand(execution)
	logCommandAccepted(execution)
	m.mu.Unlock()
	m.wakeGPUDispatcher()
	return execution, nil
}

func logCommandAccepted(current *execution) {
	slog.Info(
		"Command accepted",
		"execution_id", current.id,
		"session_id", current.sessionID,
		"gpu_pool_id", current.gpuPoolID,
		"device", current.request.Device,
		"timeout_seconds", current.request.TimeoutSeconds,
	)
}

func (m *Master) dispatchGPUCommands() {
	for {
		m.mu.Lock()
		var current *execution
		var worker *gpuWorker
		for gpuPoolID, pool := range m.gpuPoolsByID {
			queue := pool.queue
			for len(queue) > 0 {
				current = m.executionsByID[queue[0]]
				if current != nil && current.status == ExecutionStatusQueued && time.Now().Before(current.queueDeadline) {
					break
				}
				queue = queue[1:]
				if current != nil && current.status == ExecutionStatusQueued {
					m.finishExecutionLocked(current, ExecuteCommandResponse{
						ExecutionID: current.id,
						Status:      ExecutionStatusTimedOut,
						ErrorCode:   "COMMAND_TIMEOUT",
						Message:     "command timed out while waiting for a GPU Worker",
					})
				}
				current = nil
			}
			pool.queue = queue
			if current == nil || pool.deleting {
				current = nil
				continue
			}
			for _, candidate := range m.gpuWorkersByUID {
				if candidate.gpuPoolID == gpuPoolID && candidate.status == GPUWorkerIdle && candidate.pod.Ready {
					worker = candidate
					break
				}
			}
			if worker != nil {
				pool.queue = queue[1:]
				break
			}
			current = nil
		}
		if current == nil || worker == nil {
			m.mu.Unlock()
			return
		}
		current.attemptCount++
		current.status = ExecutionStatusRunning
		startedAt := time.Now()
		queueDuration := startedAt.Sub(current.queuedAt)
		current.queueDuration += queueDuration
		current.queuedAt = time.Time{}
		current.gpuWorkerUID = worker.pod.UID
		worker.status = GPUWorkerBusy
		worker.executionID = current.id
		worker.busySince = startedAt
		pod := worker.pod
		slog.Info(
			"GPU command dispatched",
			"execution_id", current.id,
			"session_id", current.sessionID,
			"gpu_pool_id", current.gpuPoolID,
			"pod", pod.Name,
			"attempt", current.attemptCount,
			"queue_duration", queueDuration,
			"queue_duration_total", current.queueDuration,
			"timeout_remaining", time.Until(current.queueDeadline),
		)
		m.mu.Unlock()
		go m.executeGPUCommand(current.id, pod)
	}
}

func (m *Master) enqueueGPUCommand(current *execution) {
	pool := m.gpuPoolsByID[current.gpuPoolID]
	if pool == nil || pool.deleting {
		m.finishExecutionLocked(current, failedExecution(current.id, "GPU_POOL_NOT_FOUND", "GPU pool was deleted"))
		return
	}
	current.status = ExecutionStatusQueued
	current.queuedAt = time.Now()
	if current.queueDeadline.IsZero() {
		current.queueDeadline = current.createdAt.Add(time.Duration(current.request.TimeoutSeconds) * time.Second)
	}
	pool.queue = append(pool.queue, current.id)
}

func (m *Master) wakeGPUDispatcher() {
	select {
	case m.gpuDispatchWakeup <- struct{}{}:
	default:
	}
}

func (m *Master) executeCPUCommand(executionID string, pod PodInfo) {
	m.mu.Lock()
	current := m.executionsByID[executionID]
	if current == nil || current.terminal() {
		m.mu.Unlock()
		return
	}
	request, sessionID := current.request, current.sessionID
	m.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(request.TimeoutSeconds)*time.Second)
	execStartedAt := time.Now()
	result := m.kubernetes.ExecuteCPU(ctx, pod, request, m.config.Execution.MaxOutputMB<<20)
	execDuration := time.Since(execStartedAt)
	cancel()
	response := responseFromCommand(executionID, result)
	m.mu.Lock()
	defer m.mu.Unlock()
	current = m.executionsByID[executionID]
	if current == nil || current.terminal() {
		return
	}
	current.execDuration += execDuration
	if currentSession := m.sessionsByID[sessionID]; currentSession != nil && currentSession.cpuGPU != nil && currentSession.cpuGPU.activeExecution == executionID {
		currentSession.cpuGPU.activeExecution = ""
	}
	m.finishExecutionLocked(current, response)
}

func (m *Master) executeGPUSessionCommand(executionID string, pod PodInfo) {
	m.mu.Lock()
	current := m.executionsByID[executionID]
	if current == nil || current.terminal() {
		m.mu.Unlock()
		return
	}
	request, sessionID := current.request, current.sessionID
	var sessionClient GPUSessionKubernetesClient
	if session := m.sessionsByID[sessionID]; session != nil && session.gpuOnly != nil {
		if client, ok := m.gpuSessionClients[session.gpuOnly.clusterName]; ok {
			sessionClient = client
		}
	}
	m.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(request.TimeoutSeconds)*time.Second)
	execStartedAt := time.Now()
	result := command.Result{Kind: command.ResultInfrastructure, Error: "GPU Session cluster client is unavailable"}
	if sessionClient != nil {
		result = sessionClient.ExecuteGPUSession(ctx, pod, request, m.config.Execution.MaxOutputMB<<20)
	}
	execDuration := time.Since(execStartedAt)
	cancel()
	response := responseFromCommand(executionID, result)
	m.mu.Lock()
	defer m.mu.Unlock()
	current = m.executionsByID[executionID]
	if current == nil || current.terminal() {
		return
	}
	current.execDuration += execDuration
	m.finishExecutionLocked(current, response)
}

func (m *Master) executeGPUCommand(executionID string, pod PodInfo) {
	m.mu.Lock()
	current := m.executionsByID[executionID]
	worker := m.gpuWorkersByUID[pod.UID]
	if current == nil || current.gpuWorkerUID != pod.UID || current.status != ExecutionStatusRunning {
		m.mu.Unlock()
		return
	}
	if worker == nil || worker.gpuPoolID != current.gpuPoolID || worker.executionID != executionID {
		m.mu.Unlock()
		m.retryGPUCommand(executionID, pod, "GPU Worker assignment disappeared")
		return
	}
	request := current.request
	remaining := time.Until(current.queueDeadline)
	m.mu.Unlock()

	precheckStartedAt := time.Now()
	idleError := m.checkGPUWorkerIdle(context.Background(), pod)
	precheckDuration := time.Since(precheckStartedAt)
	m.mu.Lock()
	current = m.executionsByID[executionID]
	if current == nil || current.gpuWorkerUID != pod.UID || current.status != ExecutionStatusRunning {
		m.mu.Unlock()
		return
	}
	current.precheckDuration += precheckDuration
	remaining = time.Until(current.queueDeadline)
	m.mu.Unlock()
	if idleError != nil {
		m.retryGPUCommand(executionID, pod, idleError.Error())
		return
	}
	if remaining <= 0 {
		m.mu.Lock()
		current = m.executionsByID[executionID]
		if current == nil || current.gpuWorkerUID != pod.UID || current.status != ExecutionStatusRunning {
			m.mu.Unlock()
			return
		}
		current.gpuWorkerUID = ""
		m.finishExecutionLocked(current, ExecuteCommandResponse{
			ExecutionID: current.id,
			Status:      ExecutionStatusTimedOut,
			ErrorCode:   "COMMAND_TIMEOUT",
			Message:     "command exceeded timeout_seconds",
		})
		m.mu.Unlock()
		go m.deleteGPUWorker(pod, executionID)
		return
	}
	request.TimeoutSeconds = max(1, int64((remaining+time.Second-1)/time.Second))
	ctx, cancel := context.WithTimeout(context.Background(), remaining)
	execStartedAt := time.Now()
	result := m.gpuPoolClientForPod(current.gpuPoolID).ExecuteGPU(ctx, pod, request, m.config.Execution.MaxOutputMB<<20)
	execDuration := time.Since(execStartedAt)
	cancel()
	m.mu.Lock()
	current = m.executionsByID[executionID]
	if current == nil || current.gpuWorkerUID != pod.UID || current.status != ExecutionStatusRunning {
		m.mu.Unlock()
		return
	}
	current.execDuration += execDuration
	m.mu.Unlock()
	if result.Kind == command.ResultInfrastructure {
		m.retryGPUCommand(executionID, pod, errorMessage(result, "GPU infrastructure failure"))
		return
	}
	m.mu.Lock()
	current = m.executionsByID[executionID]
	if current == nil || current.gpuWorkerUID != pod.UID || current.status != ExecutionStatusRunning {
		m.mu.Unlock()
		return
	}
	current.gpuWorkerUID = ""
	m.finishExecutionLocked(current, responseFromCommand(executionID, result))
	m.mu.Unlock()
	if result.Kind == command.ResultTimeout {
		go m.deleteGPUWorker(pod, executionID)
	} else {
		go m.releaseGPUWorker(pod, executionID)
	}
}

func (m *Master) retryGPUCommand(executionID string, pod PodInfo, cause string) {
	m.mu.Lock()
	current := m.executionsByID[executionID]
	if current == nil || current.gpuWorkerUID != pod.UID || current.status != ExecutionStatusRunning {
		m.mu.Unlock()
		return
	}
	slog.Warn("GPU execution failed", "execution_id", executionID, "session_id", current.sessionID, "gpu_pool_id", current.gpuPoolID, "pod", pod.Name, "attempt", current.attemptCount, "error", cause)
	current.gpuWorkerUID = ""
	if !time.Now().Before(current.queueDeadline) {
		m.finishExecutionLocked(current, ExecuteCommandResponse{
			ExecutionID: current.id,
			Status:      ExecutionStatusTimedOut,
			ErrorCode:   "COMMAND_TIMEOUT",
			Message:     "command exceeded timeout_seconds",
		})
	} else if current.attemptCount < m.config.Execution.MaxAttempts {
		m.enqueueGPUCommand(current)
	} else {
		m.finishExecutionLocked(current, failedExecution(current.id, "INFRASTRUCTURE_ERROR", fmt.Sprintf("GPU infrastructure retries exhausted after %d/%d attempts: %s", current.attemptCount, m.config.Execution.MaxAttempts, cause)))
	}
	m.mu.Unlock()
	go m.deleteGPUWorker(pod, executionID)
	m.wakeGPUDispatcher()
}

func (m *Master) finishExecutionLocked(current *execution, response ExecuteCommandResponse) {
	if current.terminal() {
		return
	}
	finishedAt := time.Now()
	if current.status == ExecutionStatusQueued && !current.queuedAt.IsZero() {
		current.queueDuration += finishedAt.Sub(current.queuedAt)
		current.queuedAt = time.Time{}
	}
	if current.request.Device == command.DeviceGPU && current.gpuPoolID != "" {
		if pool := m.gpuPoolsByID[current.gpuPoolID]; pool != nil {
			pool.queueCount++
			pool.queueTotal += current.queueDuration
			if current.queueDuration > pool.queueMax {
				pool.queueMax = current.queueDuration
			}
		}
	}
	totalDuration := finishedAt.Sub(current.createdAt)
	timing := m.executionTimingResponse(current, totalDuration)
	if current.request.Device == command.DeviceGPU {
		if parent := m.executionsByID[current.parentExecutionID]; parent != nil && parent.status == ExecutionStatusRunning {
			parent.gpuTimings = append(parent.gpuTimings, timing)
		}
	}
	current.status = response.Status
	response.Extra = executionResponseExtra(current.request.Device, timing, current.gpuTimings)
	current.result = &response
	if session := m.sessionsByID[current.sessionID]; session != nil && session.status != SessionStatusDeleting {
		m.touchSessionLocked(session)
	}
	attributes := []any{
		"execution_id", current.id,
		"session_id", current.sessionID,
		"gpu_pool_id", current.gpuPoolID,
		"device", current.request.Device,
		"status", response.Status,
		"exit_code", response.ExitCode,
		"duration", totalDuration,
		"exec_duration", current.execDuration,
		"attempts", current.attemptCount,
	}
	if response.ErrorCode != "" {
		attributes = append(attributes, "error_code", response.ErrorCode)
	}
	if response.Message != "" {
		attributes = append(attributes, "error", response.Message)
	}
	if current.request.Device == command.DeviceGPU && current.gpuPoolID != "" {
		attributes = append(attributes, "queue_duration", current.queueDuration)
	}
	if current.parentExecutionID != "" {
		attributes = append(attributes, "parent_execution_id", current.parentExecutionID)
	}
	if current.request.Device == command.DeviceGPU && current.gpuPoolID != "" {
		attributes = append(attributes, "precheck_duration", current.precheckDuration)
	}
	slog.Info("Command finished", attributes...)
	close(current.done)
}

func (m *Master) touchSessionLocked(session *session) {
	now := time.Now()
	session.lastActivityAt = now
	if session.cpuGPU != nil {
		if pool := m.gpuPoolsByID[session.cpuGPU.gpuPoolID]; pool != nil {
			pool.lastActivityAt = now
		}
	}
}

func executionResponseExtra(device command.Device, timing *ExecutionTimingResponse, gpuTimings []*ExecutionTimingResponse) *ResponseExtra {
	sandboxTimings := &SandboxTimings{}
	if device == command.DeviceCPU {
		sandboxTimings.CPUExecution = timing
		sandboxTimings.GPUExecutions = append(sandboxTimings.GPUExecutions, gpuTimings...)
	} else if timing != nil {
		sandboxTimings.GPUExecutions = []*ExecutionTimingResponse{timing}
	}
	if sandboxTimings.CPUExecution == nil && len(sandboxTimings.GPUExecutions) == 0 {
		return nil
	}
	return &ResponseExtra{SandboxTimings: sandboxTimings}
}

func (m *Master) executionTimingResponse(current *execution, totalDuration time.Duration) *ExecutionTimingResponse {
	response := &ExecutionTimingResponse{
		TotalMS:  totalDuration.Milliseconds(),
		ExecMS:   current.execDuration.Milliseconds(),
		Attempts: current.attemptCount,
	}
	if current.request.Device == command.DeviceGPU && current.gpuPoolID != "" {
		queueMS := current.queueDuration.Milliseconds()
		response.QueueMS = &queueMS
		precheckMS := current.precheckDuration.Milliseconds()
		response.PrecheckMS = &precheckMS
	}
	return response
}

func responseFromCommand(executionID string, result command.Result) ExecuteCommandResponse {
	response := ExecuteCommandResponse{ExecutionID: executionID, ExitCode: result.ExitCode, Stdout: result.Stdout, Stderr: result.Stderr}
	switch result.Kind {
	case command.ResultCompleted:
		response.Status = ExecutionStatusCompleted
	case command.ResultTimeout:
		response.Status = ExecutionStatusTimedOut
		response.ErrorCode = "COMMAND_TIMEOUT"
		response.Message = "command exceeded timeout_seconds"
	case command.ResultOutputLimit:
		response.Status = ExecutionStatusFailed
		response.ErrorCode = "OUTPUT_LIMIT_EXCEEDED"
		response.Message = "stdout and stderr exceeded the configured limit"
	case command.ResultStartFailure:
		response.Status = ExecutionStatusFailed
		response.ErrorCode = "COMMAND_START_FAILED"
		response.Message = errorMessage(result, "command could not be started")
	default:
		response.Status = ExecutionStatusFailed
		response.ErrorCode = "INFRASTRUCTURE_ERROR"
		response.Message = errorMessage(result, "command infrastructure failure")
	}
	return response
}

func failedExecution(executionID, code, message string) ExecuteCommandResponse {
	return ExecuteCommandResponse{ExecutionID: executionID, Status: ExecutionStatusFailed, ErrorCode: code, Message: message}
}
