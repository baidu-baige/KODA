package master

import (
	"context"
	"log/slog"
	"time"
)

func (m *Master) gcLoop(ctx context.Context) {
	interval := time.Duration(m.config.GC.IntervalSeconds) * time.Second
	if interval <= 0 {
		return
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		m.collectGarbage(ctx)
	}
}

func (m *Master) collectGarbage(ctx context.Context) {
	sessionTimeout := time.Duration(m.config.GC.SessionIdleTimeoutSeconds) * time.Second
	poolTimeout := time.Duration(m.config.GC.GPUPoolIdleTimeoutSeconds) * time.Second

	m.mu.Lock()
	var sessions, pools []string
	if sessionTimeout > 0 {
		for id := range m.sessionsByID {
			if m.sessionReclaimableLocked(id, time.Now(), sessionTimeout) {
				sessions = append(sessions, id)
			}
		}
	}
	if poolTimeout > 0 {
		for id := range m.gpuPoolsByID {
			if m.poolReclaimableLocked(id, time.Now(), poolTimeout) {
				pools = append(pools, id)
			}
		}
	}
	m.mu.Unlock()

	for _, sessionID := range sessions {
		m.mu.Lock()
		reclaimable := m.sessionReclaimableLocked(sessionID, time.Now(), sessionTimeout)
		m.mu.Unlock()
		if !reclaimable {
			continue
		}
		if serviceError := m.DeleteSession(ctx, sessionID); serviceError != nil {
			slog.Warn("GC session delete failed", "session_id", sessionID, "error", serviceError.Message)
			continue
		}
		slog.Info("GC session reclaimed", "session_id", sessionID)
	}
	for _, gpuPoolID := range pools {
		m.mu.Lock()
		reclaimable := m.poolReclaimableLocked(gpuPoolID, time.Now(), poolTimeout)
		m.mu.Unlock()
		if !reclaimable {
			continue
		}
		if serviceError := m.DeleteGPUPool(ctx, gpuPoolID); serviceError != nil {
			slog.Warn("GC GPU pool delete failed", "gpu_pool_id", gpuPoolID, "error", serviceError.Message)
			continue
		}
		slog.Info("GC GPU pool reclaimed", "gpu_pool_id", gpuPoolID)
	}
}

func (m *Master) sessionReclaimableLocked(sessionID string, now time.Time, timeout time.Duration) bool {
	session := m.sessionsByID[sessionID]
	if session == nil || session.status == SessionStatusDeleting {
		return false
	}
	if now.Sub(session.lastActivityAt) < timeout {
		return false
	}
	for _, execution := range m.executionsByID {
		if execution.sessionID == sessionID && !execution.terminal() {
			return false
		}
	}
	return true
}

func (m *Master) poolReclaimableLocked(gpuPoolID string, now time.Time, timeout time.Duration) bool {
	pool := m.gpuPoolsByID[gpuPoolID]
	if pool == nil || pool.deleting {
		return false
	}
	if now.Sub(pool.lastActivityAt) < timeout {
		return false
	}
	for _, session := range m.sessionsByID {
		if session.cpuGPU != nil && session.cpuGPU.gpuPoolID == gpuPoolID && session.status != SessionStatusDeleting {
			return false
		}
	}
	for _, execution := range m.executionsByID {
		if execution.gpuPoolID == gpuPoolID && !execution.terminal() {
			return false
		}
	}
	return true
}
