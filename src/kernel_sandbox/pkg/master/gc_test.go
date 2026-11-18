package master

import (
	"context"
	"testing"
	"time"

	"kernel-sandbox/pkg/config"
)

func TestCollectGarbageReclaimsIdleSessionAndPool(t *testing.T) {
	kubernetes := &fakeKubernetes{}
	cfg := testConfig()
	cfg.GC = config.GCConfig{SessionIdleTimeoutSeconds: 1, GPUPoolIdleTimeoutSeconds: 1}
	sandboxMaster := New(cfg, kubernetes, nil, nil)
	gpuPoolID := createPool(t, sandboxMaster, "image")
	sessionID := readyGPUSession(t, sandboxMaster, gpuPoolID)
	sandboxMaster.mu.Lock()
	idleSince := time.Now().Add(-2 * time.Second)
	sandboxMaster.sessionsByID[sessionID].lastActivityAt = idleSince
	sandboxMaster.gpuPoolsByID[gpuPoolID].lastActivityAt = idleSince
	sandboxMaster.mu.Unlock()

	sandboxMaster.collectGarbage(context.Background())
	if len(sandboxMaster.ListSessions()) != 0 {
		t.Fatal("idle Session was not reclaimed")
	}
	if _, serviceError := sandboxMaster.GetGPUPool(gpuPoolID); serviceError != nil {
		t.Fatal("GPU Pool was reclaimed before its Session")
	}

	sandboxMaster.collectGarbage(context.Background())
	if _, serviceError := sandboxMaster.GetGPUPool(gpuPoolID); serviceError == nil {
		t.Fatal("idle GPU Pool was not reclaimed")
	}
}
