package docker

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"kernel-sandbox/pkg/command"
	"kernel-sandbox/pkg/config"
	"kernel-sandbox/pkg/master"
)

func TestRejectUnsupportedResources(t *testing.T) {
	if validateRequest(&master.ResourceRequirements{Limits: map[string]string{"nvidia.com/gpu": "2"}}, nil) == nil {
		t.Fatal("silently accepted Kubernetes GPU allocation")
	}
	if validateRequest(nil, &master.Scheduling{NodeSelector: map[string]string{"host": "other"}}) == nil {
		t.Fatal("silently accepted node selector")
	}
	if validateRequest(&master.ResourceRequirements{}, &master.Scheduling{}) != nil {
		t.Fatal("empty maps should select Docker defaults")
	}
}
func TestDockerLifecycle(t *testing.T) {
	if os.Getenv("KODA_DOCKER_TEST") != "1" {
		t.Skip("set KODA_DOCKER_TEST=1 for real Docker integration")
	}
	image := os.Getenv("KODA_DOCKER_TEST_IMAGE")
	bin := os.Getenv("KODA_SANDBOXD_BINARY")
	if image == "" || bin == "" {
		t.Fatal("set test image and sandboxd binary")
	}
	cfg := &config.Config{MasterID: fmt.Sprintf("koda-test-%d", os.Getpid()), Server: config.ServerConfig{AdvertisedEndpoint: "http://127.0.0.1:18991"}, Docker: config.DockerConfig{Binary: "docker", StateDir: t.TempDir(), SandboxdBinary: bin, Network: "host"}, Execution: config.ExecutionConfig{MaxStdinMB: 1, MaxOutputMB: 1, MaxAttempts: 1}, CPUCluster: config.CPUClusterConfig{PodNamePrefix: "koda-test"}}
	c, e := New(cfg)
	if e != nil {
		t.Fatal(e)
	}
	defer func() { c.Close() }()
	if other, e := New(cfg); e == nil {
		other.Close()
		t.Fatal("second Master acquired same state directory")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	p, e := c.CreateCPUPod(ctx, "session-one", "pool-one", image, cfg.Server.AdvertisedEndpoint, nil, "test-prefix")
	if e != nil {
		t.Fatal(e)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if e := c.DeleteCPUPod(ctx, p); e != nil {
			t.Error(e)
		}
	}()
	p2, e := c.CreateCPUPod(ctx, "session-two", "pool-one", image, cfg.Server.AdvertisedEndpoint, nil, "test-prefix")
	if e != nil {
		t.Fatal(e)
	}
	defer c.DeleteCPUPod(context.Background(), p2)
	waitIdle := func(p master.PodInfo) {
		t.Helper()
		for i := 0; i < 80; i++ {
			s, e := c.ReadGPUPodState(ctx, p)
			if e == nil && !s.Busy {
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
		t.Fatal("sandboxd did not become idle")
	}
	waitIdle(p)
	waitIdle(p2)
	run := func(p master.PodInfo, script string, timeout int64) command.Result {
		return c.ExecuteCPU(ctx, p, command.Request{Command: []string{"/bin/sh", "-c", script}, TimeoutSeconds: timeout}, 1<<20)
	}
	result := c.ExecuteCPU(ctx, p, command.Request{Device: command.DeviceCPU, Command: []string{"/bin/sh", "-c", "printf '%s|' \"$TEST_VALUE\"; cat; printf saved > /tmp/koda-marker"}, Env: map[string]string{"TEST_VALUE": "space ' quote $ literal"}, Stdin: "stdin-value", WorkingDir: "/tmp", TimeoutSeconds: 10}, 1<<20)
	if result.Kind != command.ResultCompleted || result.ExitCode == nil || *result.ExitCode != 0 || result.Stdout != "space ' quote $ literal|stdin-value" {
		t.Fatalf("execution: %+v", result)
	}
	waitIdle(p)
	if r := run(p2, "test ! -e /tmp/koda-marker", 10); r.ExitCode == nil || *r.ExitCode != 0 {
		t.Fatalf("workspaces not isolated: %+v", r)
	}

	if r := run(p, "printf '%s' \"$KERNEL_SANDBOX_URL\"", 10); r.Stdout != cfg.Server.AdvertisedEndpoint+"/v1/session/session-one" {
		t.Fatalf("incorrect CPU session callback URL: %+v", r)
	}
	waitIdle(p)
	if r := run(p, "exit 23", 10); r.ExitCode == nil || *r.ExitCode != 23 {
		t.Fatalf("exit code lost: %+v", r)
	}
	waitIdle(p)
	if r := run(p, "sleep 60 & echo $! > /tmp/koda-child; wait", 1); r.Kind != command.ResultTimeout {
		t.Fatalf("timeout missing: %+v", r)
	}
	waitIdle(p)
	if r := run(p, "pid=$(cat /tmp/koda-child); test ! -e /proc/$pid/stat || test \"$(awk '{print $3}' /proc/$pid/stat)\" = Z", 10); r.ExitCode == nil || *r.ExitCode != 0 {
		t.Fatalf("child survived timeout: %+v", r)
	}
	waitIdle(p)
	if r := run(p, "head -c 3000000 /dev/zero", 10); r.Kind != command.ResultOutputLimit || len(r.Stdout) > 1<<20 {
		t.Fatalf("output limit: kind=%s length=%d", r.Kind, len(r.Stdout))
	}
	waitIdle(p)

	otherCfg := *cfg
	otherCfg.Docker.StateDir = t.TempDir()
	if other, err := New(&otherCfg); err == nil {
		other.Close()
		t.Fatal("same master adopted containers from another state directory")
	}
	otherCfg.MasterID = cfg.MasterID + "-other"
	otherCfg.Docker.StateDir = t.TempDir()
	other, err := New(&otherCfg)
	if err != nil {
		t.Fatal(err)
	}
	if err = other.DeleteCPUPod(ctx, p); err == nil {
		other.Close()
		t.Fatal("other master deleted foreign container")
	}
	other.Close()
	wrong := p
	wrong.Name = "someone-else"
	if c.DeleteCPUPod(ctx, wrong) == nil {
		t.Fatal("deleted wrong owner/name")
	}
	if r := run(p, "cat /tmp/koda-marker", 10); r.Stdout != "saved" {
		t.Fatal("ownership check removed original container")
	}
	// Restart the client without deleting containers. Discovery restores exact IDs and workspace contents.
	if e = c.Close(); e != nil {
		t.Fatal(e)
	}
	c, e = New(cfg)
	if e != nil {
		t.Fatal(e)
	}
	pods, e := c.ListCPUPods(ctx)
	if e != nil || len(pods) != 2 {
		t.Fatalf("discovery after restart: %v %v", pods, e)
	}
	if r := run(p, "cat /tmp/koda-marker", 10); r.Stdout != "saved" {
		t.Fatalf("workspace lost: %+v", r)
	}
	if e = c.DeleteCPUPod(ctx, p); e != nil {
		t.Fatal(e)
	}
	if e = c.DeleteCPUPod(ctx, p); e != nil {
		t.Fatal("delete should tolerate already gone", e)
	}
	// Desired pool allocation is persisted and excludes already assigned UUIDs; no GPU container is launched in this CPU test.
	cfg.Docker.GPUDevices = []string{"GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}
	if e = c.CreateGPUPool(ctx, "pool-reservation", image, 1, nil, nil); e != nil {
		t.Fatal(e)
	}
	if e = c.CreateGPUPool(ctx, "pool-overflow", image, 1, nil, nil); e == nil {
		t.Fatal("allocated a device twice")
	}
	// Validate persisted state without claiming that this CPU test has a real GPU.
	saved, e := os.ReadFile(c.statePath())
	if e != nil {
		t.Fatal(e)
	}
	if !strings.Contains(string(saved), "pool-reservation") {
		t.Fatal("pool reservation not persisted")
	}
	pools, e := c.ListGPUPools(ctx)
	if e != nil || len(pools) != 1 || pools[0].GPUPoolID != "pool-reservation" {
		t.Fatal("pool reservation not restored", pools, e)
	}
	n := int32(2)
	if c.UpdateGPUPool(ctx, "pool-reservation", "", &n, nil) == nil {
		t.Fatal("unsafe live resize accepted")
	}
	if e = c.DeleteGPUPool(ctx, "pool-reservation"); e != nil {
		t.Fatal(e)
	}
	if _, e = os.Stat(filepath.Join(cfg.Docker.StateDir, "pools.json")); e != nil {
		t.Fatal(e)
	}
	if !strings.HasPrefix(gpuName(cfg.Docker.GPUDevices[0]), "koda-gpu-") {
		t.Fatal("unstable GPU container naming")
	}
}
