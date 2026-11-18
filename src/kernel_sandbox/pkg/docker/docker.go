// Package docker implements the existing Master resource interface on a local Docker daemon.
package docker

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"kernel-sandbox/pkg/command"
	"kernel-sandbox/pkg/config"
	"kernel-sandbox/pkg/master"
)

const ownerLabel = "io.koda.sandbox.master"
const kindLabel = "io.koda.sandbox.kind"
const poolLabel = "io.koda.sandbox.pool"
const sessionLabel = "io.koda.sandbox.session"
const gpuLabel = "io.koda.sandbox.gpu"
const binary = "/tmp/koda-sandboxd"

type poolRecord struct {
	ID      string   `json:"id"`
	Image   string   `json:"image"`
	Devices []string `json:"devices"`
}
type Client struct {
	cfg   *config.Config
	mu    sync.Mutex
	lock  *os.File
	pools map[string]poolRecord
}

var _ master.KubernetesClient = (*Client)(nil)

type containerInfo struct {
	ID      string `json:"Id"`
	Name    string
	Created string
	Config  struct {
		Image  string
		Labels map[string]string
	}
	State struct {
		Running  bool
		Status   string
		ExitCode int
		Error    string
	}
	RestartCount int32
}

func New(cfg *config.Config) (*Client, error) {
	// Bind mounts and state recovery deliberately target a local daemon only.
	if h := os.Getenv("DOCKER_HOST"); h != "" && !strings.HasPrefix(h, "unix://") {
		return nil, fmt.Errorf("Docker backend requires a local unix-socket daemon")
	}
	if os.Getenv("DOCKER_CONTEXT") != "" {
		return nil, fmt.Errorf("unset DOCKER_CONTEXT; use the local daemon")
	}
	if err := os.MkdirAll(cfg.Docker.StateDir, 0700); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(filepath.Join(cfg.Docker.StateDir, "master.lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	if err = syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		return nil, fmt.Errorf("state directory is already in use: %w", err)
	}
	c := &Client{cfg: cfg, lock: f, pools: map[string]poolRecord{}}
	fail := func(e error) (*Client, error) { c.Close(); return nil, e }
	if data, e := os.ReadFile(c.statePath()); e == nil {
		if e = json.Unmarshal(data, &c.pools); e != nil {
			return fail(e)
		}
	} else if !os.IsNotExist(e) {
		return fail(e)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if _, e := c.run(ctx, "version", "--format", "{{.Server.Version}}"); e != nil {
		return fail(e)
	}

	daemonID, e := c.run(ctx, "info", "--format", "{{.ID}}")
	if e != nil {
		return fail(e)
	}
	identity := cfg.MasterID + "\n" + strings.TrimSpace(string(daemonID)) + "\n"
	identityPath := filepath.Join(cfg.Docker.StateDir, "identity")
	if b, e := os.ReadFile(identityPath); e == nil {
		if string(b) != identity {
			return fail(fmt.Errorf("state directory belongs to another master or Docker daemon"))
		}
	} else if os.IsNotExist(e) {
		if e = os.WriteFile(identityPath, []byte(identity), 0600); e != nil {
			return fail(e)
		}
	} else {
		return fail(e)
	}
	if len(cfg.Docker.GPUDevices) > 0 {
		out, e := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader").Output()
		if e != nil {
			return fail(fmt.Errorf("configured GPUs require host nvidia-smi: %w", e))
		}
		available := map[string]bool{}
		for _, d := range strings.Fields(string(out)) {
			available[d] = true
		}
		for _, d := range cfg.Docker.GPUDevices {
			if !available[d] {
				return fail(fmt.Errorf("configured GPU UUID %q is not present", d))
			}
		}
	}
	for _, p := range c.pools {
		for _, gpu := range p.Devices {
			allowed := false
			for _, g := range cfg.Docker.GPUDevices {
				if gpu == g {
					allowed = true
				}
			}
			if !allowed {
				return fail(fmt.Errorf("persisted pool uses GPU %q removed from configuration", gpu))
			}
		}
	}
	// A second state directory must not silently adopt this master's containers.
	cs, e := c.list(ctx, "")
	if e != nil {
		return fail(e)
	}
	for _, x := range cs {
		if x.Config.Labels["io.koda.sandbox.state"] != cfg.Docker.StateDir {
			return fail(fmt.Errorf("master_id already owns containers with a different state directory"))
		}
	}
	return c, nil
}
func (c *Client) Close() error {
	if c.lock == nil {
		return nil
	}
	syscall.Flock(int(c.lock.Fd()), syscall.LOCK_UN)
	e := c.lock.Close()
	c.lock = nil
	return e
}
func (c *Client) statePath() string { return filepath.Join(c.cfg.Docker.StateDir, "pools.json") }
func (c *Client) save() error {
	data, e := json.MarshalIndent(c.pools, "", "  ")
	if e != nil {
		return e
	}
	p := c.statePath() + ".tmp"
	f, e := os.OpenFile(p, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if e != nil {
		return e
	}
	if _, e = f.Write(data); e == nil {
		e = f.Sync()
	}
	ce := f.Close()
	if e == nil {
		e = ce
	}
	if e != nil {
		return e
	}
	return os.Rename(p, c.statePath())
}
func (c *Client) cmd(ctx context.Context, args ...string) *exec.Cmd {
	if os.Getenv("DOCKER_HOST") == "" {
		args = append([]string{"--host", "unix:///var/run/docker.sock"}, args...)
	}
	return exec.CommandContext(ctx, c.cfg.Docker.Binary, args...)
}
func (c *Client) run(ctx context.Context, args ...string) ([]byte, error) {
	cmd := c.cmd(ctx, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("docker %s: %w: %s", args[0], err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

func (c *Client) inspect(ctx context.Context, id string) (containerInfo, error) {
	var rows []containerInfo
	out, e := c.run(ctx, "inspect", "--type", "container", id)
	if e != nil {
		return containerInfo{}, e
	}
	if e = json.Unmarshal(out, &rows); e != nil {
		return containerInfo{}, e
	}
	if len(rows) != 1 {
		return containerInfo{}, fmt.Errorf("expected one container")
	}
	return rows[0], nil
}
func (c *Client) list(ctx context.Context, kind string) ([]containerInfo, error) {
	args := []string{"ps", "-aq", "--filter", "label=" + ownerLabel + "=" + c.cfg.MasterID}
	if kind != "" {
		args = append(args, "--filter", "label="+kindLabel+"="+kind)
	}
	out, e := c.run(ctx, args...)
	if e != nil {
		return nil, e
	}
	ids := strings.Fields(string(out))
	if len(ids) == 0 {
		return []containerInfo{}, nil
	}
	var result []containerInfo
	// Containers can disappear between ps and inspect; surface that race for retry, never adopt by name.
	for _, id := range ids {
		x, e := c.inspect(ctx, id)
		if e != nil {
			return nil, e
		}
		if x.Config.Labels["io.koda.sandbox.state"] != c.cfg.Docker.StateDir {
			return nil, fmt.Errorf("master_id owns containers from a different state directory")
		}
		result = append(result, x)
	}
	return result, nil
}
func (c *Client) owned(ctx context.Context, p master.PodInfo) (containerInfo, error) {
	if p.UID == "" {
		return containerInfo{}, fmt.Errorf("container ID is required")
	}
	x, e := c.inspect(ctx, p.UID)
	if e != nil {
		return x, e
	}
	if x.Config.Labels[ownerLabel] != c.cfg.MasterID || x.Config.Labels["io.koda.sandbox.state"] != c.cfg.Docker.StateDir || strings.TrimPrefix(x.Name, "/") != p.Name {
		return x, fmt.Errorf("container ownership mismatch")
	}
	return x, nil
}
func pod(x containerInfo) master.PodInfo {
	created, _ := time.Parse(time.RFC3339Nano, x.Created)
	l := x.Config.Labels
	return master.PodInfo{Name: strings.TrimPrefix(x.Name, "/"), UID: x.ID, SessionID: l[sessionLabel], GPUPoolID: l[poolLabel], Prefix: l["io.koda.sandbox.prefix"], Image: x.Config.Image, CreatedAt: created, Ready: x.State.Running, Failed: x.State.Status == "exited" || x.State.Status == "dead", RestartCount: x.RestartCount}
}
func gpuName(uuid string) string {
	h := sha256.Sum256([]byte(strings.ToLower(uuid)))
	return "koda-gpu-" + hex.EncodeToString(h[:12])
}
func validateRequest(r *master.ResourceRequirements, s *master.Scheduling) error {
	if r != nil && (len(r.Requests) > 0 || len(r.Limits) > 0) {
		return fmt.Errorf("Docker backend uses docker.cpu_cores/memory_mb/gpu_devices; Kubernetes resource maps are unsupported")
	}
	if s != nil && (len(s.NodeSelector) > 0 || len(s.Tolerations) > 0) {
		return fmt.Errorf("Docker backend does not support Kubernetes scheduling fields")
	}
	return nil
}
func (c *Client) create(ctx context.Context, name, image, kind, pool, session, prefix, gpu string) (master.PodInfo, error) {
	sandboxURL := c.cfg.Server.AdvertisedEndpoint
	if kind == "cpu" {
		sandboxURL = strings.TrimRight(sandboxURL, "/") + "/v1/session/" + session
	}

	args := []string{"create", "--name", name, "--network", c.cfg.Docker.Network, "--label", ownerLabel + "=" + c.cfg.MasterID, "--label", "io.koda.sandbox.state=" + c.cfg.Docker.StateDir, "--label", kindLabel + "=" + kind, "--label", poolLabel + "=" + pool, "--label", sessionLabel + "=" + session, "--label", "io.koda.sandbox.prefix=" + prefix, "--env", "KERNEL_SANDBOX_URL=" + sandboxURL, "--entrypoint", binary}
	if gpu != "" {
		args = append(args, "--gpus", "device="+gpu, "--label", gpuLabel+"="+gpu, "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility")
		for _, v := range c.cfg.Docker.GPUCapabilities {
			args = append(args, "--cap-add", v)
		}
	} else {
		args = append(args, "--env", "NVIDIA_VISIBLE_DEVICES=void")
	}
	if c.cfg.Docker.CPUCores > 0 {
		args = append(args, "--cpus", strconv.FormatFloat(c.cfg.Docker.CPUCores, 'f', -1, 64))
	}
	if c.cfg.Docker.MemoryMB > 0 {
		args = append(args, "--memory", fmt.Sprintf("%dm", c.cfg.Docker.MemoryMB))
	}
	for _, m := range c.cfg.Docker.Mounts {
		mount := "type=bind,source=" + m.Source + ",target=" + m.Target
		if m.ReadOnly {
			mount += ",readonly"
		}
		args = append(args, "--mount", mount)
	}
	args = append(args, image, "serve", "--socket", config.SandboxdSocketPath, "--terminate-grace-seconds", strconv.Itoa(config.SandboxdTerminateGraceSeconds), "--max-stdin-bytes", strconv.Itoa(c.cfg.Execution.MaxStdinMB<<20), "--max-output-bytes", strconv.Itoa(c.cfg.Execution.MaxOutputMB<<20))
	if kind == "cpu" {
		args = append(args, "--allow-cpu")
	}
	if len(c.cfg.Sandboxd.GPUProcessAllowlist) > 0 {
		b, _ := json.Marshal(c.cfg.Sandboxd.GPUProcessAllowlist)
		args = append(args, "--gpu-process-allowlist", string(b))
	}
	out, e := c.run(ctx, args...)
	if e != nil {
		return master.PodInfo{}, e
	}
	id := strings.TrimSpace(string(out))
	cleanup := func() {
		b, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		c.run(b, "rm", "-f", id)
	}
	if _, e = c.run(ctx, "cp", c.cfg.Docker.SandboxdBinary, id+":"+binary); e != nil {
		cleanup()
		return master.PodInfo{}, e
	}
	if _, e = c.run(ctx, "start", id); e != nil {
		cleanup()
		return master.PodInfo{}, e
	}
	x, e := c.inspect(ctx, id)
	if e != nil {
		cleanup()
		return master.PodInfo{}, e
	}
	return pod(x), nil
}
func (c *Client) CreateCPUPod(ctx context.Context, session, pool, image, url string, r *master.ResourceRequirements, prefix string) (master.PodInfo, error) {
	if e := validateRequest(r, nil); e != nil {
		return master.PodInfo{}, e
	}
	if prefix == "" {
		prefix = c.cfg.CPUCluster.PodNamePrefix
	}
	return c.create(ctx, c.cfg.MasterID+"-"+session, image, "cpu", pool, session, prefix, "")
}
func (c *Client) remove(ctx context.Context, p master.PodInfo) error {
	if _, e := c.owned(ctx, p); e != nil {
		// Missing exact IDs are already gone; never fall back to a reused name.
		out, le := c.run(ctx, "ps", "-aq", "--filter", "id="+p.UID)
		if le == nil && strings.TrimSpace(string(out)) == "" {
			return nil
		}
		return e
	}
	_, e := c.run(ctx, "rm", "-f", p.UID)
	return e
}
func (c *Client) DeleteCPUPod(ctx context.Context, p master.PodInfo) error { return c.remove(ctx, p) }
func (c *Client) ListCPUPods(ctx context.Context) ([]master.PodInfo, error) {
	xs, e := c.list(ctx, "cpu")
	if e != nil {
		return nil, e
	}
	ps := []master.PodInfo{}
	for _, x := range xs {
		ps = append(ps, pod(x))
	}
	return ps, nil
}
func (c *Client) CreateGPUPool(ctx context.Context, id, image string, replicas int32, r *master.ResourceRequirements, s *master.Scheduling) error {
	if e := validateRequest(r, s); e != nil {
		return e
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if replicas <= 0 {
		return fmt.Errorf("replicas must be positive")
	}
	if _, ok := c.pools[id]; ok {
		return fmt.Errorf("pool already exists")
	}
	used := map[string]bool{}
	for _, p := range c.pools {
		for _, d := range p.Devices {
			used[d] = true
		}
	}
	devices := []string{}
	for _, d := range c.cfg.Docker.GPUDevices {
		if !used[d] {
			devices = append(devices, d)
		}
	}
	if int(replicas) > len(devices) {
		return fmt.Errorf("requested %d GPUs, only %d unallocated configured devices", replicas, len(devices))
	}
	p := poolRecord{ID: id, Image: image, Devices: devices[:replicas]}
	c.pools[id] = p
	if e := c.save(); e != nil {
		delete(c.pools, id)
		return e
	}
	// Desired state is persisted first; reconciliation creates workers and recovers interruptions.
	return nil
}
func (c *Client) ListGPUPools(ctx context.Context) ([]master.GPUPoolInfo, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	ps := []master.GPUPoolInfo{}
	for _, p := range c.pools {
		ps = append(ps, master.GPUPoolInfo{GPUPoolID: p.ID, Image: p.Image, Replicas: int32(len(p.Devices))})
	}
	return ps, nil
}
func (c *Client) UpdateGPUPool(ctx context.Context, id, image string, replicas *int32, s *master.Scheduling) error {
	if e := validateRequest(nil, s); e != nil {
		return e
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	p, ok := c.pools[id]
	if !ok {
		return fmt.Errorf("pool not found")
	}
	// Replacing a pool during an execution could interrupt a kernel. Recreate explicitly after sessions close.
	if (image != "" && image != p.Image) || (replicas != nil && *replicas != int32(len(p.Devices))) {
		return fmt.Errorf("Docker pool resize/image update requires deleting its sessions and recreating the pool")
	}
	return nil
}
func (c *Client) DeleteGPUPool(ctx context.Context, id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	xs, e := c.list(ctx, "gpu")
	if e != nil {
		return e
	}
	// Remove only owned workers with this pool ID. Retain desired state on deletion failure for retry.
	for _, x := range xs {
		if x.Config.Labels[poolLabel] == id {
			if e = c.remove(ctx, pod(x)); e != nil {
				return e
			}
		}
	}
	old, ok := c.pools[id]
	delete(c.pools, id)
	if e = c.save(); e != nil {
		if ok {
			c.pools[id] = old
		}
		return e
	}
	return nil
}
func (c *Client) ListGPUPods(ctx context.Context) ([]master.PodInfo, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	xs, e := c.list(ctx, "gpu")
	if e != nil {
		return nil, e
	}
	byName := map[string]containerInfo{}
	for _, x := range xs {
		byName[strings.TrimPrefix(x.Name, "/")] = x
	}
	for _, p := range c.pools {
		for _, device := range p.Devices {
			name := gpuName(device)
			x, ok := byName[name]
			if ok && x.Config.Labels[poolLabel] != p.ID {
				return nil, fmt.Errorf("GPU reservation belongs to another pool")
			}
			if ok && x.State.Running {
				continue
			}
			if ok {
				if e = c.remove(ctx, pod(x)); e != nil {
					return nil, e
				}
			}
			if _, e = c.create(ctx, name, p.Image, "gpu", p.ID, "", "", device); e != nil {
				return nil, e
			}
		}
	}
	xs, e = c.list(ctx, "gpu")
	if e != nil {
		return nil, e
	}
	ps := []master.PodInfo{}
	for _, x := range xs {
		if _, ok := c.pools[x.Config.Labels[poolLabel]]; ok {
			ps = append(ps, pod(x))
		}
	}
	return ps, nil
}
func (c *Client) DeleteGPUWorkerPod(ctx context.Context, p master.PodInfo) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.remove(ctx, p)
}

// All command execution goes through sandboxd, including CPU commands, so timeouts kill descendants.
func (c *Client) execute(ctx context.Context, p master.PodInfo, request command.Request, maxOutput int) command.Result {
	if _, e := c.owned(ctx, p); e != nil {
		return command.Result{Kind: command.ResultInfrastructure, Error: e.Error()}
	}
	body, e := json.Marshal(request)
	if e != nil {
		return command.Result{Kind: command.ResultInfrastructure, Error: e.Error()}
	}
	// Let sandboxd enforce the deadline and return the timeout receipt before the outer transport expires.
	transport, cancel := context.WithTimeout(context.WithoutCancel(ctx), time.Duration(request.TimeoutSeconds+int64(config.SandboxdTerminateGraceSeconds)+5)*time.Second)
	defer cancel()
	output := command.NewOutput(maxOutput*6 + (1 << 20))
	cmd := c.cmd(transport, "exec", "-i", p.UID, binary, "exec", "--socket", config.SandboxdSocketPath)
	cmd.Stdin = strings.NewReader(string(body))
	cmd.Stdout = output.Stdout()
	cmd.Stderr = output.Stderr()
	done := make(chan struct{})
	go func() {
		select {
		case <-output.Exceeded():
			cancel()
		case <-done:
		}
	}()
	e = cmd.Run()
	close(done)
	stdout, stderr, overflow := output.Result()
	if e != nil || overflow {
		// Killing docker exec alone does not kill the container process; remove this owned container on transport failure.
		cleanup, stop := context.WithTimeout(context.Background(), 15*time.Second)
		defer stop()
		c.remove(cleanup, p)
		kind := command.ResultInfrastructure
		if overflow {
			kind = command.ResultOutputLimit
		} else if errors.Is(transport.Err(), context.DeadlineExceeded) {
			kind = command.ResultTimeout
		}
		return command.Result{Kind: kind, Error: fmt.Sprintf("sandboxd transport failed: %v %s", e, stderr)}
	}
	var result command.Result
	if e = json.Unmarshal([]byte(stdout), &result); e != nil {
		return command.Result{Kind: command.ResultInfrastructure, Error: "invalid sandboxd receipt: " + e.Error()}
	}
	return result
}
func (c *Client) ExecuteCPU(ctx context.Context, p master.PodInfo, r command.Request, n int) command.Result {
	r.Device = command.DeviceCPU
	return c.execute(ctx, p, r, n)
}
func (c *Client) ExecuteGPU(ctx context.Context, p master.PodInfo, r command.Request, n int) command.Result {
	r.Device = command.DeviceGPU
	return c.execute(ctx, p, r, n)
}
func (c *Client) ReadGPUPodState(ctx context.Context, p master.PodInfo) (master.GPUPodState, error) {
	if _, e := c.owned(ctx, p); e != nil {
		return master.GPUPodState{}, e
	}
	out, e := c.run(ctx, "exec", p.UID, binary, "state", "--socket", config.SandboxdSocketPath)
	if e != nil {
		return master.GPUPodState{}, e
	}
	var state master.GPUPodState
	e = json.Unmarshal(out, &state)
	return state, e
}
