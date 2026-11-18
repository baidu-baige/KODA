package config

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
)

type DockerMount struct {
	Source   string `json:"source"`
	Target   string `json:"target"`
	ReadOnly bool   `json:"read_only"`
}
type DockerConfig struct {
	Binary          string        `json:"binary"`
	StateDir        string        `json:"state_dir"`
	SandboxdBinary  string        `json:"sandboxd_binary"`
	GPUDevices      []string      `json:"gpu_devices"`
	Network         string        `json:"network"`
	Mounts          []DockerMount `json:"mounts"`
	GPUCapabilities []string      `json:"gpu_capabilities"`
	CPUCores        float64       `json:"cpu_cores"`
	MemoryMB        int64         `json:"memory_mb"`
}

func loadDocker(c *Config, base string) (*Config, error) {
	if c.Docker.Binary == "" {
		c.Docker.Binary = "docker"
	}
	if c.Docker.StateDir == "" {
		return nil, fmt.Errorf("docker.state_dir is required")
	}
	c.Docker.StateDir = resolvePath(base, c.Docker.StateDir)
	c.Docker.SandboxdBinary = resolvePath(base, c.Docker.SandboxdBinary)
	if !filepath.IsAbs(c.Docker.StateDir) {
		p, e := filepath.Abs(c.Docker.StateDir)
		if e != nil {
			return nil, e
		}
		c.Docker.StateDir = p
	}
	f, e := os.Stat(c.Docker.SandboxdBinary)
	if e != nil || !f.Mode().IsRegular() || f.Mode()&0111 == 0 {
		return nil, fmt.Errorf("docker.sandboxd_binary must be an executable file")
	}
	if !filepath.IsAbs(c.Docker.SandboxdBinary) {
		p, e := filepath.Abs(c.Docker.SandboxdBinary)
		if e != nil {
			return nil, e
		}
		c.Docker.SandboxdBinary = p
	}
	if c.Docker.Network == "" {
		c.Docker.Network = "bridge"
	}
	if c.Docker.Network != "bridge" && c.Docker.Network != "host" {
		return nil, fmt.Errorf("docker.network must be bridge or host")
	}
	u, e := url.Parse(c.Server.AdvertisedEndpoint)
	if e != nil || u.Hostname() == "" {
		return nil, fmt.Errorf("invalid advertised_endpoint")
	}
	if c.Docker.Network == "bridge" && (u.Hostname() == "localhost" || u.Hostname() == "127.0.0.1" || u.Hostname() == "::1") {
		return nil, fmt.Errorf("bridge containers need a non-loopback advertised_endpoint")
	}
	seen := map[string]bool{}
	gpuPattern := regexp.MustCompile(`^GPU-[0-9a-fA-F-]+$`)
	for _, d := range c.Docker.GPUDevices {
		if !gpuPattern.MatchString(d) || seen[d] {
			return nil, fmt.Errorf("gpu_devices must contain unique GPU UUIDs, not indices: %q", d)
		}
		seen[d] = true
	}
	for _, m := range c.Docker.Mounts {
		if !filepath.IsAbs(m.Source) || !filepath.IsAbs(m.Target) || m.Target == "/" {
			return nil, fmt.Errorf("docker mount source and target must be absolute; target cannot be /")
		}
		if _, e := os.Stat(m.Source); e != nil {
			return nil, fmt.Errorf("mount source: %w", e)
		}
	}
	if c.Docker.CPUCores < 0 || c.Docker.MemoryMB < 0 {
		return nil, fmt.Errorf("docker resource limits must not be negative")
	}
	for _, cap := range c.Docker.GPUCapabilities {
		if cap != "SYS_ADMIN" && cap != "IPC_LOCK" {
			return nil, fmt.Errorf("unsupported GPU capability %q", cap)
		}
	}
	if len(c.GPUPoolClusters) > 0 || len(c.GPUSessionClusters) > 0 {
		return nil, fmt.Errorf("docker backend does not accept Kubernetes cluster routes; gpu_only is not supported")
	}
	c.GPUPoolCluster.PoolNamePrefix = "docker-gpu-pool"
	c.CPUCluster.PodNamePrefix = "docker-cpu"
	if c.Execution.MaxAttempts <= 0 || c.Execution.MaxStdinMB <= 0 || c.Execution.MaxOutputMB <= 0 {
		return nil, fmt.Errorf("execution limits must be positive")
	}
	if c.GC.IntervalSeconds < 0 || c.GC.SessionIdleTimeoutSeconds < 0 || c.GC.GPUPoolIdleTimeoutSeconds < 0 {
		return nil, fmt.Errorf("gc timeouts must not be negative")
	}
	for _, a := range c.Sandboxd.GPUProcessAllowlist {
		if a.Process == "" {
			return nil, fmt.Errorf("empty GPU process allowlist entry")
		}
	}
	return c, nil
}
