package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func baseConfig() string {
	return `master_id: test
server:
  advertised_endpoint: http://localhost:8765
cpu_cluster:
  kubeconfig: cpu.conf
  namespace: cpu
  pod_template_file: cpu.yaml
gpu_pool_cluster:
  kubeconfig: gpu.conf
  namespace: gpu
  deployment_template_file: gpu.yaml
sandboxd:
  image: sandboxd-image
execution:
  max_attempts: 3
  max_stdin_mb: 4
  max_output_mb: 16
`
}

func writeConfig(t *testing.T, data string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadDefaults(t *testing.T) {
	path := writeConfig(t, baseConfig())
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MasterID != "test" || cfg.GPUPoolCluster.PoolNamePrefix != defaultGPUPoolPrefix || cfg.CPUCluster.PodNamePrefix != defaultSessionPodPrefix || cfg.Execution.MaxAttempts != 3 {
		t.Fatalf("unexpected config: %+v", cfg)
	}
	if cfg.GC.IntervalSeconds != 60 || cfg.GC.SessionIdleTimeoutSeconds != 1800 || cfg.GC.GPUPoolIdleTimeoutSeconds != 1800 {
		t.Fatalf("unexpected gc defaults: %+v", cfg.GC)
	}
	if cfg.CPUCluster.Kubeconfig != filepath.Join(filepath.Dir(path), "cpu.conf") {
		t.Fatalf("kubeconfig=%q", cfg.CPUCluster.Kubeconfig)
	}
}

func TestLoadGPUSessionClusters(t *testing.T) {
	path := writeConfig(t, baseConfig()+`gpu_session_clusters:
  h20:
    kubeconfig: h20.config
    namespace: h20-namespace
    pod_template_file: h20-pod.yaml
    pod_name_prefix: gpu-session-pod
    resource_names:
      - example.org/gpu_hopper
`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	h20 := cfg.GPUSessionClusters["h20"]
	if h20.Kubeconfig != filepath.Join(filepath.Dir(path), "h20.config") || h20.PodTemplateFile != filepath.Join(filepath.Dir(path), "h20-pod.yaml") || h20.PodNamePrefix != "gpu-session-pod" || h20.Namespace != "h20-namespace" || len(h20.ResourceNames) != 1 || h20.ResourceNames[0] != "example.org/gpu_hopper" {
		t.Fatalf("h20=%+v", h20)
	}
}

func TestLoadRejectsDuplicateGPUSessionResources(t *testing.T) {
	path := writeConfig(t, baseConfig()+`gpu_session_clusters:
  h20-a:
    kubeconfig: a.config
    namespace: gpu
    pod_template_file: a.yaml
    pod_name_prefix: gpu-session-a
    resource_names: [example.org/gpu]
  h20-b:
    kubeconfig: b.config
    namespace: gpu
    pod_template_file: b.yaml
    pod_name_prefix: gpu-session-b
    resource_names: [example.org/gpu]
`)
	if _, err := Load(path); err == nil || !strings.Contains(err.Error(), "configured by both") {
		t.Fatalf("error=%v", err)
	}
}

func TestLoadValidatesRequiredFields(t *testing.T) {
	for _, test := range []struct {
		name string
		data string
	}{
		{name: "master id", data: strings.Replace(baseConfig(), "master_id: test\n", "", 1)},
		{name: "sandboxd image", data: strings.Replace(baseConfig(), "image: sandboxd-image", "image: null", 1)},
		{name: "session pod template", data: baseConfig() + "gpu_session_clusters:\n  h20:\n    kubeconfig: h20.config\n    namespace: gpu\n    resource_names: [example.org/gpu_hopper]\n"},
		{name: "session pod name prefix", data: baseConfig() + "gpu_session_clusters:\n  h20:\n    kubeconfig: h20.config\n    namespace: gpu\n    pod_template_file: h20.yaml\n    resource_names: [example.org/gpu_hopper]\n"},
		{name: "session resource names", data: baseConfig() + "gpu_session_clusters:\n  h20:\n    kubeconfig: h20.config\n    namespace: gpu\n    pod_template_file: h20.yaml\n    pod_name_prefix: gpu-session\n"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := Load(writeConfig(t, test.data)); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestLoadValidatesResourceNamePrefixes(t *testing.T) {
	for _, replacement := range []string{"pool_name_prefix: GPU_POOL", "pool_name_prefix: " + strings.Repeat("a", 47)} {
		data := strings.Replace(baseConfig(), "deployment_template_file: gpu.yaml", "deployment_template_file: gpu.yaml\n  "+replacement, 1)
		if _, err := Load(writeConfig(t, data)); err == nil {
			t.Fatal("expected prefix validation error")
		}
	}
}

func TestLoadGPUProcessAllowlist(t *testing.T) {
	data := strings.Replace(baseConfig(), "image: sandboxd-image", `image: sandboxd-image
  gpu_process_allowlist:
    - process: multiprocessing.spawn
      ancestor: kerneleval.server
    - process: trusted-worker`, 1)
	cfg, err := Load(writeConfig(t, data))
	if err != nil || len(cfg.Sandboxd.GPUProcessAllowlist) != 2 || cfg.Sandboxd.GPUProcessAllowlist[0].Ancestor != "kerneleval.server" {
		t.Fatalf("config=%+v error=%v", cfg, err)
	}
}

func TestLoadPortableKubernetesTemplate(t *testing.T) {
	path := filepath.Join("..", "..", "deploy", "kubernetes", "master.yaml.tmpl")
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Backend != "kubernetes" || cfg.MasterID != "koda" || len(cfg.GPUPoolClusters) != 1 || cfg.GPUPoolClusters[0].ResourceNames[0] != "nvidia.com/gpu" {
		t.Fatalf("unexpected template: %+v", cfg)
	}
}
