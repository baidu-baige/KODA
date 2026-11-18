package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation"
	"sigs.k8s.io/yaml"
)

const (
	SandboxdSocketPath            = "/run/kernel-sandbox/sandboxd.sock"
	SandboxdTerminateGraceSeconds = 5
	defaultGPUPoolPrefix          = "gpu-pool"
	defaultSessionPodPrefix       = "kernel-sandbox"
	generatedIDSuffixLength       = 16
)

type Config struct {
	Backend            string                             `json:"backend"`
	Docker             DockerConfig                       `json:"docker"`
	MasterID           string                             `json:"master_id"`
	Server             ServerConfig                       `json:"server"`
	CPUCluster         CPUClusterConfig                   `json:"cpu_cluster"`
	GPUPoolCluster     GPUPoolClusterConfig               `json:"gpu_pool_cluster"`
	GPUPoolClusters    []GPUPoolClusterConfig             `json:"gpu_pool_clusters"`
	GPUSessionClusters map[string]GPUSessionClusterConfig `json:"gpu_session_clusters"`
	Sandboxd           SandboxdConfig                     `json:"sandboxd"`
	Execution          ExecutionConfig                    `json:"execution"`
	GC                 GCConfig                           `json:"gc"`
}

type ServerConfig struct {
	ListenAddress      string `json:"listen_address"`
	AdvertisedEndpoint string `json:"advertised_endpoint"`
}

type ClusterConfig struct {
	Kubeconfig string `json:"kubeconfig"`
	Namespace  string `json:"namespace"`
}

type CPUClusterConfig struct {
	ClusterConfig
	PodTemplateFile string `json:"pod_template_file"`
	PodNamePrefix   string `json:"pod_name_prefix"`
}

type GPUPoolClusterConfig struct {
	ClusterConfig
	Name                   string   `json:"name"`
	DeploymentTemplateFile string   `json:"deployment_template_file"`
	PoolNamePrefix         string   `json:"pool_name_prefix"`
	ResourceNames          []string `json:"resource_names"`
}

type GPUSessionClusterConfig struct {
	ClusterConfig
	PodTemplateFile string   `json:"pod_template_file"`
	PodNamePrefix   string   `json:"pod_name_prefix"`
	ResourceNames   []string `json:"resource_names"`
}

type SandboxdConfig struct {
	Image               string                     `json:"image"`
	GPUProcessAllowlist []GPUProcessAllowlistEntry `json:"gpu_process_allowlist,omitempty"`
}

type GPUProcessAllowlistEntry struct {
	Process  string `json:"process"`
	Ancestor string `json:"ancestor,omitempty"`
}

type ExecutionConfig struct {
	MaxAttempts int `json:"max_attempts"`
	MaxStdinMB  int `json:"max_stdin_mb"`
	MaxOutputMB int `json:"max_output_mb"`
}

type GCConfig struct {
	IntervalSeconds           int `json:"interval_seconds"`
	SessionIdleTimeoutSeconds int `json:"session_idle_timeout_seconds"`
	GPUPoolIdleTimeoutSeconds int `json:"gpu_pool_idle_timeout_seconds"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	cfg := &Config{GC: GCConfig{
		IntervalSeconds:           60,
		SessionIdleTimeoutSeconds: 1800,
		GPUPoolIdleTimeoutSeconds: 1800,
	}}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}
	baseDir := filepath.Dir(path)
	cfg.CPUCluster.Kubeconfig = resolvePath(baseDir, cfg.CPUCluster.Kubeconfig)
	cfg.CPUCluster.PodTemplateFile = resolvePath(baseDir, cfg.CPUCluster.PodTemplateFile)
	cfg.GPUPoolCluster.Kubeconfig = resolvePath(baseDir, cfg.GPUPoolCluster.Kubeconfig)
	cfg.GPUPoolCluster.DeploymentTemplateFile = resolvePath(baseDir, cfg.GPUPoolCluster.DeploymentTemplateFile)
	for i, cluster := range cfg.GPUPoolClusters {
		cluster.Kubeconfig = resolvePath(baseDir, cluster.Kubeconfig)
		cluster.DeploymentTemplateFile = resolvePath(baseDir, cluster.DeploymentTemplateFile)
		cfg.GPUPoolClusters[i] = cluster
	}
	for name, cluster := range cfg.GPUSessionClusters {
		cluster.Kubeconfig = resolvePath(baseDir, cluster.Kubeconfig)
		cluster.PodTemplateFile = resolvePath(baseDir, cluster.PodTemplateFile)
		cfg.GPUSessionClusters[name] = cluster
	}
	if errors := validation.IsDNS1123Label(cfg.MasterID); len(errors) != 0 {
		return nil, fmt.Errorf("master_id is invalid: %s", errors[0])
	}
	if cfg.Server.ListenAddress == "" {
		cfg.Server.ListenAddress = "0.0.0.0:8765"
	}
	if cfg.Server.AdvertisedEndpoint == "" {
		return nil, fmt.Errorf("server.advertised_endpoint is required")
	}
	if cfg.Backend == "" {
		cfg.Backend = "kubernetes"
	}
	if cfg.Backend != "kubernetes" && cfg.Backend != "docker" {
		return nil, fmt.Errorf("backend must be kubernetes or docker")
	}
	if cfg.Backend == "docker" {
		return loadDocker(cfg, baseDir)
	}
	// When gpu_pool_clusters is configured, gpu_pool_cluster is optional: the
	// first gpu_pool_clusters entry is used as the default pool cluster.
	if cfg.GPUPoolCluster.Kubeconfig == "" && len(cfg.GPUPoolClusters) > 0 {
		cfg.GPUPoolCluster = cfg.GPUPoolClusters[0]
	}
	if cfg.GPUPoolCluster.PoolNamePrefix == "" {
		cfg.GPUPoolCluster.PoolNamePrefix = defaultGPUPoolPrefix
	}
	if cfg.CPUCluster.PodNamePrefix == "" {
		cfg.CPUCluster.PodNamePrefix = defaultSessionPodPrefix
	}
	if errors := validation.IsDNS1123Label(cfg.GPUPoolCluster.PoolNamePrefix); len(errors) != 0 {
		return nil, fmt.Errorf("gpu_pool_cluster.pool_name_prefix is invalid: %s", errors[0])
	}
	if len(cfg.GPUPoolCluster.PoolNamePrefix)+1+generatedIDSuffixLength > validation.LabelValueMaxLength {
		return nil, fmt.Errorf("gpu_pool_cluster.pool_name_prefix must be at most %d characters", validation.LabelValueMaxLength-1-generatedIDSuffixLength)
	}
	if errors := validation.IsDNS1123Subdomain(cfg.CPUCluster.PodNamePrefix); len(errors) != 0 {
		return nil, fmt.Errorf("cpu_cluster.pod_name_prefix is invalid: %s", errors[0])
	}
	if len(cfg.CPUCluster.PodNamePrefix)+1+len("session-")+generatedIDSuffixLength > validation.DNS1123SubdomainMaxLength {
		return nil, fmt.Errorf("cpu_cluster.pod_name_prefix is too long")
	}
	if err := validateCluster("cpu_cluster", cfg.CPUCluster.ClusterConfig); err != nil {
		return nil, err
	}
	if cfg.CPUCluster.PodTemplateFile == "" {
		return nil, fmt.Errorf("cpu_cluster.pod_template_file is required")
	}
	if err := validateCluster("gpu_pool_cluster", cfg.GPUPoolCluster.ClusterConfig); err != nil {
		return nil, err
	}
	if cfg.GPUPoolCluster.DeploymentTemplateFile == "" {
		return nil, fmt.Errorf("gpu_pool_cluster.deployment_template_file is required")
	}
	poolResourceClusters := map[string]string{}
	for i, cluster := range cfg.GPUPoolClusters {
		name := cluster.Name
		if errors := validation.IsDNS1123Label(name); len(errors) != 0 {
			return nil, fmt.Errorf("gpu_pool_clusters[%d].name is invalid: %s", i, errors[0])
		}
		if err := validateCluster("gpu_pool_clusters."+name, cluster.ClusterConfig); err != nil {
			return nil, err
		}
		if cluster.DeploymentTemplateFile == "" {
			return nil, fmt.Errorf("gpu_pool_clusters.%s.deployment_template_file is required", name)
		}
		if cluster.PoolNamePrefix == "" {
			cluster.PoolNamePrefix = defaultGPUPoolPrefix
		}
		if errors := validation.IsDNS1123Label(cluster.PoolNamePrefix); len(errors) != 0 {
			return nil, fmt.Errorf("gpu_pool_clusters.%s.pool_name_prefix is invalid: %s", name, errors[0])
		}
		if len(cluster.ResourceNames) == 0 {
			return nil, fmt.Errorf("gpu_pool_clusters.%s.resource_names is required", name)
		}
		for _, resourceName := range cluster.ResourceNames {
			if errors := validation.IsQualifiedName(resourceName); len(errors) != 0 {
				return nil, fmt.Errorf("gpu_pool_clusters.%s.resource_names contains invalid resource %q: %s", name, resourceName, errors[0])
			}
			if other, exists := poolResourceClusters[resourceName]; exists {
				return nil, fmt.Errorf("GPU Pool resource %q is configured by both %q and %q", resourceName, other, name)
			}
			poolResourceClusters[resourceName] = name
		}
		cfg.GPUPoolClusters[i] = cluster
	}
	resourceClusters := map[string]string{}
	for name, cluster := range cfg.GPUSessionClusters {
		if errors := validation.IsDNS1123Label(name); len(errors) != 0 {
			return nil, fmt.Errorf("gpu_session_clusters.%s name is invalid: %s", name, errors[0])
		}
		if err := validateCluster("gpu_session_clusters."+name, cluster.ClusterConfig); err != nil {
			return nil, err
		}
		if cluster.PodTemplateFile == "" {
			return nil, fmt.Errorf("gpu_session_clusters.%s.pod_template_file is required", name)
		}
		if errors := validation.IsDNS1123Subdomain(cluster.PodNamePrefix); len(errors) != 0 {
			return nil, fmt.Errorf("gpu_session_clusters.%s.pod_name_prefix is invalid: %s", name, errors[0])
		}
		if len(cluster.PodNamePrefix)+1+len("gpu-session-")+generatedIDSuffixLength > validation.DNS1123SubdomainMaxLength {
			return nil, fmt.Errorf("gpu_session_clusters.%s.pod_name_prefix is too long", name)
		}
		if len(cluster.ResourceNames) == 0 {
			return nil, fmt.Errorf("gpu_session_clusters.%s.resource_names is required", name)
		}
		for _, resourceName := range cluster.ResourceNames {
			if errors := validation.IsQualifiedName(resourceName); len(errors) != 0 {
				return nil, fmt.Errorf("gpu_session_clusters.%s.resource_names contains invalid resource %q: %s", name, resourceName, errors[0])
			}
			if other, exists := resourceClusters[resourceName]; exists {
				return nil, fmt.Errorf("GPU resource %q is configured by both %q and %q", resourceName, other, name)
			}
			resourceClusters[resourceName] = name
		}
	}
	if cfg.Sandboxd.Image == "" {
		return nil, fmt.Errorf("sandboxd.image is required")
	}
	for i, entry := range cfg.Sandboxd.GPUProcessAllowlist {
		if strings.TrimSpace(entry.Process) == "" {
			return nil, fmt.Errorf("sandboxd.gpu_process_allowlist[%d].process is required", i)
		}
	}
	if cfg.Execution.MaxAttempts <= 0 {
		return nil, fmt.Errorf("execution.max_attempts must be positive")
	}
	if cfg.Execution.MaxStdinMB <= 0 {
		return nil, fmt.Errorf("execution.max_stdin_mb must be positive")
	}
	if cfg.Execution.MaxOutputMB <= 0 {
		return nil, fmt.Errorf("execution.max_output_mb must be positive")
	}
	if cfg.GC.IntervalSeconds < 0 || cfg.GC.SessionIdleTimeoutSeconds < 0 || cfg.GC.GPUPoolIdleTimeoutSeconds < 0 {
		return nil, fmt.Errorf("gc timeouts must not be negative")
	}
	return cfg, nil
}

func resolvePath(baseDir, path string) string {
	if path == "" || filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(baseDir, path)
}

func validateCluster(name string, cfg ClusterConfig) error {
	if cfg.Kubeconfig == "" || cfg.Namespace == "" {
		return fmt.Errorf("%s.kubeconfig and namespace are required", name)
	}
	return nil
}
