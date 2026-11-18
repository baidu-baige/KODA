package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDockerConfigWithoutKubernetes(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "sandboxd")
	if e := os.WriteFile(bin, []byte("test"), 0700); e != nil {
		t.Fatal(e)
	}
	base := `backend: docker
master_id: docker-test
server:
  advertised_endpoint: http://127.0.0.1:18999
docker:
  network: host
  state_dir: ./state
  sandboxd_binary: ` + bin + `
  gpu_devices: [GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa]
execution:
  max_attempts: 2
  max_stdin_mb: 4
  max_output_mb: 1
`
	c, e := Load(writeConfig(t, base))
	if e != nil {
		t.Fatal(e)
	}
	if c.Backend != "docker" || c.Docker.Binary != "docker" || !filepath.IsAbs(c.Docker.StateDir) {
		t.Fatal("wrong defaults")
	}
	for name, text := range map[string]string{"gpu_index": strings.ReplaceAll(base, "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "0"), "loopback_bridge": strings.ReplaceAll(base, "network: host", "network: bridge"), "invalid_backend": strings.ReplaceAll(base, "backend: docker", "backend: local")} {
		t.Run(name, func(t *testing.T) {
			if _, e := Load(writeConfig(t, text)); e == nil {
				t.Fatal("accepted invalid config")
			}
		})
	}
}
