package kubernetes

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"kernel-sandbox/pkg/command"
	"kernel-sandbox/pkg/config"
	"kernel-sandbox/pkg/master"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestGPUPoolSchedulingAcrossCreationUpdateAndDiscovery(t *testing.T) {
	seconds := int64(300)
	defaults := master.Scheduling{NodeSelector: map[string]string{"default": "true"}, Tolerations: []master.Toleration{{Key: "default", Operator: "Exists"}}}
	override := master.Scheduling{NodeSelector: map[string]string{"tenant": "test"}, Tolerations: []master.Toleration{{Key: "tenant", Operator: "Equal", Value: "test", Effect: "NoExecute", TolerationSeconds: &seconds}}}
	empty := master.Scheduling{NodeSelector: map[string]string{}, Tolerations: []master.Toleration{}}
	for _, test := range []struct {
		name  string
		input *master.Scheduling
		want  master.Scheduling
	}{{"omitted", nil, defaults}, {"null", &master.Scheduling{}, defaults}, {"replace", &override, override}, {"clear", &empty, empty}} {
		t.Run(test.name, func(t *testing.T) {
			clientset := fake.NewSimpleClientset()
			template := &corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "sandbox"}}}}
			applyScheduling(&template.Spec, &defaults)
			client := &Client{
				settings:                &config.Config{MasterID: "test"},
				cpuClusterClient:        &clusterClient{namespace: "test", clientset: clientset},
				gpuSessionClusterClient: &clusterClient{namespace: "test", clientset: clientset},
				gpuPoolClusterClient:    &clusterClient{namespace: "test", clientset: clientset},
				cpuPodTemplate:          template, gpuSessionPodTemplate: template,
				gpuDeploymentTemplate: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: template.Spec}}},
			}
			resources := &master.ResourceRequirements{Requests: map[string]string{"cpu": "2"}}
			if err := client.CreateGPUPool(context.Background(), "pool", "image", 1, resources, test.input); err != nil {
				t.Fatal(err)
			}
			pools, err := client.ListGPUPools(context.Background())
			if err != nil || len(pools) != 1 || !reflect.DeepEqual(pools[0].Scheduling, &test.want) || pools[0].Resources.Requests["cpu"] != "2" {
				t.Fatalf("pools=%+v error=%v", pools, err)
			}
			if err := client.UpdateGPUPool(context.Background(), "pool", "", nil, &master.Scheduling{Tolerations: []master.Toleration{}}); err != nil {
				t.Fatal(err)
			}
			pools, err = client.ListGPUPools(context.Background())
			if err != nil || !reflect.DeepEqual(pools[0].Scheduling.NodeSelector, test.want.NodeSelector) || len(pools[0].Scheduling.Tolerations) != 0 {
				t.Fatalf("partial scheduling update failed: pools=%+v error=%v", pools, err)
			}
			if err := client.UpdateGPUPool(context.Background(), "pool", "", nil, &empty); err != nil {
				t.Fatal(err)
			}
			pools, err = client.ListGPUPools(context.Background())
			if err != nil || !reflect.DeepEqual(pools[0].Scheduling, &empty) || !reflect.DeepEqual(podScheduling(&template.Spec), &defaults) {
				t.Fatalf("clear or template isolation failed: pools=%+v error=%v", pools, err)
			}
			cpu, err := client.CreateCPUPod(context.Background(), "session-1", "pool", "image", "url", nil, "custom")
			if err != nil || cpu.Name != "custom-session-1" || cpu.Prefix != "custom" {
				t.Fatalf("CPU Pod=%+v error=%v", cpu, err)
			}
			gpu, err := client.CreateGPUSessionPod(context.Background(), "gpu-session-1", "image", nil, "custom")
			if err != nil || gpu.Name != "custom-gpu-session-1" || gpu.Prefix != "custom" {
				t.Fatalf("GPU Pod=%+v error=%v", gpu, err)
			}
			for _, name := range []string{cpu.Name, gpu.Name} {
				pod, err := clientset.CoreV1().Pods("test").Get(context.Background(), name, metav1.GetOptions{})
				if err != nil || !reflect.DeepEqual(podScheduling(&pod.Spec), &defaults) {
					t.Fatalf("Session template scheduling changed: Pod=%+v error=%v", pod, err)
				}
			}
		})
	}
}

func TestLoadPodTemplateRejectsContainerCommand(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pod.yaml")
	if err := os.WriteFile(path, []byte(`apiVersion: v1
kind: Pod
spec:
  containers:
    - name: sandbox
      image: example
      command: ["sleep"]
`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadPodTemplate(path); err == nil || !strings.Contains(err.Error(), "command or args") {
		t.Fatalf("error=%v", err)
	}
}

func TestLoadPortableGPUDeploymentTemplate(t *testing.T) {
	d, err := loadDeploymentTemplate(filepath.Join("..", "..", "deploy", "templates", "gpu-deployment-template.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	c := d.Spec.Template.Spec.Containers[0]
	q := c.Resources.Limits[corev1.ResourceName("nvidia.com/gpu")]
	if q.String() != "1" {
		t.Fatal("expected one standard NVIDIA device")
	}
	if d.Spec.Template.Spec.SchedulerName != "" || len(d.Spec.Template.Spec.NodeSelector) != 0 {
		t.Fatal("portable template must not impose internal scheduling")
	}
}

func TestLoadCPUPodTemplateDisablesNVIDIADevices(t *testing.T) {
	pod, err := loadPodTemplate(filepath.Join("..", "..", "deploy", "templates", "cpu-pod-template.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, environment := range pod.Spec.Containers[0].Env {
		if environment.Name == "NVIDIA_VISIBLE_DEVICES" && environment.Value == "void" {
			return
		}
	}
	t.Fatalf("env=%+v", pod.Spec.Containers[0].Env)
}

func TestGPUResourcesRejectZeroQuantity(t *testing.T) {
	container := &corev1.Container{}
	if err := setGPUResourceRequests(container, &master.ResourceRequirements{Requests: map[string]string{"example.org/gpu_hopper": "0"}}); err == nil {
		t.Fatal("zero GPU request was accepted")
	}
}

func TestResultFromExitCodeIsCompleted(t *testing.T) {
	for _, exitCode := range []int{7, 126, 127} {
		result := resultFromExitCode("stdout", "stderr", exitCode)
		if result.Kind != command.ResultCompleted || result.ExitCode == nil || *result.ExitCode != exitCode || result.Stdout != "stdout" || result.Stderr != "stderr" {
			t.Fatalf("exit_code=%d result=%+v", exitCode, result)
		}
	}
}

func TestCreateCPUPodAppliesRuntimeFields(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	client := &Client{
		settings:         &config.Config{MasterID: "test", CPUCluster: config.CPUClusterConfig{PodNamePrefix: "custom-session"}},
		cpuClusterClient: &clusterClient{namespace: "cpu", clientset: clientset},
		cpuPodTemplate: &corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: "sandbox", Image: "placeholder", Env: []corev1.EnvVar{{Name: "NVIDIA_VISIBLE_DEVICES", Value: "void"}},
		}}}},
	}
	if _, err := client.CreateCPUPod(context.Background(), "session-1", "gpu-pool-1", "business-image", "http://master/v1/session/session-1", &master.ResourceRequirements{Requests: map[string]string{"cpu": "6", "memory": "24Gi"}}, ""); err != nil {
		t.Fatal(err)
	}
	pod, err := clientset.CoreV1().Pods("cpu").Get(context.Background(), "custom-session-session-1", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	container := pod.Spec.Containers[0]
	if container.Image != "business-image" || strings.Join(container.Command, " ") != "/bin/sh -c sleep infinity" {
		t.Fatalf("container=%+v", container)
	}
	if len(container.Env) != 2 || container.Env[0].Name != "NVIDIA_VISIBLE_DEVICES" || container.Env[0].Value != "void" || container.Env[1].Name != "KERNEL_SANDBOX_URL" || container.Env[1].Value != "http://master/v1/session/session-1" {
		t.Fatalf("env=%+v", container.Env)
	}
	if container.Resources.Requests.Cpu().String() != "6" || container.Resources.Requests.Memory().String() != "24Gi" {
		t.Fatalf("requests=%+v", container.Resources.Requests)
	}
	if pod.Labels[masterLabel] != "test" || pod.Labels[sessionLabel] != "session-1" || pod.Labels[gpuPoolLabel] != "gpu-pool-1" || pod.Labels[componentLabel] != componentCPUSession {
		t.Fatalf("labels=%v", pod.Labels)
	}
}

func TestCreateCPUPodMergesResourcesPreservingTemplate(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	client := &Client{
		settings:         &config.Config{MasterID: "test", CPUCluster: config.CPUClusterConfig{PodNamePrefix: "custom-session"}},
		cpuClusterClient: &clusterClient{namespace: "cpu", clientset: clientset},
		cpuPodTemplate: &corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: "sandbox", Image: "placeholder",
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1"), corev1.ResourceEphemeralStorage: resource.MustParse("5Gi")},
				Limits:   corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("8Gi")},
			},
		}}}},
	}
	resources := &master.ResourceRequirements{Requests: map[string]string{"cpu": "6", "memory": "24Gi"}, Limits: map[string]string{"cpu": "8"}}
	if _, err := client.CreateCPUPod(context.Background(), "session-1", "gpu-pool-1", "business-image", "http://master/v1/session/session-1", resources, ""); err != nil {
		t.Fatal(err)
	}
	pod, err := clientset.CoreV1().Pods("cpu").Get(context.Background(), "custom-session-session-1", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	container := pod.Spec.Containers[0]
	requests := container.Resources.Requests
	ephemeral := requests[corev1.ResourceEphemeralStorage]
	if requests.Cpu().String() != "6" || requests.Memory().String() != "24Gi" || ephemeral.String() != "5Gi" {
		t.Fatalf("requests=%+v", requests)
	}
	limits := container.Resources.Limits
	if limits.Cpu().String() != "8" || limits.Memory().String() != "8Gi" {
		t.Fatalf("limits=%+v", limits)
	}
}

func TestCreateGPUPoolAppliesDeploymentTemplate(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	client := &Client{
		settings:             &config.Config{MasterID: "test", Sandboxd: config.SandboxdConfig{Image: "sandboxd-image"}, Execution: config.ExecutionConfig{MaxStdinMB: 1, MaxOutputMB: 1}},
		gpuPoolClusterClient: &clusterClient{namespace: "gpu", clientset: clientset},
		gpuDeploymentTemplate: &appsv1.Deployment{
			Spec: appsv1.DeploymentSpec{
				Strategy: appsv1.DeploymentStrategy{Type: appsv1.RecreateDeploymentStrategyType},
				Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{SchedulerName: "gpu-scheduler", Containers: []corev1.Container{{Name: "sandbox"}}}},
			},
		},
	}
	gpuPoolID := "custom-pool-0000000000000001"
	if err := client.CreateGPUPool(context.Background(), gpuPoolID, "gpu-image", 3, nil, nil); err != nil {
		t.Fatal(err)
	}
	deployment, err := clientset.AppsV1().Deployments("gpu").Get(context.Background(), gpuPoolID, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if *deployment.Spec.Replicas != 3 || deployment.Spec.Strategy.Type != appsv1.RecreateDeploymentStrategyType || deployment.Spec.Template.Spec.SchedulerName != "gpu-scheduler" {
		t.Fatalf("deployment=%+v", deployment.Spec)
	}
	if deployment.Spec.Template.Spec.Containers[0].Image != "gpu-image" {
		t.Fatalf("container=%+v", deployment.Spec.Template.Spec.Containers[0])
	}
	if deployment.Labels[masterLabel] != "test" || deployment.Labels[gpuPoolLabel] != gpuPoolID || deployment.Labels[componentLabel] != componentGPUPool || deployment.Spec.Template.Labels[masterLabel] != "test" || deployment.Spec.Template.Labels[gpuPoolLabel] != gpuPoolID || deployment.Spec.Template.Labels[componentLabel] != componentGPUWorker {
		t.Fatalf("deployment labels=%v pod labels=%v", deployment.Labels, deployment.Spec.Template.Labels)
	}
}

func TestGPUWorkerDiscoveryCoversAllManagedPools(t *testing.T) {
	clientset := fake.NewSimpleClientset(
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pool-a-worker", Namespace: "gpu", Labels: map[string]string{masterLabel: "test", componentLabel: componentGPUWorker, gpuPoolLabel: "pool-a"}}},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pool-b-worker", Namespace: "gpu", Labels: map[string]string{masterLabel: "test", componentLabel: componentGPUWorker, gpuPoolLabel: "pool-b"}}},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "unmanaged", Namespace: "gpu"}},
	)
	client := &Client{
		settings:             &config.Config{MasterID: "test"},
		cpuClusterClient:     &clusterClient{namespace: "cpu", clientset: clientset},
		gpuPoolClusterClient: &clusterClient{namespace: "gpu", clientset: clientset},
	}
	pods, err := client.ListGPUPods(context.Background())
	if err != nil || len(pods) != 2 || pods[0].GPUPoolID == "" || pods[1].GPUPoolID == "" {
		t.Fatalf("pods=%+v err=%v", pods, err)
	}
}

func TestListGPUPools(t *testing.T) {
	replicas := int32(2)
	clientset := fake.NewSimpleClientset(
		&appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "pool-a", Namespace: "gpu", Labels: map[string]string{masterLabel: "test", componentLabel: componentGPUPool, gpuPoolLabel: "pool-a"}}, Spec: appsv1.DeploymentSpec{Replicas: &replicas, Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Image: "image-a"}}}}}},
		&appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "unmanaged", Namespace: "gpu"}, Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Image: "ignored"}}}}}},
	)
	client := &Client{settings: &config.Config{MasterID: "test"}, gpuPoolClusterClient: &clusterClient{namespace: "gpu", clientset: clientset}}
	pools, err := client.ListGPUPools(context.Background())
	if err != nil || len(pools) != 1 || pools[0].GPUPoolID != "pool-a" || pools[0].Image != "image-a" || pools[0].Replicas != 2 {
		t.Fatalf("pools=%+v err=%v", pools, err)
	}
}

func TestComponentLabelIsolatesManagedResources(t *testing.T) {
	replicas := int32(1)
	clientset := fake.NewSimpleClientset(
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new-cpu", Namespace: "cpu", Labels: map[string]string{masterLabel: "test", componentLabel: componentCPUSession}}},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "legacy-cpu", Namespace: "cpu", Labels: map[string]string{"kernel-sandbox/master-id": "prod", "kernel-sandbox/resource": componentCPUSession}}},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new-gpu", Namespace: "gpu", Labels: map[string]string{masterLabel: "test", componentLabel: componentGPUWorker, gpuPoolLabel: "new-pool"}}},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "legacy-gpu", Namespace: "gpu", Labels: map[string]string{"kernel-sandbox/master-id": "prod", "kernel-sandbox/resource": componentGPUWorker, gpuPoolLabel: "legacy-pool"}}},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new-gpu-session", Namespace: "gpu", Labels: map[string]string{masterLabel: "test", componentLabel: componentGPUSession, sessionLabel: "session-test"}}},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "prod-gpu-session", Namespace: "gpu", Labels: map[string]string{masterLabel: "prod", componentLabel: componentGPUSession, sessionLabel: "session-prod"}}},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "unowned-gpu-session", Namespace: "gpu", Labels: map[string]string{componentLabel: componentGPUSession, sessionLabel: "session-old"}}},
		&appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "new-pool", Namespace: "gpu", Labels: map[string]string{masterLabel: "test", componentLabel: componentGPUPool}}, Spec: appsv1.DeploymentSpec{Replicas: &replicas, Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Image: "new-image"}}}}}},
		&appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "legacy-pool", Namespace: "gpu", Labels: map[string]string{"kernel-sandbox/master-id": "prod", "kernel-sandbox/resource": componentGPUPool}}, Spec: appsv1.DeploymentSpec{Replicas: &replicas, Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Image: "legacy-image"}}}}}},
	)
	client := &Client{
		settings:                &config.Config{MasterID: "test"},
		cpuClusterClient:        &clusterClient{namespace: "cpu", clientset: clientset},
		gpuPoolClusterClient:    &clusterClient{namespace: "gpu", clientset: clientset},
		gpuSessionClusterClient: &clusterClient{namespace: "gpu", clientset: clientset},
	}
	if pods, err := client.ListCPUPods(context.Background()); err != nil || len(pods) != 1 || pods[0].Name != "new-cpu" {
		t.Fatalf("CPU pods=%+v error=%v", pods, err)
	}
	if pods, err := client.ListGPUPods(context.Background()); err != nil || len(pods) != 1 || pods[0].Name != "new-gpu" {
		t.Fatalf("GPU pods=%+v error=%v", pods, err)
	}
	if pools, err := client.ListGPUPools(context.Background()); err != nil || len(pools) != 1 || pools[0].GPUPoolID != "new-pool" {
		t.Fatalf("GPU pools=%+v error=%v", pools, err)
	}
	if pods, err := client.ListGPUSessionPods(context.Background()); err != nil || len(pods) != 1 || pods[0].Name != "new-gpu-session" {
		t.Fatalf("GPU Session pods=%+v error=%v", pods, err)
	}
}

func TestConfigureSandboxd(t *testing.T) {
	client := &Client{settings: &config.Config{
		Sandboxd: config.SandboxdConfig{
			Image:               "sandboxd-image",
			GPUProcessAllowlist: []config.GPUProcessAllowlistEntry{{Process: "multiprocessing.spawn", Ancestor: "kerneleval.server"}},
		},
		Execution: config.ExecutionConfig{MaxStdinMB: 1, MaxOutputMB: 2},
	}}
	spec := &corev1.PodSpec{Containers: []corev1.Container{{Name: "sandbox"}}}
	if err := client.configureSandboxd(spec, &spec.Containers[0]); err != nil {
		t.Fatal(err)
	}
	container := spec.Containers[0]
	if len(spec.InitContainers) != 1 || spec.InitContainers[0].Image != "sandboxd-image" {
		t.Fatalf("init containers=%+v", spec.InitContainers)
	}
	if len(container.Command) != 1 || container.Command[0] != sandboxdBinary || !contains(container.Args, "--max-stdin-bytes") {
		t.Fatalf("command=%v args=%v", container.Command, container.Args)
	}
	if args := strings.Join(container.Args, " "); !strings.Contains(args, "--gpu-process-allowlist") || !strings.Contains(args, `"ancestor":"kerneleval.server"`) {
		t.Fatalf("args=%v", container.Args)
	}
	if container.ReadinessProbe == nil || container.ReadinessProbe.Exec == nil {
		t.Fatal("sandboxd readiness probe was not configured")
	}
}

func TestCreateGPUSessionPodAppliesResourcesAndLabels(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	client := &Client{
		settings:                &config.Config{MasterID: "test", GPUPoolCluster: config.GPUPoolClusterConfig{PoolNamePrefix: "gpu-pool"}},
		gpuSessionClusterClient: &clusterClient{namespace: "gpu", clientset: clientset},
		gpuSessionPodNamePrefix: "koda-sandbox-test",
		gpuSessionPodTemplate: &corev1.Pod{Spec: corev1.PodSpec{SchedulerName: "gpu-scheduler", Containers: []corev1.Container{{Name: "sandbox", Image: "placeholder", Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{corev1.ResourceName("rdma/hca"): resource.MustParse("1")},
			Limits:   corev1.ResourceList{corev1.ResourceName("rdma/hca"): resource.MustParse("1")},
		}}}}},
	}
	resources := &master.ResourceRequirements{Requests: map[string]string{"cpu": "4", "memory": "16Gi", "example.org/gpu_hopper": "1"}, Limits: map[string]string{"example.org/gpu_hopper": "1"}}
	if _, err := client.CreateGPUSessionPod(context.Background(), "session-1", "gpu-image", resources, ""); err != nil {
		t.Fatal(err)
	}
	pod, err := clientset.CoreV1().Pods("gpu").Get(context.Background(), "koda-sandbox-test-session-1", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if pod.Spec.SchedulerName != "gpu-scheduler" || pod.Spec.RestartPolicy != corev1.RestartPolicyNever {
		t.Fatalf("spec=%+v", pod.Spec)
	}
	container := pod.Spec.Containers[0]
	if container.Image != "gpu-image" || strings.Join(container.Command, " ") != "/bin/sh -c sleep infinity" {
		t.Fatalf("container=%+v", container)
	}
	if container.Resources.Requests.Cpu().String() != "4" || container.Resources.Requests.Memory().String() != "16Gi" {
		t.Fatalf("requests=%+v", container.Resources.Requests)
	}
	gpuName := corev1.ResourceName("example.org/gpu_hopper")
	if request := container.Resources.Requests[gpuName]; request.String() != "1" {
		t.Fatalf("gpu request=%v", request.String())
	}
	if limit := container.Resources.Limits[gpuName]; limit.String() != "1" {
		t.Fatalf("gpu limit=%v", limit.String())
	}
	if request := container.Resources.Requests[corev1.ResourceName("rdma/hca")]; request.String() != "1" {
		t.Fatalf("rdma request=%v", request.String())
	}
	if pod.Labels[masterLabel] != "test" || pod.Labels[sessionLabel] != "session-1" || pod.Labels[componentLabel] != componentGPUSession {
		t.Fatalf("labels=%v", pod.Labels)
	}
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
