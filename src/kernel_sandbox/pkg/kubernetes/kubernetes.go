package kubernetes

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"strings"

	"kernel-sandbox/pkg/config"
	"kernel-sandbox/pkg/master"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/yaml"
)

const (
	masterLabel         = "koda.local/master-id"
	componentLabel      = "koda.local/component"
	sessionLabel        = "koda.local/session-id"
	gpuPoolLabel        = "koda.local/pool-id"
	componentCPUSession = "cpu-session"
	componentGPUPool    = "gpu-pool"
	componentGPUWorker  = "gpu-worker"
	componentGPUSession = "gpu-session"
	sandboxdBinary      = "/opt/kernel-sandbox/sandboxd"
)

var _ master.KubernetesClient = (*Client)(nil)

type Client struct {
	settings                *config.Config
	cpuClusterClient        *clusterClient
	gpuPoolClusterClient    *clusterClient
	gpuSessionClusterClient *clusterClient
	cpuPodTemplate          *corev1.Pod
	gpuDeploymentTemplate   *appsv1.Deployment
	gpuSessionPodTemplate   *corev1.Pod
	gpuSessionPodNamePrefix string
}

type clusterClient struct {
	namespace  string
	restConfig *rest.Config
	clientset  kubernetes.Interface
}

func New(settings *config.Config) (*Client, error) {
	cpuClusterClient, err := newClusterClient(settings.CPUCluster.ClusterConfig)
	if err != nil {
		return nil, fmt.Errorf("CPU Kubernetes client: %w", err)
	}
	gpuPoolClusterClient, err := newClusterClient(settings.GPUPoolCluster.ClusterConfig)
	if err != nil {
		return nil, fmt.Errorf("GPU Kubernetes client: %w", err)
	}
	cpuPodTemplate, err := loadPodTemplate(settings.CPUCluster.PodTemplateFile)
	if err != nil {
		return nil, fmt.Errorf("CPU PodTemplate: %w", err)
	}
	gpuDeploymentTemplate, err := loadDeploymentTemplate(settings.GPUPoolCluster.DeploymentTemplateFile)
	if err != nil {
		return nil, fmt.Errorf("GPU DeploymentTemplate: %w", err)
	}
	return &Client{settings: settings, cpuClusterClient: cpuClusterClient, gpuPoolClusterClient: gpuPoolClusterClient, cpuPodTemplate: cpuPodTemplate, gpuDeploymentTemplate: gpuDeploymentTemplate}, nil
}

// NewGPUSessionClient builds a Client for gpu_only Session Pods on one GPU
// Session cluster.
func NewGPUSessionClient(settings *config.Config, cluster config.GPUSessionClusterConfig) (*Client, error) {
	gpuSessionClusterClient, err := newClusterClient(cluster.ClusterConfig)
	if err != nil {
		return nil, fmt.Errorf("GPU Kubernetes client: %w", err)
	}
	gpuSessionPodTemplate, err := loadPodTemplate(cluster.PodTemplateFile)
	if err != nil {
		return nil, fmt.Errorf("GPU Session PodTemplate: %w", err)
	}
	return &Client{
		settings:                settings,
		gpuSessionClusterClient: gpuSessionClusterClient,
		gpuSessionPodTemplate:   gpuSessionPodTemplate,
		gpuSessionPodNamePrefix: cluster.PodNamePrefix,
	}, nil
}

// NewGPUPoolClient builds a Client for GPU Pool Deployments on one GPU Pool cluster.
func NewGPUPoolClient(settings *config.Config, cluster config.GPUPoolClusterConfig) (*Client, error) {
	gpuPoolClusterClient, err := newClusterClient(cluster.ClusterConfig)
	if err != nil {
		return nil, fmt.Errorf("GPU Pool Kubernetes client: %w", err)
	}
	gpuDeploymentTemplate, err := loadDeploymentTemplate(cluster.DeploymentTemplateFile)
	if err != nil {
		return nil, fmt.Errorf("GPU Pool DeploymentTemplate: %w", err)
	}
	return &Client{
		settings:              settings,
		gpuPoolClusterClient:  gpuPoolClusterClient,
		gpuDeploymentTemplate: gpuDeploymentTemplate,
	}, nil
}

func newClusterClient(clusterConfig config.ClusterConfig) (*clusterClient, error) {
	restConfig, err := clientcmd.BuildConfigFromFlags("", clusterConfig.Kubeconfig)
	if err != nil {
		return nil, err
	}
	restConfig.QPS = 1024
	restConfig.Burst = 1024
	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, err
	}
	return &clusterClient{namespace: clusterConfig.Namespace, restConfig: restConfig, clientset: clientset}, nil
}

func loadPodTemplate(path string) (*corev1.Pod, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	pod := &corev1.Pod{}
	if err := yaml.Unmarshal(data, pod); err != nil {
		return nil, err
	}
	if pod.Kind != "" && pod.Kind != "Pod" {
		return nil, fmt.Errorf("template kind must be Pod")
	}
	if len(pod.Spec.Containers) != 1 {
		return nil, fmt.Errorf("PodTemplate must contain exactly one container")
	}
	container := &pod.Spec.Containers[0]
	if len(container.Command) != 0 || len(container.Args) != 0 {
		return nil, fmt.Errorf("PodTemplate container must not set command or args")
	}
	return pod, nil
}

func loadDeploymentTemplate(path string) (*appsv1.Deployment, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	deployment := &appsv1.Deployment{}
	if err := yaml.Unmarshal(data, deployment); err != nil {
		return nil, err
	}
	if deployment.Kind != "" && deployment.Kind != "Deployment" {
		return nil, fmt.Errorf("template kind must be Deployment")
	}
	if len(deployment.Spec.Template.Spec.Containers) != 1 {
		return nil, fmt.Errorf("DeploymentTemplate must contain exactly one container")
	}
	container := &deployment.Spec.Template.Spec.Containers[0]
	if len(container.Command) != 0 || len(container.Args) != 0 {
		return nil, fmt.Errorf("DeploymentTemplate container must not set command or args")
	}
	return deployment, nil
}

func (k *Client) CreateCPUPod(ctx context.Context, sessionID, gpuPoolID, image, sandboxURL string, resources *master.ResourceRequirements, prefix string) (master.PodInfo, error) {
	if prefix == "" {
		prefix = k.settings.CPUCluster.PodNamePrefix
	}
	pod := k.cpuPodTemplate.DeepCopy()
	pod.TypeMeta = metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"}
	pod.Name = prefix + "-" + sessionID
	pod.GenerateName = ""
	pod.Namespace = k.cpuClusterClient.namespace
	labels := k.componentLabels(componentCPUSession)
	labels[sessionLabel], labels[gpuPoolLabel] = sessionID, gpuPoolID
	pod.Labels = mergeLabels(pod.Labels, labels)
	pod.Spec.RestartPolicy = corev1.RestartPolicyNever
	container := &pod.Spec.Containers[0]
	container.Image = image
	container.Command = []string{"/bin/sh", "-c", "sleep infinity"}
	container.Args = nil
	container.Env = append(container.Env, corev1.EnvVar{Name: "KERNEL_SANDBOX_URL", Value: sandboxURL})
	if err := setCPUResourceRequests(container, resources); err != nil {
		return master.PodInfo{}, err
	}
	created, err := k.cpuClusterClient.clientset.CoreV1().Pods(k.cpuClusterClient.namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		return master.PodInfo{}, err
	}
	return podInfo(created), nil
}

func (k *Client) DeleteCPUPod(ctx context.Context, pod master.PodInfo) error {
	return deletePod(ctx, k.cpuClusterClient, pod.Name)
}

func (k *Client) ListCPUPods(ctx context.Context) ([]master.PodInfo, error) {
	return listPods(ctx, k.cpuClusterClient, k.componentSelector(componentCPUSession))
}

func (k *Client) CreateGPUPool(ctx context.Context, gpuPoolID, image string, replicas int32, resources *master.ResourceRequirements, scheduling *master.Scheduling) error {
	deployment := k.gpuDeploymentTemplate.DeepCopy()
	workerLabels := k.componentLabels(componentGPUWorker)
	workerLabels[gpuPoolLabel] = gpuPoolID
	deployment.TypeMeta = metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"}
	deployment.Name = gpuPoolID
	deployment.GenerateName = ""
	deployment.Namespace = k.gpuPoolClusterClient.namespace
	poolLabels := k.componentLabels(componentGPUPool)
	poolLabels[gpuPoolLabel] = gpuPoolID
	deployment.Labels = mergeLabels(deployment.Labels, poolLabels)
	deployment.Spec.Replicas = &replicas
	deployment.Spec.Selector = &metav1.LabelSelector{MatchLabels: workerLabels}
	deployment.Spec.Template.Labels = mergeLabels(deployment.Spec.Template.Labels, workerLabels)
	deployment.Spec.Template.Spec.RestartPolicy = corev1.RestartPolicyAlways
	applyScheduling(&deployment.Spec.Template.Spec, scheduling)
	container := &deployment.Spec.Template.Spec.Containers[0]
	container.Image = image
	if resources != nil {
		if err := setGPUResourceRequests(container, resources); err != nil {
			return err
		}
	}
	if err := k.configureSandboxd(&deployment.Spec.Template.Spec, container); err != nil {
		return err
	}
	_, err := k.gpuPoolClusterClient.clientset.AppsV1().Deployments(k.gpuPoolClusterClient.namespace).Create(ctx, deployment, metav1.CreateOptions{})
	return err
}

func (k *Client) ListGPUPools(ctx context.Context) ([]master.GPUPoolInfo, error) {
	deployments, err := k.gpuPoolClusterClient.clientset.AppsV1().Deployments(k.gpuPoolClusterClient.namespace).List(ctx, metav1.ListOptions{LabelSelector: k.componentSelector(componentGPUPool)})
	if err != nil {
		return nil, err
	}
	result := make([]master.GPUPoolInfo, 0, len(deployments.Items))
	for index := range deployments.Items {
		deployment := &deployments.Items[index]
		if deployment.DeletionTimestamp != nil {
			continue
		}
		replicas := int32(1)
		if deployment.Spec.Replicas != nil {
			replicas = *deployment.Spec.Replicas
		}
		container := deployment.Spec.Template.Spec.Containers[0]
		result = append(result, master.GPUPoolInfo{GPUPoolID: deployment.Name, Image: container.Image, Replicas: replicas,
			Resources:  &master.ResourceRequirements{Requests: resourceListToMap(container.Resources.Requests), Limits: resourceListToMap(container.Resources.Limits)},
			Scheduling: podScheduling(&deployment.Spec.Template.Spec)})
	}
	return result, nil
}

func (k *Client) UpdateGPUPool(ctx context.Context, gpuPoolID, image string, replicas *int32, scheduling *master.Scheduling) error {
	deployments := k.gpuPoolClusterClient.clientset.AppsV1().Deployments(k.gpuPoolClusterClient.namespace)
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		deployment, err := deployments.Get(ctx, gpuPoolID, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if image != "" {
			deployment.Spec.Template.Spec.Containers[0].Image = image
		}
		if replicas != nil {
			deployment.Spec.Replicas = replicas
		}
		applyScheduling(&deployment.Spec.Template.Spec, scheduling)
		_, err = deployments.Update(ctx, deployment, metav1.UpdateOptions{})
		return err
	})
}

func (k *Client) configureSandboxd(spec *corev1.PodSpec, container *corev1.Container) error {
	const volumeName = "kernel-sandbox-runtime"
	spec.Volumes = append(spec.Volumes, corev1.Volume{Name: volumeName, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}})
	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{Name: volumeName, MountPath: "/opt/kernel-sandbox"})
	spec.InitContainers = append(spec.InitContainers, corev1.Container{
		Name: "kernel-sandbox-injector", Image: k.settings.Sandboxd.Image,
		Command:      []string{"sh", "-c", "cp /usr/local/bin/sandboxd /runtime/sandboxd && chmod 0755 /runtime/sandboxd"},
		VolumeMounts: []corev1.VolumeMount{{Name: volumeName, MountPath: "/runtime"}},
	})
	container.Command = []string{sandboxdBinary}
	container.Args = []string{
		"serve",
		"--socket", config.SandboxdSocketPath,
		"--terminate-grace-seconds", fmt.Sprint(config.SandboxdTerminateGraceSeconds),
		"--max-stdin-bytes", fmt.Sprint(k.settings.Execution.MaxStdinMB << 20),
		"--max-output-bytes", fmt.Sprint(k.settings.Execution.MaxOutputMB << 20),
	}
	if len(k.settings.Sandboxd.GPUProcessAllowlist) != 0 {
		allowlist, err := json.Marshal(k.settings.Sandboxd.GPUProcessAllowlist)
		if err != nil {
			return err
		}
		container.Args = append(container.Args, "--gpu-process-allowlist", string(allowlist))
	}
	container.ReadinessProbe = &corev1.Probe{
		ProbeHandler:        corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{sandboxdBinary, "health", "--socket", config.SandboxdSocketPath}}},
		InitialDelaySeconds: 1, PeriodSeconds: 2, TimeoutSeconds: 2, FailureThreshold: 5, SuccessThreshold: 1,
	}
	return nil
}

func (k *Client) DeleteGPUPool(ctx context.Context, gpuPoolID string) error {
	policy := metav1.DeletePropagationForeground
	err := k.gpuPoolClusterClient.clientset.AppsV1().Deployments(k.gpuPoolClusterClient.namespace).Delete(ctx, gpuPoolID, metav1.DeleteOptions{PropagationPolicy: &policy})
	if apierrors.IsNotFound(err) {
		return nil
	}
	return err
}

func (k *Client) ListGPUPods(ctx context.Context) ([]master.PodInfo, error) {
	return listPods(ctx, k.gpuPoolClusterClient, k.componentSelector(componentGPUWorker))
}

func (k *Client) DeleteGPUWorkerPod(ctx context.Context, pod master.PodInfo) error {
	return deletePod(ctx, k.gpuPoolClusterClient, pod.Name)
}

func (k *Client) DeleteGPUSessionPod(ctx context.Context, pod master.PodInfo) error {
	return deletePod(ctx, k.gpuSessionClusterClient, pod.Name)
}

// CreateGPUSessionPod launches a persistent GPU Session Pod from the selected GPU
// Session cluster's Pod template.
func (k *Client) CreateGPUSessionPod(ctx context.Context, sessionID, image string, resources *master.ResourceRequirements, prefix string) (master.PodInfo, error) {
	if prefix == "" {
		prefix = k.gpuSessionPodNamePrefix
	}
	pod := k.gpuSessionPodTemplate.DeepCopy()
	pod.TypeMeta = metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"}
	pod.Name = prefix + "-" + sessionID
	pod.GenerateName = ""
	pod.Namespace = k.gpuSessionClusterClient.namespace
	labels := k.componentLabels(componentGPUSession)
	labels[sessionLabel] = sessionID
	pod.Labels = mergeLabels(pod.Labels, labels)
	pod.Spec.RestartPolicy = corev1.RestartPolicyNever
	container := &pod.Spec.Containers[0]
	container.Image = image
	container.Command = []string{"/bin/sh", "-c", "sleep infinity"}
	container.Args = nil
	if err := setGPUResourceRequests(container, resources); err != nil {
		return master.PodInfo{}, err
	}
	created, err := k.gpuSessionClusterClient.clientset.CoreV1().Pods(k.gpuSessionClusterClient.namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		return master.PodInfo{}, err
	}
	return podInfo(created), nil
}

func (k *Client) ListGPUSessionPods(ctx context.Context) ([]master.PodInfo, error) {
	return listPods(ctx, k.gpuSessionClusterClient, k.componentSelector(componentGPUSession))
}

func (k *Client) componentLabels(component string) map[string]string {
	return map[string]string{masterLabel: k.settings.MasterID, componentLabel: component}
}

func (k *Client) componentSelector(component string) string {
	return masterLabel + "=" + k.settings.MasterID + "," + componentLabel + "=" + component
}

func setCPUResourceRequests(container *corev1.Container, request *master.ResourceRequirements) error {
	if request == nil {
		return nil
	}
	if container.Resources.Requests == nil {
		container.Resources.Requests = corev1.ResourceList{}
	}
	for name, value := range map[corev1.ResourceName]string{corev1.ResourceCPU: request.Requests["cpu"], corev1.ResourceMemory: request.Requests["memory"]} {
		if value == "" {
			continue
		}
		quantity, err := resource.ParseQuantity(value)
		if err != nil || quantity.Sign() <= 0 {
			return fmt.Errorf("invalid %s request %q", name, value)
		}
		container.Resources.Requests[name] = quantity
	}

	if container.Resources.Limits != nil {
		for name, value := range map[corev1.ResourceName]string{corev1.ResourceCPU: request.Limits["cpu"], corev1.ResourceMemory: request.Limits["memory"]} {
			if value == "" {
				continue
			}
			quantity, err := resource.ParseQuantity(value)
			if err != nil || quantity.Sign() <= 0 {
				return fmt.Errorf("invalid %s request %q", name, value)
			}
			container.Resources.Limits[name] = quantity
		}
	}
	return nil
}
func setGPUResourceRequests(container *corev1.Container, request *master.ResourceRequirements) error {
	if request == nil {
		return nil
	}
	if err := applyResourceList(&container.Resources.Requests, request.Requests); err != nil {
		return err
	}
	return applyResourceList(&container.Resources.Limits, request.Limits)
}

// applyResourceList merges the given name/quantity values into the resource
// list, overriding any template defaults for the same resource name.
func applyResourceList(list *corev1.ResourceList, values map[string]string) error {
	for name, value := range values {
		if value == "" {
			continue
		}
		quantity, err := resource.ParseQuantity(value)
		if err != nil || quantity.Sign() <= 0 {
			return fmt.Errorf("invalid %s resource %q", name, value)
		}
		if *list == nil {
			*list = corev1.ResourceList{}
		}
		(*list)[corev1.ResourceName(name)] = quantity
	}
	return nil
}

func mergeLabels(current, extra map[string]string) map[string]string {
	result := map[string]string{}
	for key, value := range current {
		result[key] = value
	}
	for key, value := range extra {
		result[key] = value
	}
	return result
}

func applyScheduling(spec *corev1.PodSpec, scheduling *master.Scheduling) {
	if scheduling == nil {
		return
	}
	if scheduling.NodeSelector != nil {
		spec.NodeSelector = maps.Clone(scheduling.NodeSelector)
	}
	if scheduling.Tolerations != nil {
		spec.Tolerations = make([]corev1.Toleration, 0, len(scheduling.Tolerations))
		for _, t := range scheduling.Tolerations {
			spec.Tolerations = append(spec.Tolerations, corev1.Toleration{Key: t.Key, Operator: corev1.TolerationOperator(t.Operator), Value: t.Value, Effect: corev1.TaintEffect(t.Effect), TolerationSeconds: t.TolerationSeconds})
		}
	}
}

func podScheduling(spec *corev1.PodSpec) *master.Scheduling {
	scheduling := &master.Scheduling{NodeSelector: map[string]string{}, Tolerations: []master.Toleration{}}
	maps.Copy(scheduling.NodeSelector, spec.NodeSelector)
	for _, t := range spec.Tolerations {
		scheduling.Tolerations = append(scheduling.Tolerations, master.Toleration{Key: t.Key, Operator: string(t.Operator), Value: t.Value, Effect: string(t.Effect), TolerationSeconds: t.TolerationSeconds})
	}
	return scheduling
}

func podInfo(pod *corev1.Pod) master.PodInfo {
	ready := false
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
			ready = true
			break
		}
	}
	restartCount := int32(0)
	if len(pod.Status.ContainerStatuses) > 0 {
		restartCount = pod.Status.ContainerStatuses[0].RestartCount
	}
	result := master.PodInfo{Name: pod.Name, UID: string(pod.UID), SessionID: pod.Labels[sessionLabel], GPUPoolID: pod.Labels[gpuPoolLabel], CreatedAt: pod.CreationTimestamp.Time, Ready: ready, Failed: pod.Status.Phase == corev1.PodFailed, Deleting: pod.DeletionTimestamp != nil, RestartCount: restartCount}
	if result.SessionID != "" {
		result.Prefix = strings.TrimSuffix(pod.Name, "-"+result.SessionID)
	}
	if len(pod.Spec.Containers) > 0 {
		container := pod.Spec.Containers[0]
		result.Image = container.Image
		if len(container.Resources.Requests) > 0 || len(container.Resources.Limits) > 0 {
			result.Resources = &master.ResourceRequirements{Requests: resourceListToMap(container.Resources.Requests), Limits: resourceListToMap(container.Resources.Limits)}
		}
	}
	return result
}

func resourceListToMap(list corev1.ResourceList) map[string]string {
	if len(list) == 0 {
		return nil
	}
	result := make(map[string]string, len(list))
	for name, quantity := range list {
		result[string(name)] = quantity.String()
	}
	return result
}

func listPods(ctx context.Context, cluster *clusterClient, selector string) ([]master.PodInfo, error) {
	items, err := cluster.clientset.CoreV1().Pods(cluster.namespace).List(ctx, metav1.ListOptions{LabelSelector: selector})
	if err != nil {
		return nil, err
	}
	result := make([]master.PodInfo, 0, len(items.Items))
	for index := range items.Items {
		result = append(result, podInfo(&items.Items[index]))
	}
	return result, nil
}

func deletePod(ctx context.Context, cluster *clusterClient, name string) error {
	grace := int64(0)
	err := cluster.clientset.CoreV1().Pods(cluster.namespace).Delete(ctx, name, metav1.DeleteOptions{GracePeriodSeconds: &grace})
	if apierrors.IsNotFound(err) {
		return nil
	}
	return err
}
