package master

import (
	"time"

	"kernel-sandbox/pkg/command"
)

type ErrorKind string

const (
	ErrorInvalid     ErrorKind = "invalid"
	ErrorNotFound    ErrorKind = "not_found"
	ErrorConflict    ErrorKind = "conflict"
	ErrorInternal    ErrorKind = "internal"
	ErrorUnavailable ErrorKind = "unavailable"
	ErrorTimeout     ErrorKind = "timeout"
)

type ServiceError struct {
	Kind        ErrorKind
	Code        string
	Message     string
	ExecutionID string
	Extra       *ResponseExtra
}

func (e *ServiceError) Error() string { return e.Message }

type SessionType string

const (
	SessionTypeGPUOnly SessionType = "gpu_only"
	SessionTypeCPUGPU  SessionType = "cpu_gpu"
)

type SessionStatus string

const (
	SessionStatusCreating SessionStatus = "creating"
	SessionStatusReady    SessionStatus = "ready"
	SessionStatusFailed   SessionStatus = "failed"
	SessionStatusDeleting SessionStatus = "deleting"
)

type ExecutionStatus string

const (
	ExecutionStatusQueued    ExecutionStatus = "queued"
	ExecutionStatusRunning   ExecutionStatus = "running"
	ExecutionStatusCompleted ExecutionStatus = "completed"
	ExecutionStatusFailed    ExecutionStatus = "failed"
	ExecutionStatusTimedOut  ExecutionStatus = "timed_out"
)

type GPUWorkerStatus string

const (
	GPUWorkerPending  GPUWorkerStatus = "pending"
	GPUWorkerIdle     GPUWorkerStatus = "idle"
	GPUWorkerBusy     GPUWorkerStatus = "busy"
	GPUWorkerDeleting GPUWorkerStatus = "deleting"
)

type CreateGPUPoolRequest struct {
	Image     string                `json:"image"`
	Replicas  int32                 `json:"replicas"`
	Resources *ResourceRequirements `json:"resources,omitempty"`
	Scheduling
}

type UpdateGPUPoolRequest struct {
	Image    string `json:"image,omitempty"`
	Replicas *int32 `json:"replicas,omitempty"`
	Scheduling
}

type GPUPoolResponse struct {
	GPUPoolID string                  `json:"gpu_pool_id"`
	Cluster   string                  `json:"cluster,omitempty"`
	Image     string                  `json:"image"`
	Replicas  int32                   `json:"replicas"`
	Workers   GPUWorkerCounts         `json:"workers"`
	Pods      []GPUWorkerResponse     `json:"pods"`
	Timing    *ResourceTimingResponse `json:"timing,omitempty"`
	Usage     *GPUPoolUsageResponse   `json:"usage,omitempty"`
}

type GPUPoolUsageResponse struct {
	BusyMS      int64                             `json:"busy_ms"`
	CapacityMS  int64                             `json:"capacity_ms"`
	Workers     map[string]GPUWorkerUsageResponse `json:"workers"`
	Queue       GPUPoolQueueUsageResponse         `json:"queue"`
	QueueLength int                               `json:"queue_length"`
}

type GPUPoolQueueUsageResponse struct {
	Count   int64 `json:"count"`
	TotalMS int64 `json:"total_ms"`
	MaxMS   int64 `json:"max_ms"`
}

type GPUWorkerUsageResponse struct {
	BusyMS     int64 `json:"busy_ms"`
	CapacityMS int64 `json:"capacity_ms"`
}

type GPUWorkerCounts struct {
	Pending int `json:"pending"`
	Idle    int `json:"idle"`
	Busy    int `json:"busy"`
}

type GPUWorkerResponse struct {
	Pod          string          `json:"pod"`
	Status       GPUWorkerStatus `json:"status"`
	Ready        bool            `json:"ready"`
	RestartCount int32           `json:"restart_count"`
}

type CreateSessionRequest struct {
	SessionType SessionType           `json:"session_type"`
	GPUPoolID   string                `json:"gpu_pool_id,omitempty"`
	Image       string                `json:"image,omitempty"`
	Resources   *ResourceRequirements `json:"resources,omitempty"`
	Prefix      string                `json:"prefix,omitempty"`
}

// Nil preserves the template/current value; an empty map or slice clears it.
type Scheduling struct {
	NodeSelector map[string]string `json:"node_selector"`
	Tolerations  []Toleration      `json:"tolerations"`
}

type Toleration struct {
	Key               string `json:"key,omitempty"`
	Operator          string `json:"operator,omitempty"`
	Value             string `json:"value,omitempty"`
	Effect            string `json:"effect,omitempty"`
	TolerationSeconds *int64 `json:"toleration_seconds,omitempty"`
}

// ResourceRequirements describes a session Pod's resource requirements as raw
// Kubernetes requests and limits maps, each keyed by the full resource name
// (e.g. "cpu", "memory", "example.org/gpu_count_best_effort"). Limits may be
// empty or omitted.
type ResourceRequirements struct {
	Requests map[string]string `json:"requests,omitempty"`
	Limits   map[string]string `json:"limits,omitempty"`
}

type SessionResponse struct {
	SessionID   string                  `json:"session_id"`
	GPUPoolID   string                  `json:"gpu_pool_id,omitempty"`
	SessionType SessionType             `json:"session_type"`
	Status      SessionStatus           `json:"status"`
	CreatedAt   time.Time               `json:"created_at"`
	CPU         *CPUResponse            `json:"cpu,omitempty"`
	GPU         *GPUSessionResponse     `json:"gpu,omitempty"`
	Error       string                  `json:"error,omitempty"`
	Timing      *ResourceTimingResponse `json:"timing,omitempty"`
}

type CPUResponse struct {
	Image     string                `json:"image"`
	Pod       string                `json:"pod,omitempty"`
	Ready     bool                  `json:"ready"`
	Resources *ResourceRequirements `json:"resources,omitempty"`
}

type GPUSessionResponse struct {
	Image     string                `json:"image"`
	Pod       string                `json:"pod,omitempty"`
	Ready     bool                  `json:"ready"`
	Resources *ResourceRequirements `json:"resources,omitempty"`
}

type ExecuteCommandResponse struct {
	ExecutionID string          `json:"execution_id"`
	Status      ExecutionStatus `json:"status"`
	ExitCode    *int            `json:"exit_code"`
	Stdout      string          `json:"stdout"`
	Stderr      string          `json:"stderr"`
	ErrorCode   string          `json:"error_code,omitempty"`
	Message     string          `json:"message,omitempty"`
	Extra       *ResponseExtra  `json:"extra,omitempty"`
}

type ResponseExtra struct {
	SandboxTimings *SandboxTimings `json:"sandbox_timings,omitempty"`
}

type SandboxTimings struct {
	CPUExecution  *ExecutionTimingResponse   `json:"cpu_execution,omitempty"`
	GPUExecutions []*ExecutionTimingResponse `json:"gpu_executions,omitempty"`
}

type ResourceTimingResponse struct {
	CreateMS *int64 `json:"create_ms,omitempty"`
	ReadyMS  *int64 `json:"ready_ms,omitempty"`
}

type ExecutionTimingResponse struct {
	TotalMS    int64  `json:"total_ms"`
	QueueMS    *int64 `json:"queue_ms,omitempty"`
	PrecheckMS *int64 `json:"precheck_ms,omitempty"`
	ExecMS     int64  `json:"exec_ms"`
	Attempts   int    `json:"attempts"`
}

type PodInfo struct {
	Name         string
	Prefix       string
	UID          string
	SessionID    string
	GPUPoolID    string
	Image        string
	Resources    *ResourceRequirements
	CreatedAt    time.Time
	Ready        bool
	Failed       bool
	Deleting     bool
	RestartCount int32
}

type GPUPoolInfo struct {
	GPUPoolID  string
	Image      string
	Replicas   int32
	Resources  *ResourceRequirements
	Scheduling *Scheduling
}

type GPUPodState struct {
	Busy         bool         `json:"busy"`
	ProcessCount int          `json:"process_count"`
	Processes    []GPUProcess `json:"processes"`
}

type GPUProcess struct {
	PID          string `json:"pid"`
	ProcessName  string `json:"process_name"`
	UsedMemoryMB string `json:"used_memory_mb"`
}

type gpuPool struct {
	id             string
	image          string
	replicas       int32
	clusterName    string
	resources      *ResourceRequirements
	scheduling     *Scheduling
	queue          []string
	deleting       bool
	recreating     bool
	createdAt      time.Time
	lastActivityAt time.Time
	createDuration *time.Duration
	readyDuration  *time.Duration
	workerUsage    map[string]gpuWorkerUsage
	queueCount     int64
	queueTotal     time.Duration
	queueMax       time.Duration
}

type gpuWorkerUsage struct {
	busy     time.Duration
	capacity time.Duration
}

type session struct {
	id             string
	prefix         string
	sessionType    SessionType
	status         SessionStatus
	errorMessage   string
	createdAt      time.Time
	lastActivityAt time.Time
	createDuration *time.Duration
	readyDuration  *time.Duration

	cpuGPU  *cpuGPUSession
	gpuOnly *gpuOnlySession
}

type cpuGPUSession struct {
	gpuPoolID       string
	cpuPod          *PodInfo
	cpuImage        string
	cpuResources    *ResourceRequirements
	probeRunning    bool
	probeAttempts   int
	activeExecution string
}

type gpuOnlySession struct {
	clusterName  string
	gpuPod       *PodInfo
	gpuImage     string
	gpuResources *ResourceRequirements
}

type execution struct {
	id                string
	sessionID         string
	gpuPoolID         string
	parentExecutionID string
	gpuTimings        []*ExecutionTimingResponse
	request           command.Request
	status            ExecutionStatus
	attemptCount      int
	gpuWorkerUID      string
	result            *ExecuteCommandResponse
	done              chan struct{}
	createdAt         time.Time
	queuedAt          time.Time
	queueDeadline     time.Time
	queueDuration     time.Duration
	precheckDuration  time.Duration
	execDuration      time.Duration
}

func (e *execution) terminal() bool {
	return e.status == ExecutionStatusCompleted || e.status == ExecutionStatusTimedOut || e.status == ExecutionStatusFailed
}

type gpuWorker struct {
	pod            PodInfo
	gpuPoolID      string
	status         GPUWorkerStatus
	executionID    string
	discoveredAt   time.Time
	busySince      time.Time
	availableSince time.Time
}
