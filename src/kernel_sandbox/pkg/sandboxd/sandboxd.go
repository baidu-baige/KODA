package sandboxd

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"kernel-sandbox/pkg/command"
	"kernel-sandbox/pkg/config"
	"kernel-sandbox/pkg/master"

	"github.com/prometheus/procfs"
)

const defaultSocketPath = "/run/kernel-sandbox/sandboxd.sock"

type server struct {
	allowCPU            bool
	terminateGrace      time.Duration
	maxStdin            int
	maxOutput           int
	procRoot            string
	gpuProcessAllowlist []config.GPUProcessAllowlistEntry

	mu   sync.Mutex
	busy bool
}

func Run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: sandboxd serve|exec|health|state")
	}
	switch args[0] {
	case "serve":
		return serve(args[1:])
	case "exec":
		return call(args[1:], http.MethodPost, "/exec", os.Stdin)
	case "health":
		return call(args[1:], http.MethodGet, "/health", nil)
	case "state":
		return call(args[1:], http.MethodGet, "/state", nil)
	default:
		return fmt.Errorf("unknown sandboxd command %q", args[0])
	}
}

func serve(args []string) error {
	flags := flag.NewFlagSet("sandboxd serve", flag.ContinueOnError)
	socketPath := flags.String("socket", defaultSocketPath, "Unix socket path")
	graceSeconds := flags.Int("terminate-grace-seconds", 5, "seconds between TERM and KILL")
	allowCPU := flags.Bool("allow-cpu", false, "allow CPU commands in a CPU-only Docker session")
	maxStdin := flags.Int("max-stdin-bytes", 0, "stdin byte limit")
	maxOutput := flags.Int("max-output-bytes", 0, "combined stdout and stderr byte limit")
	allowlistJSON := flags.String("gpu-process-allowlist", "[]", "JSON GPU process allowlist")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *graceSeconds < 0 || *maxStdin <= 0 || *maxOutput <= 0 {
		return fmt.Errorf("invalid sandboxd limits")
	}
	allowlist, err := parseGPUProcessAllowlist(*allowlistJSON)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(*socketPath), 0o755); err != nil {
		return err
	}
	if err := os.Remove(*socketPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	listener, err := net.Listen("unix", *socketPath)
	if err != nil {
		return err
	}
	defer listener.Close()
	if err := os.Chmod(*socketPath, 0o666); err != nil {
		return err
	}
	handler := &server{
		allowCPU:            *allowCPU,
		terminateGrace:      time.Duration(*graceSeconds) * time.Second,
		maxStdin:            *maxStdin,
		maxOutput:           *maxOutput,
		procRoot:            "/proc",
		gpuProcessAllowlist: allowlist,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /exec", handler.handleExec)
	mux.HandleFunc("GET /health", handler.handleHealth)
	mux.HandleFunc("GET /state", handler.handleState)
	return (&http.Server{Handler: mux}).Serve(listener)
}

func (s *server) handleExec(writer http.ResponseWriter, request *http.Request) {
	var input command.Request
	decoder := json.NewDecoder(http.MaxBytesReader(writer, request.Body, int64(s.maxStdin)*6+1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&input); err != nil {
		writeJSON(writer, http.StatusBadRequest, command.Result{Kind: command.ResultInfrastructure, Error: err.Error()})
		return
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		writeJSON(writer, http.StatusBadRequest, command.Result{Kind: command.ResultInfrastructure, Error: "request body must contain one JSON object"})
		return
	}
	if err := input.Validate(s.maxStdin); err != nil || (input.Device != command.DeviceGPU && !s.allowCPU) {
		if err == nil {
			err = fmt.Errorf("device must be gpu")
		}
		writeJSON(writer, http.StatusBadRequest, command.Result{Kind: command.ResultInfrastructure, Error: err.Error()})
		return
	}
	s.mu.Lock()
	if s.busy {
		s.mu.Unlock()
		writeJSON(writer, http.StatusConflict, command.Result{Kind: command.ResultInfrastructure, Error: "sandboxd is busy"})
		return
	}
	s.busy = true
	s.mu.Unlock()
	result, cleanup := s.execute(request.Context(), input)
	writeJSON(writer, http.StatusOK, result)
	if cleanup == nil {
		s.mu.Lock()
		s.busy = false
		s.mu.Unlock()
		return
	}
	go func() {
		cleanup()
		s.mu.Lock()
		s.busy = false
		s.mu.Unlock()
	}()
}

func (s *server) handleHealth(writer http.ResponseWriter, _ *http.Request) {
	writeJSON(writer, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *server) handleState(writer http.ResponseWriter, _ *http.Request) {
	s.mu.Lock()
	busy := s.busy
	s.mu.Unlock()
	state, err := readGPUState(s.procRoot, busy, s.gpuProcessAllowlist)
	if err != nil {
		writeJSON(writer, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(writer, http.StatusOK, state)
}

type processInfo struct {
	pid     int
	ppid    int
	cmdline string
	gpu     bool
}

func parseGPUProcessAllowlist(value string) ([]config.GPUProcessAllowlistEntry, error) {
	var allowlist []config.GPUProcessAllowlistEntry
	if err := json.Unmarshal([]byte(value), &allowlist); err != nil {
		return nil, fmt.Errorf("invalid GPU process allowlist: %w", err)
	}
	for i, entry := range allowlist {
		if strings.TrimSpace(entry.Process) == "" {
			return nil, fmt.Errorf("GPU process allowlist entry %d requires process", i)
		}
	}
	return allowlist, nil
}

func readGPUState(procRoot string, busy bool, allowlist []config.GPUProcessAllowlistEntry) (master.GPUPodState, error) {
	if procRoot == "" {
		procRoot = "/proc"
	}
	processes, err := readProcesses(procRoot)
	if err != nil {
		return master.GPUPodState{}, err
	}
	var residuals []processInfo
	for _, process := range processes {
		if process.gpu && !ignoreGPUProcess(process, processes, allowlist) {
			residuals = append(residuals, process)
		}
	}
	sort.Slice(residuals, func(i, j int) bool { return residuals[i].pid < residuals[j].pid })
	state := master.GPUPodState{Busy: busy, ProcessCount: len(residuals), Processes: make([]master.GPUProcess, 0, len(residuals))}
	for _, process := range residuals {
		state.Processes = append(state.Processes, master.GPUProcess{PID: strconv.Itoa(process.pid), ProcessName: process.cmdline})
	}
	return state, nil
}

func readProcesses(procRoot string) (map[int]processInfo, error) {
	fs, err := procfs.NewFS(procRoot)
	if err != nil {
		return nil, err
	}
	procs, err := fs.AllProcs()
	if err != nil {
		return nil, err
	}
	processes := make(map[int]processInfo)
	for _, proc := range procs {
		stat, err := proc.Stat()
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("read process %d: %w", proc.PID, err)
		}
		cmdline, err := proc.CmdLine()
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("read process %d: %w", proc.PID, err)
		}
		fds, err := proc.FileDescriptorTargets()
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("read process %d: %w", proc.PID, err)
		}
		process := processInfo{pid: proc.PID, ppid: stat.PPID, cmdline: strings.Join(cmdline, " ")}
		for _, target := range fds {
			suffix, found := strings.CutPrefix(target, "/dev/nvidia")
			if found && (suffix == "-uvm" || suffix != "" && strings.Trim(suffix, "0123456789") == "") {
				process.gpu = true
				break
			}
		}
		processes[proc.PID] = process
	}
	return processes, nil
}

func ignoreGPUProcess(process processInfo, processes map[int]processInfo, allowlist []config.GPUProcessAllowlistEntry) bool {
	for current := process; current.pid != 1; {
		if current.ppid == 0 {
			return true
		}
		parent, exists := processes[current.ppid]
		if !exists {
			break
		}
		current = parent
	}
	for _, entry := range allowlist {
		if !strings.Contains(process.cmdline, entry.Process) {
			continue
		}
		if entry.Ancestor == "" {
			return true
		}
		for ppid := process.ppid; ; {
			ancestor, exists := processes[ppid]
			if !exists {
				break
			}
			if strings.Contains(ancestor.cmdline, entry.Ancestor) {
				return true
			}
			ppid = ancestor.ppid
		}
	}
	return false
}

func (s *server) execute(ctx context.Context, request command.Request) (command.Result, func()) {
	process := exec.Command(request.Command[0], request.Command[1:]...)
	process.Dir = request.WorkingDir
	process.Env = mergeEnvironment(request.Env)
	process.Stdin = strings.NewReader(request.Stdin)
	process.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	output := command.NewOutput(s.maxOutput)
	process.Stdout = output.Stdout()
	process.Stderr = output.Stderr()
	if err := process.Start(); err != nil {
		return command.Result{Kind: command.ResultStartFailure, Error: err.Error()}, nil
	}
	wait := make(chan error, 1)
	go func() { wait <- process.Wait() }()
	timer := time.NewTimer(time.Duration(request.TimeoutSeconds) * time.Second)
	defer timer.Stop()

	var kind command.ResultKind
	var waitErr error
	var cleanup func()
	select {
	case waitErr = <-wait:
		kind = command.ResultCompleted
		cleanup = func() {
			cleanupProcessGroup(process.Process.Pid, s.terminateGrace)
			reapChildren()
		}
	case <-timer.C:
		kind = command.ResultTimeout
		cleanup = func() {
			_ = stopProcessGroup(process.Process.Pid, wait, s.terminateGrace)
			reapChildren()
		}
	case <-output.Exceeded():
		kind = command.ResultOutputLimit
		cleanup = func() {
			_ = stopProcessGroup(process.Process.Pid, wait, s.terminateGrace)
			reapChildren()
		}
	case <-ctx.Done():
		kind = command.ResultInfrastructure
		cleanup = func() {
			_ = stopProcessGroup(process.Process.Pid, wait, s.terminateGrace)
			reapChildren()
		}
	}
	stdout, stderr, overflow := output.Result()
	if overflow && kind == command.ResultCompleted {
		kind = command.ResultOutputLimit
	}
	result := command.Result{Kind: kind, Stdout: stdout, Stderr: stderr}
	if kind == command.ResultCompleted && process.ProcessState != nil {
		exitCode := process.ProcessState.ExitCode()
		result.ExitCode = &exitCode
	}
	if waitErr != nil && result.ExitCode == nil {
		result.Error = waitErr.Error()
	}
	return result, cleanup
}

func stopProcessGroup(pid int, wait <-chan error, grace time.Duration) error {
	_ = syscall.Kill(-pid, syscall.SIGTERM)
	timer := time.NewTimer(grace)
	defer timer.Stop()
	select {
	case err := <-wait:
		cleanupProcessGroup(pid, 0)
		return err
	case <-timer.C:
		_ = syscall.Kill(-pid, syscall.SIGKILL)
		return <-wait
	}
}

func cleanupProcessGroup(pid int, grace time.Duration) {
	if syscall.Kill(-pid, 0) != nil {
		return
	}
	_ = syscall.Kill(-pid, syscall.SIGTERM)
	deadline := time.Now().Add(grace)
	for time.Now().Before(deadline) {
		if syscall.Kill(-pid, 0) != nil {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	_ = syscall.Kill(-pid, syscall.SIGKILL)
}

func reapChildren() {
	for {
		var status syscall.WaitStatus
		pid, err := syscall.Wait4(-1, &status, syscall.WNOHANG, nil)
		if pid > 0 {
			continue
		}
		if errors.Is(err, syscall.ECHILD) || pid == 0 {
			return
		}
		return
	}
}

func mergeEnvironment(extra map[string]string) []string {
	values := map[string]string{}
	for _, item := range os.Environ() {
		key, value, ok := strings.Cut(item, "=")
		if ok {
			values[key] = value
		}
	}
	for key, value := range extra {
		values[key] = value
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]string, 0, len(keys))
	for _, key := range keys {
		result = append(result, key+"="+values[key])
	}
	return result
}

func call(args []string, method, path string, body io.Reader) error {
	flags := flag.NewFlagSet("sandboxd "+strings.TrimPrefix(path, "/"), flag.ContinueOnError)
	socketPath := flags.String("socket", defaultSocketPath, "Unix socket path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	client := &http.Client{Transport: &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "unix", *socketPath)
	}}}
	request, err := http.NewRequest(method, "http://sandboxd"+path, body)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(response.Body)
	if err != nil {
		return err
	}
	if response.StatusCode/100 != 2 {
		return fmt.Errorf("sandboxd returned %s: %s", response.Status, payload)
	}
	_, err = os.Stdout.Write(payload)
	return err
}

func writeJSON(writer http.ResponseWriter, status int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}
