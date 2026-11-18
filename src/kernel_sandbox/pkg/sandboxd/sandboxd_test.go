package sandboxd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"kernel-sandbox/pkg/command"
	"kernel-sandbox/pkg/config"
	"kernel-sandbox/pkg/master"
)

func testServer(maxOutput int) *server {
	return &server{terminateGrace: 20 * time.Millisecond, maxStdin: 1024, maxOutput: maxOutput}
}

func executeRequest(t *testing.T, server *server, request command.Request) (int, command.Result) {
	t.Helper()
	payload, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	httpRequest := httptest.NewRequest(http.MethodPost, "/exec", bytes.NewReader(payload))
	recorder := httptest.NewRecorder()
	server.handleExec(recorder, httpRequest)
	var result command.Result
	if err := json.NewDecoder(recorder.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	return recorder.Code, result
}

func TestExecuteCommand(t *testing.T) {
	status, result := executeRequest(t, testServer(1024), command.Request{
		Device: command.DeviceGPU, Command: []string{"/bin/sh", "-c", "cat; echo error >&2; exit 7"},
		Stdin: "input", TimeoutSeconds: 5,
	})
	if status != http.StatusOK || result.Kind != command.ResultCompleted || result.ExitCode == nil || *result.ExitCode != 7 {
		t.Fatalf("status=%d result=%+v", status, result)
	}
	if !strings.Contains(result.Stdout, "input") || !strings.Contains(result.Stderr, "error") {
		t.Fatalf("stdout=%q stderr=%q", result.Stdout, result.Stderr)
	}
}

func TestResultDoesNotWaitForResidualProcessCleanup(t *testing.T) {
	server := testServer(1024)
	server.terminateGrace = 500 * time.Millisecond
	started := time.Now()
	status, result := executeRequest(t, server, command.Request{
		Device:         command.DeviceGPU,
		Command:        []string{"/bin/sh", "-c", "(trap '' TERM; while :; do sleep 1; done) >/dev/null 2>&1 &"},
		TimeoutSeconds: 5,
	})
	if status != http.StatusOK || result.Kind != command.ResultCompleted || time.Since(started) >= server.terminateGrace {
		t.Fatalf("status=%d result=%+v duration=%s", status, result, time.Since(started))
	}
	server.mu.Lock()
	busy := server.busy
	server.mu.Unlock()
	if !busy {
		t.Fatal("sandboxd became idle before residual process cleanup")
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		server.mu.Lock()
		busy = server.busy
		server.mu.Unlock()
		if !busy {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("sandboxd remained busy after residual process cleanup")
}

func TestExecuteLimitsAndValidation(t *testing.T) {
	_, output := executeRequest(t, testServer(32), command.Request{
		Device: command.DeviceGPU, Command: []string{"/bin/sh", "-c", "yes output"}, TimeoutSeconds: 5,
	})
	if output.Kind != command.ResultOutputLimit {
		t.Fatalf("output result=%+v", output)
	}
	status, invalid := executeRequest(t, testServer(1024), command.Request{Device: command.DeviceCPU, Command: []string{"true"}, TimeoutSeconds: 1})
	if status != http.StatusBadRequest || invalid.Kind != command.ResultInfrastructure {
		t.Fatalf("status=%d result=%+v", status, invalid)
	}
}

func TestRejectsMultipleJSONValues(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/exec", strings.NewReader(`{"device":"gpu","command":["true"],"timeout_seconds":1} {}`))
	recorder := httptest.NewRecorder()
	testServer(1024).handleExec(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status=%d", recorder.Code)
	}
}

func TestMissingCommandIsStartFailure(t *testing.T) {
	_, result := executeRequest(t, testServer(1024), command.Request{Device: command.DeviceGPU, Command: []string{"/missing-command"}, TimeoutSeconds: 1})
	if result.Kind != command.ResultStartFailure || result.ExitCode != nil || result.Error == "" {
		t.Fatalf("result=%+v", result)
	}
}

func TestStartedCommandExitCodesAreCompleted(t *testing.T) {
	tests := []struct {
		name     string
		command  string
		exitCode int
	}{
		{name: "exit 126", command: "exit 126", exitCode: 126},
		{name: "exit 127", command: "exit 127", exitCode: 127},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, result := executeRequest(t, testServer(1024), command.Request{Device: command.DeviceGPU, Command: []string{"/bin/sh", "-c", test.command}, TimeoutSeconds: 1})
			if result.Kind != command.ResultCompleted || result.ExitCode == nil || *result.ExitCode != test.exitCode {
				t.Fatalf("result=%+v", result)
			}
		})
	}
}

func TestExecuteTimeoutAndBusy(t *testing.T) {
	server := testServer(1024)
	server.terminateGrace = 500 * time.Millisecond
	started := time.Now()
	_, result := executeRequest(t, server, command.Request{
		Device: command.DeviceGPU, Command: []string{"/bin/sh", "-c", "trap '' TERM; sleep 5"}, TimeoutSeconds: 1,
	})
	if result.Kind != command.ResultTimeout || result.ExitCode != nil || time.Since(started) >= time.Second+server.terminateGrace {
		t.Fatalf("result=%+v", result)
	}
	server.mu.Lock()
	busy := server.busy
	server.mu.Unlock()
	if !busy {
		t.Fatal("sandboxd became idle before timed out process cleanup")
	}
	server = testServer(1024)
	server.busy = true
	status, result := executeRequest(t, server, command.Request{Device: command.DeviceGPU, Command: []string{"true"}, TimeoutSeconds: 1})
	if status != http.StatusConflict || !strings.Contains(result.Error, "busy") {
		t.Fatalf("status=%d result=%+v", status, result)
	}
}

func writeProcess(t *testing.T, procRoot string, pid, ppid int, cmdline, gpuDevice string) {
	t.Helper()
	directory := filepath.Join(procRoot, fmt.Sprint(pid))
	if err := os.MkdirAll(filepath.Join(directory, "fd"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "stat"), []byte(fmt.Sprintf("%d (test) S %d%s\n", pid, ppid, strings.Repeat(" 0", 50))), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "cmdline"), []byte(strings.ReplaceAll(cmdline, " ", "\x00")+"\x00"), 0o600); err != nil {
		t.Fatal(err)
	}
	if gpuDevice != "" {
		if err := os.Symlink(gpuDevice, filepath.Join(directory, "fd", "3")); err != nil {
			t.Fatal(err)
		}
	}
}

func testProc(t *testing.T) string {
	t.Helper()
	procRoot := t.TempDir()
	writeProcess(t, procRoot, 1, 0, "/opt/kernel-sandbox/sandboxd serve", "")
	writeProcess(t, procRoot, 561, 1, "/usr/bin/python3 -m kerneleval.server", "")
	writeProcess(t, procRoot, 564, 561, "/usr/bin/python3 -c multiprocessing.resource_tracker", "")
	writeProcess(t, procRoot, 761, 561, "/usr/bin/python3 -c multiprocessing.spawn spawn_main", "/dev/nvidia0")
	writeProcess(t, procRoot, 762, 561, "/usr/bin/python3 unexpected_gpu_child.py", "/dev/nvidia-uvm")
	writeProcess(t, procRoot, 800, 1, "/usr/bin/python3 -m other.server", "")
	writeProcess(t, procRoot, 801, 800, "/usr/bin/python3 -c multiprocessing.spawn spawn_main", "/dev/nvidia1")
	writeProcess(t, procRoot, 900, 0, "bash", "/dev/nvidia2")
	writeProcess(t, procRoot, 901, 900, "python exec_child.py", "/dev/nvidia3")
	writeProcess(t, procRoot, 902, 1, "python non_gpu.py", "")
	return procRoot
}

func TestGPUProcessAllowlist(t *testing.T) {
	procRoot := testProc(t)
	tests := []struct {
		name      string
		allowlist []config.GPUProcessAllowlistEntry
		wantPIDs  string
	}{
		{name: "empty allowlist", wantPIDs: "761,762,801"},
		{name: "process and ancestor", allowlist: []config.GPUProcessAllowlistEntry{{Process: "multiprocessing.spawn", Ancestor: "kerneleval.server"}}, wantPIDs: "762,801"},
		{name: "ancestor mismatch", allowlist: []config.GPUProcessAllowlistEntry{{Process: "multiprocessing.spawn", Ancestor: "missing.server"}}, wantPIDs: "761,762,801"},
		{name: "process only", allowlist: []config.GPUProcessAllowlistEntry{{Process: "multiprocessing.spawn"}}, wantPIDs: "762"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state, err := readGPUState(procRoot, false, test.allowlist)
			if err != nil {
				t.Fatal(err)
			}
			pids := make([]string, 0, len(state.Processes))
			for _, process := range state.Processes {
				pids = append(pids, process.PID)
			}
			if strings.Join(pids, ",") != test.wantPIDs || state.ProcessCount != len(state.Processes) {
				t.Fatalf("state=%+v", state)
			}
		})
	}
}

func TestStateDoesNotRequireNvidiaSMI(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	server := testServer(1024)
	server.procRoot = testProc(t)
	server.busy = true
	recorder := httptest.NewRecorder()
	server.handleState(recorder, httptest.NewRequest(http.MethodGet, "/state", nil))
	var state master.GPUPodState
	if err := json.NewDecoder(recorder.Body).Decode(&state); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusOK || !state.Busy || state.ProcessCount != 3 || !strings.Contains(state.Processes[0].ProcessName, "multiprocessing.spawn") || state.Processes[0].UsedMemoryMB != "" {
		t.Fatalf("status=%d state=%+v", recorder.Code, state)
	}
}

func TestParseGPUProcessAllowlist(t *testing.T) {
	allowlist, err := parseGPUProcessAllowlist(`[{"process":"multiprocessing.spawn","ancestor":"kerneleval.server"}]`)
	if err != nil || len(allowlist) != 1 || allowlist[0].Ancestor != "kerneleval.server" {
		t.Fatalf("allowlist=%+v error=%v", allowlist, err)
	}
	for _, value := range []string{`{}`, `[{"ancestor":"kerneleval.server"}]`} {
		if _, err := parseGPUProcessAllowlist(value); err == nil {
			t.Fatalf("value=%s", value)
		}
	}
}

func TestHealthDoesNotInspectGPU(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	recorder := httptest.NewRecorder()
	testServer(1024).handleHealth(recorder, httptest.NewRequest(http.MethodGet, "/health", nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status=%d", recorder.Code)
	}
}
