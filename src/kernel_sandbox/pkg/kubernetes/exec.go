package kubernetes

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"kernel-sandbox/pkg/command"
	"kernel-sandbox/pkg/config"
	"kernel-sandbox/pkg/master"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/remotecommand"
	utilexec "k8s.io/client-go/util/exec"
)

func (k *Client) ExecuteCPU(ctx context.Context, pod master.PodInfo, request command.Request, maxOutput int) command.Result {
	return k.cpuClusterClient.exec(ctx, pod, wrappedCommand(request.Command, request.WorkingDir, request.Env), request.Stdin, maxOutput)
}

func (k *Client) ExecuteGPU(ctx context.Context, pod master.PodInfo, request command.Request, maxOutput int) command.Result {
	payload, err := json.Marshal(request)
	if err != nil {
		return command.Result{Kind: command.ResultInfrastructure, Error: err.Error()}
	}
	result := k.gpuPoolClusterClient.exec(ctx, pod, []string{sandboxdBinary, "exec", "--socket", config.SandboxdSocketPath}, string(payload), maxOutput+1<<20)
	if result.Kind != command.ResultCompleted || result.ExitCode == nil || *result.ExitCode != 0 {
		if result.Kind == command.ResultCompleted {
			result.Kind = command.ResultInfrastructure
			result.Error = "sandboxd exec returned without an exit code"
			if result.ExitCode != nil {
				result.Error = fmt.Sprintf("sandboxd exec exited with code %d", *result.ExitCode)
			}
			if stderr := strings.TrimSpace(result.Stderr); stderr != "" {
				result.Error += ": " + stderr
			}
		}
		return result
	}
	decoded := command.Result{}
	if err := json.Unmarshal([]byte(result.Stdout), &decoded); err != nil {
		return command.Result{Kind: command.ResultInfrastructure, Error: "decode sandboxd result: " + err.Error(), Stdout: result.Stdout, Stderr: result.Stderr}
	}
	return decoded
}

// ExecuteGPUSession runs a command directly inside a gpu_only session's
// persistent Pod. The Pod is launched with a plain "sleep infinity" entrypoint
// (no sandboxd), so the command is executed with an ordinary streaming exec in
// the GPU cluster.
func (k *Client) ExecuteGPUSession(ctx context.Context, pod master.PodInfo, request command.Request, maxOutput int) command.Result {
	return k.gpuSessionClusterClient.exec(ctx, pod, wrappedCommand(request.Command, request.WorkingDir, request.Env), request.Stdin, maxOutput)
}

func (k *Client) ReadGPUPodState(ctx context.Context, pod master.PodInfo) (master.GPUPodState, error) {
	result := k.gpuPoolClusterClient.exec(ctx, pod, []string{sandboxdBinary, "state", "--socket", config.SandboxdSocketPath}, "", 1<<20)
	if result.Kind != command.ResultCompleted || result.ExitCode == nil || *result.ExitCode != 0 {
		return master.GPUPodState{}, fmt.Errorf("GPU state command failed (%s): %s", result.Kind, strings.TrimSpace(strings.Join([]string{result.Error, result.Stdout, result.Stderr}, " ")))
	}
	state := master.GPUPodState{}
	if err := json.Unmarshal([]byte(result.Stdout), &state); err != nil {
		return master.GPUPodState{}, err
	}
	return state, nil
}

func (c *clusterClient) exec(ctx context.Context, pod master.PodInfo, args []string, stdin string, maxOutput int) command.Result {
	request := c.clientset.CoreV1().RESTClient().Post().Resource("pods").Name(pod.Name).Namespace(c.namespace).SubResource("exec")
	request.VersionedParams(&corev1.PodExecOptions{Command: args, Stdin: stdin != "", Stdout: true, Stderr: true, TTY: false}, scheme.ParameterCodec)
	executor, err := remotecommand.NewSPDYExecutor(c.restConfig, "POST", request.URL())
	if err != nil {
		return command.Result{Kind: command.ResultInfrastructure, Error: err.Error()}
	}
	execCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	output := command.NewOutput(maxOutput)
	watchDone := make(chan struct{})
	go func() {
		select {
		case <-output.Exceeded():
			cancel()
		case <-watchDone:
		}
	}()
	options := remotecommand.StreamOptions{Stdout: output.Stdout(), Stderr: output.Stderr(), Tty: false}
	if stdin != "" {
		options.Stdin = strings.NewReader(stdin)
	}
	err = executor.StreamWithContext(execCtx, options)
	close(watchDone)
	stdoutText, stderrText, overflow := output.Result()
	result := command.Result{Stdout: stdoutText, Stderr: stderrText}
	if overflow {
		result.Kind = command.ResultOutputLimit
		return result
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		result.Kind = command.ResultTimeout
		return result
	}
	if err == nil {
		return resultFromExitCode(stdoutText, stderrText, 0)
	}
	var exitError utilexec.ExitError
	if errors.As(err, &exitError) {
		return resultFromExitCode(stdoutText, stderrText, exitError.ExitStatus())
	}
	var upgradeError *httpstream.UpgradeFailureError
	if errors.As(err, &upgradeError) || errors.Is(err, io.EOF) {
		result.Kind = command.ResultInfrastructure
		result.Error = err.Error()
		return result
	}
	result.Kind = command.ResultInfrastructure
	result.Error = err.Error()
	return result
}

func resultFromExitCode(stdout, stderr string, exitCode int) command.Result {
	return command.Result{Kind: command.ResultCompleted, ExitCode: &exitCode, Stdout: stdout, Stderr: stderr}
}

func wrappedCommand(args []string, workingDir string, environment map[string]string) []string {
	keys := make([]string, 0, len(environment))
	for key := range environment {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := []string{"/bin/sh", "-c", `if [ -n "$1" ]; then cd -- "$1" || exit 127; fi; shift; exec env "$@"`, "kernel-sandbox", workingDir}
	for _, key := range keys {
		result = append(result, key+"="+environment[key])
	}
	return append(result, args...)
}
