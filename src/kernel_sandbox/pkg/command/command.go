package command

import (
	"bytes"
	"fmt"
	"io"
	"regexp"
	"strings"
	"sync"
)

type Device string

const (
	DeviceCPU Device = "cpu"
	DeviceGPU Device = "gpu"
)

type ResultKind string

const (
	ResultCompleted      ResultKind = "result"
	ResultInfrastructure ResultKind = "infrastructure"
	ResultTimeout        ResultKind = "timeout"
	ResultOutputLimit    ResultKind = "output_limit"
	ResultStartFailure   ResultKind = "start_failure"
)

var envNamePattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

type Request struct {
	Device         Device            `json:"device"`
	Command        []string          `json:"command"`
	WorkingDir     string            `json:"working_dir,omitempty"`
	Env            map[string]string `json:"env,omitempty"`
	Stdin          string            `json:"stdin,omitempty"`
	TimeoutSeconds int64             `json:"timeout_seconds"`
}

func (r Request) Validate(maxStdin int) error {
	if r.Device != DeviceCPU && r.Device != DeviceGPU {
		return fmt.Errorf("device must be cpu or gpu")
	}
	if len(r.Command) == 0 || strings.TrimSpace(r.Command[0]) == "" {
		return fmt.Errorf("command must not be empty")
	}
	if r.TimeoutSeconds <= 0 {
		return fmt.Errorf("timeout_seconds must be positive")
	}
	if len(r.Stdin) > maxStdin {
		return fmt.Errorf("stdin exceeds %d bytes", maxStdin)
	}
	for key := range r.Env {
		if !envNamePattern.MatchString(key) {
			return fmt.Errorf("invalid environment variable %q", key)
		}
	}
	return nil
}

type Result struct {
	ExitCode *int       `json:"exit_code"`
	Stdout   string     `json:"stdout"`
	Stderr   string     `json:"stderr"`
	Kind     ResultKind `json:"kind"`
	Error    string     `json:"error,omitempty"`
}

type Output struct {
	mu       sync.Mutex
	max      int
	written  int
	overflow bool
	exceeded chan struct{}
	once     sync.Once
	stdout   bytes.Buffer
	stderr   bytes.Buffer
}

func NewOutput(max int) *Output {
	return &Output{max: max, exceeded: make(chan struct{})}
}

func (o *Output) Stdout() io.Writer         { return outputWriter{output: o, buffer: &o.stdout} }
func (o *Output) Stderr() io.Writer         { return outputWriter{output: o, buffer: &o.stderr} }
func (o *Output) Exceeded() <-chan struct{} { return o.exceeded }

func (o *Output) Result() (stdout, stderr string, overflow bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.stdout.String(), o.stderr.String(), o.overflow
}

type outputWriter struct {
	output *Output
	buffer *bytes.Buffer
}

func (w outputWriter) Write(data []byte) (int, error) {
	w.output.mu.Lock()
	defer w.output.mu.Unlock()
	remaining := w.output.max - w.output.written
	if remaining <= 0 {
		w.output.markExceeded()
		return len(data), nil
	}
	part := data
	if len(part) > remaining {
		part = part[:remaining]
		w.output.markExceeded()
	}
	_, _ = w.buffer.Write(part)
	w.output.written += len(part)
	return len(data), nil
}

func (o *Output) markExceeded() {
	o.overflow = true
	o.once.Do(func() { close(o.exceeded) })
}
