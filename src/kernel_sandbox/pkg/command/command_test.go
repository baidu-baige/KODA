package command

import (
	"io"
	"testing"
)

func TestRequestValidation(t *testing.T) {
	request := Request{Device: DeviceGPU, Command: []string{"python"}, TimeoutSeconds: 10, Env: map[string]string{"CUDA_VISIBLE_DEVICES": "0"}}
	if err := request.Validate(1024); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	request.Command = nil
	if err := request.Validate(1024); err == nil {
		t.Fatal("expected empty command error")
	}
}

func TestOutputSharesLimit(t *testing.T) {
	output := NewOutput(6)
	_, _ = io.WriteString(output.Stdout(), "abcd")
	_, _ = io.WriteString(output.Stderr(), "EFGH")
	stdout, stderr, overflow := output.Result()
	if stdout != "abcd" || stderr != "EF" || !overflow {
		t.Fatalf("stdout=%q stderr=%q overflow=%v", stdout, stderr, overflow)
	}
	select {
	case <-output.Exceeded():
	default:
		t.Fatal("expected exceeded signal")
	}
}
