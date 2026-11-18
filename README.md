# KODA: Kernel Optimization via Domain Adaptation

[English](README.md) | [简体中文](README.zh-CN.md)

[**Hugging Face · KODA-35B**](https://huggingface.co/Baidu-Baige/KODA-35B) · [**知乎 · KODA 技术文章**](https://zhuanlan.zhihu.com/p/2106387936149361761)

KODA is a framework for optimizing GPU kernels with language models. It generates CUDA code, compiles and evaluates candidates in managed sandboxes, and feeds correctness, timing, and profiling results back to the model for iterative improvement.

KODA is built on [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent), extended with GPU kernel optimization controllers, sandbox execution, and evaluation/profiling tools.

[Quickstart overview](docs/quickstart.md) · For first-time setup, choose the [single-host Docker Quickstart](docs/docker-quickstart.md) or the [Kubernetes Quickstart](docs/kubernetes-quickstart.md). Both explain server setup, automatic pool/session creation, a smoke run, and cleanup.

## Choose a sandbox backend

| Backend | Requirements | Setup |
| --- | --- | --- |
| **Docker (single host)** | Linux, Docker, NVIDIA Container Toolkit, dedicated GPU UUIDs | [Docker Quickstart](docs/docker-quickstart.md) |
| **Kubernetes (default)** | GPU-enabled cluster, kubeconfig, CPU/GPU templates | [Kubernetes Quickstart](docs/kubernetes-quickstart.md) |

Set `backend: docker` in the Sandbox Server configuration to use local Docker containers. Omitting `backend` preserves the existing Kubernetes behavior. Both backends use the same agent-facing HTTP API and automatically create CPU sessions and GPU workers when evaluation starts.

Start the Sandbox Server before running evaluation. The model service and agent runner may run on separate machines. Docker mode currently supports `cpu_gpu` sessions on a local daemon; `gpu_only`, remote Docker daemons and live pool resizing are not supported. GPU devices must be reserved for the sandbox and excluded from model-serving allocations.

## Release scope

This repository contains only the kernel controller workflow. Retired loop/ReAct/OpenCode agent modes, the unused LLM proxy, training scripts, internal cluster configurations, and unrelated upstream guides are excluded. Shared runtime/model helpers are retained only where required by the controller. LiteLLM still supports compatible model APIs.

Model endpoints, credentials and output locations are configured at runtime. Controller prompts are packaged as `controller_h.yaml` and `controller_b.yaml`, with repository-level copies under `configs/`. Tool caches default to the isolated sandbox workspace's `.cache` directory. Tools installed under `/opt/kerneltools` and `/workspace/kerneleval` are paths inside the sandbox image, not host/cloud mounts; override them for a different image.

## How it works

```text
Evaluation runner ── model API ── candidate kernel.py / kernel.cu
        │
        └── Sandbox Server
               ├── per-task CPU session: workspace and compilation
               └── shared GPU workers: evaluation and profiling
                         │
                         └── feedback → next model iteration → best candidate
```

The model service may be vLLM, SGLang, an OpenAI-compatible gateway, or another provider supported by the configured model adapter. SGLang is not required. The model inference GPUs and kernel evaluation GPUs are separate resources; the evaluation runner itself can run on a CPU host.

This Quickstart describes the native **single-workload controller**. Each dataset row is an independent optimization task. A run with one selected workload per operator does not establish correctness across all workloads of that operator, and its speedup is not an official SOL leaderboard score.

## Prerequisites

- Python 3.10 or newer on the runner.
- A reachable model API, model identifier, and API credential if required.
- A running Kernel Sandbox Server configured for your Docker host or Kubernetes cluster, and compatible CPU/GPU sandbox images.
- The checkout includes a 270-task KernelBench dataset. Run from the repository root and use `--subset kernelbench`. For other inputs and dataset licensing, see [Benchmark datasets](dataset/README.md).

You do **not** manually create task sandboxes. Starting evaluation creates a GPU worker pool and individual CPU sessions automatically. Server operators can choose [Docker setup](docs/docker-quickstart.md) or [Kubernetes setup](docs/kubernetes-quickstart.md).

## Container images

| Image | Address |
| --- | --- |
| Agent | `ghcr.io/baidu-baige/koda-agent-clean:20261118-clean` |
| CPU/GPU sandbox | `ghcr.io/baidu-baige/koda-sandbox-clean:20261118-clean` |

If a GHCR image is private, sign in with a GitHub account that has package access and a classic token with `read:packages`.

```bash
read -rp 'GitHub username: ' GHCR_USER
read -rsp 'GitHub package token: ' GHCR_TOKEN
printf '\n'
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
unset GHCR_TOKEN
docker pull ghcr.io/baidu-baige/koda-agent-clean:20261118-clean
docker pull ghcr.io/baidu-baige/koda-sandbox-clean:20261118-clean
```

The agent image contains the code at `/workspace/mini-swe-agent`. The sandbox image supplies the compilation/evaluation environment; it is not the Sandbox Server. The agent image includes the controller-only code and both ckpt140 H/B configurations. For Docker mode, build the Master and sandboxd from the current source checkout. Docker mode copies the built sandboxd into containers and needs no injector image. Kubernetes mode still uses a sandboxd injector image and `imagePullSecrets` for private registry access.

The agent image passes the 320-test regression suite. The Docker backend has also passed B200 model → compile → eval → profile validation, GPU isolation, timeout, worker recovery, Master restart and cleanup checks. Follow the deployment checks for your own environment.

## Install

For the Docker workflow, follow the [Docker Quickstart](docs/docker-quickstart.md): it uses the agent image for Python dependencies and includes a standalone smoke task. The host Python installation below is optional.

```bash
git clone https://github.com/baidu-baige/KODA.git
cd KODA
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m minisweagent.run.benchmarks.kernelbench --help
```

## ckpt140 / Qwen3.6 configuration

The KODA model weights (ckpt140) are hosted on [Hugging Face](https://huggingface.co/Baidu-Baige/KODA-35B).

| Target evaluation GPU | Controller YAML | CUDA compile architecture |
| --- | --- | --- |
| H100 / H20 (Hopper) | [controller_h.yaml](configs/controller_h.yaml) | `9.0` |
| B200 (Blackwell) | [controller_b.yaml](configs/controller_b.yaml) | `10.0a` |

These are two hardware variants of the same Qwen3.6/ckpt140 controller, not different agent modes. Both use temperature `0.6`, top-p `0.95`, top-k `20`, min-p `0`, presence penalty `0`, repetition penalty `1`, thinking enabled, 65,536 maximum output tokens, and a 262,144-token conversation budget. The full-run budgets are 70 rounds, 50 evaluation submissions and 7,200 seconds, with dynamic output sizing and a 10,240-token margin.

The default served-model alias is `ckpt140`; set it to the actual model ID exposed by your endpoint. Configure API/Master URLs and credentials locally. H/B selection refers to the **sandbox evaluation GPU**, independently of the hardware hosting the model service. The prompt uses configured architecture and measured GPU information; target-specific CUTLASS examples are selected by probing the actual device. Remove or adapt the Qwen-specific extra-body fields for providers that do not support them.

## Configure

Copy [the example configuration](configs/controller_h.yaml) to `eval.local.yaml` and replace all `REPLACE_...` values. Keep API credentials out of Git. The controller uses two configuration files: the base supplies prompts and tool definitions; your override supplies model, budget, and environment settings.

- `model.model_name`: `openai/<served-model-name>` for an OpenAI-compatible service. Use the identifier advertised by your provider.
- `model.model_kwargs.api_base`: API base URL, normally ending in `/v1`.
- `model.model_kwargs.api_key`: provider credential, or `EMPTY` for an unauthenticated local service.
- `agent.master_url`: the same reachable Sandbox Server URL passed with `--sandbox-server-url`. CPU sessions must also be able to reach it.
- `environment.cpu_image`: a CPU compilation image compatible with the GPU image's Python, PyTorch, and CUDA ABI.
- `agent.compile_arch`: the target GPU architecture; the example uses H20/Hopper `9.0`. Adapt it for your hardware.

The override clears inherited GPU resource requests so that the server's default GPU cluster/template is used. Configure resource keys explicitly if your server requires cluster routing. Sampling, reasoning, and output-token options depend on the provider; the example avoids provider-specific thinking flags. Set them explicitly for reproducible model comparisons.

```bash
export CONTROLLER_CONFIG=configs/controller_h.yaml  # B200: configs/controller_b.yaml
chmod 600 eval.local.yaml
export SANDBOX_URL='http://YOUR_SANDBOX_SERVER:PORT'
export SANDBOX_IMAGE='ghcr.io/baidu-baige/koda-sandbox-clean:20261118-clean'
export DATASET=kernelbench
```

Do not set `environment.gpu_pool_id` for the automatic pool lifecycle. To reuse an existing pool, supply its ID deliberately; its image must match `--sandbox-image`, and the runner will not delete that pool.

## Run one task

```bash
python -u -m minisweagent.run.benchmarks.kernelbench \
  --config "$CONTROLLER_CONFIG" \
  --config eval.local.yaml \
  --subset "$DATASET" --split train --slice 0:1 \
  --backend cuda \
  --sandbox-server-url "$SANDBOX_URL" --sandbox-image "$SANDBOX_IMAGE" \
  --workers 1 --pool-replicas 1 --output output/smoke --no-wandb \
  -c agent.num_rounds=2 -c agent.max_submit_eval=2 \
  -c agent.wall_time_limit_seconds=600
```

Verify an actual model response, successful candidate compilation, and a GPU evaluation result. For a correct candidate, inspect the profiling result for captured operations and warnings; command success alone does not prove that useful profiling metrics were collected. Two rounds may be insufficient for a model to generate correct code. Distinguish candidate compilation errors from infrastructure failures.

## Run a batch

Use a new output directory and remove the slice. Start with conservative concurrency:

```bash
python -u -m minisweagent.run.benchmarks.kernelbench \
  --config "$CONTROLLER_CONFIG" \
  --config eval.local.yaml \
  --subset "$DATASET" --split train \
  --backend cuda \
  --sandbox-server-url "$SANDBOX_URL" --sandbox-image "$SANDBOX_IMAGE" \
  --workers 4 --pool-replicas 1 --output output/batch --no-wandb
```

`--workers` controls concurrent task threads; `--pool-replicas` requests GPU worker replicas. Docker mode assigns one configured UUID per replica; Kubernetes mode uses the resource allocation in its deployment template. If omitted, pool replicas default to the task worker count. Multiple tasks can share a smaller GPU pool and queue for execution. Pool startup waits for at least one idle worker, not for every requested replica to become ready.

The example allows 70 rounds, 50 submitted evaluations, a 7,200-second wall-time budget and a 262,144-token budget, with up to 65,536 output tokens per request. These are separate limits; the model service must support the requested context/output sizes. Review evaluator tolerances, correctness trials, reference timing, and hardware settings before comparing benchmark results.

## Results and cleanup

Inspect `preds.json`, `eval_result.json`, per-task trajectories, evaluation receipts, and candidate source files under the output directory. Keep the original tool metadata when reporting correctness and performance. Preserve failed workloads in aggregate statistics.

Task cleanup deletes its CPU session; batch finalization deletes a pool created by that run. Reused pools remain under the caller's management. Abrupt termination or network failures can leave resources behind. Use the server API to inspect and remove only the exact session/pool IDs owned by that run. An interrupt cancels pending tasks and may wait for running tasks to finish; do not assume immediate termination.

## Release checks

Before publishing source changes, run `python3 scripts/audit_release.py`. The check reports file locations, never matched credential values, and rejects internal network/registry settings, cloud path names, private IPv4/IPv6 endpoints, credentials embedded in URLs, hardcoded credential literals, known token/private-key formats, missing local imports and retired components. It is also configured as a local pre-commit hook. This is a source-tree check: Git history and container layers must be reviewed separately before making previously private artifacts public.

## Roadmap

- [x] **Single-host Docker sandbox backend:** `cpu_gpu` sessions with UUID allocation, timeout cleanup, recovery and real GPU validation. See the [Docker Quickstart](docs/docker-quickstart.md).
- [ ] Docker `gpu_only` sessions and live GPU pool resizing.

## Acknowledgements and license

KODA builds on [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent), originally developed by Kilian A. Lieret, Carlos E. Jimenez, and contributors. We thank the upstream authors and community for their work.

The upstream code is licensed under the MIT License. Its original copyright notice and license text are preserved in [LICENSE.md](LICENSE.md). Third-party components, model weights, and benchmark datasets remain subject to their respective licenses.
