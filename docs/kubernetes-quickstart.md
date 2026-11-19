# Kubernetes Quickstart: run a GPU kernel optimization experiment

[English](kubernetes-quickstart.md) | [简体中文](kubernetes-quickstart.zh-CN.md)

This guide follows one experiment from setup to saved results. You can place the model service, evaluation runner, and Sandbox Server on different machines. This guide uses the native single-workload CUDA controller and a Kubernetes-backed sandbox.

> **This guide covers the Kubernetes backend.** For a single host without Kubernetes, use the [Docker Quickstart](docker-quickstart.md). Kubernetes remains the default when `backend` is omitted; Docker is selected explicitly in the Server configuration.

## 1. Understand the components

```text
Model service (vLLM / SGLang / compatible provider)
           ↕ model requests and replies
Evaluation runner (Python, CPU is sufficient)
           ↕ HTTP requests
Sandbox Server / Master (Go, started before evaluation)
           ↕ Kubernetes API
           ├── CPU session Pod for each task: workspace and compilation
           └── GPU worker pool: correctness, timing, and profiling
```

The Sandbox Server is **not started by the evaluation command**. Start it once before evaluating; it can serve multiple experiments. When evaluation starts, the runner asks that server to create a GPU pool, waits for an idle worker, and creates individual CPU sessions. The server creates the Kubernetes resources using your configured templates. You do not manually create each task sandbox.

Three kinds of concurrency are independent: model-server request capacity, concurrent agent tasks (`--workers`), and GPU worker replicas (`--pool-replicas`). A worker replica's GPU allocation comes from its Kubernetes template.

## 2. Prepare the resources and machines

You need:

| Resource | Requirement |
| --- | --- |
| Runner | Linux, Python 3.10+, network access to model API and Sandbox Server |
| Master host | Linux, Go 1.23+ and Make to build; kubeconfig for the target cluster |
| Kubernetes | Schedulable CPU/GPU nodes, working NVIDIA drivers and GPU device plugin, permission to create sandbox resources |
| Model service | Model identifier, API base URL, and credential if required |
| GPU business image | CUDA/PyTorch runtime, compiler/toolchain, evaluator, and profiling tools required by the controller |
| CPU business image | Matching compilation environment and kernel tool wrappers; compatible Python/PyTorch/CUDA ABI |
| sandboxd injector image | Contains `/usr/local/bin/sandboxd` plus `sh`, `cp`, and `chmod` |
| Dataset | A runner-compatible dataset, including the reference `code` for every task |

**Images:** use `docker.io/loongsage/koda:agent-20261118-clean` for the runner and `docker.io/loongsage/koda:sandbox-20261118-clean` for both CPU compilation and GPU execution. Configure `imagePullSecrets` if your cluster requires registry authentication. The sandboxd injector remains a separate image built in section 5.2.

**Dataset:** KernelBench is bundled. See [Benchmark datasets](../dataset/README.md) for SOL-ExecBench inputs.

All Kubernetes nodes must be able to pull the images. Set `imagePullSecrets` in the templates if needed. Use a cluster with GPU allocation already functioning; installing Kubernetes, drivers, and the NVIDIA device plugin is outside this application guide.

The model GPUs and evaluation GPUs can be separate machines. If sharing a node, allocate non-overlapping devices through your scheduler. Do not assume that four GPU worker replicas mean four cards unless the template requests one GPU per replica.

## 3. Install the code on the runner

```bash
git clone https://github.com/baidu-baige/KODA.git
cd KODA
export KODA_ROOT="$PWD"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m minisweagent.run.benchmarks.kernelbench --help
```

Keep this shell for the runner steps. A private repository requires your own GitHub authentication. If the Master runs elsewhere, clone the same code revision there too.

## 4. Prepare and test the model service

Use an existing compatible API. The serving engine does not have to be SGLang. For an OpenAI-compatible API, record:

```bash
export MODEL_API_BASE='https://YOUR_MODEL_API/v1'
export MODEL_NAME='YOUR_SERVED_MODEL_NAME'
read -rsp 'Model API key (EMPTY if not required): ' MODEL_API_KEY
printf '\n'
export MODEL_API_KEY
```

Test actual generation without putting the credential in command-line arguments:

```bash
python - <<'PY'
import os
from openai import OpenAI
client = OpenAI(base_url=os.environ['MODEL_API_BASE'], api_key=os.environ['MODEL_API_KEY'])
r = client.chat.completions.create(
    model=os.environ['MODEL_NAME'],
    messages=[{'role': 'user', 'content': 'Reply with a short greeting.'}],
    max_tokens=256,
)
print('generation finished:', r.choices[0].finish_reason)
PY
```

A model listing alone does not verify generation. For a reasoning model that uses the entire test budget before producing text, inspect the reasoning response or adjust this test according to the provider. Confirm the service supports the context/output budgets used later. Configure thinking and sampling explicitly for the chosen model; do not copy another provider's unsupported flags.

## 5. Deploy the Sandbox Server — first-time setup

Run this section on the **Master host**, not inside a task sandbox. The Master host must reach the Kubernetes API. Runner and CPU session Pods must reach the Master's advertised HTTP endpoint. `localhost` is only appropriate when every caller genuinely shares that network namespace.

### 5.1 Check cluster access

```bash
export KUBE_CONFIG='/absolute/path/to/your/kubeconfig'
export SANDBOX_NS='your-sandbox-namespace'
kubectl --kubeconfig "$KUBE_CONFIG" -n "$SANDBOX_NS" get pods
kubectl --kubeconfig "$KUBE_CONFIG" -n "$SANDBOX_NS" auth can-i create pods
kubectl --kubeconfig "$KUBE_CONFIG" -n "$SANDBOX_NS" auth can-i create pods/exec
kubectl --kubeconfig "$KUBE_CONFIG" -n "$SANDBOX_NS" auth can-i create deployments.apps
```

The namespace must already exist. The Master identity also needs resource read/list/watch/update/delete permissions for the resources it manages, including Pods, Deployments and their ReplicaSets. Use your cluster administrator's scoped credentials; do not put kubeconfig contents in Git.

### 5.2 Build the Master and sandboxd

```bash
cd "$KODA_ROOT/src/kernel_sandbox"
make build
```

This creates `bin/kernel-sandbox-master`, `bin/sandboxd`, and the pool monitor. Build on a platform compatible with the execution nodes.

The injector supplies only the sandbox supervisor binary, not the CUDA/evaluation environment. The checked-in `Dockerfile.sandboxd` uses a public shell-capable base. You can build the injector as follows:

```bash
cat > /tmp/Dockerfile.koda-sandboxd <<'DOCKER'
FROM alpine:3.20
COPY bin/sandboxd /usr/local/bin/sandboxd
ENTRYPOINT ["/usr/local/bin/sandboxd"]
DOCKER
export SANDBOXD_IMAGE='YOUR_REGISTRY/koda-sandboxd:YOUR_VERSION'
docker build -f /tmp/Dockerfile.koda-sandboxd -t "$SANDBOXD_IMAGE" .
docker push "$SANDBOXD_IMAGE"
```

The Go Makefile builds with `CGO_ENABLED=0`. The Master injects the binary through an initContainer and an `emptyDir`, then runs it as the business container's entrypoint. Do not add a command or args to the business container in your Pod/Deployment templates: the server rejects such templates.

### 5.3 Adapt the CPU and GPU templates

Create a private deployment directory; preserve the checked-in examples:

```bash
mkdir -p deploy/local/templates
cp deploy/templates/cpu-pod-template.yaml deploy/local/templates/cpu.yaml
cp deploy/templates/gpu-deployment-template.yaml deploy/local/templates/gpu.yaml
```

Edit both copies before starting the server. The supplied templates use standard NVIDIA GPU resource names. Configure these fields for your cluster:

| Field | What to configure |
| --- | --- |
| `schedulerName`, queue annotations | Your scheduler; remove internal queue settings on a standard cluster |
| `nodeSelector`, affinity, tolerations | Your actual nodes and taints; remove internal labels and hostnames |
| GPU resources | Your device-plugin key and allocation; for a standard NVIDIA setup, requests/limits commonly use `nvidia.com/gpu: "1"` |
| `NVIDIA_VISIBLE_DEVICES` | Remove hardcoded physical device lists; let the device plugin allocate devices |
| CPU/memory resources | Values that fit your cluster; internal templates may request very large resources |
| Volumes/PVCs and mount paths | Storage available in your namespace; replace internal PVC names and cache paths |
| lifecycle hooks | Remove the internal Hugging Face cache symlink or replace it with your cache location |
| Security context | Permissions required by the chosen image, CUDA execution and NSYS/NCU profiling |
| `imagePullSecrets` | Registry authentication when required |

Keep exactly one business container in each template. Keep CPU and GPU roles separate: the CPU Pod does not request a GPU. GPU worker templates must request the intended GPU allocation. Do not mount all task workspaces onto one shared writable directory: tasks use identical workspace paths inside separate CPU Pods. Preserve the storage paths expected by your business image and its tool wrappers; shared caches and task-local workspaces are different things.

These are templates consumed by the Master, **not resources you should deploy with `kubectl apply`**. The Master fills in names, labels, image, replica count and sandboxd injection when evaluation requests resources.

### 5.4 Write the Master configuration

Set an IP/DNS name reachable from both the runner and CPU Pods:

```bash
export SANDBOX_URL='http://YOUR_MASTER_HOST:18766'
python3 - <<'PY'
import os
from pathlib import Path
import yaml
local = Path('deploy/local').resolve()
config = {
    'master_id': 'koda',
    'server': {'listen_address': '0.0.0.0:18766', 'advertised_endpoint': os.environ['SANDBOX_URL']},
    'cpu_cluster': {
        'kubeconfig': str(Path(os.environ['KUBE_CONFIG']).resolve()),
        'namespace': os.environ['SANDBOX_NS'],
        'pod_template_file': str(local / 'templates/cpu.yaml'),
        'pod_name_prefix': 'koda-cpu',
    },
    'gpu_pool_clusters': [{
        'name': 'gpu',
        'kubeconfig': str(Path(os.environ['KUBE_CONFIG']).resolve()),
        'namespace': os.environ['SANDBOX_NS'],
        'deployment_template_file': str(local / 'templates/gpu.yaml'),
        'pool_name_prefix': 'koda-gpu',
        'resource_names': ['nvidia.com/gpu'],
    }],
    'sandboxd': {'image': os.environ['SANDBOXD_IMAGE']},
    'execution': {'max_attempts': 5, 'max_stdin_mb': 4, 'max_output_mb': 1},
    'gc': {'interval_seconds': 60, 'session_idle_timeout_seconds': 5400, 'gpu_pool_idle_timeout_seconds': 5400},
}
path = local / 'master.yaml'
path.write_text(yaml.safe_dump(config, sort_keys=False))
path.chmod(0o600)
print(path)
PY
bash deploy/scripts/master.sh start deploy/local/master.yaml
bash deploy/scripts/master.sh status deploy/local/master.yaml
curl --fail --silent --show-error "$SANDBOX_URL/healthz"
curl --fail --silent --show-error "$SANDBOX_URL/v1/gpu_pool"
```

Use the resource name matching your GPU template if it is not `nvidia.com/gpu`. Relative paths in the YAML resolve against the YAML's directory; the example uses absolute paths to avoid ambiguity. If Python/PyYAML is not installed on the Master host, install them or write the equivalent YAML manually.

The health endpoint checks that the server responds. It does not prove a GPU can run the evaluator. No GPU pool is expected yet; an empty pool list is normal. Logs are under `src/kernel_sandbox/.run/koda/master.log` unless `KERNEL_SANDBOX_RUN_DIR` overrides the location. Keep the server on a trusted network: access to it permits executing sandbox commands.

## 6. Prepare the dataset on the runner

From the repository root, select the bundled 270-task KernelBench dataset. See [Benchmark datasets](../dataset/README.md) for other inputs.

```bash
cd "$KODA_ROOT"
source .venv/bin/activate
export DATASET=kernelbench
```

Use a dataset prepared for this controller, including its task reference/shape conventions and correctness metadata. A raw upstream SOL dataset is not automatically equivalent. A 202-row dataset with one workload per operator means 202 single-workload tasks; it does not evaluate every official workload for each operator.

## 7. Configure the agent

On the runner, set `SANDBOX_URL` to the already-started Master. CPU session Pods must be able to reach that same URL. Use images compatible with the current controller:

```bash
export SANDBOX_URL='http://YOUR_MASTER_HOST:18766'
export SANDBOX_IMAGE='docker.io/loongsage/koda:sandbox-20261118-clean'
export CPU_IMAGE="$SANDBOX_IMAGE"
export RUN_TAG="koda-$(date +%Y%m%d-%H%M%S)"
export CONTROLLER_CONFIG=configs/controller_h.yaml  # B200: configs/controller_b.yaml
cp "$CONTROLLER_CONFIG" eval.local.yaml
python - <<'PY'
import os
from pathlib import Path
import yaml
p = Path('eval.local.yaml')
c = yaml.safe_load(p.read_text())
c['agent']['master_url'] = os.environ['SANDBOX_URL']
c['model']['model_name'] = 'openai/' + os.environ['MODEL_NAME']
c['model']['model_kwargs'].update(api_base=os.environ['MODEL_API_BASE'], api_key=os.environ['MODEL_API_KEY'])
c['environment'].update(cpu_image=os.environ['CPU_IMAGE'], session_prefix=os.environ['RUN_TAG'])
# Empty mappings use the Master default GPU template and its resource requests.
c['environment']['resource_request'] = {}
c['environment']['resource_limit'] = {}
p.write_text(yaml.safe_dump(c, sort_keys=False))
p.chmod(0o600)
PY
```

The base `configs/controller_h.yaml` provides prompts and tool definitions. `eval.local.yaml` overrides model/environment/budgets. It is ignored by Git and may contain credentials. The example targets an OpenAI-compatible endpoint; for another protocol, configure its supported model adapter instead.

Before running, review `agent.compile_arch`: `9.0` targets Hopper/H20. Select the value accepted by your toolchain for your actual GPU; do not use a Hopper binary on a different architecture. The example's 262,144-token budget and 65,536-token output cap require model-server support. Adapt them deliberately for a smaller-context model and record the change.

Leave `environment.gpu_pool_id` unset for automatic pool creation. Set it only when deliberately reusing a pool; the GPU image must match and the runner will not delete the reused pool.

## 8. Run one task and observe automatic deployment

```bash
python -u -m minisweagent.run.benchmarks.kernelbench \
  --config "$CONTROLLER_CONFIG" --config eval.local.yaml \
  --subset "$DATASET" --split train --slice 0:1 \
  --backend cuda \
  --sandbox-server-url "$SANDBOX_URL" --sandbox-image "$SANDBOX_IMAGE" \
  --workers 1 --pool-replicas 1 --output "output/$RUN_TAG-smoke" --no-wandb \
  -c agent.num_rounds=2 -c agent.max_submit_eval=2 \
  -c agent.wall_time_limit_seconds=600
```

In a second terminal on a host with cluster credentials:

```bash
kubectl --kubeconfig "$KUBE_CONFIG" -n "$SANDBOX_NS" get pods,deployments -w
```

Expected sequence:

1. Runner sends `POST /v1/gpu_pool`; Master creates a GPU Deployment and injects sandboxd.
2. Runner polls until at least one GPU worker is idle. This does not wait for all replicas.
3. Runner sends `POST /v1/session`; Master creates a CPU Pod and waits for readiness.
4. Runner initializes the task workspace and requests model-generated candidate code.
5. The controller compiles on CPU, requests GPU evaluation, and profiles eligible candidates.
6. Tool feedback is returned to the model; subsequent iterations revise the candidate.
7. Cleanup deletes the task session and, on batch finalization, the pool owned by this run.

Two rounds are a smoke budget, not a guarantee that a model writes correct code. A correct result establishes more than API reachability. Useful profiling additionally requires captured operations/metrics: exit code zero with empty profiling output is not a successful profiling validation. Preserve the logs if compilation fails; distinguish candidate errors from image/ABI/network errors.

## 9. Inspect the result

```bash
export OUT="output/$RUN_TAG-smoke"
python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['OUT']) / 'preds.json'
assert p.exists(), 'No completed task summary; inspect runner logs first'
d = json.loads(p.read_text())
print('finished tasks:', len(d))
print('solved:', sum(x.get('solved') is True for x in d.values()))
PY
```

| Output | Meaning |
| --- | --- |
| `preds.json` | Per-task final result and selected candidate summary |
| `eval_result.json` | Batch evaluation summary |
| `<task>/trajectory.json` | Model messages, tool observations and exit details |
| `<task>/trajectory.readable.yaml` | Readable trajectory |
| `<task>/eval_turn_N.json` | Per-round evaluation receipt when emitted by the controller |
| `<task>/kernel_turn_N.py`, `.cu` | Candidate source for that round when emitted |

Do not treat process exit, Pod Ready, or HTTP 200 as task correctness. Review `solved`, the raw evaluation receipts, failed trials and profiling output. The evaluator's tolerances, reference timing, correctness/performance trials, image versions and GPU determine comparability. Local speedup is not the official SOL leaderboard score.

## 10. Run a batch

After the smoke test succeeds, use a new output directory and remove `--slice`:

```bash
python -u -m minisweagent.run.benchmarks.kernelbench \
  --config "$CONTROLLER_CONFIG" --config eval.local.yaml \
  --subset "$DATASET" --split train \
  --backend cuda \
  --sandbox-server-url "$SANDBOX_URL" --sandbox-image "$SANDBOX_IMAGE" \
  --workers 4 --pool-replicas 2 --output "output/$RUN_TAG-batch" --no-wandb
```

This runs up to four agents sharing two GPU worker replicas. Requests queue when both workers are busy. If `--pool-replicas` is omitted, replica count defaults to the task worker count (bounded by the selected task count). Model generation and GPU execution queues are separate bottlenecks.

Default example budgets are 70 rounds, 50 submitted evaluations, 7,200 seconds, and 262,144 tokens; they are independent stopping conditions. Changing a configuration file after launch does not hot-update the running process. Retain original attempts when restarting and do not assume a native run resumes unfinished conversations automatically.

## 11. Stop and clean up

For a foreground run, interrupt once and let active tasks reach cleanup; the runner may wait for running tasks while cancelling pending tasks. Normal per-task cleanup deletes the CPU session. Batch finalization deletes only the GPU pool that this invocation created. A reused pool remains.

After an abrupt exit, inspect `GET /v1/session` and `GET /v1/gpu_pool`, identify exact resources from this run's logs/session prefix, and delete only those session IDs. Delete the owned GPU pool after its sessions are gone. See [API reference](../src/kernel_sandbox/docs/api.md) for response formats. Avoid broad namespace/prefix deletion when resources may be shared.

Stop your dedicated Master only after its active work has finished:

```bash
cd "$KODA_ROOT/src/kernel_sandbox"
bash deploy/scripts/master.sh stop deploy/local/master.yaml
```

Stopping the Master is separate from stopping one evaluation. Do not stop a shared server to terminate a single run. Keep result directories and the exact code/config/image versions for reproducibility.

## 12. Troubleshooting

| Symptom | Check / next action |
| --- | --- |
| Cannot connect to Master | Server process/log, advertised address, port, routing from runner and CPU Pods; `localhost` may refer to the wrong host |
| Server healthy, pool never ready | Kubernetes events, resource availability, device-plugin key, scheduling rules, image pulls, injector image |
| CPU session never ready | CPU quota, PVC status, image pull failures, template constraints and events |
| Model listing works but generation fails | Actual model identifier, provider credentials, supported request parameters, context/output limits |
| CUDA compilation fails | Compiler stderr, candidate code versus missing dependencies, correct target architecture |
| Compiled extension cannot load | CPU/GPU Python, PyTorch and CUDA ABI compatibility; missing runtime libraries |
| Eval works, profile is empty/fails | NSYS/NCU installation, driver profiling permissions, captured kernels/NVTX/CUDA graph visibility |
| CPU tools cannot request GPU work | CPU Pod → advertised Master connectivity and matching `agent.master_url` |
| Pods stuck Pending | `kubectl describe pod` and namespace events; do not infer readiness from requested replica count |
| Restart appears to repeat work | Preserve separate attempts and check this controller's existing-result handling before resuming |

## Completion checklist

An environment is ready when one real task shows: an actual model response; automatic GPU pool and CPU session creation; a compiled candidate; a successful GPU correctness/performance evaluation; useful profile data for a correct candidate; feedback used in a subsequent model iteration; saved trajectory/source/receipts; and verified resource cleanup. The commands and configuration relationships in this guide have been checked against the source. The generic external deployment path has not yet been validated end to end with publicly published business images and data.
