# Docker Quickstart: run your first kernel optimization task

[English](docker-quickstart.md) | [简体中文](docker-quickstart.zh-CN.md)

This guide runs one small task from a fresh checkout using a local Docker sandbox. You do not need Kubernetes or a separate sandboxd injector image. The model API can run on another machine or be provided by a compatible service.

```text
Existing model API ↔ agent container
                         ↓ HTTP
                   Sandbox Master on the GPU host
                         ↓ local Docker daemon
                   CPU task container → GPU worker container
                         compile         eval / profile
```

The Master must be started first. Starting the agent then automatically creates its GPU pool and CPU session. The agent image supplies Python dependencies; you do not need to install PyTorch or the agent's Python packages on the host.

## 1. Check the host

Use a Linux x86_64 host with Git, Python 3, Docker, NVIDIA drivers, NVIDIA Container Toolkit, and Go 1.23 or newer. The published images below target `linux/amd64`; other host architectures need separately built compatible images. The account running the Master must be able to execute Docker commands. This guide supports a local Docker daemon, not remote Docker contexts. Reserve at least one GPU for the sandbox; do not select GPUs used by your model server or another workload.

```bash
git --version
python3 --version
docker version
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
go version
```

If Docker or the GPU toolkit is missing, follow the official [Docker Engine installation](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) guides for your distribution, then repeat these checks. The toolkit guide also links driver installation instructions. If only Go is missing, install it from the [official Go distribution](https://go.dev/doc/install). For example, the following installs the Linux AMD64 Go 1.23.6 toolchain in your user directory, with its published SHA256:

```bash
mkdir -p "$HOME/.local/koda-go-1.23.6"
curl -fL https://go.dev/dl/go1.23.6.linux-amd64.tar.gz -o /tmp/koda-go.tar.gz
printf '%s\n' '9379441ea310de000f33a4dc767bd966e72ab2826270e038e78b2c53c2e7802d  /tmp/koda-go.tar.gz' | sha256sum -c -
tar -xzf /tmp/koda-go.tar.gz -C "$HOME/.local/koda-go-1.23.6"
export PATH="$HOME/.local/koda-go-1.23.6/go/bin:$PATH"
go version
```

If building your own images on another architecture, use its matching official Go archive. Go module downloads require network access during the initial build. On restricted networks, see section 11 before retrying failed downloads.

## 2. Clone the repository and choose a fresh run directory

```bash
git clone https://github.com/baidu-baige/KODA.git
cd KODA
export KODA_ROOT="$PWD"
export KODA_RUN_DIR="$HOME/koda-runs/smoke-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$KODA_RUN_DIR"
chmod 700 "$KODA_RUN_DIR"
git rev-parse HEAD
```

While the repository is private, Git needs an account/token with repository read access. Authenticate using your existing Git credential helper; do not put tokens in clone URLs.

Run the remaining commands in the same Bash shell so the exported variables and helper function remain available. The Master is started in the background; a second terminal is not required.

## 3. Pull the images

The published KODA images are available from Docker Hub. You can pull them without signing in.

```bash
export AGENT_IMAGE=docker.io/loongsage/koda:agent-20261118-clean
export SANDBOX_IMAGE=docker.io/loongsage/koda:sandbox-20261118-clean
docker pull "$AGENT_IMAGE"
docker pull "$SANDBOX_IMAGE"
```

Validation used Docker Engine 29.1.3 with its default security configuration. An older Docker 20.10 installation rejected Python thread creation (`can't start new thread`) under its default seccomp profile. If you encounter this error before evaluation starts, update Docker Engine and its container runtime/security packages before retrying.

The sandbox image provides the compiler and evaluation/profiling tools. The agent image provides Python dependencies. The agent image includes the controller-only code and ckpt140 H/B configurations. Build the Master and sandboxd from this checkout in the next step.

For reproducibility, the published manifests have these digests:

| Image | Digest |
| --- | --- |
| Agent | `sha256:6e9ec2d4d424705f0988a0a10fed5dcbfbeabbe439263880ca7bfcac047a5b9b` |
| Sandbox | `sha256:69e69e4f352c0c5308acbe60b19b60938bc56785ce2e7babe5ca2c4354de67cb` |

## 4. Build the Server and select a GPU

```bash
cd "$KODA_ROOT/src/kernel_sandbox"
mkdir -p bin
CGO_ENABLED=0 go build -buildvcs=false -o bin/sandboxd ./cmd/sandboxd
CGO_ENABLED=0 go build -buildvcs=false -o bin/kernel-sandbox-master ./cmd/kernel-sandbox-master
cd "$KODA_ROOT"

export GPU_UUID='GPU-REPLACE-WITH-AN-ID-FROM-NVIDIA-SMI'
export CONTROLLER_CONFIG='configs/controller_b.yaml'
export CUDA_ARCH='10.0a'  # B200; use 9.0 for H100/H20 with this toolchain.
export MASTER_IP='YOUR_GPU_HOST_IP'
export MASTER_PORT=18766
export SANDBOX_URL="http://$MASTER_IP:$MASTER_PORT"
export MASTER_ID="koda-smoke-$(date +%s)"
```

The B200 example selects `configs/controller_b.yaml`. For H100/H20, select `configs/controller_h.yaml` and set `CUDA_ARCH=9.0`. Both presets target ckpt140/Qwen3.6; see the README for sampling and budget details.

Use a complete UUID, not a numeric device index. Choose a reachable host IP, not `localhost`: CPU containers use Docker bridge networking in this guide and must reach the Master on the host. Ensure the port is free and reachable from those containers and from the agent.

Validate the actual Docker GPU assignment before starting the Server:

```bash
docker run --rm --gpus "device=$GPU_UUID" \
  --entrypoint nvidia-smi "$SANDBOX_IMAGE" \
  --query-gpu=uuid,name --format=csv,noheader
```

The output must contain only the selected GPU. If Docker reports that it cannot select a GPU device driver, fix NVIDIA Container Toolkit/runtime configuration first. Do not proceed by exposing all GPUs or enabling privileged mode.

## 5. Define the agent Python helper

The following helper runs the current checkout inside the agent image. Source is mounted read-only; generated data, configuration and outputs are written under the new run directory. The agent container itself is not assigned a GPU.

```bash
koda_python() {
  docker run --rm -i --network host \
    -e KODA_HOST_UID="$(id -u)" -e KODA_HOST_GID="$(id -g)" \
    -e HOME=/tmp/koda-home -e NVIDIA_VISIBLE_DEVICES=void \
    -e HF_HOME=/run/koda/cache/huggingface \
    -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
    -e PYTHONPATH=/opt/koda/src \
    -e KODA_ROOT -e KODA_RUN_DIR -e GPU_UUID -e CUDA_ARCH -e CONTROLLER_CONFIG \
    -e MASTER_IP -e MASTER_PORT -e MASTER_ID -e SANDBOX_URL \
    -e SANDBOX_IMAGE -e MODEL_API_BASE -e MODEL_NAME -e MODEL_API_KEY \
    -e MODEL_EXTRA_BODY \
    -v "$KODA_ROOT:/opt/koda:ro" -v "$KODA_RUN_DIR:/run/koda" \
    -w /opt/koda --entrypoint /opt/koda-agent-venv/bin/python "$AGENT_IMAGE" "$@"
}
koda_python -m minisweagent.run.benchmarks.kernelbench --help
```

The agent image includes Python under `/opt/koda-agent-venv`. This helper uses the default container user; generated files may be root-owned. The Master config below is explicitly assigned back to your host UID/GID.

The helper explicitly selects the Python environment bundled in the published image; using its system `python3` can miss dependencies. It also disables GPU exposure for the CPU-only agent. The helper uses host networking only for the agent's API connections. CPU/GPU sandbox containers are created separately by the Master using its configured Docker network.

## 6. Generate and start the Docker Master

```bash
koda_python - <<'PY'
import os
from pathlib import Path
import yaml
c = yaml.safe_load(Path('src/kernel_sandbox/deploy/docker/master.yaml.tmpl').read_text())
c['master_id'] = os.environ['MASTER_ID']
c['server'].update(listen_address='0.0.0.0:' + os.environ['MASTER_PORT'], advertised_endpoint=os.environ['SANDBOX_URL'])
c['docker'].update(
    state_dir=os.environ['KODA_RUN_DIR'] + '/state',
    sandboxd_binary=os.environ['KODA_ROOT'] + '/src/kernel_sandbox/bin/sandboxd',
    gpu_devices=[os.environ['GPU_UUID']],
    network='bridge',
)
p = Path('/run/koda/master.yaml')
p.write_text(yaml.safe_dump(c, sort_keys=False))
p.chmod(0o600)
os.chown(p, int(os.environ['KODA_HOST_UID']), int(os.environ['KODA_HOST_GID']))
PY

"$KODA_ROOT/src/kernel_sandbox/bin/kernel-sandbox-master" \
  --config "$KODA_RUN_DIR/master.yaml" > "$KODA_RUN_DIR/master.log" 2>&1 &
MASTER_PID=$!
printf '%s\n' "$MASTER_PID" > "$KODA_RUN_DIR/master.pid"

for attempt in $(seq 1 30); do
  curl --noproxy "$MASTER_IP" --fail --silent "$SANDBOX_URL/healthz" && break
  sleep 1
done
curl --noproxy "$MASTER_IP" --fail --silent --show-error "$SANDBOX_URL/healthz"
curl --noproxy "$MASTER_IP" --fail --silent --show-error "$SANDBOX_URL/v1/gpu_pool"
```

A healthy server returns `{"status":"ok"}`; the initial GPU pool list should be empty. If startup fails, read `master.log` before retrying. The Master must remain running throughout evaluation.

The template limits each container to 8 CPU cores and 32 GiB of memory. Adjust these for larger tasks. It grants GPU containers `SYS_ADMIN` for the tested Nsight setup; use only the capabilities your profiling setup requires. It also preserves a narrowly scoped allowlist for the evaluator's documented persistent helper. It does not use privileged containers, bind the Docker socket into task containers, or expose all GPUs.

Keep the Master API on a trusted network. Each Master requires a unique ID and durable state directory. The state is locked and bound to the daemon/Master identity; keep it while its resources exist.

## 7. Configure and check the model API

Use an existing vLLM, SGLang, or compatible OpenAI-style API:

```bash
export MODEL_API_BASE='http://YOUR_MODEL_HOST:PORT/v1'
export MODEL_NAME='YOUR_SERVED_MODEL_NAME'
read -rsp 'Model API key (EMPTY if not required): ' MODEL_API_KEY
printf '\n'
export MODEL_API_KEY
export MODEL_EXTRA_BODY='{}'

koda_python - <<'PY'
import os
from openai import OpenAI
client = OpenAI(base_url=os.environ['MODEL_API_BASE'], api_key=os.environ['MODEL_API_KEY'])
r = client.chat.completions.create(
    model=os.environ['MODEL_NAME'],
    messages=[{'role': 'user', 'content': 'Reply with a short greeting.'}],
    max_tokens=256,
)
print('Model API response received; finish_reason:', r.choices[0].finish_reason)
PY
```

A reasoning model may consume this small diagnostic budget before producing final text. This tests API access; the actual kernel run below validates generation and tool feedback. The example controller uses a 262,144-token budget and a 65,536-token output cap; adjust them if your service supports less. Provider-specific thinking/sampling options belong in `MODEL_EXTRA_BODY` only when supported, for example `{"chat_template_kwargs":{"enable_thinking":true}}` for a compatible service.

## 8. Create the included smoke dataset and agent configuration

No SOL download or internal storage path is required for this smoke test. The repository includes a single GELU reference task. The preparation script reads its source without importing PyTorch on the host:

```bash
koda_python examples/docker_smoke/prepare_dataset.py --output /run/koda/data

koda_python - <<'PY'
import json, os
from pathlib import Path
import yaml
c = yaml.safe_load(Path(os.environ['CONTROLLER_CONFIG']).read_text())
c['agent'].update(
    master_url=os.environ['SANDBOX_URL'], compile_arch=os.environ['CUDA_ARCH'],
    num_rounds=2, max_submit_eval=2, wall_time_limit_seconds=900,
    token_limit=262144,
)
c['model']['model_name'] = 'openai/' + os.environ['MODEL_NAME']
c['model']['model_kwargs'].update(
    api_base=os.environ['MODEL_API_BASE'], api_key=os.environ['MODEL_API_KEY'],
    extra_body=(c['model']['model_kwargs'].get('extra_body') or {}) | json.loads(os.environ.get('MODEL_EXTRA_BODY', '{}')),
)
c['environment'].update(
    cpu_image=os.environ['SANDBOX_IMAGE'], session_prefix=os.environ['MASTER_ID'],
    resource_request={}, resource_limit={},
)
c['environment'].pop('gpu_pool_id', None)
p = Path('/run/koda/eval.yaml')
p.write_text(yaml.safe_dump(c, sort_keys=False))
p.chmod(0o600)
PY
```

Writing a complete configuration ensures inherited Kubernetes resource maps are actually cleared. Docker rejects nonempty Kubernetes resource/scheduling maps. Leave `gpu_pool_id` unset so this run owns and cleans up its pool.

## 9. Run the task and verify results

```bash
set -o pipefail
koda_python -u -m minisweagent.run.benchmarks.kernelbench \
  --config /run/koda/eval.yaml --subset /run/koda/data --split train \
  --backend cuda \
  --sandbox-server-url "$SANDBOX_URL" --sandbox-image "$SANDBOX_IMAGE" \
  --workers 1 --pool-replicas 1 --output /run/koda/output --no-wandb \
  2>&1 | tee "$KODA_RUN_DIR/agent.log"

koda_python - <<'PY'
import json
from pathlib import Path
p = Path('/run/koda/output/preds.json')
assert p.exists(), 'No result summary; inspect agent.log and master.log'
for name, row in json.loads(p.read_text()).items():
    print(name, {k: row.get(k) for k in ['solved', 'correctness', 'num_turns', 'exit_status']})
PY
```

During the run, the Master creates a GPU worker container, waits for it to become idle, creates a CPU session container, and executes compile/eval/profile through sandboxd. The CPU container calls the Master back for GPU work. Source and artifacts are transferred by the existing tools; no shared task-workspace mount is needed.

Check `output/<task>/trajectory.json` or `trajectory.readable.yaml` for:

- A real model reply and successful compilation.
- An evaluation receipt with `correctness: true`.
- Profiling with captured kernels/operations (for this task, nonzero `num_ops`).
- A subsequent model iteration using the feedback.

`RoundLimitReached` is expected for a two-round smoke budget. It is not an error if a valid candidate was saved. Two rounds cannot guarantee correctness for every model: candidate errors and infrastructure errors must be distinguished. A process exit or an HTTP 200 alone is not proof that the complete toolchain worked.

After this succeeds, use your prepared benchmark dataset and a new run directory. Increase `--workers` and `--pool-replicas` deliberately: Docker assigns one configured GPU UUID per pool replica, and task count may exceed GPU count. A full benchmark dataset is separate from this minimal toolchain test.

## 10. Verify cleanup and stop the Server

Normal evaluation cleanup deletes its CPU session and owned GPU pool. Confirm that this dedicated Master's lists are empty:

```bash
curl --noproxy "$MASTER_IP" --fail --silent --show-error "$SANDBOX_URL/v1/session"
curl --noproxy "$MASTER_IP" --fail --silent --show-error "$SANDBOX_URL/v1/gpu_pool"
docker ps -a --filter "label=io.koda.sandbox.master=$MASTER_ID"
```

If an interrupted run left resources, delete its exact session IDs through `/v1/session/<id>`, then its pool through `/v1/gpu_pool/<id>`. Do not delete other users' resources or remove state files to force a restart.

Stop only the Server started for this run, after checking its process identity:

```bash
python3 - <<'PY'
import os, signal
from pathlib import Path
r = Path(os.environ['KODA_RUN_DIR'])
pid = int((r / 'master.pid').read_text())
p = Path('/proc') / str(pid)
if p.exists():
    args = p.joinpath('cmdline').read_bytes().split(b'\0')
    assert (os.environ['KODA_ROOT'] + '/src/kernel_sandbox/bin/kernel-sandbox-master').encode() in args
    assert str(r / 'master.yaml').encode() in args
    os.kill(pid, signal.SIGTERM)
    print('Stopped this run\'s Master')
PY
unset MODEL_API_KEY
```

Retain the logs and results. Protect `eval.yaml`: it may contain your model credential. Stopping the Master alone leaves containers for restart recovery; it is not equivalent to deleting a pool.

## 11. Restricted networks and common setup failures

Different tools use different trust/proxy configuration. Apply your network's approved settings to the tool that failed:

| Tool | Proxy / certificate configuration |
| --- | --- |
| curl | Shell proxy variables and `CURL_CA_BUNDLE` or `--cacert` |
| Git over HTTPS | Shell proxy variables and `GIT_SSL_CAINFO` |
| Go / crane | Shell proxy variables and `SSL_CERT_FILE` |
| Docker login/pull | Docker daemon proxy and registry CA configuration; shell exports alone may not apply |

If the model API needs a proxy inside the agent container, explicitly pass the relevant proxy variables to the helper's `docker run` command. Include the Master address in `NO_PROXY` so sandbox requests stay on the local network.

- **Docker login/pull times out but curl works:** Docker daemon network settings are separate from shell proxy variables. Ask your administrator to configure daemon proxy/CA settings, or use a userspace registry client with your approved proxy and CA, then load the downloaded archive into Docker.
- **TLS verification fails:** use the CA bundle approved for your network. Do not disable certificate verification.
- **Go module/PyPI downloads fail:** distinguish connection/CA errors from missing packages. This guide's containerized agent path avoids host pip installation entirely.
- **GPU container cannot start:** check the single-UUID Docker test in section 4 and NVIDIA runtime setup.
- **CPU session not ready:** check its access to `SANDBOX_URL`; host loopback is not reachable as host loopback from a bridge container. Check `master.log` before retrying.
- **Compile succeeds but extension load fails:** CPU/GPU images must match Python, PyTorch and CUDA ABI.
- **Eval succeeds but profile fails/has zero operations:** check Nsight availability, driver profiling permissions and actual kernel capture; do not count an empty profile as success.

For the userspace registry-client alternative, install `crane` from the official `google/go-containerregistry` releases and verify the archive checksum. The following is the tested Linux AMD64 v0.22.1 installation; use the appropriate asset on other architectures. Set your approved shell proxy/CA variables before these downloads if needed.

```bash
mkdir -p "$KODA_RUN_DIR/tools"
(
  cd "$KODA_RUN_DIR/tools"
  curl -fL https://github.com/google/go-containerregistry/releases/download/v0.22.1/go-containerregistry_Linux_x86_64.tar.gz -o go-containerregistry_Linux_x86_64.tar.gz
  curl -fL https://github.com/google/go-containerregistry/releases/download/v0.22.1/checksums.txt -o checksums.txt
  python3 - <<'PY'
import hashlib, tarfile
from pathlib import Path
name = 'go-containerregistry_Linux_x86_64.tar.gz'
expected = next(line.split()[0] for line in Path('checksums.txt').read_text().splitlines() if line.split()[-1] == name)
assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
with tarfile.open(name) as archive:
    member = archive.getmember('crane')
    assert member.isfile()
    Path('crane').write_bytes(archive.extractfile(member).read())
Path('crane').chmod(0o755)
PY
)
export PATH="$KODA_RUN_DIR/tools:$PATH"
```

Then pull with the same shell proxy/CA settings:

```bash
crane digest "$SANDBOX_IMAGE"
crane pull --cache_path "$KODA_RUN_DIR/registry-cache" "$SANDBOX_IMAGE" "$KODA_RUN_DIR/sandbox.tar"
docker load -i "$KODA_RUN_DIR/sandbox.tar"
crane pull --cache_path "$KODA_RUN_DIR/registry-cache" "$AGENT_IMAGE" "$KODA_RUN_DIR/agent.tar"
docker load -i "$KODA_RUN_DIR/agent.tar"
```

If your proxy reports an HTTP/2 `PROTOCOL_ERROR`, retry this userspace download with `export GODEBUG=http2client=0`. Keep verified complete layers in the cache. If a transfer ends with `unexpected EOF`, a partially written cache file may remain: remove only the incomplete layer entry identified in the error before retrying, or use a fresh cache directory. This changes the HTTP protocol, not TLS certificate verification.

This preserves TLS verification and does not require changing a shared Docker daemon. Allow enough disk space for the archives plus extracted images; remove the archives after successful import if space is limited. An imported archive may not populate `RepoDigests`; keep the source manifest digest and compare image configuration/layer identities when auditing the import.

## Supported scope

This backend supports local Linux Docker and `cpu_gpu` sessions. `gpu_only`, remote Docker daemons, multi-host Docker and live pool image/replica changes are not implemented. Omitted `backend` still selects Kubernetes.

Pool allocation and owned CPU workspaces recover after a Master restart. Missing GPU workers are recreated on their reserved UUIDs. Recovery is not exactly-once replay of in-flight model/tool calls; retain trajectories and explicitly retry interrupted tasks. Unexplained residual GPU processes cause worker recycling.

## Tested configuration

The application sequence above was exercised from a fresh checkout and newly built binaries on Linux x86_64, Docker 29.1.3, Go 1.23.6, and one B200 with driver 580.95.05. It used a new bridge-networked Master/state directory, the included smoke dataset, and an existing model API. Both iterations compiled successfully, passed correctness, and returned nonempty profiling data; the run then removed its sessions/pool and stopped the Master.

The host already had Docker, the driver and NVIDIA Container Toolkit installed. Published image content was verified against GHCR; cached layers were reused where the network proxy interrupted large downloads. This validation covers the application workflow, not installation of every possible Linux/driver combination or an entirely cold download through every proxy.

To try the bundled KernelBench tasks after the GELU smoke test, use `--subset kernelbench --split train --slice 0:1` in the same command and choose a fresh output directory. Remove the slice to run all 270 tasks.
