# Docker 快速入门：运行第一个 kernel 优化任务

[English](docker-quickstart.md) | [简体中文](docker-quickstart.zh-CN.md)

本教程从新克隆的代码开始，使用本机 Docker sandbox 跑一个小任务。不需要 Kubernetes，也不需要单独的 sandboxd 注入镜像。模型 API 可以部署在其他机器上，也可以使用兼容的在线服务。

```text
Existing model API ↔ agent container
                         ↓ HTTP
                   Sandbox Master on the GPU host
                         ↓ local Docker daemon
                   CPU task container → GPU worker container
                         compile         eval / profile
```

必须先启动 Master。启动 agent 后，它会自动创建 GPU pool 和 CPU session。agent 镜像提供 Python 依赖，因此无需在宿主机安装 PyTorch 或 agent 的 Python 包。

## 1. 检查宿主机

使用安装了 Git、Python 3、Docker、NVIDIA 驱动、NVIDIA Container Toolkit 和 Go 1.23 或更高版本的 Linux x86_64 主机。下文发布的镜像面向 `linux/amd64`，其他架构需要另行构建兼容镜像。运行 Master 的账号必须能执行 Docker 命令。本教程支持本地 Docker daemon，不支持远程 Docker context。至少为 sandbox 预留一张 GPU，不要选择模型服务或其他任务正在使用的卡。

```bash
git --version
python3 --version
docker version
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
go version
```

如果尚未安装 Docker 或 GPU toolkit，按操作系统参照官方 [Docker Engine 安装文档](https://docs.docker.com/engine/install/)和 [NVIDIA Container Toolkit 安装文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)完成安装，再重复检查。Toolkit 文档也提供驱动安装入口。如果只缺 Go，从 [Go 官方发行](https://go.dev/doc/install)安装。下面以 Linux AMD64 的 Go 1.23.6 为例，安装到个人目录，并校验官方 SHA256：

```bash
mkdir -p "$HOME/.local/koda-go-1.23.6"
curl -fL https://go.dev/dl/go1.23.6.linux-amd64.tar.gz -o /tmp/koda-go.tar.gz
printf '%s\n' '9379441ea310de000f33a4dc767bd966e72ab2826270e038e78b2c53c2e7802d  /tmp/koda-go.tar.gz' | sha256sum -c -
tar -xzf /tmp/koda-go.tar.gz -C "$HOME/.local/koda-go-1.23.6"
export PATH="$HOME/.local/koda-go-1.23.6/go/bin:$PATH"
go version
```

如果在其他架构上自行构建镜像，请下载对应架构的 Go 安装包。首次构建需要联网下载 Go 模块。在受限网络中遇到下载失败时，先查看第 11 节再重试。

## 2. 克隆仓库并创建独立运行目录

```bash
git clone https://github.com/baidu-baige/KODA.git
cd KODA
export KODA_ROOT="$PWD"
export KODA_RUN_DIR="$HOME/koda-runs/smoke-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$KODA_RUN_DIR"
chmod 700 "$KODA_RUN_DIR"
git rev-parse HEAD
```

仓库为私有时，Git 需要具有仓库读取权限的账号/token。使用现有 Git 凭据管理器认证，不要把 token 写进克隆 URL。GHCR package 权限与代码仓库权限是两回事。

后续命令请在同一个 Bash shell 中执行，确保导出的变量和辅助函数继续可用。Master 会在后台运行，无需另开终端。

## 3. 登录并拉取镜像

如果 GHCR 镜像为私有，请使用有镜像包访问权限的 GitHub 账号及带 `read:packages` 的 classic 令牌登录。

```bash
read -rp 'GitHub username: ' GHCR_USER
read -rsp 'GitHub package token: ' GHCR_TOKEN
printf '\n'
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
unset GHCR_TOKEN

export AGENT_IMAGE=ghcr.io/baidu-baige/koda-agent-clean:20261118-clean
export SANDBOX_IMAGE=ghcr.io/baidu-baige/koda-sandbox-clean:20261118-clean
docker pull "$AGENT_IMAGE"
docker pull "$SANDBOX_IMAGE"
```

验收使用 Docker Engine 29.1.3 的默认安全配置。一套旧 Docker 20.10 环境在默认 seccomp 下无法创建 Python 线程，报错 `can't start new thread`。如果评测启动前遇到此错误，请先更新 Docker Engine 及其容器运行时、安全组件，再重试。

sandbox 镜像提供编译器和评测/性能分析工具，agent 镜像提供 Python 依赖。Agent 镜像包含精简后的 controller 代码和 ckpt140 的 H/B 配置。下一步需要从当前代码构建 Master 和 sandboxd。

为便于复现，发布镜像的 manifest digest 如下：

| 镜像 | Digest |
| --- | --- |
| Agent | `sha256:6e9ec2d4d424705f0988a0a10fed5dcbfbeabbe439263880ca7bfcac047a5b9b` |
| Sandbox | `sha256:69e69e4f352c0c5308acbe60b19b60938bc56785ce2e7babe5ca2c4354de67cb` |

## 4. 构建 Server 并选择 GPU

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

B200 示例选择 `configs/controller_b.yaml`；H100/H20 改为 `configs/controller_h.yaml`，同时将 `CUDA_ARCH` 改为 `9.0`。两份配置均已适配 ckpt140/Qwen3.6，详细预算与采样参数见中文 README。

使用完整 UUID，不要使用数字设备编号。填写可访问的宿主机 IP，不要使用 `localhost`：本教程的 CPU 容器使用 Docker bridge 网络，需要通过宿主机地址访问 Master。确认端口未被占用，且容器与 agent 都能访问。

启动 Server 前，验证 Docker 实际分配到的 GPU：

```bash
docker run --rm --gpus "device=$GPU_UUID" \
  --entrypoint nvidia-smi "$SANDBOX_IMAGE" \
  --query-gpu=uuid,name --format=csv,noheader
```

输出应只包含选定的 GPU。如果 Docker 提示无法选择 GPU device driver，先修复 NVIDIA Container Toolkit/runtime 配置。不要通过暴露所有 GPU 或开启 privileged 模式跳过问题。

## 5. 定义 agent Python 辅助函数

以下辅助函数在 agent 镜像中运行当前克隆的代码。源码只读挂载，生成的数据、配置和结果写入新运行目录。agent 容器本身不分配 GPU。

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

Agent 镜像的 Python 位于 `/opt/koda-agent-venv`。此辅助函数使用容器默认用户，生成文件可能属于 root。下面生成的 Master 配置会明确归还宿主用户的 UID/GID。

辅助函数明确使用发布镜像自带的 Python 环境；直接调用系统 `python3` 可能缺少依赖。它也关闭了 CPU agent 的 GPU 暴露。host 网络仅用于 agent 连接 API；CPU/GPU sandbox 容器由 Master 根据其 Docker 网络配置另外创建。

## 6. 生成配置并启动 Docker Master

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

健康 Server 返回 `{"status":"ok"}`，初始 GPU pool 列表应为空。如果启动失败，先查看 `master.log` 再重试。整个评估期间 Master 必须保持运行。

模板将每个容器限制为 8 个 CPU 核和 32 GiB 内存，大任务应按需调整。测试用 Nsight 环境需要 GPU 容器具有 `SYS_ADMIN`，请仅授予实际性能分析所需的 capability。模板还保留了对评测器已知常驻辅助进程的精确白名单。它不会使用 privileged 容器、向任务容器挂载 Docker socket 或暴露全部 GPU。

将 Master API 放在可信网络中。每个 Master 都需要唯一 ID 和持久 state 目录。state 目录会加锁并绑定 daemon/Master 身份，在相关资源仍存在时必须保留。

## 7. 配置并检查模型 API

使用已有的 vLLM、SGLang 或兼容 OpenAI 协议的 API：

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

推理模型可能先耗尽这段小诊断预算才输出最终文本。这里主要测试 API 访问，后续真实 kernel 任务才验证生成和工具反馈。示例 controller 使用 262,144 token 预算与 65,536 token 单次输出上限；服务支持更小时应相应调整。仅在服务支持时，通过 `MODEL_EXTRA_BODY` 设置服务特有的思考/采样参数，例如兼容服务的 `{"chat_template_kwargs":{"enable_thinking":true}}`。

## 8. 生成自带冒烟数据和 agent 配置

本次冒烟不需要下载 SOL 数据或访问内部存储路径。仓库包含一道 GELU 参考任务，准备脚本只读取源码，无需在宿主机导入 PyTorch：

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

生成完整配置可以确保继承的 Kubernetes 资源映射真正被清空。Docker 后端拒绝非空的 Kubernetes 资源/调度映射。不设置 `gpu_pool_id`，让本次运行拥有并清理自己的 pool。

## 9. 运行任务并检查结果

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

运行时，Master 创建 GPU worker 容器，等待其空闲，再创建 CPU session 容器，并通过 sandboxd 执行 compile/eval/profile。CPU 容器会回调 Master 请求 GPU 工作。现有工具负责传递源码和产物，不需要挂载共享任务工作区。

检查 `output/<task>/trajectory.json` 或 `trajectory.readable.yaml` 中是否有：

- 真实模型回复和成功的编译。
- `correctness: true` 的评测回执。
- 捕获到 kernel/算子的性能分析结果（本题应有非零 `num_ops`）。
- 模型收到反馈后继续执行下一轮。

两轮冒烟预算结束时，`RoundLimitReached` 是预期结果；如果已保存有效候选，它不属于错误。不能保证所有模型都能在两轮内生成正确代码，需要区分候选错误与基础设施错误。进程退出或 HTTP 200 本身不足以证明完整工具链成功。

成功后，再使用准备好的正式 benchmark 数据集和新的运行目录。按资源容量增加 `--workers` 与 `--pool-replicas`：Docker 为每个 pool replica 分配一个配置中的 GPU UUID，任务数可以多于 GPU 数。完整 benchmark 与这个最小工具链测试是不同范围。

## 10. 检查清理结果并停止 Server

正常收尾会删除本次 CPU session 和拥有的 GPU pool。确认这个专用 Master 的资源列表已为空：

```bash
curl --noproxy "$MASTER_IP" --fail --silent --show-error "$SANDBOX_URL/v1/session"
curl --noproxy "$MASTER_IP" --fail --silent --show-error "$SANDBOX_URL/v1/gpu_pool"
docker ps -a --filter "label=io.koda.sandbox.master=$MASTER_ID"
```

如果中断运行后有残留，通过 `/v1/session/<id>` 删除本次明确的 session ID，再通过 `/v1/gpu_pool/<id>` 删除其 pool。不要删除别人的资源，也不要通过删除 state 文件强行重启。

先核对进程身份，再只停止本次启动的 Server：

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

保留日志和结果。注意保护 `eval.yaml`，它可能包含模型凭据。只停止 Master 会保留容器以便重启恢复，并不等于删除 pool。

## 11. 受限网络与常见安装问题

不同工具使用不同的信任证书与代理配置。请给发生故障的工具配置所在网络批准的设置：

| 工具 | 代理/证书配置 |
| --- | --- |
| curl | shell 代理变量，以及 `CURL_CA_BUNDLE` 或 `--cacert` |
| Git HTTPS | shell 代理变量及 `GIT_SSL_CAINFO` |
| Go / crane | shell 代理变量及 `SSL_CERT_FILE` |
| Docker login/pull | Docker daemon 的代理及 registry CA 配置；仅在 shell 导出变量可能不生效 |

如果 agent 容器访问模型 API 也需要代理，需明确把相关代理变量传入辅助函数的 `docker run`。将 Master 地址加入 `NO_PROXY`，使 sandbox 请求继续走本地网络。

- **GitHub/GHCR 私有访问：**代码仓库和 package 权限分别配置，两者都需认证。不要把 token 存进仓库或克隆 URL。
- **Docker login/pull 超时而 curl 正常：**Docker daemon 网络配置与 shell 代理变量不同。请管理员配置 daemon 代理/CA，或使用用户态 registry 客户端通过批准的代理和 CA 下载归档后导入 Docker。
- **TLS 校验失败：**使用所在网络批准的 CA bundle，不要关闭证书校验。
- **Go 模块/PyPI 下载失败：**先区分连接/CA 错误和包缺失。本教程的容器化 agent 路线完全无需宿主机 pip 安装。
- **GPU 容器启动失败：**检查第 4 节单 UUID Docker 测试及 NVIDIA runtime。
- **CPU session 不就绪：**检查能否访问 `SANDBOX_URL`；bridge 容器的 loopback 并不是宿主机 loopback。先查看 `master.log` 再重试。
- **编译成功但扩展加载失败：**CPU/GPU 镜像必须匹配 Python、PyTorch 和 CUDA ABI。
- **eval 成功但 profile 失败或零算子：**检查 Nsight、驱动性能分析权限和实际 kernel 捕获，空 profile 不能计为成功。

用户态 registry 客户端可以使用 `google/go-containerregistry` 官方发布的 `crane`，安装时校验归档。下面是已测试的 Linux AMD64 v0.22.1 安装方式；其他架构需选择相应资产。受限网络中先配置批准的 shell 代理/CA 变量。

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

然后在同一套 shell 代理/CA 设置下登录并拉取：

```bash
# Read a package token from a protected file; crane writes Docker-compatible credentials.
cat /path/to/protected/package-token | crane auth login ghcr.io -u YOUR_GITHUB_USER --password-stdin
crane digest "$SANDBOX_IMAGE"
crane pull --cache_path "$KODA_RUN_DIR/registry-cache" "$SANDBOX_IMAGE" "$KODA_RUN_DIR/sandbox.tar"
docker load -i "$KODA_RUN_DIR/sandbox.tar"
crane pull --cache_path "$KODA_RUN_DIR/registry-cache" "$AGENT_IMAGE" "$KODA_RUN_DIR/agent.tar"
docker load -i "$KODA_RUN_DIR/agent.tar"
```

如果代理返回 HTTP/2 `PROTOCOL_ERROR`，可设置 `export GODEBUG=http2client=0` 后重试该用户态下载。保留已验证完整的层缓存。遇到 `unexpected EOF` 时可能留下不完整缓存文件：仅删除错误对应的未完成层，再重试，或换用新缓存目录。这只改变 HTTP 协议，不关闭 TLS 证书校验。

该方式保留 TLS 校验，也无需修改共享 Docker daemon。为归档和导入后的镜像留足空间，成功导入后可删除归档以节省磁盘。导入归档可能不填充 `RepoDigests`；审计时保留源 manifest digest，并比较镜像配置和各层身份。

## 支持范围

本后端支持本地 Linux Docker 和 `cpu_gpu` session。尚未实现 `gpu_only`、远程 Docker daemon、多主机 Docker 及运行中的 pool 镜像/副本数变更。不填写 `backend` 时仍选择 Kubernetes。

Master 重启后可以恢复 pool 分配和自己创建的 CPU 工作区；丢失的 GPU worker 会在预留 UUID 上重建。恢复并不保证在途模型/工具调用恰好执行一次，应保留轨迹并明确重试中断任务。不明残留 GPU 进程会触发 worker 回收。

## 已验证环境

上述应用流程已在新克隆源码和新编译二进制上测试：Linux x86_64、Docker 29.1.3、Go 1.23.6、一张 B200、驱动 580.95.05。测试使用新建的 bridge 网络 Master/state 目录、自带冒烟数据和已有模型 API。两轮都编译成功、正确性通过且 profile 非空，之后删除 session/pool 并停止 Master。

测试主机事先已安装 Docker、驱动和 NVIDIA Container Toolkit。发布镜像内容已与 GHCR 核对；代理中断大文件下载时复用了缓存层。该验证覆盖应用流程，不代表验证了所有 Linux/驱动组合的安装或所有代理下的完全冷下载。

GELU 冒烟完成后，可将同一命令的数据参数换成 `--subset kernelbench --split train --slice 0:1`，并使用新的输出目录，试跑内置 KernelBench 题目。移除切片参数即可运行全部 270 题。
