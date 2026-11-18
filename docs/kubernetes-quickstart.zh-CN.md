# Kubernetes 快速入门：运行 GPU kernel 优化实验

[English](kubernetes-quickstart.md) | [简体中文](kubernetes-quickstart.zh-CN.md)

本教程从准备环境开始，直到保存实验结果。模型服务、评估 runner 和 Sandbox Server 可以位于不同机器。本教程使用原生单 workload CUDA controller 和 Kubernetes sandbox 后端。

> **本教程针对 Kubernetes 后端。** 无 Kubernetes 的单机部署请看 [Docker 快速入门](docker-quickstart.zh-CN.md)。不填写 `backend` 时默认使用 Kubernetes；Docker 必须在 Server 配置中明确启用。

## 1. 理解组件关系

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

**评估命令不会启动 Sandbox Server。** 先启动一次 Server，它可以服务多次实验。评估启动后，runner 请求 Server 创建 GPU pool，等待空闲 worker，再创建每题 CPU session。Server 根据配置的模板创建 Kubernetes 资源，无需手动创建每道题的 sandbox。

三种并发互相独立：模型服务的请求容量、agent 任务并发（`--workers`）和 GPU worker 副本数（`--pool-replicas`）。每个副本分配几张 GPU，由 Kubernetes 模板决定。

## 2. 准备资源和机器

需要准备：

| 资源 | 要求 |
| --- | --- |
| Runner | Linux、Python 3.10+，可访问模型 API 和 Sandbox Server |
| Master 主机 | Linux、Go 1.23+、Make，以及目标集群 kubeconfig |
| Kubernetes | 可调度的 CPU/GPU 节点、正常 NVIDIA 驱动和 GPU device plugin、创建 sandbox 资源的权限 |
| 模型服务 | 模型标识、API base URL，必要时提供凭据 |
| GPU 业务镜像 | CUDA/PyTorch、编译工具链、评测器及性能分析工具 |
| CPU 业务镜像 | 配套编译环境及 kernel 工具包装器，Python/PyTorch/CUDA ABI 兼容 |
| sandboxd 注入镜像 | 包含 `/usr/local/bin/sandboxd`、`sh`、`cp` 和 `chmod` |
| 数据集 | runner 兼容格式，每题含参考 `code` |

**镜像：**runner 使用 `ghcr.io/baidu-baige/koda-agent-release:20261117-controller-8afc1e2`，CPU 编译与 GPU 执行均可使用 `ghcr.io/baidu-baige/koda-sandbox:py312-cuda129-torch28-2026110922`。私有镜像的拉取方式参见 [镜像登录/拉取说明](../README.zh-CN.md#容器镜像)。CPU/GPU 模板都需配置 `imagePullSecrets`，供 Kubernetes 拉取。sandboxd 注入镜像单独按第 5.2 节构建。

**数据集：**仓库内置 KernelBench。SOL-ExecBench 输入说明见[评测数据](../dataset/README.md)。

所有 Kubernetes 节点都必须能拉取镜像，必要时在模板中配置 `imagePullSecrets`。请使用已能正常分配 GPU 的集群；Kubernetes、驱动和 NVIDIA device plugin 的安装不属于本应用教程范围。

模型 GPU 和评测 GPU 可以位于不同机器。同节点部署时应由调度器分配不重叠设备。只有模板明确每个 worker 请求一张卡时，四个 worker 副本才代表四张卡。

## 3. 在 runner 安装代码

```bash
git clone https://github.com/baidu-baige/KODA.git
cd KODA
export KODA_ROOT="$PWD"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m minisweagent.run.benchmarks.kernelbench --help
```

后续 runner 步骤继续使用此 shell。私有仓库需要自己的 GitHub 认证。如果 Master 在其他机器，也需在那里克隆相同代码版本。

## 4. 准备并测试模型服务

使用已有兼容 API，不限定 SGLang。OpenAI 兼容接口需记录：

```bash
export MODEL_API_BASE='https://YOUR_MODEL_API/v1'
export MODEL_NAME='YOUR_SERVED_MODEL_NAME'
read -rsp 'Model API key (EMPTY if not required): ' MODEL_API_KEY
printf '\n'
export MODEL_API_KEY
```

实际测试生成，同时避免把凭据放到命令行参数里：

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

模型列表可用不等于生成可用。推理模型可能在这个小测试预算内只输出思考，可检查 reasoning 返回或按服务方式调整测试。确认服务支持后文的上下文/输出预算。根据模型明确配置思考和采样，不要照搬另一个服务不支持的参数。

## 5. 首次部署 Sandbox Server

在 **Master 主机**执行本节，不是在任务 sandbox 中执行。Master 主机必须能访问 Kubernetes API。runner 和 CPU session Pod 必须能访问 Master 对外公布的 HTTP 地址。只有调用者确实共享同一网络命名空间时才适合使用 `localhost`。

### 5.1 检查集群权限

```bash
export KUBE_CONFIG='/absolute/path/to/your/kubeconfig'
export SANDBOX_NS='your-sandbox-namespace'
kubectl --kubeconfig "$KUBE_CONFIG" -n "$SANDBOX_NS" get pods
kubectl --kubeconfig "$KUBE_CONFIG" -n "$SANDBOX_NS" auth can-i create pods
kubectl --kubeconfig "$KUBE_CONFIG" -n "$SANDBOX_NS" auth can-i create pods/exec
kubectl --kubeconfig "$KUBE_CONFIG" -n "$SANDBOX_NS" auth can-i create deployments.apps
```

namespace 必须已存在。Master 身份还需要读取、列出、watch、更新和删除受管资源的权限，包括 Pod、Deployment 及 ReplicaSet。使用管理员提供的适当范围凭据，不要把 kubeconfig 内容提交到 Git。

### 5.2 构建 Master 和 sandboxd

```bash
cd "$KODA_ROOT/src/kernel_sandbox"
make build
```

构建生成 `bin/kernel-sandbox-master`、`bin/sandboxd` 和 pool monitor。编译平台必须与执行节点兼容。

注入镜像只提供 sandbox 监督进程二进制，并不提供 CUDA/评测环境。仓库的 `Dockerfile.sandboxd` 使用公开的 shell 基础镜像，可按以下方式构建：

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

Go Makefile 使用 `CGO_ENABLED=0`。Master 通过 initContainer 和 `emptyDir` 注入二进制，再将其作为业务容器入口。Pod/Deployment 模板的业务容器不要填写 command 或 args，否则 Server 会拒绝。

### 5.3 适配 CPU/GPU 模板

创建本地部署目录，保留仓库原始示例：

```bash
mkdir -p deploy/local/templates
cp deploy/templates/cpu-pod-template.yaml deploy/local/templates/cpu.yaml
cp deploy/templates/gpu-deployment-template.yaml deploy/local/templates/gpu.yaml
```

启动 Server 前编辑两个副本。现有模板使用标准 NVIDIA GPU 资源键，仍需逐项按实际集群配置：

| 字段 | 配置方式 |
| --- | --- |
| `schedulerName`、队列注解 | 使用自己的调度器；标准集群删除内部队列设置 |
| `nodeSelector`、affinity、tolerations | 使用真实节点/污点规则，删除内部标签和主机名 |
| GPU resources | 使用本集群 device plugin 的资源键和数量；标准 NVIDIA 环境通常为 requests/limits 中 `nvidia.com/gpu: "1"` |
| `NVIDIA_VISIBLE_DEVICES` | 删除写死的物理设备列表，让 device plugin 分配 |
| CPU/memory resources | 设为集群可满足的值，内部示例可能申请很大资源 |
| volume/PVC 和挂载路径 | 使用 namespace 可用的存储，替换内部 PVC 和缓存路径 |
| lifecycle hook | 删除或替换内部 Hugging Face 缓存软链接 |
| securityContext | 配置镜像、CUDA、NSYS/NCU 实际所需权限 |
| `imagePullSecrets` | 私有镜像仓库认证 |

每份模板保留且只保留一个业务容器。CPU Pod 不申请 GPU，GPU worker 模板明确申请所需 GPU。不要把所有任务工作区挂到同一个共享可写目录：各独立 CPU Pod 内会使用相同工作区路径。保留业务镜像和工具依赖的存储路径，并区分共享缓存与任务局部工作区。

这些文件由 Master 作为模板读取，**不是让你用 `kubectl apply` 手工创建的资源**。评估请求到来时，Master 填入名称、标签、镜像、副本数并注入 sandboxd。

### 5.4 编写 Master 配置

填写 runner 和 CPU Pod 都能访问的 IP/DNS：

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

如果 GPU 模板使用的资源键不是 `nvidia.com/gpu`，同步修改配置。YAML 中的相对路径以该 YAML 所在目录为基准，示例用绝对路径避免歧义。Master 主机未装 Python/PyYAML 时先安装，或手工写出等价 YAML。

健康接口只表示 Server 有响应，不代表 GPU 已能执行评测器。此时尚未创建 GPU pool，空列表正常。默认日志位于 `src/kernel_sandbox/.run/koda/master.log`，可由 `KERNEL_SANDBOX_RUN_DIR` 覆盖。Server 应位于可信网络，因为访问接口即可执行 sandbox 命令。

## 6. 在 runner 准备数据集

在仓库根目录选择内置的 KernelBench 270 题。其他输入格式见[评测数据](../dataset/README.md)。

```bash
cd "$KODA_ROOT"
source .venv/bin/activate
export DATASET=kernelbench
```

使用为当前 controller 准备的数据，包含参考程序、形状约定和正确性元数据。原始上游 SOL 数据不一定等价。202 行、每算子选一个 workload 的数据表示 202 个单 workload 任务，并不覆盖各算子的所有官方 workload。

## 7. 配置 agent

在 runner 设置 `SANDBOX_URL` 指向已启动的 Master，CPU session Pod 也必须能访问同一地址。使用与 controller 兼容的镜像：

```bash
export SANDBOX_URL='http://YOUR_MASTER_HOST:18766'
export SANDBOX_IMAGE='ghcr.io/baidu-baige/koda-sandbox:py312-cuda129-torch28-2026110922'
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

基础配置 `configs/controller_h.yaml` 提供 prompt 和工具定义；`eval.local.yaml` 覆盖模型、环境和预算。它已被 Git 忽略，可能含凭据。示例使用 OpenAI 兼容接口；其他协议应配置对应模型适配器。

运行前检查 `agent.compile_arch`：`9.0` 面向 Hopper/H20。按实际 GPU 和工具链设置，不要把 Hopper 二进制用于其他架构。262,144 token 预算和 65,536 token 输出上限需要模型服务支持；较小上下文模型应主动调整并记录变化。

不填写 `environment.gpu_pool_id` 时自动建池。只有确实要复用 pool 才设置它；GPU 镜像必须匹配，runner 不会删除复用的池。

## 8. 跑一题并观察自动部署

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

在另一个具有集群凭据的终端观察：

```bash
kubectl --kubeconfig "$KUBE_CONFIG" -n "$SANDBOX_NS" get pods,deployments -w
```

预期过程：

1. runner 发送 `POST /v1/gpu_pool`，Master 创建 GPU Deployment 并注入 sandboxd。
2. runner 轮询到至少一个 worker 空闲，不会等待所有副本就绪。
3. runner 发送 `POST /v1/session`，Master 创建 CPU Pod 并等待就绪。
4. runner 初始化任务工作区，请求模型生成代码。
5. controller 在 CPU 编译，请求 GPU eval，并为符合条件的候选执行 profile。
6. 将工具反馈交给模型，继续修改候选。
7. 每题收尾删除 session，批次收尾删除本次拥有的 pool。

两轮只是冒烟预算，不保证模型一定写对代码。正确候选比 API 可达更能证明链路有效；profile 还必须真正采到算子/指标，退出码零但输出为空不能算成功。编译失败时保留日志，区分代码问题和镜像/ABI/网络错误。

## 9. 查看结果

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

| 输出 | 用途 |
| --- | --- |
| `preds.json` | 每题最终结果及选中候选摘要 |
| `eval_result.json` | 批次汇总 |
| `<task>/trajectory.json` | 模型消息、工具反馈和退出详情 |
| `<task>/trajectory.readable.yaml` | 易读轨迹 |
| `<task>/eval_turn_N.json` | controller 输出的逐轮评测回执（如有） |
| `<task>/kernel_turn_N.py`、`.cu` | controller 输出的对应轮候选源码（如有） |

不要用进程退出、Pod Ready 或 HTTP 200 代替正确性判定。检查 `solved`、原始回执、失败 trial 和 profile 输出。容差、参考计时、正确性/性能 trial、镜像及 GPU 都会影响可比性。本地 speedup 不是官方 SOL 榜单分数。

## 10. 批量评测

冒烟成功后，使用新的输出目录，移除 `--slice`：

```bash
python -u -m minisweagent.run.benchmarks.kernelbench \
  --config "$CONTROLLER_CONFIG" --config eval.local.yaml \
  --subset "$DATASET" --split train \
  --backend cuda \
  --sandbox-server-url "$SANDBOX_URL" --sandbox-image "$SANDBOX_IMAGE" \
  --workers 4 --pool-replicas 2 --output "output/$RUN_TAG-batch" --no-wandb
```

示例最多并发四个 agent，共享两个 GPU worker 副本。两个 worker 都忙时 GPU 请求排队。不填写 `--pool-replicas` 时默认等于任务并发数（不超过选中题数）。模型生成队列和 GPU 执行队列是不同瓶颈。

示例默认预算为 70 轮、50 次提交评测、7200 秒、262144 token，分别独立触发结束。进程启动后修改配置文件不会热更新。重启时保留旧 attempt，不要假设原生运行一定能自动续上未结束的对话。

## 11. 停止与清理

前台运行时先中断一次，让活跃任务进入清理；runner 取消待启动任务时可能仍等待在途任务。每题正常清理删除 CPU session，批次收尾只删除本次创建的 pool，复用的 pool 保留。

异常退出后检查 `GET /v1/session` 和 `GET /v1/gpu_pool`，按本次日志/session 前缀核对明确资源，只删除这些 session ID，再删除拥有的 pool。响应格式参见 [API 文档](../src/kernel_sandbox/docs/api.md)。共享资源环境中不要按整个 namespace/宽泛前缀清理。

仅在活跃任务全部结束后停止专用 Master：

```bash
cd "$KODA_ROOT/src/kernel_sandbox"
bash deploy/scripts/master.sh stop deploy/local/master.yaml
```

停止 Master 与停止某个评估任务不同。不要为结束一项任务停止共享 Server。保留结果以及精确的代码、配置、镜像版本以便复现。

## 12. 故障排查

| 现象 | 检查与处理 |
| --- | --- |
| 无法连接 Master | 查看进程/日志、公布地址、端口，以及 runner/CPU Pod 到 Server 的路由；`localhost` 可能指向错误机器 |
| Server 健康但 pool 不就绪 | Kubernetes events、GPU 资源、device plugin 键、调度规则、镜像和 injector 拉取 |
| CPU session 不就绪 | CPU 配额、PVC、镜像拉取、模板约束和 events |
| 模型列表正常但生成失败 | 模型名、凭据、请求参数支持、上下文/输出限制 |
| CUDA 编译失败 | compiler stderr、代码错误或依赖缺失、目标架构 |
| 扩展无法加载 | CPU/GPU Python、PyTorch、CUDA ABI 及运行库 |
| eval 正常但 profile 失败/为空 | NSYS/NCU、驱动权限、kernel/NVTX/CUDA graph 可见性 |
| CPU 工具无法请求 GPU | CPU Pod 到 Master 的连通性和 `agent.master_url` 是否一致 |
| Pod 一直 Pending | 查看 `kubectl describe pod` 和 namespace events，不按申请副本数推断就绪 |
| 重启似乎重复执行 | 保留 attempt 并核对 controller 的已有结果处理机制 |

## 验收清单

一个真实任务应具有：实际模型回复；自动创建 GPU pool 和 CPU session；成功编译；GPU 正确性与性能回执；正确候选的有效 profile；模型收到反馈后继续下一轮；保存的轨迹、源码及回执；最后资源清理通过。本文命令和配置关系已按源码检查，通用外部集群部署路径尚未结合公开资源完成端到端验收。
