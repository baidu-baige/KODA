# KODA: Kernel Optimization via Domain Adaptation

[English](README.md) | [简体中文](README.zh-CN.md)

[**Hugging Face · KODA-35B**](https://huggingface.co/Baidu-Baige/KODA-35B) · [**知乎 · KODA 技术文章**](https://zhuanlan.zhihu.com/p/2106387936149361761)

KODA 是一个使用语言模型优化 GPU kernel 的框架。它生成 CUDA 代码，在受管 sandbox 中编译、评测候选实现，并将正确性、计时和性能分析结果反馈给模型，持续迭代优化。

KODA 基于 [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)，扩展了 GPU kernel 优化 controller、sandbox 执行和评测/性能分析工具。

[快速入门总览](docs/quickstart.zh-CN.md) · 首次部署可选择[单机 Docker 教程](docs/docker-quickstart.zh-CN.md)或 [Kubernetes 教程](docs/kubernetes-quickstart.zh-CN.md)，涵盖 Server 部署、自动创建 pool/session、冒烟测试和清理。

## 选择 sandbox 后端

| 后端 | 环境要求 | 教程 |
| --- | --- | --- |
| **Docker（单机）** | Linux、Docker、NVIDIA Container Toolkit、专用 GPU UUID | [Docker 快速入门](docs/docker-quickstart.zh-CN.md) |
| **Kubernetes（默认）** | GPU 集群、kubeconfig、CPU/GPU 模板 | [Kubernetes 快速入门](docs/kubernetes-quickstart.zh-CN.md) |

在 Sandbox Server 配置中设置 `backend: docker`，即可使用本地 Docker 容器。不填写 `backend` 时保持原有 Kubernetes 行为。两种后端提供相同的 agent HTTP API，评估启动后自动创建 CPU session 和 GPU worker。

先启动 Sandbox Server，再运行评估。模型服务和 agent runner 可以部署在其他机器。Docker 模式目前支持本地 daemon 上的 `cpu_gpu` session，不支持 `gpu_only`、远程 Docker daemon 和运行中扩缩 pool。GPU 必须预留给 sandbox，不应与模型服务分配重叠。

## 发布范围

本仓库只保留 kernel controller 流程。旧 loop/ReAct/OpenCode agent 模式、未使用的 LLM proxy、训练脚本、内部集群配置和无关上游教程均已移除。共享运行时和模型辅助逻辑仅按 controller 实际依赖保留；兼容模型 API 仍可通过 LiteLLM 使用。

模型地址、凭据和输出目录在运行时配置。controller prompt 分别为 `controller_h.yaml` 和 `controller_b.yaml`，`configs/` 下保留仓库级配套副本。工具缓存默认放在独立 sandbox 工作区的 `.cache` 下。`/opt/kerneltools` 和 `/workspace/kerneleval` 是 sandbox 镜像内部工具路径，不是宿主机/云盘挂载；使用其他镜像时可覆盖。

## 工作流程

```text
Evaluation runner ── model API ── candidate kernel.py / kernel.cu
        │
        └── Sandbox Server
               ├── per-task CPU session: workspace and compilation
               └── shared GPU workers: evaluation and profiling
                         │
                         └── feedback → next model iteration → best candidate
```

模型服务可以是 vLLM、SGLang、OpenAI 兼容中转接口，或模型适配器支持的其他服务，不限定 SGLang。模型推理 GPU 与 kernel 评测 GPU 是不同资源；评估 runner 本身可以运行在 CPU 主机上。

本快速入门介绍原生的**单 workload controller**。数据集每行对应一个独立优化任务。每算子只选一个 workload 的评估，不能证明该算子全部 workload 都正确，本地 speedup 也不是官方 SOL 榜单分数。

## 前置条件

- runner 中有 Python 3.10 或更高版本（Docker 路线使用镜像内 Python）。
- 可访问的模型 API、模型标识，以及必要的访问凭据。
- 已按 Docker 主机或 Kubernetes 集群配置并启动的 Kernel Sandbox Server，以及兼容的 CPU/GPU 镜像。
- runner 支持的 benchmark 数据集，可用本地 Hugging Face 数据或数据集标识；不要照搬历史内部文件路径。

**不需要手动创建每题 sandbox。** 启动评估会自动创建 GPU worker pool 和独立 CPU session。Server 部署可选择 [Docker 方式](docs/docker-quickstart.zh-CN.md)或 [Kubernetes 方式](docs/kubernetes-quickstart.zh-CN.md)。

## 容器镜像

| 镜像 | 地址 |
| --- | --- |
| Agent | `docker.io/loongsage/koda:agent-20261118-clean` |
| CPU/GPU sandbox | `docker.io/loongsage/koda:sandbox-20261118-clean` |

KODA 发布镜像托管于 Docker Hub，可以匿名拉取。

```bash
docker pull docker.io/loongsage/koda:agent-20261118-clean
docker pull docker.io/loongsage/koda:sandbox-20261118-clean
```

agent 镜像内代码位于 `/workspace/mini-swe-agent`。sandbox 镜像提供编译/评测环境，本身不是 Sandbox Server。Agent 镜像包含精简后的 controller 代码及 ckpt140 的 H/B 两份配置。Docker 模式需从当前源码构建 Master 和 sandboxd。Docker 模式将编译好的 sandboxd 拷入容器，不需要注入镜像。Kubernetes 仍使用 sandboxd 注入镜像和用于私有仓库认证的 `imagePullSecrets`。

Agent 镜像通过了 320 项回归测试。Docker 后端已通过 B200 模型 → compile → eval → profile 验证，以及 GPU 隔离、超时、worker 恢复、Master 重启和清理检查。请按教程检查自己的部署环境。

## 安装

Docker 路线请按 [Docker 快速入门](docs/docker-quickstart.zh-CN.md)操作，使用 agent 镜像提供 Python 依赖，并运行自带独立冒烟任务。以下宿主机 Python 安装是可选方式。

```bash
git clone https://github.com/baidu-baige/KODA.git
cd KODA
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m minisweagent.run.benchmarks.kernelbench --help
```

## ckpt140 / Qwen3.6 配置

KODA 模型权重（ckpt140）托管在 [Hugging Face](https://huggingface.co/Baidu-Baige/KODA-35B)。

| 评测 GPU | Controller YAML | CUDA 编译架构 |
| --- | --- | --- |
| H100 / H20（Hopper） | [controller_h.yaml](configs/controller_h.yaml) | `9.0` |
| B200（Blackwell） | [controller_b.yaml](configs/controller_b.yaml) | `10.0a` |

两份都是同一个 ckpt140/Qwen3.6 controller，只区分目标硬件，不是不同 agent 模式。模型参数均为 temperature `0.6`、top-p `0.95`、top-k `20`、min-p `0`、presence penalty `0`、repetition penalty `1`、开启思考、单次最多输出 65536 token、对话预算 262144 token。正式运行预算为 70 轮、50 次评测提交、7200 秒，启用动态输出额度并保留 10240 token margin。

默认服务模型别名为 `ckpt140`，应改为接口实际提供的模型 ID。API/Master 地址和凭据只在本地配置。H/B 指 **sandbox 执行评测的 GPU**，与模型服务用什么卡无关。prompt 使用配置架构和实测 GPU 信息，CUTLASS 参考按实际设备探测选择。使用不支持相应参数的其他服务时，需要移除或调整 Qwen 专用 extra-body 字段。

## 配置

将[示例配置](configs/controller_h.yaml)复制为 `eval.local.yaml`，替换所有 `REPLACE_...`。不要将 API 凭据提交到 Git。controller 使用两份配置：基础配置提供 prompt 和工具定义，覆盖配置提供模型、预算和环境参数。

- `model.model_name`：OpenAI 兼容服务使用 `openai/<served-model-name>`，模型标识以服务实际提供为准。
- `model.model_kwargs.api_base`：API 基础地址，通常以 `/v1` 结尾。
- `model.model_kwargs.api_key`：访问凭据，无认证本地服务可填写 `EMPTY`。
- `agent.master_url`：与 `--sandbox-server-url` 相同且可访问的 Server 地址，CPU session 也必须能访问。
- `environment.cpu_image`：CPU 编译镜像，需与 GPU 镜像的 Python、PyTorch、CUDA ABI 兼容。
- `agent.compile_arch`：目标 GPU 架构，示例 H20/Hopper 为 `9.0`，按硬件修改。

示例覆盖配置清除继承的 GPU 资源请求，使用 Server 默认集群/模板；需要集群路由时明确配置资源键。采样、思考和输出 token 参数取决于服务；示例不强制服务特有的思考标记。做模型对照时应明确设置这些参数以保证可复现。

```bash
export CONTROLLER_CONFIG=configs/controller_h.yaml  # B200: configs/controller_b.yaml
chmod 600 eval.local.yaml
export SANDBOX_URL='http://YOUR_SANDBOX_SERVER:PORT'
export SANDBOX_IMAGE='docker.io/loongsage/koda:sandbox-20261118-clean'
export DATASET=kernelbench
```

在仓库根目录运行时，`--subset kernelbench` 使用内置的 270 道 KernelBench 题。其他输入格式和许可说明见[评测数据](dataset/README.md)。

自动管理 pool 时不设置 `environment.gpu_pool_id`。需要复用已有池时才填写 ID，池镜像必须匹配 `--sandbox-image`；runner 不会删除复用的池。

## 运行一题

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

确认有真实模型回复、候选编译成功和 GPU eval 回执。正确候选还应检查 profile 中的实际算子与警告；命令成功本身不能证明采到了有效指标。两轮可能不足以让模型写出正确实现，应区分候选编译错误与基础设施错误。

## 批量运行

使用新的输出目录，移除 slice，先采用较低并发：

```bash
python -u -m minisweagent.run.benchmarks.kernelbench \
  --config "$CONTROLLER_CONFIG" \
  --config eval.local.yaml \
  --subset "$DATASET" --split train \
  --backend cuda \
  --sandbox-server-url "$SANDBOX_URL" --sandbox-image "$SANDBOX_IMAGE" \
  --workers 4 --pool-replicas 1 --output output/batch --no-wandb
```

`--workers` 控制任务并发，`--pool-replicas` 申请 GPU worker 副本。Docker 每副本分配一个配置 UUID；Kubernetes 根据模板分配资源。不填写时副本数默认等于任务并发数。多个任务可共享较小 pool 并排队。启动时只等至少一个 worker 空闲，不会等待全部副本就绪。

示例预算为 70 轮、50 次提交评测、7200 秒、262144 token，每次输出最多 65536 token。各限制独立生效，模型服务必须支持对应上下文/输出额度。比较性能前应核对容差、正确性 trial、参考计时和硬件。

## 结果与清理

在输出目录查看 `preds.json`、`eval_result.json`、逐题轨迹、评测回执和候选源码。报告正确性/性能时保留原始工具元数据，汇总统计不要排除失败 workload。

每题清理删除其 CPU session，批次收尾删除本次创建的 pool；复用池由调用方管理。强制退出或网络故障可能留下资源，使用 Server API 只检查和删除本次明确的 session/pool ID。一次中断会取消待启动任务，但可能等待在途任务结束，不要假设立即退出。

## 发布检查

发布源码变更前运行 `python3 scripts/audit_release.py`。检查只报告文件位置，不输出命中的凭据值；会拒绝内部网络/镜像仓库配置、云盘路径名、私网 IPv4/IPv6 地址、URL 内嵌凭据、硬编码凭据值、已知 token/私钥格式、缺失的本地导入和退役组件。仓库也配置了本地 pre-commit 入口。该检查针对当前源码树；将已有私有产物改为公开前，还需单独检查 Git 历史和容器镜像层。

## 开发计划

- [x] **单机 Docker sandbox 后端：**支持 `cpu_gpu` session、UUID 分配、超时清理、恢复，并通过真实 GPU 验证。参见 [Docker 快速入门](docs/docker-quickstart.zh-CN.md)。
- [ ] Docker `gpu_only` session 和运行中 GPU pool 扩缩容。

## 致谢与许可证

KODA 基于 [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)，其上游作者包括 Kilian A. Lieret、Carlos E. Jimenez 和其他贡献者。感谢上游作者和社区的工作。

上游代码采用 MIT License，原始版权声明和许可正文保留在 [LICENSE.md](LICENSE.md)。第三方组件、模型权重和 benchmark 数据分别遵守各自的许可证。
