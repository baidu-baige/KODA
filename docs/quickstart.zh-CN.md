# 快速入门

[English](quickstart.md) | [简体中文](quickstart.zh-CN.md)

按部署环境选择教程。两种方式使用相同的 agent 流程：模型生成 kernel 代码，sandbox 编译、评测并做性能分析，再把反馈交给模型继续迭代。

| 教程 | 适用场景 | 环境要求 |
| --- | --- | --- |
| [Docker 快速入门](docker-quickstart.zh-CN.md) | 希望不依赖 Kubernetes，在一台 GPU 主机运行 sandbox | Linux、本地 Docker daemon、NVIDIA Container Toolkit、专用 GPU UUID |
| [Kubernetes 快速入门](kubernetes-quickstart.zh-CN.md) | 已有可用的 GPU Kubernetes 集群 | 集群权限、kubeconfig、CPU/GPU 模板、镜像仓库访问权限 |

两种方式都需准备兼容模型 API 和数据集，并在评估前启动 Sandbox Server。评估命令会自动创建 GPU pool 和 CPU session。模型服务和 agent runner 可以运行在其他机器上。

Docker 模式通过 `backend: docker` 启用，不填写时默认走 Kubernetes。Docker 教程目前支持单机本地 daemon 的 `cpu_gpu` session。配置、验收和清理细节见对应教程。

镜像地址及私有仓库登录说明见 [README](../README.zh-CN.md#容器镜像)。

## 已有中文部署资料

- [Kernel Sandbox 服务说明](../src/kernel_sandbox/README.md)
- [Sandbox HTTP API](../src/kernel_sandbox/docs/api.md)
- [Sandbox 架构与资源生命周期](../src/kernel_sandbox/docs/design.md)
