# Kernel Sandbox

[English deployment guides](../../docs/quickstart.md) | [中文部署教程](../../docs/quickstart.zh-CN.md)

Sandbox Master 提供统一 HTTP API，自动管理每题 CPU session 和共享 GPU worker pool。

- [Docker 部署](../../docs/docker-quickstart.zh-CN.md)：单机本地 Docker，无需 Kubernetes；配置 `backend: docker`。
- [Kubernetes 部署](../../docs/kubernetes-quickstart.zh-CN.md)：使用 kubeconfig 和 Pod/Deployment 模板；默认后端。
- [HTTP API](docs/api.md) 与[架构设计](docs/design.md)。

先启动 Master，再运行评估。未指定已有 pool 时，评估自动创建并清理 pool/session。所有镜像、资源、网络和存储配置由部署者提供；不要在配置中保存共享凭据。

```bash
make build
bash deploy/scripts/master.sh start /absolute/path/to/master.yaml
bash deploy/scripts/master.sh status /absolute/path/to/master.yaml
```

服务地址必须能从 runner 和 CPU sandbox 访问。发布的业务镜像见仓库 README；Docker 使用本地 sandboxd 二进制，Kubernetes 使用 Dockerfile.sandboxd 构建的注入镜像。
