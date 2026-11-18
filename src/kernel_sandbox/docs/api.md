# Kernel Sandbox API

基础地址：

- dev：`http://127.0.0.1:8766`
- prod：`http://127.0.0.1:8767`

下文示例默认使用 dev。测试 prod 时将端口替换为 `8767`。

请求体和响应体均为 JSON。带请求体的接口需设置 `Content-Type: application/json`。

## GPU 资源池

### `POST /v1/gpu_pool`

创建 GPU 资源池。接口返回 `202` 后，需要继续查询资源池，等待 `workers.idle > 0`。

`node_selector` 和 `tolerations` 整体替换 Worker Pod 模板配置；省略或 `null` 保留默认值，`{}`／`[]` 清空。`tolerations` 元素字段为 `key`、`operator`、`value`、`effect`、`toleration_seconds`；最后一项仅适用于 `NoExecute`，省略表示无限期容忍。

请求参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `image` | string | 是 | GPU Worker 业务镜像，不能为空 |
| `replicas` | integer | 是 | Worker 副本数，必须大于 0 |
| `node_selector` | object | 否 | 替换 Worker Pod 的节点选择配置 |
| `tolerations` | array | 否 | 替换 Worker Pod 的污点容忍配置 |

示例命令：

```bash
curl -sS -X POST http://127.0.0.1:8766/v1/gpu_pool \
  -H 'Content-Type: application/json' \
  -d '{
    "image":"ghcr.io/baidu-baige/koda-sandbox:py312-cuda129-torch28-2026110922",
    "replicas":1
  }'
```

成功响应：HTTP `202`

```json
{
  "gpu_pool_id": "koda-sandbox-gpu-pool-a1b2c3d4e5f67890",
  "image": "ghcr.io/baidu-baige/koda-sandbox:py312-cuda129-torch28-2026110922",
  "replicas": 1,
  "workers": {"pending": 0, "idle": 0, "busy": 0},
  "pods": [],
  "timing": {"create_ms": 48}
}
```

错误响应：

| HTTP 状态 | 错误码 | 说明 |
| --- | --- | --- |
| `400` | `BAD_REQUEST` | `image` 为空或 `replicas <= 0` |
| `500` | `GPU_POOL_CREATE_FAILED` | Kubernetes Deployment 创建失败 |

### `GET /v1/gpu_pool`

列出全部 GPU 资源池。

示例命令：

```bash
curl -sS http://127.0.0.1:8766/v1/gpu_pool
```

成功响应：HTTP `200`

```json
[
  {
    "gpu_pool_id": "koda-sandbox-gpu-pool-a1b2c3d4e5f67890",
    "image": "ghcr.io/baidu-baige/koda-sandbox:py312-cuda129-torch28-2026110922",
    "replicas": 1,
    "workers": {"pending": 0, "idle": 1, "busy": 0},
    "pods": [
      {
        "pod": "koda-sandbox-gpu-pool-a1b2c3d4e5f67890-6f56b85c7f-abcd1",
        "status": "idle",
        "ready": true,
        "restart_count": 0
      }
    ],
    "timing": {"create_ms": 48, "ready_ms": 17120}
  }
]
```

没有资源池时返回 `[]`。

字段说明：

- `workers.pending`：尚不可调度的 Worker 数量。
- `workers.idle`：当前可接收命令的 Worker 数量。
- `workers.busy`：正在执行命令或执行后检查的 Worker 数量。
- `pods[].status`：`pending`、`idle`、`busy` 或 `deleting`。
- `timing.ready_ms`：从创建资源池到首次出现可调度 Worker 的耗时。
- `usage.busy_ms`：累计 Worker 占用毫秒数，包含预检、命令执行和后检。
- `usage.capacity_ms`：累计可用 Worker 容量毫秒数；每个 Worker 从首次 ready 到退出积分，pending 不计入。
- `usage.workers`：按 Pod 名保存 `{busy_ms, capacity_ms}`；退出 Worker 的历史保留，同名替换继续累计。扩缩容不会重置池计时。
- `usage.queue_length`：当前等待调度的 GPU 命令数。
- `usage.queue.count`：已结束（完成、失败或超时）的 GPU 命令数，重试只计一次。
- `usage.queue.total_ms`：上述命令累计排队毫秒数；一次命令多次重试时，累计各次等待时间。
- `usage.queue.max_ms`：Master 进程生命周期内单条 GPU 命令的最大累计排队毫秒数。
- `cluster`：Pool 所属的 `gpu_pool_clusters[].name`；默认集群为空。
- `usage` 是 Master 进程内计数；Master 重启后重新开始。旧版 Master 不返回该字段。

### `GET /v1/gpu_pool/{gpu_pool_id}`

查询指定 GPU 资源池。

路径参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `gpu_pool_id` | string | 是 | GPU 资源池 ID |

示例命令：

```bash
curl -sS \
  http://127.0.0.1:8766/v1/gpu_pool/koda-sandbox-gpu-pool-a1b2c3d4e5f67890
```

成功响应：HTTP `200`

```json
{
  "gpu_pool_id": "koda-sandbox-gpu-pool-a1b2c3d4e5f67890",
  "image": "ghcr.io/baidu-baige/koda-sandbox:py312-cuda129-torch28-2026110922",
  "replicas": 1,
  "workers": {"pending": 0, "idle": 1, "busy": 0},
  "pods": [
    {
      "pod": "koda-sandbox-gpu-pool-a1b2c3d4e5f67890-6f56b85c7f-abcd1",
      "status": "idle",
      "ready": true,
      "restart_count": 0
    }
  ],
  "timing": {"create_ms": 48, "ready_ms": 17120}
}
```

错误响应：

| HTTP 状态 | 错误码 | 说明 |
| --- | --- | --- |
| `404` | `GPU_POOL_NOT_FOUND` | 资源池不存在或正在删除 |

### `PATCH /v1/gpu_pool/{gpu_pool_id}`

更新资源池镜像、副本数或调度配置，至少提供一个更新字段。

路径参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `gpu_pool_id` | string | 是 | GPU 资源池 ID |

请求参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `image` | string | 否 | 新的 GPU Worker 业务镜像 |
| `replicas` | integer | 否 | 新副本数，必须大于 0 |
| `node_selector` | object | 否 | 整体替换，`{}` 清空；省略／`null` 保留当前值 |
| `tolerations` | array | 否 | 整体替换，`[]` 清空；省略／`null` 保留当前值 |

示例命令：

```bash
curl -sS -X PATCH \
  http://127.0.0.1:8766/v1/gpu_pool/<gpu_pool_id> \
  -H 'Content-Type: application/json' \
  -d '{
    "image":"YOUR_REGISTRY/sandbox:VERSION",
    "replicas":2
  }'
```

成功响应：HTTP `200`

```json
{
  "gpu_pool_id": "koda-sandbox-gpu-pool-a1b2c3d4e5f67890",
  "image": "YOUR_REGISTRY/sandbox:VERSION",
  "replicas": 2,
  "workers": {"pending": 0, "idle": 1, "busy": 0},
  "pods": [
    {
      "pod": "koda-sandbox-gpu-pool-a1b2c3d4e5f67890-6f56b85c7f-abcd1",
      "status": "idle",
      "ready": true,
      "restart_count": 0
    }
  ],
  "timing": {"create_ms": 48, "ready_ms": 17120}
}
```

镜像滚动更新和新 Pod Ready 是异步过程，应继续调用 GET 检查状态。

错误响应：

| HTTP 状态 | 错误码 | 说明 |
| --- | --- | --- |
| `400` | `BAD_REQUEST` | 没有提供有效字段或 `replicas <= 0` |
| `404` | `GPU_POOL_NOT_FOUND` | 资源池不存在或正在删除 |
| `500` | `GPU_POOL_UPDATE_FAILED` | Kubernetes Deployment 更新失败 |

### `DELETE /v1/gpu_pool/{gpu_pool_id}`

删除 GPU 资源池。必须先删除绑定该资源池的全部 Session。

路径参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `gpu_pool_id` | string | 是 | GPU 资源池 ID |

示例命令：

```bash
curl -sS -X DELETE \
  http://127.0.0.1:8766/v1/gpu_pool/<gpu_pool_id>
```

成功响应：HTTP `202`

```json
{"status":"deletion_requested"}
```

错误响应：

| HTTP 状态 | 错误码 | 说明 |
| --- | --- | --- |
| `404` | `GPU_POOL_NOT_FOUND` | 资源池不存在或正在删除 |
| `409` | `GPU_POOL_IN_USE` | 仍有 Session 绑定该资源池 |
| `500` | `GPU_POOL_DELETE_FAILED` | Kubernetes Deployment 删除失败 |

## Session

### `POST /v1/session`

创建 Session。`gpu_only` 创建独占 GPU Pod；`cpu_gpu` 创建持久化 CPU Pod，并通过共享 GPU Pool 执行 GPU 命令。

创建 Session 时可传 `prefix`，省略或为空时使用服务端配置，须满足 Kubernetes Pod 命名和总长度限制。

请求参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `session_type` | string | 是 | `gpu_only` 或 `cpu_gpu` |
| `gpu_pool_id` | string | 条件必填 | 仅 `cpu_gpu` 必填；`gpu_only` 不绑定资源池 |
| `image` | string | 条件必填 | `gpu_only` 必填；`cpu_gpu` 可选，省略时使用资源池镜像 |
| `resources.requests` | object | 条件必填 | `gpu_only` 必填；覆盖独占 GPU Pod 模板中的同名 requests，并用于选择 GPU 集群 |
| `resources.limits` | object | 否 | 覆盖 Pod 模板中的同名 limits；`cpu_gpu` 仅保留 `cpu`、`memory` |
| `prefix` | string | 否 | 本 Session 的 Pod 名称前缀，Session ID 保持原格式 |

`resources.requests` 和 `resources.limits` 的 key 是完整 Kubernetes 资源名。H20 模板默认使用 `example.org/gpu_hopper: "1"`，用户请求中的同名字段会覆盖模板默认值。请求中仍须显式携带该资源名用于集群路由。Master 只从 `gpu_session_clusters` 中选择集群：必须恰好匹配一个集群；未匹配或同时匹配多个集群都会返回 `BAD_REQUEST`，不会回落到共享 `gpu_pool_cluster`。

创建 GPU-only Session：

```bash
curl -sS -X POST http://127.0.0.1:8766/v1/session \
  -H 'Content-Type: application/json' \
  -d '{
    "session_type":"gpu_only",
    "image":"ghcr.io/baidu-baige/koda-sandbox:py312-cuda129-torch28-2026110922",
    "resources":{
      "requests":{
        "cpu":"10",
        "memory":"100G",
        "example.org/gpu_hopper":"1"
      },
      "limits":{"example.org/gpu_hopper":"1"}
    }
  }'
```

成功响应：HTTP `202`

```json
{
  "session_id": "gpu-session-a1b2c3d4e5f67890",
  "session_type": "gpu_only",
  "status": "creating",
  "created_at": "2026-08-03T14:08:00.123456789+08:00",
  "gpu": {
    "image": "ghcr.io/baidu-baige/koda-sandbox:py312-cuda129-torch28-2026110922",
    "pod": "koda-sandbox-dev-gpu-session-a1b2c3d4e5f67890",
    "ready": false,
    "resources": {
      "requests": {
        "cpu":"10",
        "memory":"100G",
        "example.org/gpu_hopper":"1"
      },
      "limits":{"example.org/gpu_hopper":"1"}
    }
  },
  "timing": {"create_ms": 51}
}
```

`gpu_only` Session 也需要继续调用 GET，等待独占 GPU Pod Ready 后 `status=ready`。

创建 CPU-GPU Session：

```bash
curl -sS -X POST http://127.0.0.1:8766/v1/session \
  -H 'Content-Type: application/json' \
  -d '{
    "session_type":"cpu_gpu",
    "gpu_pool_id":"<gpu_pool_id>",
    "image":"YOUR_REGISTRY/sandbox:VERSION",
    "resources":{"requests":{"cpu":"1","memory":"1Gi"}}
  }'
```

成功响应：HTTP `202`

```json
{
  "session_id": "session-a1b2c3d4e5f67890",
  "gpu_pool_id": "koda-sandbox-gpu-pool-a1b2c3d4e5f67890",
  "session_type": "cpu_gpu",
  "status": "creating",
  "created_at": "2026-08-03T14:08:00.123456789+08:00",
  "cpu": {
    "image": "YOUR_REGISTRY/sandbox:VERSION",
    "pod": "koda-sandbox-cpu-session-a1b2c3d4e5f67890",
    "ready": false,
    "resources": {"requests":{"cpu":"1","memory":"1Gi"}}
  },
  "timing": {"create_ms": 51}
}
```

`cpu_gpu` Session 需要继续调用 GET，等待 `status=ready`。

错误响应：

| HTTP 状态 | 错误码 | 说明 |
| --- | --- | --- |
| `400` | `BAD_REQUEST` | Session 类型无效、`cpu_gpu` 缺少资源池 ID，或 `gpu_only` 缺少 image/resources.requests |
| `404` | `GPU_POOL_NOT_FOUND` | `cpu_gpu` 指定的 GPU 资源池不存在 |
| `409` | `SESSION_DELETED` | Pod 创建期间 Session 被删除 |
| `500` | `CPU_POD_CREATE_FAILED` | CPU Pod 创建失败 |
| `500` | `GPU_POD_CREATE_FAILED` | 独占 GPU Pod 创建失败 |

### `GET /v1/session`

列出全部 Session。

示例命令：

```bash
curl -sS http://127.0.0.1:8766/v1/session
```

成功响应：HTTP `200`

```json
[
  {
    "session_id": "session-a1b2c3d4e5f67890",
    "gpu_pool_id": "koda-sandbox-gpu-pool-a1b2c3d4e5f67890",
    "session_type": "cpu_gpu",
    "status": "ready",
    "created_at": "2026-08-03T14:08:00.123456789+08:00",
    "cpu": {
      "image": "YOUR_REGISTRY/sandbox:VERSION",
      "pod": "koda-sandbox-cpu-session-a1b2c3d4e5f67890",
      "ready": true,
      "resources": {"requests":{"cpu":"1","memory":"1Gi"}}
    },
    "timing": {"create_ms":51,"ready_ms":2984}
  }
]
```

没有 Session 时返回 `[]`。

Master 重启后会根据 Kubernetes 中仍存在的 CPU Session Pod 和 `gpu_only` 独占 GPU Pod 恢复 Session。恢复后的 `timing` 为空；重启前正在排队或执行的命令不会恢复。运行期间独占 GPU Pod 变为 NotReady 时 Session 回到 `creating`，Pod 失败、正在删除或消失时 Session 进入 `failed`。

### `GET /v1/session/{session_id}`

查询指定 Session。

路径参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `session_id` | string | 是 | Session ID |

示例命令：

```bash
curl -sS http://127.0.0.1:8766/v1/session/<session_id>
```

成功响应：HTTP `200`

```json
{
  "session_id": "session-a1b2c3d4e5f67890",
  "gpu_pool_id": "koda-sandbox-gpu-pool-a1b2c3d4e5f67890",
  "session_type": "cpu_gpu",
  "status": "ready",
  "created_at": "2026-08-03T14:08:00.123456789+08:00",
  "cpu": {
    "image": "YOUR_REGISTRY/sandbox:VERSION",
    "pod": "koda-sandbox-cpu-session-a1b2c3d4e5f67890",
    "ready": true,
    "resources": {"requests":{"cpu":"1","memory":"1Gi"}}
  },
  "timing": {"create_ms":51,"ready_ms":2984}
}
```

错误响应：

| HTTP 状态 | 错误码 | 说明 |
| --- | --- | --- |
| `404` | `SESSION_NOT_FOUND` | Session 不存在或正在删除 |

### `DELETE /v1/session/{session_id}`

删除指定 Session 及其独占 CPU/GPU Pod，并终止该 Session 正在排队或执行的命令。

路径参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `session_id` | string | 是 | Session ID |

示例命令：

```bash
curl -sS -X DELETE \
  http://127.0.0.1:8766/v1/session/<session_id>
```

成功响应：HTTP `202`

```json
{"status":"deletion_requested"}
```

错误响应：

| HTTP 状态 | 错误码 | 说明 |
| --- | --- | --- |
| `404` | `SESSION_NOT_FOUND` | Session 不存在或正在删除 |
| `500` | `SESSION_DELETE_FAILED` | 关联 Pod 删除失败 |

### `DELETE /v1/session`

删除全部 Session，保留 GPU 资源池。

可传查询参数 `prefix`，只删除 Pod 名称以 `<prefix>-` 开头的 Session，覆盖 `cpu_gpu` 和 `gpu_only`，也包括创建中的 Session。空值或非法前缀返回 `400 BAD_REQUEST`；未匹配时返回 `202`。省略该参数仍删除全部 Session。

```bash
curl -sS -X DELETE 'http://127.0.0.1:8766/v1/session?prefix=my-job'
```

示例命令：

```bash
curl -sS -X DELETE http://127.0.0.1:8766/v1/session
```

成功响应：HTTP `202`

```json
{"status":"deletion_requested"}
```

错误响应：

| HTTP 状态 | 错误码 | 说明 |
| --- | --- | --- |
| `500` | `CLEANUP_FAILED` | 至少一个 Session 删除失败 |

## 命令执行

### `POST /v1/session/{session_id}/command_exec`

在 Session 中同步执行 CPU 或 GPU 命令。`cpu_gpu` 的 GPU 命令进入绑定资源池的共享队列；`gpu_only` 的命令直接在独占 GPU Pod 中执行。

路径参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `session_id` | string | 是 | Session ID |

请求参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `device` | string | `cpu_gpu` 必填 | `cpu_gpu` 使用 `cpu` 或 `gpu`；`gpu_only` 由 Session 类型固定为 GPU，可省略 |
| `command` | string[] | 是 | argv 数组，不能为空 |
| `working_dir` | string | 否 | 容器内工作目录 |
| `env` | object | 否 | 环境变量，名称需合法 |
| `stdin` | string | 否 | 标准输入；当前上限 4 MiB |
| `timeout_seconds` | integer | 是 | 总超时时间，必须大于 0 |

命令按 argv 直接执行，不经过 shell。需要管道、重定向或变量展开时使用 `bash -lc`。

`gpu_only` 会忽略请求中的 `device` 并固定在独占 GPU Pod 中执行。

GPU 示例命令：

```bash
curl -sS -X POST \
  http://127.0.0.1:8766/v1/session/<session_id>/command_exec \
  -H 'Content-Type: application/json' \
  -d '{
    "device":"gpu",
    "command":["hostname"],
    "timeout_seconds":60
  }'
```

CPU 示例命令：

```bash
curl -sS -X POST \
  http://127.0.0.1:8766/v1/session/<session_id>/command_exec \
  -H 'Content-Type: application/json' \
  -d '{
    "device":"cpu",
    "command":["bash","-lc","echo cpu-ok"],
    "working_dir":"/tmp",
    "env":{"CASE":"1"},
    "stdin":"input",
    "timeout_seconds":60
  }'
```

成功响应：HTTP `200`

```json
{
  "execution_id": "execution-a1b2c3d4e5f67890",
  "status": "completed",
  "exit_code": 0,
  "stdout": "worker-pod-name\n",
  "stderr": "",
  "extra": {
    "sandbox_timings": {
      "gpu_executions": [
        {
          "total_ms": 412,
          "queue_ms": 18,
          "precheck_ms": 72,
          "exec_ms": 315,
          "attempts": 1
        }
      ]
    }
  }
}
```

返回说明：

- `status=completed` 表示拿到了退出码，`exit_code` 仍可能非零。
- `status=timed_out` 表示超时。
- `status=failed` 表示启动失败、输出超限或执行基础设施失败。
- 当前 `stdout` 与 `stderr` 合计上限为 1 MiB。
- GPU 的 `timeout_seconds` 包含排队、重试和执行时间。
- `queue_ms` 和 `precheck_ms` 仅用于 `cpu_gpu` 的共享 GPU Pool 执行。
- `gpu_only` 命令没有共享队列、Worker 前后检查或基础设施自动重试。
- 同一 Session 同时只允许一条 CPU 命令，否则返回 `CPU_BUSY`。
- 同一 `gpu_only` Session 支持多条命令并发执行。

CPU 命令触发 GPU 命令时，响应中的 timing 形式为：

```json
{
  "sandbox_timings": {
    "cpu_execution": {
      "total_ms": 15459,
      "exec_ms": 15459,
      "attempts": 1
    },
    "gpu_executions": [
      {
        "total_ms": 15269,
        "queue_ms": 12036,
        "precheck_ms": 300,
        "exec_ms": 2932,
        "attempts": 2
      }
    ]
  }
}
```

错误响应：

| HTTP 状态 | 错误码 | 说明 |
| --- | --- | --- |
| `400` | `BAD_REQUEST` | 请求字段无效 |
| `404` | `SESSION_NOT_FOUND` | Session 不存在 |
| `404` | `GPU_POOL_NOT_FOUND` | 绑定的 GPU 资源池不存在 |
| `409` | `SESSION_NOT_READY` | Session 未 Ready |
| `409` | `CPU_NOT_AVAILABLE` | Session 没有 CPU Pod |
| `409` | `CPU_BUSY` | 已有 CPU 命令正在运行 |
| `503` | `INFRASTRUCTURE_ERROR` | GPU 基础设施重试耗尽 |

以下命令级错误以 HTTP `200` 返回：

| `status` | `error_code` | 说明 |
| --- | --- | --- |
| `timed_out` | `COMMAND_TIMEOUT` | 排队或执行超时 |
| `failed` | `OUTPUT_LIMIT_EXCEEDED` | stdout 与 stderr 合计超限 |
| `failed` | `COMMAND_START_FAILED` | 命令无法启动 |
| `failed` | `SESSION_DELETED` | 执行期间 Session 被删除 |

## 健康检查

### `GET /healthz`

检查 Master HTTP 服务是否存活。

示例命令：

```bash
curl -sS http://127.0.0.1:8766/healthz
```

成功响应：HTTP `200`

```json
{"status":"ok"}
```

## 通用错误格式

```json
{
  "error_code": "SESSION_NOT_FOUND",
  "message": "session does not exist"
}
```

JSON 格式错误、未知字段或一个请求体中包含多个 JSON 值时，返回 HTTP `400`、`INVALID_JSON`。
