"""CPU session workspace and command execution through the Sandbox Master API."""

import atexit
import logging
import platform
import pathlib
import re
import shlex
import time
from typing import Any

import requests
from pydantic import BaseModel, Field

from minisweagent.exceptions import SandboxInfraError, Submitted
from minisweagent.utils.serialize import recursive_merge

logger = logging.getLogger(__name__)


def _is_retryable(error: Exception | None = None, status_code: int | None = None) -> bool:
    """Return True if the error/status indicates a transient infrastructure issue."""
    if error is not None:
        return isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))
    if status_code is not None:
        return status_code >= 500 or status_code == 429
    return False


def _is_infra_fatal(response: requests.Response) -> bool:
    """True only for the unrecoverable "session gone" case (session pod evicted).

    Two shapes both mean the session no longer exists server-side, so retrying command_exec here is
    pure空转 — the caller must re-run the problem on a fresh session:
      1. HTTP 404 on the session-scoped command_exec URL (session id not registered anymore);
      2. ``error_code == "INFRASTRUCTURE_ERROR"`` **and** a message shaped like
         ``pods "...-session-XXX" not found``.
    Other INFRASTRUCTURE_ERROR variants (transient scheduling hiccups etc.) fall through to the
    normal 5xx retry path rather than aborting the session.
    """
    if response.status_code == 404:
        return True
    try:
        payload = response.json()
    except (ValueError, AttributeError):
        return False
    if not isinstance(payload, dict) or payload.get("error_code") != "INFRASTRUCTURE_ERROR":
        return False
    return bool(re.search(r'pods?\b.*\bnot found', str(payload.get("message", "")), re.IGNORECASE))


class KernelSandboxEnvironmentConfig(BaseModel):
    """Configuration for a Kernel Sandbox environment."""

    base_url: str
    """Kernel Sandbox master endpoint."""
    image: str
    """Image for the shared GPU worker pool."""
    gpu_pool_id: str
    """Id of the GPU worker pool this session is bound to (required by POST /v1/session)."""
    cpu_image: str = ""
    """Optional CPU pod image (sent as cpu_image on session create). Empty = master default."""
    session_prefix: str = ""
    """Pod name prefix for this session (sent as `prefix` on session create); the session id keeps its
    format. Must be non-empty — an empty value is rejected at session creation. Must satisfy k8s Pod
    naming + total length limits, and enables prefix-scoped bulk deletion (DELETE /v1/session?prefix=...)."""
    problem_source: str
    """KernelBench problem source; sent to the pod's instance builder on stdin."""
    workspace_name: str
    """Task id recorded in the workspace metadata."""
    modeling: str = ""
    """HF modeling_<arch>.py source for level_4 tasks (empty otherwise); injected into the loop prompt."""
    config: str = ""
    """HF config.json source for level_4 tasks (empty otherwise); injected into the loop prompt."""
    backend: str = "triton"
    """Candidate kernel backend."""
    gpu: str | None = None
    """Target GPU name recorded in the task."""
    gpu_replicas: int = 1
    """Replica count for the shared GPU worker pool."""
    resource_request: dict = Field(default_factory=dict)
    """k8s requests applied to the shared GPU worker (GPU pod) deployment template on pool create:
    they override the same-named resources in the selected cluster's template (names not listed keep
    the template default), and a GPU resource name also routes the pool to the cluster that owns it
    (gpu_pool_clusters.resource_names). Empty (default) = the master's default cluster with that
    template's default resources. Consumed by the runner when it creates the pool, NOT by the cpu_gpu
    session (the session inherits the pool's cluster via gpu_pool_id and never routes on its own)."""
    resource_limit: dict = Field(default_factory=dict)
    """k8s limits paired with resource_request; override the GPU pod template's same-named limits
    (quantities stringified). Empty = omit."""
    cwd: str = "/workspace"
    """Working directory on the CPU pod; the workspace root."""
    timeout: int = 600
    """Per-command timeout in seconds."""
    session_ready_timeout: int = 600
    """How long to wait for the CPU pod session to become ready."""
    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables set for every executed command."""
    max_retries: int = 5
    """Maximum number of retries for transient sandbox failures."""
    retry_backoff: float = 60.0
    """Base backoff in seconds between retries (doubles each attempt)."""


class KernelSandboxEnvironment:
    """Execute agent commands in a managed Kernel Sandbox session."""

    def __init__(self, *, config_class: type = KernelSandboxEnvironmentConfig, **kwargs):
        """Ensure the GPU pool, create a session, then build the workspace on its CPU pod."""
        self.config = config_class(**kwargs)
        self.base_url = self.config.base_url.rstrip("/")
        self.client = requests.Session()
        self.session_id = ""
        self._closed = False
        atexit.register(self.cleanup)
        try:
            self._create_session()
            self._build_workspace()
        except Exception:
            self.cleanup()
            raise

    def _create_session(self) -> None:
        """Create a cpu_gpu session and wait until it is ready."""
        _payload = {"session_type": "cpu_gpu", "gpu_pool_id": self.config.gpu_pool_id}
        if self.config.cpu_image:
            # master schema 的 CPU pod 镜像字段是 `image`（CreateSessionRequest json:"image"），
            # `cpu_image` 只是 master 日志的 log field —— 2026-08-31 实测 400 unknown field
            # 该字段留空 = 用 pool 镜像 (request.Image == "" → pool.image)。
            _payload["image"] = self.config.cpu_image
        if not self.config.session_prefix:
            raise ValueError("session_prefix must not be empty")
        _payload["prefix"] = self.config.session_prefix
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.client.post(f"{self.base_url}/v1/session", json=_payload, timeout=30)
                if _is_retryable(status_code=response.status_code) and attempt < self.config.max_retries:
                    logger.warning(f"Session create returned {response.status_code}, retry {attempt}")
                    time.sleep(self.config.retry_backoff * (2 ** (attempt - 1)))
                    continue
                if not response.ok:
                    logger.error(f"Session create HTTP {response.status_code}: {response.text}")
                response.raise_for_status()
                self.session_id = response.json()["session_id"]
                break
            except requests.exceptions.RequestException as error:
                if _is_retryable(error=error) and attempt < self.config.max_retries:
                    logger.warning(f"Session create failed ({error}), retry {attempt}")
                    time.sleep(self.config.retry_backoff * (2 ** (attempt - 1)))
                    continue
                raise
        deadline = time.monotonic() + self.config.session_ready_timeout
        while time.monotonic() < deadline:
            try:
                response = self.client.get(f"{self.base_url}/v1/session/{self.session_id}", timeout=30)
                response.raise_for_status()
                session = response.json()
            except requests.exceptions.RequestException:
                time.sleep(2)
                continue
            if session["status"] == "ready":
                return
            if session["status"] == "failed":
                raise RuntimeError(session.get("error") or "CPU Session creation failed")
            time.sleep(1)
        raise TimeoutError(f"Session {self.session_id} was not ready within {self.config.session_ready_timeout}s")

    def _build_workspace(self) -> None:
        """Lay down task.py (the reference source) at the workspace root on the CPU pod.

        cpu_gpu layout = task.py only (no docs/problem.json), mirroring kernel_opencode's
        _setup_workspace; the eval envelope reads it back via `--ref-file task.py`.

        沙箱 wrapper 的脚本第 1 行就是 ``cd $working_dir`` —— working_dir 指向
        尚不存在的目录时 cd 先挂 (rc=127), 命令体(含 mkdir)根本不会执行
        (2026-08-31 实证: cwd=/workspace/kb_ws 首次 build 报 can't cd)。
        故建目录这一步的 working_dir 用已存在的**父目录**, 命令内再 mkdir+cd。
        """
        ws = shlex.quote(self.config.cwd)
        parent = shlex.quote(str(pathlib.Path(self.config.cwd).parent))
        d_task = "KB_TASK_EOF"
        while d_task in self.config.problem_source:
            d_task = "_" + d_task
        command = (
            "set -e\n"
            f"mkdir -p {ws}\n"
            f"cd {ws}\n"
            f"cat > task.py <<'{d_task}'\n{self.config.problem_source}\n{d_task}\n"
            "echo WROTE_OK"
        )
        for attempt in range(1, self.config.max_retries + 1):
            output = self.execute({"command": command, "device": "cpu"}, cwd=parent)
            if output["returncode"] == 0 and "WROTE_OK" in output.get("output", ""):
                return
            if output.get("extra", {}).get("exception_type") and attempt < self.config.max_retries:
                logger.warning(f"Workspace build infra error, retry {attempt}: {output.get('exception_info', '')}")
                time.sleep(self.config.retry_backoff * (2 ** (attempt - 1)))
                continue
            break
        if output["returncode"] != 0:
            raise RuntimeError(f"Workspace build failed in sandbox session {self.session_id}: {output}")

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute one shell command in the active sandbox session on the action's device."""
        command_timeout = timeout if timeout is not None else self.config.timeout
        body = {
            "device": action.get("device", "cpu"),
            "command": ["bash", "-lc", action.get("command", "")],
            "working_dir": cwd or self.config.cwd,
            "env": {**self.config.env, **(action.get("env") or {})},
            "timeout_seconds": command_timeout,
        }
        if "stdin" in action:
            body["stdin"] = action["stdin"]
        last_error: Exception | None = None
        payload = {}
        for attempt in range(1, self.config.max_retries + 1):
            payload = {}
            try:
                response = self.client.post(
                    f"{self.base_url}/v1/session/{self.session_id}/command_exec",
                    json=body,
                    timeout=command_timeout + 60,
                )
                if not response.ok and _is_infra_fatal(response):
                    logger.error(
                        f"command_exec infra-fatal (HTTP {response.status_code}) for session "
                        f"{self.session_id}: {response.text}"
                    )
                    raise SandboxInfraError(
                        f"sandbox session {self.session_id} unavailable (HTTP {response.status_code})"
                    )
                if _is_retryable(status_code=response.status_code) and attempt < self.config.max_retries:
                    logger.warning(f"command_exec returned {response.status_code}, retry {attempt}")
                    time.sleep(self.config.retry_backoff * (2 ** (attempt - 1)))
                    continue
                payload = response.json()
                if not response.ok:
                    logger.error(
                        f"command_exec HTTP {response.status_code} for session {self.session_id}: "
                        f"{payload}"
                    )
                response.raise_for_status()
                output = {
                    "output": payload.get("stdout", "") + payload.get("stderr", ""),
                    "returncode": payload.get("exit_code") if payload.get("exit_code") is not None else -1,
                    "exception_info": payload.get("error_code", ""),
                    "extra": {
                        **(payload.get("extra") or {}),
                        "execution_id": payload.get("execution_id"),
                        "status": payload.get("status"),
                        "error_code": payload.get("error_code"),
                    },
                }
                self._check_finished(output)
                return output
            except requests.exceptions.RequestException as error:
                last_error = error
                if _is_retryable(error=error) and attempt < self.config.max_retries:
                    logger.warning(f"command_exec failed ({error}), retry {attempt}")
                    time.sleep(self.config.retry_backoff * (2 ** (attempt - 1)))
                    continue
                logger.error(f"command_exec request failed for session {self.session_id}: {error}")
                break
        output = {
            "output": "",
            "returncode": -1,
            "exception_info": str(last_error),
            "extra": {
                **(payload.get("extra") or {}),
                "error_code": payload.get("error_code"),
                "exception_type": type(last_error).__name__, "exception": str(last_error),
            },
        }
        self._check_finished(output)
        return output

    def write_candidate(self, kernel_py: str, kernel_cu: str | None = None) -> None:
        """Write kernel.py (+ optional kernel.cu) at the workspace root on the CPU pod.

        Round candidate layout mirrors kernel_gpu_only.write_candidate: the candidate
        file(s) at the workspace root, stale build artifacts cleared (workspace copy
        and site-packages copy) so a round never reuses an old cuda_extension.so.
        The opencode-tool executors read kernel.py back via `--model-file`.

        清 eval_result.json 同理: CPU Pod 在会话内保留文件状态 (design.md:31/308), 而
        run_eval 只在跑完才写 --output (run_eval.py:137)。本轮 eval 若超时被 kill / 崩在
        写文件前, exec_eval 的 cat 兜底会读到**上一轮**的旧裁决冒充本轮。开轮即删, 兜底就
        只可能读到本轮真正写出的结果, 否则文件不存在 → 合成明确错误。
        """
        ws = shlex.quote(self.config.cwd)
        d_py = "KB_KPY_EOF"
        while d_py in kernel_py:
            d_py = "_" + d_py
        parts = [
            "set -e",
            f"cd {ws}",
            "rm -rf cuda_extension.so build kernel.cu kernels eval_result.json",
            'SP=$(python3 -c "import site;print(site.getsitepackages()[0])" 2>/dev/null); '
            '[ -n "$SP" ] && rm -f "$SP/cuda_extension.so" 2>/dev/null || true',
            f"cat > kernel.py <<'{d_py}'\n{kernel_py}\n{d_py}",
        ]
        if kernel_cu:
            d_cu = "KB_KCU_EOF"
            while d_cu in kernel_cu:
                d_cu = "_" + d_cu
            parts.append(f"cat > kernel.cu <<'{d_cu}'\n{kernel_cu}\n{d_cu}")
        parts.append("echo WROTE_OK")
        out = self.execute({"command": "\n".join(parts)})
        if out["returncode"] != 0 or "WROTE_OK" not in out.get("output", ""):
            raise RuntimeError(f"write_candidate failed in cpu_gpu session {self.session_id}: {out}")

    @staticmethod
    def _check_finished(output: dict) -> None:
        """Raise Submitted if the output indicates task completion (same contract as LocalEnvironment)."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        """Return environment values available to prompt templates."""
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        """Serialize the environment configuration and session metadata."""
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json") | {"session_id": self.session_id},
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def cleanup(self) -> None:
        """Close the active sandbox session and its HTTP client."""
        if self._closed:
            return
        if self.session_id:
            try:
                self.client.delete(f"{self.base_url}/v1/session/{self.session_id}", timeout=30).raise_for_status()
            except requests.exceptions.RequestException:
                return
        self._closed = True
        self.client.close()
