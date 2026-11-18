"""kernel_loop._read_env_info 单测：版本提示只是 prompt 增益，永不阻断 run。

三个不变量：
1. 探测必须落在**真正跑 eval 的 GPU worker**（device="gpu"），且不能继承会话的
   6000s 超时、不能用 CPU pod 的 cwd —— prompt 断言的是 "the eval environment"。
2. 探测走 sandbox 的 env.execute，会因 HTTP 抖动抛 RequestException；一次抖动带走整批
   270 题是不可接受的，故失败必须降级成空串。
3. 空结果**不进永久缓存**：否则一次抖动让整批 run 永久失去版本提示且不可恢复。但也不能
   每题立刻重试（镜像真没 python3 时是 270 次 GPU 槽位空转），故失败压制 5 分钟后自愈。
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from minisweagent.agents import controller_runtime
from minisweagent.agents.controller_runtime import (
    _ENV_INFO_CACHE,
    _PROBE_FAILED_AT,
    _PROBE_RETRY_AFTER_S,
    _PROBE_TIMEOUT_S,
    _read_env_info,
)


class _Env:
    def __init__(self, result, config=None):
        self.config = config
        self.result = result
        self.calls = 0
        self.action: dict = {}
        self.kwargs: dict = {}

    def execute(self, action, cwd: str = "", **kwargs) -> dict:
        self.calls += 1
        self.action, self.kwargs = action, {"cwd": cwd, **kwargs}
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


# _read_env_info 恒回满形 dict; CUTLASS/手写骨架探测缺失时为空串 (compile_dir=None 时整块不注入)。
_EMPTY = {"versions": "", "triton_api_hints": "", "gpu_specs": "",
          "cutlass_version": "", "cutlass_example": "", "handwritten_example": "",
          "cutlass3x_example": ""}


@pytest.fixture(autouse=True)
def _clear_cache():
    warmup = controller_runtime._PROBE_COLD_WARMUP_S
    controller_runtime._PROBE_COLD_WARMUP_S = 0
    _ENV_INFO_CACHE.clear()
    _PROBE_FAILED_AT.clear()
    yield
    controller_runtime._PROBE_COLD_WARMUP_S = warmup
    _ENV_INFO_CACHE.clear()
    _PROBE_FAILED_AT.clear()


def test_probe_success_parses_both_lines_despite_stderr_noise():
    """stderr 与 stdout 合并进 output，解析必须靠前缀而非行号。"""
    env = _Env({
        "returncode": 0,
        "output": "UserWarning: TF32 is on\nVERSIONS:Python 3.12.12 / torch 2.9.0+cu129 / triton 3.5.0\n"
                  "GPU_SPECS:NVIDIA H20 | compute capability 9.0 | 132 SM | 227.0 KB shared mem/SM | 96.0 GB device memory\n"
                  "TRITON_HINTS:Available: dot(input, other, acc) | NOT available: tl.num_programs\n",
    })
    assert _read_env_info(env) == _EMPTY | {
        "versions": "Python 3.12.12 / torch 2.9.0+cu129 / triton 3.5.0",
        "triton_api_hints": "Available: dot(input, other, acc) | NOT available: tl.num_programs",
        "gpu_specs": "NVIDIA H20 | compute capability 9.0 | 132 SM | 227.0 KB shared mem/SM | 96.0 GB device memory",
    }


def test_probe_runs_on_the_gpu_worker_with_its_own_short_timeout():
    """cpu_gpu 默认 device=cpu 会探到控制器 pod；cwd 是 CPU pod 的工作区，GPU worker 上可能不存在。"""
    env = _Env({"returncode": 0, "output": "VERSIONS:torch 2.9.0\nTRITON_HINTS:\n"})
    _read_env_info(env)
    assert (env.action["device"], env.kwargs["cwd"], env.kwargs["timeout"]) == ("gpu", "/", _PROBE_TIMEOUT_S)
    assert _PROBE_TIMEOUT_S < 600  # 占的是 eval 的 pool 槽位，不能继承会话的 6000s


@pytest.mark.parametrize(
    ("result",),
    [
        (RuntimeError("command_exec HTTP 503"),),
        ({"returncode": 1, "output": "python3: command not found"},),
        ({"returncode": 0, "output": ""},),
    ],
)
def test_probe_failure_degrades_to_empty_without_poisoning_the_cache(result):
    env = _Env(result, config=None)
    assert _read_env_info(env) == _EMPTY
    assert "local" not in _ENV_INFO_CACHE  # 空结果绝不永久生效
    assert "local" in _PROBE_FAILED_AT


def test_failure_is_suppressed_briefly_then_retried_and_recovers():
    """静默期内不重探（否则 270 题各撞一次 GPU 槽位），期满后下一题自动重探并把成功结果永久缓存。"""
    env = _Env(RuntimeError("command_exec HTTP 503"))
    assert _read_env_info(env) == _EMPTY
    assert _read_env_info(env) == _EMPTY
    assert env.calls == 1
    _PROBE_FAILED_AT["local"] = time.monotonic() - _PROBE_RETRY_AFTER_S - 1
    env.result = {"returncode": 0, "output": "VERSIONS:torch 2.9.0\nTRITON_HINTS:\n"}
    assert _read_env_info(env) == _EMPTY | {"versions": "torch 2.9.0"}
    assert env.calls == 2
    assert _read_env_info(env)["versions"] == "torch 2.9.0"
    assert env.calls == 2  # 成功后永久缓存，不再受静默期影响
