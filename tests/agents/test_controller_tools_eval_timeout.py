"""exec_eval 超时/降级路径: 命令级超时不得回退读 eval_result.json (跨轮旧裁决防冒充)。

沙箱命令超时 (design.md:251, status=timed_out) 时 run_eval 没写出结果; CPU Pod 在会话内
保留文件状态 (design.md:31/308), 上一轮的 eval_result.json 还在, 若 cat 兜底就会把旧裁决
当成本轮结果 —— gate 会据此误放行。三条不变量: 超时→合成 timeout 且**不** cat; 旧引擎无
JSON→cat 本轮文件; 无文件→environment_error。
"""

from types import SimpleNamespace

import pytest

from minisweagent.agents.utils.controller_tools import exec_eval, run_controller_tool
from minisweagent.agents.utils.timing import PhaseTimer, build_phase_timings

_CFG = SimpleNamespace(
    master_url="http://m", eval_server_token="REPLACE_EVAL_TOKEN", kerneleval_dir="/opt/ke",
    kernelprofile_dir="/opt/kp", kernelcompile_dir="/opt/kc", kerneltest_dir="/opt/kt",
    gpu_id="0", backend="triton",
)


class _Session:
    """按调用顺序回放预置结果的假会话; 记录每条命令以便断言是否走了 cat 兜底。"""

    session_id = "sess"

    def __init__(self, *results: dict):
        self.results = list(results)
        self.commands: list[str] = []

    def execute(self, action: dict, timeout: int | None = None) -> dict:
        self.commands.append(action["command"])
        return self.results.pop(0)


def _cat_issued(session: _Session) -> bool:
    return any(c.startswith("cat ") for c in session.commands)


def test_command_timeout_synthesizes_timeout_and_never_reads_stale_file():
    session = _Session({"output": "partial log, killed", "returncode": -1,
                        "exception_info": "", "extra": {"status": "timed_out"}})
    out = exec_eval(_CFG, session, "/ws")
    assert '"status": "timeout"' in out["output"]
    assert out["extra"]["status"] == "timed_out"
    assert not _cat_issued(session)  # 关键: 超时绝不回退读上一轮的 eval_result.json
    assert session.commands == [session.commands[0]]  # 只发了 eval 一条, 没有第二条 cat


def test_normal_json_stdout_passes_through_without_cat():
    verdict = '{"status": "completed", "correctness": true}'
    session = _Session({"output": verdict, "returncode": 0, "exception_info": "", "extra": {}})
    out = exec_eval(_CFG, session, "/ws")
    assert out["output"] == verdict
    assert not _cat_issued(session)


def test_old_engine_no_stdout_json_falls_back_to_current_round_file():
    """非超时且 stdout 无 JSON (旧引擎): 允许 cat, 读到的是本轮 write_candidate 后新写的文件。"""
    file_verdict = '{"status": "completed", "correctness": false}'
    session = _Session(
        {"output": "[KernelEval] done", "returncode": 0, "exception_info": "", "extra": {"execution_id": "eval"}},
        {"output": file_verdict, "returncode": 0, "exception_info": "", "extra": {"execution_id": "cat"}},
    )
    out = exec_eval(_CFG, session, "/ws")
    assert out["output"] == file_verdict
    assert out["extra"]["execution_id"] == "eval"
    assert _cat_issued(session)


def test_no_json_and_no_file_reports_environment_error():
    session = _Session(
        {"output": "boom", "returncode": 1, "exception_info": "", "extra": {}},
        {"output": "", "returncode": 1, "exception_info": "", "extra": {}},
    )
    out = exec_eval(_CFG, session, "/ws")
    assert '"status": "environment_error"' in out["output"]


@pytest.mark.parametrize(
    ("timed_out_extra",),
    [({"status": "timed_out"},), ({"status": "timed_out", "execution_id": "x"},)],
)
def test_timeout_takes_priority_over_stale_file_even_when_file_exists(timed_out_extra):
    """即便旧 eval_result.json 仍在 (会 cat 出 completed), 超时判定优先, 不给它 cat 的机会。"""
    session = _Session({"output": "", "returncode": -1, "exception_info": "", "extra": timed_out_extra})
    out = exec_eval(_CFG, session, "/ws")
    assert '"status": "timeout"' in out["output"] and not _cat_issued(session)


@pytest.mark.parametrize(
    ("tool", "stdout"),
    [("compile", '{"compiled": true}'), ("eval", '{"status": "completed"}'),
     ("profile", '{"stage_timings": {"nsys_model_load": 1.2}}'), ("test", "passed")],
)
def test_tool_wrappers_preserve_sandbox_timings(tool, stdout):
    timings = {"cpu_execution": {"duration_ms": 1500},
               "gpu_executions": [{"queue_ms": 200, "exec_ms": 1000, "attempts": 1}]}
    extra = {"execution_id": "execution-1", "status": "completed", "sandbox_timings": timings}
    session = _Session({"output": stdout, "returncode": 0, "exception_info": "", "extra": extra})
    out = run_controller_tool(tool, cfg=_CFG, session=session, workspace="/ws", arg="tests/test_kernel.py")
    assert out["extra"] == (extra | {"compiled": True} if tool == "compile" else extra)
    assert "compiled" not in extra
    assert build_phase_timings(PhaseTimer(), [(tool, out)])["tools"][0]["sandbox"] == timings
