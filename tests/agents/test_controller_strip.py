"""kernel_loop._strip_tool_output 单文档契约单测。

2026-08-25 契约：pod 工具 stdout 是单个全量文档、顶层内嵌 ``agent_view``
（工具侧瘦身视图，瘦身规则唯一权威在工具包内）；loop 只做提取——观测 =
agent_view（保 indent=2 多行），原文全量经 raw_output 落盘零丢失。
旧 pod 输出（无 agent_view）原样透传，harness 不再维护第二套剥除规则。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from minisweagent.agents.controller_runtime import _strip_tool_output


def _full_doc(view: dict, extra: dict | None = None) -> str:
    """拼一条工具侧 full 输出：顶层内嵌 agent_view 的完整文档。"""
    doc = {"status": "completed", **(extra or {})}
    doc["agent_view"] = view
    return json.dumps(doc, indent=2)


def _run(raw: str, tool: str = "eval") -> dict:
    return _strip_tool_output({"tool": tool, "output": raw, "extra": {}})


def test_extracts_agent_view_as_observation():
    """观测 = agent_view（工具侧瘦身产物原样），indent=2 多行与工具侧一致。"""
    view = {"status": "completed", "speedup": 2.05, "kernel_wall_time": 0.2}
    doc = _full_doc(view, {"kernel_gpu_time": 0.1, "anti_hacking_report": {"summary": {}}})
    out = _run(doc)
    obs = json.loads(out["output"])
    assert obs == view  # 观测就是 agent_view，不存在顶层机制字段
    assert "\n  " in out["output"]  # 保持 indent=2，traj yaml 渲染不丢缩进


def test_raw_output_preserves_full_document():
    """原文全量落盘：含 agent_view 与机制字段，一个字节不少。"""
    view = {"speedup": 2.05}
    doc = _full_doc(view, {"kernel_gpu_time": 0.2, "metadata": {"hardware": "H800"}})
    stripped = _run(doc)
    raw = json.loads(stripped["extra"]["raw_output"])
    assert raw["agent_view"]["speedup"] == 2.05
    assert raw["kernel_gpu_time"] == 0.2
    assert raw["metadata"]["hardware"] == "H800"
    # 观测就是从原文 agent_view 提取的（提取语义，零转换）
    assert json.loads(stripped["output"]) == raw["agent_view"]


def test_legacy_output_without_agent_view_passthrough():
    """旧 pod 输出（无 agent_view）：原样透传，不再剥除。"""
    raw = json.dumps(
        {"status": "completed", "metadata": {"dur_trial_ms": [1, 2]}, "ncu": {"single_pass": {"error": "x"}}},
        indent=2,
    )
    stripped = _run(raw)
    assert json.loads(stripped["output"]) == json.loads(raw)  # 内容一字未动
    assert stripped["extra"]["raw_output"] == raw


def test_prefix_before_json_is_preserved():
    """容忍工具 stdout 的明文前缀（旧壳残留），JSON 从首个 { 解析。"""
    raw = "[eval_out]\n" + _full_doc({"speedup": 1.5})
    stripped = _run(raw)
    assert stripped["output"].startswith("[eval_out]\n")
    assert json.loads(stripped["output"].split("\n", 1)[1])["speedup"] == 1.5
    assert stripped["extra"]["raw_output"] == raw


def test_non_json_or_non_dict_output_passthrough():
    """无 JSON / 非 dict / 非对象输出：原样返回，不写 raw_output。"""
    for raw in ("", "no json here", json.dumps([1, 2, 3])):
        out = {"tool": "eval", "output": raw, "extra": {}}
        assert _strip_tool_output(out) is out
        assert "raw_output" not in out["extra"]