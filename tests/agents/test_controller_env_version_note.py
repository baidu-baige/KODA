"""kernel_loop._env_version_note 单测：用 eval worker 自报的版本给 prompt 断言做对账。

探测只在 round 1 之前跑一次；eval 的裁决 JSON 里 metadata 带的版本是刚跑完这次 eval 的
worker 自报的，是唯一权威。不一致时必须把真值回填进观测（模型正照着 prompt 里的版本写
kernel），一致时必须闭嘴（18 轮 × 270 题重复一行纯属污染）。旧镜像的引擎不写这些字段，
缺失时同样闭嘴 —— 不能把"没数据"当成"不一致"。

比对是**逐字段全等**：子串包含会让 `torch 2.9.0` 被 `torch 2.9.0+cu129` 吞掉，正好漏掉
「worker 装的不是同一个 torch」这个本功能唯一要兜底的场景。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from minisweagent.agents.controller_runtime import _env_version_note

_PROMPT = "Python 3.12.12 / torch 2.9.0+cu129 / triton 3.5.0 / transformers 4.57.1 / CUDA 12.9"


def _eval_doc(**md) -> str:
    return json.dumps({"status": "completed", "correctness": True, "metadata": md})


def test_mismatched_worker_versions_are_backfilled_into_the_observation():
    note = _env_version_note(_PROMPT, _eval_doc(torch_version="2.10.0+cu130", triton_version="3.6.0"))
    assert "torch 2.10.0+cu130" in note and "triton 3.6.0" in note
    assert "trust these versions" in note


@pytest.mark.parametrize(
    ("field", "worker_version"),
    [
        ("torch_version", "2.9.0"),  # prompt 是 2.9.0+cu129: worker 装的是 CPU 版 torch
        ("triton_version", "3.5.0.post1"),  # prompt 是 3.5.0
        ("python_version", "3.12.1"),  # prompt 是 3.12.12
        ("transformers_version", "4.57"),  # prompt 是 4.57.1
    ],
)
def test_prefix_versions_are_not_swallowed(field, worker_version):
    """真值是 prompt 值的前缀（或反之）时，子串判断会误判为一致 —— 必须报出来。"""
    assert worker_version in _env_version_note(_PROMPT, _eval_doc(**{field: worker_version}))


def test_empty_prompt_versions_get_the_worker_truth():
    """探测失败 → prompt 没有版本段；此时 worker 自报的必须补上，而不是判为"一致"。"""
    assert "torch 2.9.0+cu129" in _env_version_note("", _eval_doc(torch_version="2.9.0+cu129"))


@pytest.mark.parametrize(
    ("eval_output",),
    [
        # 与 prompt 逐字段全等（prompt 多带 CUDA 无妨：只比 worker 报了的字段）
        (_eval_doc(python_version="3.12.12", torch_version="2.9.0+cu129", triton_version="3.5.0"),),
        (_eval_doc(),),  # 旧镜像: metadata 在但没有版本字段
        (_eval_doc(torch_version=None),),  # 字段在但为 None (triton 缺失时 _triton_version 的返回)
        (json.dumps({"status": "completed"}),),  # 连 metadata 都没有
        ("environment_error: eval produced no result.json",),  # 非 JSON
        ('{"status": "completed", "metadata": ',),  # 截断的 JSON
    ],
)
def test_silent_when_consistent_or_no_data(eval_output):
    assert _env_version_note(_PROMPT, eval_output) == ""
