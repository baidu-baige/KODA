"""kernel_loop._env_cache_key 单测：缓存键必须认**实际执行探测命令的**镜像。

_read_env_info 的探测命令显式带 device="gpu" → 落在 GPU pool worker（config.image），
不是控制器 CPU pod（config.cpu_image）。键取错会让同一 pool 镜像被重复探测 N 次，
更糟的是若 cpu_image 与 image 不同源，键会宣称"这是 CPU 镜像的版本"而内容来自 GPU。
image 为空（纯 CPU 会话）时回落到 cpu_image 才是对的。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from minisweagent.agents.controller_runtime import _env_cache_key


class _Env:
    def __init__(self, config):
        self.config = config


class _Config:
    def __init__(self, **fields):
        self.__dict__.update(fields)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_Config(image="gpu:pool", cpu_image="cpu:a"), "gpu:pool"),
        (_Config(image="gpu:pool", cpu_image="cpu:b"), "gpu:pool"),
        (_Config(image="", cpu_image="cpu:a"), "cpu:a"),
        (_Config(image="gpu_only:v3"), "gpu_only:v3"),
        (None, "local"),
        (_Config(), "local"),
    ],
)
def test_cache_key_follows_executing_pod_image(config, expected):
    assert _env_cache_key(_Env(config)) == expected


def test_same_gpu_pool_image_shares_cache_across_cpu_images():
    """同一 GPU pool 镜像 + 不同 cpu_image → 必须是同一键：探测跑在 pool 上，版本必然相同，
    分键只会白跑 N 次探测（本次 fix 的核心回归点）。"""
    assert _env_cache_key(_Env(_Config(image="gpu:pool", cpu_image="cpu:a"))) == _env_cache_key(
        _Env(_Config(image="gpu:pool", cpu_image="cpu:b"))
    )
