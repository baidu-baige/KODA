"""实例容差透传: cfg.eval_atol/eval_rtol 成对给出才注入 KERNELEVAL_ATOL/RTOL (run_eval.py 读 env 回落)。"""

from types import SimpleNamespace

import pytest

from minisweagent.agents.controller_runtime import ToolRuntimeConfig
from minisweagent.agents.utils.controller_tools import tool_env

_BASE = dict(
    master_url="http://m", eval_server_token="REPLACE_EVAL_TOKEN", kerneleval_dir="/opt/ke",
    kernelprofile_dir="/opt/kp", kernelcompile_dir="/opt/kc", kerneltest_dir="/opt/kt",
    gpu_id="0", backend="cuda",
)


@pytest.mark.parametrize(
    ("atol", "rtol", "expected"),
    [
        (1e-05, 0.05, {"KERNELEVAL_ATOL": "1e-05", "KERNELEVAL_RTOL": "0.05"}),
        (1e-05, None, {}),
        (None, None, {}),
    ],
)
def test_tool_env_tolerance(atol, rtol, expected):
    env = tool_env(SimpleNamespace(**_BASE, eval_atol=atol, eval_rtol=rtol), "sess")
    assert {k: v for k, v in env.items() if k in ("KERNELEVAL_ATOL", "KERNELEVAL_RTOL")} == expected


def test_config_accepts_instance_tolerance():
    cfg = ToolRuntimeConfig(system_template="", instance_template="", eval_atol=1e-5, eval_rtol=0.05)
    assert (cfg.eval_atol, cfg.eval_rtol) == (1e-5, 0.05)
    assert ToolRuntimeConfig(system_template="", instance_template="").eval_atol is None
