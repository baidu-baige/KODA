"""exec_compile 的 arch 透传: cfg.compile_arch 非空时作 compile.sh 第 2 参, 空则不传。

B 卡 (Blackwell sm_100a) 适配的核心: harness 注入 compile_arch=10.0a 必须真的落到
compile.sh 的 TORCH_CUDA_ARCH_LIST, 否则 .so 编成 sm_90a 在 sm_100 上 no kernel image
加载失败 (multi-arch-support-design.md §1.3)。空值必须逐字保持旧行为 (compile.sh 回落
9.0a), 不能凭空补默认值。
"""

from types import SimpleNamespace

from minisweagent.agents.utils.controller_tools import exec_compile, tool_env

_BASE = dict(
    master_url="http://m", eval_server_token="REPLACE_EVAL_TOKEN", kerneleval_dir="/opt/ke",
    kernelprofile_dir="/opt/kp", kernelcompile_dir="/opt/kc", kerneltest_dir="/opt/kt",
    gpu_id="0", backend="hybrid",
)


class _Session:
    session_id = "sess"

    def __init__(self, result: dict):
        self.result = result
        self.commands: list[str] = []

    def execute(self, action: dict, timeout: int | None = None) -> dict:
        self.commands.append(action["command"])
        return self.result


def _ok_session() -> _Session:
    return _Session({"output": '{"compiled": true, "status": "success"}',
                     "returncode": 0, "exception_info": "", "extra": {}})


def test_compile_arch_passed_as_second_arg():
    cfg = SimpleNamespace(**_BASE, compile_arch="10.0a")
    session = _ok_session()
    exec_compile(cfg, session, "/ws")
    assert session.commands[0] == "bash /opt/kc/compile.sh /ws 10.0a"


def test_empty_compile_arch_omits_arg():
    cfg = SimpleNamespace(**_BASE, compile_arch="")
    session = _ok_session()
    exec_compile(cfg, session, "/ws")
    assert session.commands[0] == "bash /opt/kc/compile.sh /ws"


def test_missing_compile_arch_attr_omits_arg():
    """旧 config 无 compile_arch 字段: getattr 兜底为空, 不炸也不补默认。"""
    cfg = SimpleNamespace(**_BASE)
    session = _ok_session()
    exec_compile(cfg, session, "/ws")
    assert session.commands[0] == "bash /opt/kc/compile.sh /ws"


def test_tool_env_injects_kernel_compile_arch():
    """test 脚本自己编译时也需要 arch: KERNEL_COMPILE_ARCH 随信封 env 透传到 worker。"""
    cfg = SimpleNamespace(**_BASE, compile_arch="10.0a")
    assert tool_env(cfg, "sess")["KERNEL_COMPILE_ARCH"] == "10.0a"


def test_tool_env_omits_kernel_compile_arch_when_empty():
    cfg = SimpleNamespace(**_BASE, compile_arch="")
    assert "KERNEL_COMPILE_ARCH" not in tool_env(cfg, "sess")


def test_tool_env_omits_kernel_compile_arch_when_missing():
    cfg = SimpleNamespace(**_BASE)
    assert "KERNEL_COMPILE_ARCH" not in tool_env(cfg, "sess")
