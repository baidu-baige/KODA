"""Compile/eval/profile envelope tools used by the controller; no OpenCode process is required."""

from __future__ import annotations

import json
import re
import shlex

# 工具超时(与插件 EVAL/PROFILE/COMPILE/TEST_TIMEOUT_MS 对齐, 秒)。
EVAL_TIMEOUT_S = 600
PROFILE_TIMEOUT_S = 900
COMPILE_TIMEOUT_S = 600
TEST_TIMEOUT_S = 90


def tool_env(cfg, session_id: str) -> dict[str, str]:
    """信封环境变量: remote 转发 + 本地禁 CUDA(= 插件 DEVICE_GPU() 的 remote 分支)。

    编译缓存 env 写死透传 (2026-09-03): 云盘 torch.compile 缓存路径必须随信封
    进 GPU worker 编译进程 —— 走 kerneltool_common._post_via_master 透传, 冷编译
    225s → 命中 ~50s。写死保证确定 (不依赖 yaml environment.env 合并链路是否生效)。

    KERNEL_COMPILE_ARCH (2026-09-23): cfg.compile_arch 非空时透传目标 CUDA arch ——
    经 master 转发到 GPU worker, 供 test 脚本自己调 utils/compile.py 时使用 (test.sh
    读它设 TORCH_CUDA_ARCH_LIST)。B 卡 (sm_100a) 必需, 否则 worker 自动探测得 sm_100。
    """
    env = {
        "KERNELEVAL_MODE": "remote",
        "KERNELEVAL_MASTER_URL": cfg.master_url,
        "KERNELEVAL_SESSION_ID": session_id,
        "KERNELEVAL_SERVER_TOKEN": cfg.eval_server_token,
        "CUDA_VISIBLE_DEVICES": "-1",
        "KERNELEVAL_DIR": cfg.kerneleval_dir,
        "KERNELPROFILE_DIR": cfg.kernelprofile_dir,
        "KERNELCOMPILE_DIR": cfg.kernelcompile_dir,
        "KERNTEST_DIR": cfg.kerneltest_dir,
        "GPU_ID": cfg.gpu_id,
        "KERNEL_BACKEND": cfg.backend,
        "TORCHINDUCTOR_CACHE_DIR": getattr(cfg, "inductor_cache_dir", "")
        or f"{getattr(cfg, 'workspace_dir', '') or '/workspace/kb_ws'}/.cache/inductor",
        "TRITON_CACHE_DIR": getattr(cfg, "triton_cache_dir", "")
        or f"{getattr(cfg, 'workspace_dir', '') or '/workspace/kb_ws'}/.cache/triton",
    }
    if getattr(cfg, "compile_arch", ""):
        env["KERNEL_COMPILE_ARCH"] = cfg.compile_arch
    if getattr(cfg, "eval_atol", None) is not None and getattr(cfg, "eval_rtol", None) is not None:
        env |= {"KERNELEVAL_ATOL": repr(cfg.eval_atol), "KERNELEVAL_RTOL": repr(cfg.eval_rtol)}
    return env


def _cmd_eval(cfg, workspace: str, output_path: str, kernel_file: str) -> str:
    """eval 引擎命令: 与插件 evaluate run() 参数逐字对齐(含 --compiled-reference)。"""
    argv = [
        "python3",
        "run_eval.py",
        "--project-dir",
        workspace,
        "--ref-file",
        "task.py",
        "--model-file",
        kernel_file,
        "--backend",
        cfg.backend,
        "--device",
        f"cuda:{cfg.gpu_id}",
        "--entry-point",
        "Model",
        "--num-correct-trials",
        "5",
        "--num-perf-trials",
        "50",
        "--num-warmup-trials",
        "10",
        "--compiled-reference",
        "--output-mode",
        "full",
        "--output",
        output_path,
    ]
    return f"cd {shlex.quote(cfg.kerneleval_dir)} && " + " ".join(shlex.quote(a) for a in argv)


def _cmd_profile(cfg, workspace: str, kernel_file: str) -> str:
    """profile 引擎命令: 与插件 profile run() 参数一致(默认 10 个采样窗)。

    2026-08-25 单文档契约: --output-mode full 下 stdout 是单个全量文档、顶层
    内嵌 agent_view (瘦身权威在工具包内, harness 只提取)。不再传 --output
    落盘文件 —— 与 gpu_only 的 _mb_profile_wrapper 对齐, full 文档本身即
    全量原文 (raw 落盘走 observation 的 extra.raw_output)。
    """
    argv = [
        "bash",
        f"{cfg.kernelprofile_dir}/profile.sh",
        kernel_file,
        "--device",
        cfg.gpu_id,
        "--sample-windows",
        "10",
        "--output-mode",
        "full",
        # loop 式自动化 (对齐 gpu_only 链路的 profile.sh --ncu-auto): 工具
        # 自动从 nsys ops 按耗时降序选 top-3 label 做 ncu 采样 —— 模型无需
        # 预写 NVTX 也有 kernel 级 ncu 指标 (2026-08-31)。
        "--ncu-auto",
    ]
    return " ".join(shlex.quote(a) for a in argv)


def exec_eval(cfg, session, workspace: str, timeout_s: int = EVAL_TIMEOUT_S) -> dict:
    """执行 eval 信封工具: 引擎裁决 slim JSON 走 stdout (不归档, 见模块 docstring)。

    超时区分: 命令级超时 (design.md:251, status=timed_out) 时**不**回退 cat result.json ——
    该文件本轮已被 write_candidate 清掉 (跨轮防陈旧), 读它只会把上一轮旧裁决冒充本轮; 直接
    合成 timeout 状态, gate 与轨迹据此可辨。其余 stdout 无 JSON 才走 cat 兜底 (旧引擎)。
    """
    result_out = f"{workspace}/eval_result.json"
    cmd = _cmd_eval(cfg, workspace, result_out, f"{workspace}/kernel.py")
    r = session.execute({"command": cmd, "env": tool_env(cfg, session.session_id)}, timeout=timeout_s)
    out = (r.get("output") or "").strip()
    if not out.startswith("{"):
        if r.get("extra", {}).get("status") == "timed_out":
            out = json.dumps({"status": "timeout", "error_message": f"eval exceeded {timeout_s}s and was killed"})
        else:
            # 旧引擎 stdout 无 JSON 降级路径: 从 pod 拉 --output 全量 (本轮真正写出的结果)。
            r2 = session.execute({"command": f"cat {shlex.quote(result_out)} 2>/dev/null"})
            if r2["returncode"] == 0 and r2.get("output", "").strip():
                out = r2["output"].strip()
            else:
                out = json.dumps(
                    {
                        "status": "environment_error",
                        "error_message": f"eval produced no result.json; rc={r.get('returncode')}",
                    }
                )
    return {
        "tool": "eval",
        "output": out,
        "returncode": r.get("returncode", -1),
        "exception_info": r.get("exception_info", ""),
        "extra": r.get("extra", {}),
    }


def exec_profile(cfg, session, workspace: str, timeout_s: int = PROFILE_TIMEOUT_S) -> dict:
    """执行 profile 信封工具: nsys/ncu 报告文本(stdout 给 agent 简单视图)。

    输出不在这里截断: 全量经 _strip_tool_output 落 raw_output 存档, 模型观测的
    瘦身由 observation 模板 elide(>40k 截头尾 5k)兜底 —— 截断若发生在此处,
    trajectory 里就永远没有 full 了 (2026-08-31)。--output-mode full 下
    stdout 是单文档契约(顶层内嵌 agent_view), 与 gpu_only wrapper 对齐。
    """
    cmd = _cmd_profile(cfg, workspace, f"{workspace}/kernel.py")
    r = session.execute({"command": cmd, "env": tool_env(cfg, session.session_id)}, timeout=timeout_s)
    out = r.get("output") or r.get("exception_info") or "(no output)"
    return {
        "tool": "profile",
        "output": out,
        "returncode": r.get("returncode", -1),
        "exception_info": r.get("exception_info", ""),
        "extra": r.get("extra", {}),
    }


def _distill_compile_error(text: str, max_lines: int = 40, max_chars: int = 2000) -> str:
    """从 nvcc/ninja 编译错误里抽出真正的诊断行, 剥掉淹没重点的超长编译命令行。

    编译失败原文里, `[1/3] c++ -MMD -I... -isystem ...` 这类 invocation 行动辄上千字符,
    把 `kernel.cu(44): error: ...` / `static assertion failed` / `N errors detected` 挤到末尾,
    模型常连续几轮看不清同一个错 (如 Sm90 的 DefaultGemmConfiguration incomplete type)。
    这里只保留诊断行前置回灌, 不改原始 error 字段。"""
    keep = re.compile(
        r"error:|warning:|note:|static assertion|required (from|by)|"
        r"instantiation of|detected during|errors? detected|ninja:|\.(cu|cpp|cc|h|cuh)[:(]\d+"
    )
    drop = re.compile(r"\s-I|\s-isystem|\s-MMD|\s-MF|^\s*\[\d+/\d+\]|^\s*(c\+\+|nvcc|gcc|cc1plus)\s")
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    picked = [ln for ln in lines if keep.search(ln) and not drop.search(ln)]
    digest = "\n".join(picked[:max_lines])[:max_chars]
    return digest.strip()


def exec_compile(cfg, session, workspace: str, timeout_s: int = COMPILE_TIMEOUT_S) -> dict:
    """执行 compile 信封工具: native 后端构建 cuda_extension.so(恒 CUDA_VISIBLE_DEVICES=-1)。

    arch 透传: cfg.compile_arch 非空时作 compile.sh 第 2 参 (TORCH_CUDA_ARCH_LIST), 空则不传
    (compile.sh 回落 9.0a, 保持旧 config 逐字兼容)。B 卡配 compile_arch=10.0a 消灭 sm_90a
    二进制在 sm_100 上 no kernel image 的加载失败。
    """
    arch = getattr(cfg, "compile_arch", "") or ""
    arch_arg = f" {shlex.quote(arch)}" if arch else ""
    cmd = f"bash {shlex.quote(cfg.kernelcompile_dir)}/compile.sh {shlex.quote(workspace)}{arch_arg}"
    r = session.execute({"command": cmd, "env": tool_env(cfg, session.session_id)}, timeout=timeout_s)
    output = (r.get("output") or "").strip()
    compiled = False
    i = output.find("{")
    try:
        parsed = json.JSONDecoder().raw_decode(output, i)[0] if i >= 0 else None
    except ValueError:
        parsed = None
    if parsed is None:
        ok = r.get("returncode") == 0
        compiled = ok
        if ok:
            output = json.dumps({"compiled": True, "status": "success", "artifact": "cuda_extension.so"})
        else:
            output = json.dumps(
                {
                    "error_brief": _distill_compile_error(f"{output}\n{r.get('exception_info') or ''}")
                    or "(no diagnostic lines parsed — see stdout/stderr)",
                    "compiled": False,
                    "status": "compilation_failed",
                    "stdout": output[-1500:],
                    "stderr": (r.get("exception_info") or "")[-2000:],
                }
            )
    else:
        compiled = bool(parsed.get("compiled")) or parsed.get("status") == "success"
        if compiled:
            output = json.dumps(parsed, ensure_ascii=False)
        else:
            output = json.dumps(
                {
                    "error_brief": _distill_compile_error(f"{parsed.get('error', '')}\n{r.get('exception_info') or ''}")
                    or "(no diagnostic lines parsed — see error/stderr)",
                    **parsed,
                    "stderr": (r.get("exception_info") or "")[-2000:],
                },
                ensure_ascii=False,
            )
    return {
        "tool": "compile",
        "output": output,
        "returncode": r.get("returncode", -1),
        "exception_info": r.get("exception_info", ""),
        "extra": {**r.get("extra", {}), "compiled": compiled},
    }


def exec_test(cfg, session, workspace: str, rel_path: str, timeout_s: int = TEST_TIMEOUT_S) -> dict:
    """执行 test 信封工具: tests/test_*.py 快测(not an eval verdict)。"""
    rel = (rel_path or "").strip()
    if not rel or not rel.startswith("tests/"):
        return {
            "tool": "test",
            "output": json.dumps({"error": f"test path must be inside workspace: {rel!r}"}),
            "returncode": 0,
            "exception_info": "",
            "extra": {},
        }
    cmd = f"bash {shlex.quote(cfg.kerneltest_dir)}/test.sh {shlex.quote(rel)}"
    r = session.execute({"command": cmd, "env": tool_env(cfg, session.session_id)}, timeout=timeout_s)
    out = (r.get("output") or "").strip()
    return {
        "tool": "test",
        "output": out,
        "returncode": r.get("returncode", -1),
        "exception_info": r.get("exception_info", ""),
        "extra": r.get("extra", {}),
    }


_EXECUTORS = {
    "compile": exec_compile,
    "eval": exec_eval,
    "profile": exec_profile,
    "test": exec_test,
}


def run_controller_tool(name: str, *, cfg, session, workspace: str, arg: str | None = None) -> dict:
    """按名执行一个信封工具(全局分发表, 新增工具 = 加一张表项)。

    `arg` 给 test 的 rel path; 其余工具无参数(eval 直接读 workspace/kernel.py)。
    """
    executor = _EXECUTORS.get(name)
    if executor is None:
        return {
            "tool": name,
            "output": json.dumps({"error": f"unknown tool {name!r}"}),
            "returncode": 0,
            "exception_info": "",
            "extra": {},
        }
    if name == "test":
        return executor(cfg, session, workspace, arg or "")
    return executor(cfg, session, workspace)
