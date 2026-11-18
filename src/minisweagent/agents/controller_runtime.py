"""Shared runtime and environment probes required by the controller."""

import base64

import hashlib

import json

import time

from minisweagent import Environment, Model

from minisweagent.agents.base import AgentRuntimeConfig, AgentRuntime

from minisweagent.agents.utils.env_probe import BASE_PROBE_SCRIPT, normalize_gpu_name

from minisweagent.agents.utils.kernel_workspace import read_workspace_file


from minisweagent.utils.log import logger

from minisweagent.utils.metrics import COLLECTOR


def model_specs_markdown(cfg) -> str:
    """Assemble the level_4 HF model-structure block from the env config.

    Sources are the dataset's `modeling` / `config` columns (empty for
    non-level_4 tasks). Returns "" for empty input so the instance_template's
    `{% if model_specs %}` renders nothing and StrictUndefined never trips.
    """
    modeling = (getattr(cfg, "modeling", "") or "").strip()
    config = (getattr(cfg, "config", "") or "").strip()
    if not (modeling or config):
        return ""
    parts = [
        "### Model architecture reference (transformers 4.57.1, same as the eval environment)",
        "This task is an HF pretrained model. Replicate the structure from the modeling source and config below:",
        "attribute names / module hierarchy / forward signature must match it exactly — do not "
        "guess them from memory (guessing wrong surfaces as `has no attribute` / "
        "`unexpected keyword argument` / shape mismatch at instantiate time).",
    ]
    if config:
        parts.append("config.json:\n```json\n" + config + "\n```")
    if modeling:
        parts.append(
            "modeling source (Embeddings / Attention / Block / forward / LM head):\n```python\n" + modeling + "\n```"
        )
    return "\n\n".join(parts)


def _compile_ok(comp: dict) -> bool:
    """compile 判据：output 里的 JSON status == "success" (2026-08-22)。

    不再 grep 明文字符串（旧判据 `"Compile success" in output`）——compile.py
    已收敛为 stdout 唯一 JSON、进程不再向 stderr 写业务日志，明文依赖会
    在无 stderr 合流时永久误判失败。与 _eval_json 同法容忍明文前缀（旧壳/
    wrapper 残留），rc 非 0 或 JSON 解析失败均按未过处理。
    """
    if comp.get("returncode", 1) != 0:
        return False
    i = comp.get("output", "").find("{")
    if i < 0:
        return False
    try:
        obj, _ = json.JSONDecoder().raw_decode(comp.get("output", ""), i)
    except ValueError:
        return False
    return isinstance(obj, dict) and obj.get("status") == "success"


_ENV_INFO_CACHE: dict[str, dict] = {}

_PROBE_FAILED_AT: dict[str, float] = {}

_PROBE_RETRY_AFTER_S = 300

_PROBE_TIMEOUT_S = 60

_PROBE_COLD_WARMUP_S = 120

_PROBE_COLD_SLEEP_S = 8


def _env_cache_key(env: Environment) -> str:
    """取**实际跑探测命令的**镜像名做 cache key; 无镜像 (LocalEnv) 用 'local'。

    探测显式走 ``device="gpu"`` (见 _read_env_info), 落在 GPU pool worker, 故 ``image``
    (pool 镜像) 优先于 ``cpu_image``: 同一 pool 镜像的版本必然相同, 跨 run 共享缓存是对的;
    反过来按 cpu_image 分键, 只会让同一 pool 镜像被重复探测。``image`` 为空 (纯 CPU 会话)
    时回落 ``cpu_image``, 仍无则 'local'。

    GPU pool 型号差异并入 key: CUTLASS 3x 骨架按实际 GPU arch 选 (Sm90 wgmma / Sm100 tcgen05),
    同一镜像若同时跑 H20 与 B200 两个 pool, 探测结果 (尤其 cutlass3x_example) 会串。config 里
    只有 ``resource_request`` 带 GPU 资源名 (example.org/gpu_hopper vs example.org/gpu_blackwell) 能可靠区分
    pool 卡型, 故把它拼进 key; 无 GPU 资源名 (纯 CPU / gpu_only 无 resource_request) 时回落原 key。
    """
    cfg = getattr(env, "config", None)
    if cfg is None:
        return "local"
    key = getattr(cfg, "image", None) or getattr(cfg, "cpu_image", None) or "local"
    rr = getattr(cfg, "resource_request", None) or {}
    gpu_res = next((k for k in sorted(rr) if "gpu" in k.lower()), "")
    return f"{key}@{gpu_res}" if gpu_res else key


def _read_env_info(env: Environment, compile_dir: str | None = None) -> dict:
    """Read versions + triton API surface from the GPU worker that runs eval, cached per image.

    Thin wrapper over ``agents.utils.env_probe.probe_env_info`` (single source of truth shared
    with the opencode agent): runs the probe on ``device="gpu"`` (the eval GPU worker) and parses
    the tagged output. Failures yield empty strings so the run is never blocked; cached per image
    so a 270-instance batch probes once.

    探测显式 ``device="gpu"``: prompt 断言的是 "the eval environment", 而 cpu_gpu 下 execute
    默认 device=cpu 会落在控制器 CPU pod —— 它与 GPU pool 同源只是约定 (``--sandbox-image``
    只覆盖 ``image``, 不覆盖 ``cpu_image``), 拿它的 triton 去断言 eval 的 API surface 是在猜。
    master 支持在 worker 上跑任意命令 (见 kernel_sandbox/examples/bench.py 的 device=gpu +
    python3)。gpu_only 会话本来就把所有命令强制到唯一 GPU pod, 这里的 device 键对它是冗余的。
    用 heredoc 传多行脚本, 避免多层 shell 转义。

    API 探测取代硬编码版本提示: 直接 hasattr / inspect.signature 查真实安装的
    triton, 永不过期, 不依赖模型记忆版本→API 映射。
    """
    cache_key = _env_cache_key(env)
    if cache_key in _ENV_INFO_CACHE:
        return _ENV_INFO_CACHE[cache_key]
    if time.monotonic() - _PROBE_FAILED_AT.get(cache_key, float("-inf")) < _PROBE_RETRY_AFTER_S:
        return {
            "versions": "",
            "triton_api_hints": "",
            "gpu_specs": "",
            "cutlass_version": "",
            "cutlass_example": "",
            "handwritten_example": "",
            "cutlass3x_example": "",
        }

    script = BASE_PROBE_SCRIPT
    # CUTLASS 探测 (仅 opencode 信封链路的 config 有 kernelcompile_dir): 从镜像里的
    # 编译栈读真实 CUTLASS 版本号 + 编译验证过的 golden example (examples/gemm_tf32/kernel.cu),
    # base64 单行传出避免多行解析。硬编码版本/签名会随镜像升级变谎 (见 env_versions 契约),
    # 故版本与骨架都从镜像现读; compile_dir 为 None (gpu_only scripts 链路) 时整块不注入。
    # compile_dir 来源: cpu_gpu 下 kernelcompile_dir 只在 AGENT config (ToolRuntimeConfig),
    # env.config (KernelSandboxEnvironmentConfig) 无此字段 → 由调用方 (agent.run) 显式传入;
    # gpu_only 下 env.config 有该字段, 走 fallback 读 env.config 保持原行为。
    compile_dir = compile_dir or getattr(getattr(env, "config", None), "kernelcompile_dir", None)
    if compile_dir:
        script += (
            "import os, re, base64\n"
            f"cc_dir = {compile_dir!r}\n"
            "try:\n"
            "    txt = open(os.path.join(cc_dir, 'cutlass', 'include', 'cutlass', 'version.h')).read()\n"
            "    mm = [re.search(r'CUTLASS_' + k + r'\\s+(\\d+)', txt) for k in ('MAJOR', 'MINOR', 'PATCH')]\n"
            "    if all(mm):\n"
            "        print('CUTLASS_VERSION:' + '.'.join(m.group(1) for m in mm))\n"
            "except Exception:\n"
            "    pass\n"
            "try:\n"
            "    data = open(os.path.join(cc_dir, 'examples', 'gemm_tf32', 'kernel.cu'), 'rb').read()\n"
            "    print('CUTLASS_EXAMPLE:' + base64.b64encode(data).decode())\n"
            "except Exception:\n"
            "    pass\n"
            # 手写 CUDA 参考: 从镜像 examples/*/kernel.cu 里挑一个"编译验证过、非 CUTLASS、含真实
            # __global__ kernel"的最小样例注入。模型默认路径 (手写 CUDA) 因此也有一份现读的锚点,
            # 直击 ~160 例手写内存/launch bug。纯数据驱动 + 体积上限 + 找不到静默为空, 随镜像升级不撒谎。
            "try:\n"
            "    import glob\n"
            "    cands = sorted(glob.glob(os.path.join(cc_dir, 'examples', '*', 'kernel.cu')))\n"
            "    best = None\n"
            "    for c in cands:\n"
            "        try:\n"
            "            b = open(c, 'rb').read()\n"
            "        except Exception:\n"
            "            continue\n"
            "        if len(b) > 8000 or b'__global__' not in b:\n"
            "            continue\n"
            "        low = b.lower()\n"
            # 判"真用了 CUTLASS"要看命名空间/头引用, 不能用裸子串 'cutlass':
            # binding_stack (理想手写锚点) 注释里提了一句 cutlass_usage_analysis.md, 裸子串会误杀。
            "        if b'cutlass::' in low or b'#include <cutlass' in low or "
            "b'#include \"cutlass' in low or b'cute::' in low or b'collectivebuilder' in low:\n"
            "            continue\n"
            "        if best is None or len(b) < best[1]:\n"
            "            best = (b, len(b))\n"
            "    if best is not None:\n"
            "        print('HANDWRITTEN_EXAMPLE:' + base64.b64encode(best[0]).decode())\n"
            "except Exception:\n"
            "    pass\n"
            # CUTLASS 3.x golden (Scheme B): 自研 3.x GemmUniversal/CollectiveBuilder GEMM 启动
            # cutlass::device_kernel<GemmKernel> → 分类器判 OWN (非 cuBLAS 内部 cutlass3x_sm90_ NATIVE),
            # 是真自研 kernel 且吃到 tensor-core 峰值 (Hopper wgmma / Blackwell tcgen05),
            # 专补 correct-but-slow 的投影/matmul 桶。
            # 数据驱动扫 examples/*/kernel.cu 找含 3.x 标记 (collectivebuilder / cute:: / gemmuniversal)
            # 的那份 —— 与 2.x gemm_tf32 (device::Gemm, 无这些标记) 天然区分。镜像里没有
            # (未 bake 对应 arch 的 3x 骨架 → examples/) 时静默为空, YAML {% if %} 块不渲染, 零回归。
            # 按 GPU 实际 compute-capability major 过滤 3.x 骨架: 3.x 是 arch 专属 API
            # (Sm90 wgmma / Sm100 tcgen05), 发错卡要么编译失败要么运行期 device-side assert。
            # 骨架目录名关键词 → 所需 sm major (可扩展: 将来新 arch 加一条映射即可);
            # 目录名无关键词的 3.x 骨架视为通用, 不过滤。
            "try:\n"
            "    import torch\n"
            "    _sm_major = int(torch.cuda.get_device_properties(0).major)\n"
            "except Exception:\n"
            "    _sm_major = 0\n"
            "try:\n"
            "    import glob\n"
            "    cands = sorted(glob.glob(os.path.join(cc_dir, 'examples', '*', 'kernel.cu')))\n"
            "    best3x = None\n"
            "    for c in cands:\n"
            "        try:\n"
            "            b = open(c, 'rb').read()\n"
            "        except Exception:\n"
            "            continue\n"
            "        if len(b) > 16000:\n"
            "            continue\n"
            "        low = b.lower()\n"
            "        if not (b'collectivebuilder' in low or b'cute::' in low or b'gemmuniversal' in low):\n"
            "            continue\n"
            "        base = os.path.basename(os.path.dirname(c)).lower()\n"
            "        arch = 10 if 'blackwell' in base else (9 if 'hopper' in base else 0)\n"
            "        if arch and _sm_major != arch:\n"
            "            continue\n"
            "        hint = 0 if arch == 10 else (1 if (arch == 9 or '3x' in base) else 2)\n"
            "        key = (hint, len(b))\n"
            "        if best3x is None or key < best3x[1]:\n"
            "            best3x = (b, key)\n"
            "    if best3x is not None:\n"
            "        print('CUTLASS3X_EXAMPLE:' + base64.b64encode(best3x[0]).decode())\n"
            "except Exception:\n"
            "    pass\n"
        )
    deadline = time.monotonic() + _PROBE_COLD_WARMUP_S
    while True:
        # 并发批启动: 别的线程可能已探成 (入缓存) 或已判失败 (进静默期)。每轮先复查, 一有结果就
        # 跟随, 避免 N 个 worker 线程对冷 pool 齐刷重探 (雷鸣群 + 空转 GPU 槽)。
        if cache_key in _ENV_INFO_CACHE:
            return _ENV_INFO_CACHE[cache_key]
        if time.monotonic() - _PROBE_FAILED_AT.get(cache_key, float("-inf")) < _PROBE_RETRY_AFTER_S:
            return {
                "versions": "",
                "triton_api_hints": "",
                "gpu_specs": "",
                "cutlass_version": "",
                "cutlass_example": "",
                "handwritten_example": "",
                "cutlass3x_example": "",
            }
        try:
            # cwd 显式 "/": 会话 config.cwd 是 CPU pod 的业务工作区 (/workspace/kb_ws),
            # GPU worker 上不一定存在, working_dir 不存在会让探测无谓失败。
            out = env.execute(
                {"command": f"python3 <<'PYEOF'\n{script}PYEOF", "device": "gpu"},
                cwd="/",
                timeout=_PROBE_TIMEOUT_S,
            )
            COLLECTOR.record_sandbox_timings(
                (out.get("extra") or {}).get("sandbox_timings") or {},
                getattr(getattr(env, "config", None), "workspace_name", ""),
            )
        except Exception as error:  # 版本提示只是 prompt 增益, sandbox 抖动不该带走整批题
            logger.warning(f"env info probe raised on {cache_key}: {error}")
            out = {"returncode": -1, "output": ""}
        result = {
            "versions": "",
            "triton_api_hints": "",
            "gpu_specs": "",
            "cutlass_version": "",
            "cutlass_example": "",
            "handwritten_example": "",
            "cutlass3x_example": "",
        }
        if out["returncode"] == 0:
            for line in out["output"].strip().splitlines():
                if line.startswith("VERSIONS:"):
                    result["versions"] = line[len("VERSIONS:") :].strip()
                elif line.startswith("TRITON_HINTS:"):
                    hints = line[len("TRITON_HINTS:") :].strip()
                    if hints:
                        result["triton_api_hints"] = hints
                elif line.startswith("GPU_SPECS:"):
                    specs = line[len("GPU_SPECS:") :].strip()
                    if specs:
                        result["gpu_specs"] = normalize_gpu_name(specs)
                elif line.startswith("CUTLASS_VERSION:"):
                    result["cutlass_version"] = line[len("CUTLASS_VERSION:") :].strip()
                elif line.startswith("CUTLASS_EXAMPLE:"):
                    b64 = line[len("CUTLASS_EXAMPLE:") :].strip()
                    if b64:
                        try:
                            result["cutlass_example"] = base64.b64decode(b64).decode("utf-8", "replace")
                        except Exception:
                            pass
                elif line.startswith("HANDWRITTEN_EXAMPLE:"):
                    b64 = line[len("HANDWRITTEN_EXAMPLE:") :].strip()
                    if b64:
                        try:
                            result["handwritten_example"] = base64.b64decode(b64).decode("utf-8", "replace")
                        except Exception:
                            pass
                elif line.startswith("CUTLASS3X_EXAMPLE:"):
                    b64 = line[len("CUTLASS3X_EXAMPLE:") :].strip()
                    if b64:
                        try:
                            result["cutlass3x_example"] = base64.b64decode(b64).decode("utf-8", "replace")
                        except Exception:
                            pass
        # 探测结果只经 prompt 露出, 静默为空时无从分辨"镜像没装 triton"与"探测挂了", 故记一行。
        # rc==0 的脚本必然至少打出 Python 版本, 故"三项全空" ⟺ 探测本身没跑成 → 不进永久缓存。
        if result["versions"] or result["triton_api_hints"] or result["gpu_specs"]:
            logger.info(
                f"env info [{cache_key}]: {result['versions'] or '(versions unavailable)'}"
                f"{'' if result['gpu_specs'] else ' (no gpu specs)'}"
                f"{'' if result['triton_api_hints'] else ' (no triton API hints)'}"
            )
            _ENV_INFO_CACHE[cache_key] = result
            return result
        # 空结果多半是批启动瞬间 GPU pool 未预热完 (rc=-1)。预热窗口内短睡重探让 pool 焐热,
        # 而不是立刻判失败——否则本题会整轮带空 env (见 _PROBE_COLD_WARMUP_S)。超窗仍空才落
        # 静默期 (镜像真没 python3 时也不无限空转; 静默态由并发线程下轮复查即跟随)。
        if time.monotonic() >= deadline:
            logger.warning(
                f"env info probe empty on {cache_key} (rc={out['returncode']}); "
                f"预热窗口 {_PROBE_COLD_WARMUP_S}s 内仍探不到, {_PROBE_RETRY_AFTER_S}s 后由下一题重探"
            )
            _PROBE_FAILED_AT[cache_key] = time.monotonic()
            return result
        logger.info(
            f"env info probe empty on {cache_key} (rc={out['returncode']}); "
            f"GPU pool 冷启动预热中, {_PROBE_COLD_SLEEP_S}s 后重探"
        )
        time.sleep(_PROBE_COLD_SLEEP_S)


def _norm_gpu_name(output: dict) -> dict:
    """Use the public GPU model name in observations sent to the model."""
    if isinstance(output.get("output"), str):
        output["output"] = normalize_gpu_name(output["output"])
    return output


def _strip_tool_output(output: dict) -> dict:
    """Extract the tool-side slim view; keep the full original for disk.

    worker 的 stdout report 是唯一数据源（raw_output 落盘全量），观测文本是
    它的一个视图。2026-08-25 单文档契约：pod 工具 stdout 是单个全量文档、
    顶层内嵌 ``agent_view``（工具侧瘦身视图——瘦身规则的唯一权威已收敛在
    工具包内，harness 不再维护第二套剥除规则表）；这里只做提取：观测 =
    agent_view，未修改的原文经 ``output["extra"]["raw_output"]`` 保存——
    format_observation_messages 把它带到 trajectory，flat_eval 从它解析，
    任何数据都不会丢。旧 pod 输出（无 agent_view）原样透传，不再剥除
    （旧工具版本不维护）。
    """
    raw = output.get("output", "")
    i = raw.find("{")
    if i >= 0:
        try:
            obj, end = json.JSONDecoder().raw_decode(raw, i)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            view = obj.get("agent_view")
            if isinstance(view, dict):
                # 保持与 worker 落盘一致的多行缩进 (工具侧 agent_view 为 indent=2
                # 产物), 免得观测变成 compact 单行, traj yaml 渲染时丢换行/缩进.
                output["output"] = raw[:i] + json.dumps(view, indent=2, ensure_ascii=False) + raw[end:]
            output.setdefault("extra", {})["raw_output"] = raw
    # 卡名归一化: eval/profile 每轮 JSON 的 model-facing 文本统一改名,
    # raw_output 已在上面保存了 worker 原文 (含原始卡名) 供打分, 不受影响.
    return _norm_gpu_name(output)


class ControllerRuntimeConfig(AgentRuntimeConfig):
    """Config for the fixed-round kernel optimization loop. Paths are relative to the workspace."""

    num_rounds: int = 5
    """Number of write/eval/profile rounds before the loop stops."""
    max_submit_eval: int = 50
    """Stop the loop after this many eval submissions (rounds that produced a kernel candidate and sent it
    to the eval pipeline, regardless of compile/correctness outcome); 0 disables. Unlike num_rounds this
    ignores no-code controller rounds (DIAGNOSE/REVERT/REQUEST_ENV_INFO), so it bounds the GPU-expensive,
    data-meaningful unit — the number of real kernel attempts — rather than total thinking turns."""
    early_stop_identical_rounds: int = 0
    """Stop the loop once the submitted kernel bytes (kernel.py + kernel.cu) repeat for this many
    consecutive rounds; 0 disables. A converged model re-emits the identical kernel every remaining
    round (the "平台期冻结" waste). Trade-off: identical rounds still re-roll eval noise and the scorer
    keeps the max across rounds, so early-stopping forgoes that free re-sampling upside — keep it off
    (or raise the threshold) when late noise draws of a frozen kernel are the best recorded result."""
    problem_file: str = "docs/problem.json"
    """Task metadata (task_id / backend / gpu) rendered into the templates."""
    reference_file: str = "docs/reference.py"
    """Reference implementation source rendered into the instance template."""


class ControllerRuntime(AgentRuntime):
    """Fixed-round loop agent; see the module docstring for the round structure."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type = ControllerRuntimeConfig,
        instance_id: str = "",
        **kwargs,
    ):
        """See `ControllerRuntimeConfig` for permitted keyword arguments.

        `config_class` lets subclasses swap in a richer config (`ToolRuntimeConfig`).
        """
        super().__init__(model, env, config_class=config_class, instance_id=instance_id, **kwargs)
        self.round = 0
        self._last_submission_key: str | None = None
        self._identical_streak = 0

    @staticmethod
    def _submission_key(action: dict) -> str:
        """Canonical submission bytes for the early-stop identity hash. Multi-file rounds must fold in
        kernel.cu: `command` only mirrors kernel.py (see ControllerModel), so a cu-only change
        would otherwise look identical and trip a false early-stop."""
        return action.get("kernel_py", action.get("command", "")) + (action.get("kernel_cu") or "")

    def _loop_exit(self, action: dict, submission: str) -> dict | None:
        """Submit-limit / round-limit / convergence early-stop (whichever fires first), else None to keep
        looping. Early-stop triggers once `_submission_key(action)` repeats for `early_stop_identical_rounds`
        consecutive rounds. Call exactly once per round — it advances the identical-streak counter and the
        eval-submit counter (every fixed-round loop round sends a candidate to eval)."""
        self._eval_submits = getattr(self, "_eval_submits", 0) + 1
        if 0 < self.config.max_submit_eval <= self._eval_submits:
            return {
                "role": "exit",
                "content": "EvalSubmitLimitReached",
                "extra": {"exit_status": "EvalSubmitLimitReached", "submission": submission},
            }
        if self.round >= self.config.num_rounds:
            return {
                "role": "exit",
                "content": "RoundLimitReached",
                "extra": {"exit_status": "RoundLimitReached", "submission": submission},
            }
        if (n := self.config.early_stop_identical_rounds) > 0:
            key = hashlib.sha256(self._submission_key(action).encode()).hexdigest()
            self._identical_streak = self._identical_streak + 1 if key == self._last_submission_key else 1
            self._last_submission_key = key
            if self._identical_streak >= n:
                return {
                    "role": "exit",
                    "content": "ConvergedEarlyStop",
                    "extra": {"exit_status": "ConvergedEarlyStop", "submission": submission},
                }
        return None

    def _log_round_timings(self, timings: dict) -> None:
        """One line per round: LLM latency + each tool's wall-clock (sandbox transport included).

        Uses the `minisweagent` logger (not AgentRuntime's unattached "agent" one) so the line
        reaches both the console and the run's minisweagent.log. No square brackets in the
        message — the console RichHandler runs with markup=True and would eat them as tags.
        """
        tools = " ".join(f"{t['name']}={t['total']}s" for t in timings["tools"])
        total = round((timings["llm"] or 0.0) + sum(t["total"] or 0.0 for t in timings["tools"]), 3)
        # tokens=<当前上下文>/<上限>: 当前上下文 = n_tokens = 最近一次请求的 total_tokens (全量 prompt +
        # 本次回复, 即当前对话规模, 非累计); 上限 = token_limit (0 时省略)。流式且下游未回 usage 时为 0。
        # 当前上下文越过上限即在下一轮开头抛 LimitsTokenExceeded。
        limit = getattr(self.config, "token_limit", 0)
        tokens = f"{self.n_tokens}/{limit}" if limit else f"{self.n_tokens}"
        # dyn_cap=<下一轮输出上限>/margin=<安全余量>: 开启动态输出时, 下一轮的 max_completion_tokens =
        # min(配置 max_completion_tokens, token_limit - 当前上下文 - margin); ≤0 表示没空间, 届时不下发。
        dyn = ""
        if getattr(self.config, "is_dynamic_completion", False) and limit:
            mk = getattr(getattr(self.model, "config", None), "model_kwargs", None) or {}
            if base := mk.get("max_completion_tokens"):
                margin = self.config.dynamic_completion_margin_tokens
                dyn = f" dyn_cap={min(base, limit - self.n_tokens - margin)}/margin={margin}"
        logger.info(
            f"{self.instance_id or 'kernel'} round {self.round}/{self.config.num_rounds} "
            f"timings: llm={timings['llm']}s {tools} total={total}s "
            f"tokens={tokens}{dyn}"
        )


_EVAL_VERSION_FIELDS = (
    ("Python", "python_version"),
    ("torch", "torch_version"),
    ("triton", "triton_version"),
    ("transformers", "transformers_version"),
)


def _parse_versions(versions: str) -> dict[str, str]:
    """把探测串 "Python 3.12.12 / torch 2.9.0+cu129 / CUDA 12.9" 拆回 {标签: 版本号}。

    不认识的形状 (静态 env_versions 兜底是自由文本) 只会拆不出对应标签, 后续按"prompt
    没声明过这个包"处理 —— 宁可多报一行真值, 不可拿没校验过的自由文本冒充一致。
    """
    return dict(part.split(" ", 1) for part in versions.split(" / ") if " " in part)


def _env_version_note(prompt_versions: str, eval_output: str) -> str:
    """eval worker 自报版本与 prompt 断言不符时, 回填一行纠正; 相符或无数据返回 ""。

    数据来自刚跑完这次 eval 的 worker (裁决 JSON 的 metadata, AI-DRL-299 落盘侧字段) ——
    比任何探测都权威, 且免费。用途是探测失败(prompt 无版本)或 pool 镜像不齐时兜底,
    不替代探测: metadata 给不出 triton 的 hasattr / signature API surface。

    比对**逐字段全等**, 不做子串包含也不做版本号规范化: `"torch 2.9.0" in "torch 2.9.0+cu129"`
    为真, 会把"worker 装的是 CPU 版 torch"这种正要兜底的不同源场景判成一致 (同理
    `3.5.0` 被 `3.5.0.post1` 吞); 而规范化掉 `+cu129` / `.post1` 恰好是主动丢掉这个差异。
    prompt 未声明某标签 = 模型从没被告知 → 也按不符处理, 补上真值。
    旧镜像的引擎不写这些字段 → 返回 "", 静默不告警。
    """
    i = eval_output.find("{")
    if i < 0:
        return ""
    try:
        obj, _ = json.JSONDecoder().raw_decode(eval_output, i)
    except ValueError:
        return ""
    md = obj.get("metadata") if isinstance(obj, dict) else None
    if not isinstance(md, dict):
        return ""
    measured = {label: str(md[key]) for label, key in _EVAL_VERSION_FIELDS if md.get(key)}
    stated = _parse_versions(prompt_versions)
    if not measured or all(stated.get(label) == version for label, version in measured.items()):
        return ""
    shown = " / ".join(f"{label} {version}" for label, version in measured.items())
    logger.warning(f"eval worker versions disagree with the prompt: {shown} vs {prompt_versions!r}")
    return (
        f"\n\nNOTE: the GPU worker that just ran this eval reports {shown}"
        " — trust these versions over the ones stated in your instructions."
    )


class ToolRuntimeConfig(ControllerRuntimeConfig):
    """loop × opencode 信封工具配置 (cpu_gpu): 每轮调**哪些**工具由 `pipeline_tools` 决定。

    工具实现见 ``agents/utils/opencode_tools.py``(kernel-tools.ts 的 python 对等，
    命令在 CPU pod 本地执行 + KERNELEVAL_MODE=remote 经 master 转发 GPU)——
    本类只管编排(gating)与配置, 不在代码写死工具集。
    """

    pipeline_tools: list[str] = ["eval", "profile"]
    """每轮按序调用的信封工具表 (compile/eval/profile/test 的任意有序子集)。
    loop+triton 默认只需 eval+profile (compile 是 native 后端的事, test 是
    opencode 侧快测)。gating 固定语义: compile 失败或 eval 不过 → 后续工具
    全部 [skipped] 占位。"""
    master_url: str = ""
    """cpu_gpu master 转发通道 (注入 KERNELEVAL_MASTER_URL + SESSION_ID)。"""
    eval_server_token: str = ""
    """GPU worker 内自愈 eval server 的 bearer token (kerneltool_common._token() 强校验)。"""
    workspace_dir: str = ""
    """pod 内 workspace 根 (kernel.py 落这里, eval --project-dir 同址); 空 = 用环境 cwd。"""
    kerneleval_dir: str = "/workspace/kerneleval"
    """eval 引擎目录 (run_eval.py 所在), 与 opencode 插件 env 对齐。"""
    kernelprofile_dir: str = "/opt/kerneltools/kernelprofile"
    """profile 引擎目录 (profile.sh 所在; 镜像真路径, /workspace/kernelprofile 是旧假设)。"""
    kernelcompile_dir: str = "/opt/kerneltools/kernelcompile"
    """compile 引擎目录 (compile.sh 所在, native 后端用)。"""
    kerneltest_dir: str = "/opt/kerneltools/kerneltest"
    """test 引擎目录 (test.sh 所在)。"""
    gpu_id: str = "0"
    """目标 GPU 编号 (eval --device cuda:{gpu_id} / profile --device)。"""
    backend: str = "triton"
    """eval --backend (KERNEL_BACKEND 注入); loop+triton 恒 triton。"""
    compile_arch: str = ""
    """CUDA arch for TORCH_CUDA_ARCH_LIST (如 9.0a=Hopper / 10.0a=Blackwell); 空 = 不传,
    compile.sh 回落 9.0a (向后兼容)。B 卡必须显式设 10.0a, 否则 .so 编成 sm_90a 加载不了。"""
    env_versions: str = ""
    """静态版本信息 fallback; 探测成功时用 pod 实测值, 只在探测拿不到时兜底。

    顺序是动态优先: 硬编码版本号会随镜像升级变成静默的谎, 而 prompt 里一个错版本号
    比没有更坏 —— 模型会照着不存在的 API 写 kernel。
    """
    inductor_cache_dir: str = ""
    triton_cache_dir: str = ""
    eval_atol: float | None = None
    eval_rtol: float | None = None
    """实例自带的正确性容差 (数据集 tolerance.max_atol/max_rtol, kernelbench.py 逐实例注入);
    经 env KERNELEVAL_ATOL/RTOL → run_eval.py → 信封 params 进 GPU worker。None = 引擎全局默认 1e-2。"""


class ToolRuntime(ControllerRuntime):
    """loop × opencode 信封工具 (cpu_gpu): 每轮 model 写 kernel.py → 按 pipeline_tools
    配置依次跑信封工具, 命令全在 CPU pod 本地执行 (CUDA_VISIBLE_DEVICES=-1),
    引擎经 KERNELEVAL_MODE=remote + master 转发 GPU pool worker。工具调用实现
    见 opencode_tools.py; 本类只做编排与 gating。"""

    @staticmethod
    def _skipped(tool: str, msg: str) -> dict:
        return {"tool": tool, "output": msg, "returncode": 0, "exception_info": "", "extra": {}}

    def __init__(self, model: Model, env: Environment, *, instance_id: str = "", **kwargs):
        """See `ToolRuntimeConfig` for permitted keyword arguments."""
        super().__init__(model, env, config_class=ToolRuntimeConfig, instance_id=instance_id, **kwargs)
        self._env_versions = ""
        """prompt 里实际断言的版本串 (run 填); step 用它跟 eval worker 自报的对账。"""

    def _workspace(self) -> str:
        """pod 内 workspace 根: 显式配置优先, 否则取环境 cwd。"""
        if self.config.workspace_dir:
            return self.config.workspace_dir
        return getattr(getattr(self.env, "config", None), "cwd", "/workspace")

    def run(self, task: str = "", **kwargs) -> dict:
        """cpu_gpu workspace 只有 task.py (无 docs/problem.json): 模板变量同
        KernelLoopCudaAgent 从 task.py + 环境配置取。"""
        self.round = 0
        cfg = getattr(self.env, "config", None)
        env_info = _read_env_info(self.env, getattr(self.config, "kernelcompile_dir", None))
        template_vars = {
            "task_id": getattr(cfg, "workspace_name", "") or "task",
            "backend": getattr(cfg, "backend", "") or self.config.backend,
            "gpu": getattr(cfg, "gpu", None) or "the CUDA device selected by the evaluator",
            "reference_code": read_workspace_file(self.env, "task.py"),
            "model_specs": model_specs_markdown(cfg),
            "env_versions": env_info["versions"] or self.config.env_versions,
            "triton_api_hints": env_info["triton_api_hints"],
            "gpu_specs": env_info["gpu_specs"],
            # CUTLASS 骨架/版本: 只有多后端 CUDA 的 cpu_gpu YAML 引用; triton loop 用不到
            # 便是未用 Jinja 变量 (StrictUndefined 不介意), 探测缺失时为空串 → {% if %} 整块隐去。
            "cutlass_version": env_info.get("cutlass_version", ""),
            "cutlass_example": env_info.get("cutlass_example", ""),
            # 手写 CUDA 锚点: 同样只有 cpu_gpu YAML 引用; 探测缺失时为空 → {% if %} 整块隐去。
            "handwritten_example": env_info.get("handwritten_example", ""),
            # CUTLASS 3.x (Scheme B) 骨架: 镜像里 bake 了 3.x golden 时注入 → 允许纯 GEMM 走
            # Hopper wgmma 峰值; 未 bake 时为空 → {% if %} 隐去, 模型退回 2.x Sm80 骨架。
            "cutlass3x_example": env_info.get("cutlass3x_example", ""),
        }
        self._env_versions = template_vars["env_versions"]
        return AgentRuntime.run(self, task, **(template_vars | kwargs))
