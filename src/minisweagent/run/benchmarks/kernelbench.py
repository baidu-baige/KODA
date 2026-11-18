#!/usr/bin/env python3

"""Run kernel optimization agents on KernelBench instances in batch mode.

Structured like run/benchmarks/swebench.py's original SWEBench runner: a thread pool
processes one instance per worker. The agent runs here on the control node; execution
happens in the Kernel Sandbox -- each instance gets its own sandbox session whose CPU
pod hosts a fresh workspace, and eval/profile commands are queued onto the shared GPU
worker pool. Per-instance artifacts (instances root: trajectory.json, reference.py,
kernel_turn_N.py, eval_turn_N.json + unpacked `kb_ws/` workspace dir and `workdir_meta/` run
artifacts) and the aggregate `preds.json` / `eval_result.json` are written
locally under the output directory; the workspaces themselves live and die with their
sandbox sessions.
"""

import concurrent.futures
import io
import json
import os
import random
import secrets
import re
import statistics
import tarfile
import threading
import time
import traceback
from enum import Enum
from pathlib import Path

import typer
from rich.live import Live

from minisweagent.agents.utils.timing import PhaseTimer
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments.kernel_sandbox import KernelSandboxEnvironment
from minisweagent.exceptions import SandboxInfraError
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.common import ProgressControllerAgent
from minisweagent.run.benchmarks.utils.sandbox_client import SandboxClient, SandboxClientConfig
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.metrics import COLLECTOR, fastn_distribution
from minisweagent.utils.readable_yaml import render_trajectory_yaml
from minisweagent.utils.serialize import UNSET, recursive_merge


class KernelBackend(str, Enum):
    """Supported candidate kernel backends (CLI --backend)."""

    triton = "triton"
    cuda = "cuda"



_HELP_TEXT = """Run kernel optimization agents on KernelBench instances in batch mode via the Kernel Sandbox.

[not dim]
More information about the usage: [bold green]https://github.com/baidu-baige/KODA[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c kernel.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot the default config file! See above.[/red]

[bold green]-c kernel.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c kernel.yaml -c agent.step_limit=50[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "controller_h.yaml"

DATASET_MAPPING = {
    "kernelbench": "dataset/kernelbench.jsonl",
}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()


def normalize_instance(instance: dict) -> dict:
    """Fill in 'instance_id'/'problem_statement' for datasets that don't use SWE-bench's schema."""
    if "level" in instance and "name" in instance:
        return {**instance, "instance_id": f"level_{instance['level']}_{instance['name']}"}
    if "instance_id" in instance and "problem_statement" in instance:
        return instance
    extra_info = instance.get("extra_info") or {}
    instance_id = instance.get("instance_id") or extra_info.get("name") or extra_info.get("problem_id")
    problem_statement = instance.get("problem_statement")
    if problem_statement is None and (prompt := instance.get("prompt")):
        problem_statement = next((m["content"] for m in prompt if m.get("role") == "user"), "")
    return {**instance, "instance_id": str(instance_id), "problem_statement": problem_statement or ""}


INSTANCE_META_KEYS = ("instance_id", "name", "level", "op_type")


def write_instance_meta(output_path: Path, instances: list[dict]) -> None:
    """Append per-instance metadata (instance_id → name/level/op_type) to instance_meta.jsonl.

    Lets downstream (SFT cleaning) map a trajectory dir back to its seed metadata without
    re-deriving instance ids. Missing columns are written as null. Reruns into the same
    output dir append again; readers take the last row per instance_id.
    """
    path = output_path / "instance_meta.jsonl"
    # an interrupted append leaves a half line without "\n"; start a fresh line so only that line is lost
    torn = path.exists() and path.read_bytes()[-1:] not in (b"", b"\n")
    with path.open("a", encoding="utf-8") as f:
        f.write(
            "\n" * torn
            + "".join(
                json.dumps({k: inst.get(k) for k in INSTANCE_META_KEYS}, ensure_ascii=False, default=str) + "\n"
                for inst in instances
            )
        )


def update_preds_file(output_path: Path, instance_id: str, result: dict) -> None:
    """Thread-safe read-modify-write of the batch predictions file (preds.json).

    Top-level instance ids are sorted for stable diffs, but each entry's fields keep
    their insertion order (the readable flat layout) rather than being alphabetized.
    """
    with _OUTPUT_FILE_LOCK:
        preds = json.loads(output_path.read_text()) if output_path.exists() else {}
        preds[instance_id] = result
        ordered = {k: preds[k] for k in sorted(preds)}
        output_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False))


def update_progress_file(
    output_path: Path, completed: int, total: int, instance_id: str, exit_status: str | None
) -> None:
    """Overwrite progress.json after each instance finishes (completed/total + last instance)."""
    with _OUTPUT_FILE_LOCK:
        output_path.write_text(
            json.dumps(
                {
                    "completed": completed,
                    "total": total,
                    "percentage": round(100 * completed / total, 2) if total else 0.0,
                    "last_instance": instance_id,
                    "last_exit_status": exit_status,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def _speedup_stats(pairs: list[tuple[str, float]]) -> dict:
    """Mean, median, and the top 10 (instance_id, speedup) for a list of speedup pairs."""
    speedups = [s for _, s in pairs]
    top10 = sorted(pairs, key=lambda p: p[1], reverse=True)[:10]
    return {
        "mean": round(statistics.mean(speedups), 4) if speedups else None,
        "median": round(statistics.median(speedups), 4) if speedups else None,
        "top10": [{"instance_id": iid, "speedup": round(s, 4)} for iid, s in top10],
    }


def write_eval_result_file(preds_path: Path, total: int) -> None:
    """Aggregate preds.json into an eval_result.json summary."""
    with _OUTPUT_FILE_LOCK:
        preds = json.loads(preds_path.read_text())
        eval_total = 0
        compiled_count = 0
        correct_count = 0
        eager_ref_speedups: list[tuple[str, float]] = []
        compiled_ref_speedups: list[tuple[str, float]] = []
        judge_speedups: list[tuple[str, float]] = []
        for instance_id, entry in preds.items():
            if not entry.get("solved"):
                continue
            if entry.get("compilation") is None:
                continue
            eval_total += 1
            if entry.get("compilation"):
                compiled_count += 1
                if entry.get("correctness"):
                    correct_count += 1
                    # eager_ref = 与 eager 参考对比 (eager_wall_speedup); compiled_ref = 与
                    # torch.compile 参考对比 (compiled_wall_speedup) —— 二者是裁决 speedup
                    # (= min, 取耗时最短参考当基线, 见 _speedup) 的构成项。
                    if (esp := entry.get("eager_ref_speedup")) is not None:
                        eager_ref_speedups.append((instance_id, esp))
                    if (crs := entry.get("compiled_ref_speedup")) is not None and crs > 0:
                        compiled_ref_speedups.append((instance_id, crs))
                    # 裁决口径: preds.flatten_eval 的 speedup 即服务端裁决值, 只对 correct 轮收
                    # (>0 守卫滤掉未裁决占位)。分布/统计键与 flat_eval 同名 (speedup)。
                    if (sp := entry.get("speedup")) is not None and sp > 0:
                        judge_speedups.append((instance_id, sp))
            elif entry.get("correctness"):
                correct_count += 1

        eval_result = {
            "total": total,
            "eval_total": eval_total,
            "compiled_rate": compiled_count / total if total else 0.0,
            "correct_rate": correct_count / total if total else 0.0,
            "compiled_rate_eval_total": compiled_count / eval_total if eval_total else 0.0,
            "correct_rate_eval_total": correct_count / eval_total if eval_total else 0.0,
            "eager_ref_speedup_distribution": fastn_distribution([s for _, s in eager_ref_speedups]),
            "compiled_ref_speedup_distribution": fastn_distribution([s for _, s in compiled_ref_speedups]),
            "eager_ref_speedup": _speedup_stats(eager_ref_speedups),
            "compiled_ref_speedup": _speedup_stats(compiled_ref_speedups),
            # 最终裁决口径 speedup (= min(eager, compiled) 双基线) 的分布与统计;
            # 相同时刻的 key 名与 flat_eval/preds 一致 (speedup), 一眼可对比。
            "speedup_distribution": fastn_distribution([s for _, s in judge_speedups]),
            "speedup": _speedup_stats(judge_speedups),
        }
        preds_path.with_name("eval_result.json").write_text(json.dumps(eval_result, indent=2))


def render_run_readme(output_dir: Path, planned: int | None = None) -> None:
    """Describe controller artifacts without referring to retired proxy/OpenCode layouts."""
    text = """# Controller evaluation output

- `preds.json`: selected candidate and correctness summary for each completed task.
- `eval_result.json`: aggregate compilation/correctness and speedup statistics.
- `<task>/trajectory.json`: model replies, controller decisions and tool receipts.
- `<task>/trajectory.readable.yaml`: readable rendering of the same trajectory.
- `<task>/kernel_turn_N.py`, `.cu`, `eval_turn_N.json`: exported per-round artifacts when available.

Use the raw eval receipts to interpret correctness and timing. Failed workloads remain in the denominator.
This is a single-workload optimization run, not an official SOL leaderboard score.
"""
    if planned is not None:
        text += f"\nPlanned tasks: {planned}.\n"
    try:
        (output_dir / "README.md").write_text(text, encoding="utf-8")
    except OSError as error:
        logger.warning(f"Could not write output README: {error}")


_EVAL_JSON_DECODER = json.JSONDecoder()


def _parse_eval_output(content: str) -> dict | None:
    """Extract the eval-result JSON object embedded in an eval round's <output> block."""
    i = content.find("<output>")
    if i < 0:
        return None
    j = content.find("{", i)
    if j < 0:
        return None
    try:
        obj, _ = _EVAL_JSON_DECODER.raw_decode(content, j)
    except Exception:
        return None
    return obj if isinstance(obj, dict) and "compiled" in obj else None


def _code_of(msg: dict) -> str | None:
    """Recover the kernel source an assistant turn proposed (action command, else a ```python fence)."""
    actions = (msg.get("extra") or {}).get("actions") or []
    if actions and actions[0].get("command"):
        return actions[0]["command"]
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(str(x) for x in content)
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", content, re.S)
    if not blocks:
        return None
    # reasoning models emit draft blocks before the final file; pick the submission
    return next((b for b in reversed(blocks) if "ModelNew" in b), blocks[-1])


def _cu_of(msg: dict) -> str | None:
    """Recover the native kernel.cu an assistant turn proposed (multi-file action `kernel_cu`,
    else the last ```cpp / ```cuda fence). Only the loop multi-file agent carries `kernel_cu`.

    The fence fallback captures the full language token then checks membership (mirroring the model's
    ControllerModel parser) so a prefix like ```curl / ```csharp is not mistaken for CUDA."""
    actions = (msg.get("extra") or {}).get("actions") or []
    if actions and actions[0].get("kernel_cu"):
        return actions[0]["kernel_cu"]
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(str(x) for x in content)
    cu_langs = {"cpp", "c++", "cxx", "cc", "c", "cu", "cuda"}
    blocks = [
        m.group(2)
        for m in re.finditer(r"```[ \t]*([A-Za-z0-9_+.\-]*)[^\n]*\n(.*?)```", content, re.S)
        if m.group(1).lower() in cu_langs
    ]
    return blocks[-1].strip() if blocks else None


def _failed_layers(ev: dict) -> list[str]:
    layers = (ev.get("anti_hacking_report") or {}).get("layers") or {}
    return [k for k, v in layers.items() if not (v or {}).get("passed", True)]


def _anti_passed(ev: dict) -> bool:
    summary = (ev.get("anti_hacking_report") or {}).get("summary") or {}
    return bool(summary["passed"]) if "passed" in summary else (len(_failed_layers(ev)) == 0)


def _genuine(ev: dict) -> bool:
    """A real solution: correct, compiled, and neither a decoy nor an anti-hacking failure."""
    return bool(ev.get("correctness") and ev.get("compiled") and not ev.get("decoy_kernel") and _anti_passed(ev))


def _r(x, n: int = 4):
    return round(x, n) if isinstance(x, int | float) else x


def _pos(x):
    """A runtime/time value only if it is a real positive measurement (eval defaults to -1.0/None)."""
    return x if isinstance(x, int | float) and x > 0 else None


def _speedup(ev: dict):
    """Canonical speedup for scoring: the eval's ``speedup`` —— kerneleval 2026-08-25
    收敛后裁决口径 = min(eager_wall_speedup, compiled_wall_speedup)（交错墙钟;
    2026-08-28 修正方向, 原 max 取最慢参考）。"""
    return ev.get("speedup")


def _ratio(a, b):
    try:
        return round(a / b, 4) if b else None
    except Exception:
        return None


def flat_eval(ev: dict, turn_id: int | None, finish_reason: str | None, timings: dict | None = None) -> dict:
    """Flatten one round's raw eval into a readable, kernelgym-styled record.

    ``timings`` is the agent-loop ``phase_timings`` captured in the trajectory and stored
    verbatim: ``{"llm": <s>, "tools": [{"name", "total", "stage_timings", "sandbox"}, ...]}``.
    Each tool's ``total`` is its wall-clock (incl. sandbox); ``stage_timings`` is the eval
    pipeline's own per-stage breakdown (the fine-grained inner view), and their gap is the
    sandbox queue/transport/startup overhead.
    """
    md = ev.get("metadata") or {}
    cov = md.get("triton_kernel_coverage") or {}
    krs = md.get("kernel_runtime_stats") or {}
    sp = _speedup(ev)
    trials = md.get("correctness_trials")
    if isinstance(trials, str):
        trials = trials.strip("() ")
    nc, nt = cov.get("num_custom_kernels"), cov.get("num_total_kernels")
    ct, tt = cov.get("custom_kernel_cuda_time_in_profiling_us"), cov.get("total_kernel_run_time_in_profiling_us")
    return {
        "turn_id": turn_id,
        "sample_id": 0,
        "finish_reason": finish_reason,
        "correctness": bool(ev.get("correctness")),
        "compilation": bool(ev.get("compiled")),
        "is_decoy_kernel": bool(ev.get("decoy_kernel")),
        "success": ev.get("status") == "completed" and not ev.get("error_message"),
        "status": ev.get("status"),
        "error": ev.get("error_message") or ev.get("error_code"),
        "is_speedup_positive": (sp or 0) > 1.0,
        "speedup": _r(sp),
        "score": (_r(sp) if _genuine(ev) else 0.0),
        "speedup_stats": ev.get("speedup_stats"),
        # Two reference baselines (kerneleval 2026-08-25 收敛后唯一 schema:
        # 线性交错墙钟双基线, 裁决 speedup = min —— 写 preds 的键名不变, 读的源键名
        # 即当前 schema 的 *_wall_speedup / *_wall_time, 不再兼容旧 pod 输出字段。
        #   compiled_ref_speedup = wall vs the torch.compile'd reference;
        #   eager_ref_speedup    = wall vs the eager (uncompiled) reference.
        "compiled_ref_speedup": _r(ev.get("compiled_wall_speedup")),
        "eager_ref_speedup": _r(ev.get("eager_wall_speedup")),
        "kernel_runtime_ms": _pos(ev.get("kernel_wall_time")),
        "reference_runtime_ms": _pos(ev.get("eager_wall_time")),
        "compiled_reference_runtime_ms": _pos(ev.get("compiled_wall_time")),
        "kernel_runtime_stats": {k: krs.get(k) for k in ("mean", "std", "min", "max", "num_trials")} if krs else None,
        # per-trial 明细（event 口径原始样本，100 trials）：事后对比用，不参与 speedup
        "kernel_trial_ms": md.get("kernel_trial_ms"),
        "reference_trial_ms": md.get("reference_trial_ms"),
        "compiled_reference_trial_ms": md.get("compiled_reference_trial_ms"),
        "num_correct_trials": md.get("num_correct_trials"),
        "num_perf_trials": md.get("num_perf_trials"),
        "correctness_trials": trials,
        "phase_timings": timings or None,
        "num_custom_kernels": nc,
        "num_total_kernels": nt,
        "num_coverage": _ratio(nc, nt),
        "custom_kernel_cuda_time_us": ct,
        "total_kernel_run_time_us": tt,
        "time_coverage": _ratio(ct, tt),
        "triton_kernels_in_profiling": cov.get("triton_kernels_in_profiling"),
        "triton_kernels_not_in_profiling": cov.get("triton_kernels_not_in_profiling"),
        "anti_hacking_passed": _anti_passed(ev),
        "anti_hacking_failed_layers": _failed_layers(ev),
        "anti_hacking_report": ev.get("anti_hacking_report"),
        "hardware": md.get("hardware"),
        "device": md.get("device"),
        "cuda_version": md.get("cuda_version"),
        "backend": md.get("backend"),
        "seed": md.get("seed"),
        # 环境快照 (AI-DRL-299): 同一 kernel 跨镜像表现不同时先对这几个版本。
        # 采集端是 pod 内 kerneleval 的 _stage_environment, 经 to_full_dict 平铺进 metadata。
        "torch_version": md.get("torch_version"),
        "triton_version": md.get("triton_version"),
        "transformers_version": md.get("transformers_version"),
        "python_version": md.get("python_version"),
    }


def _walk_turns(data: dict) -> list[tuple]:
    """Extract controller eval observations and their immutable candidate source."""
    turns = []
    pending_code = pending_cu = finish_reason = None
    for msg in data.get("messages", []):
        if msg.get("role") == "assistant":
            pending_code = _code_of(msg) or pending_code
            pending_cu = _cu_of(msg) or pending_cu
            finish_reason = (msg.get("extra") or {}).get("finish_reason") or msg.get("finish_reason")
        elif msg.get("role") == "user" and re.match(r"^### Round \d+/\d+: eval", msg.get("content", "")):
            ev = _parse_eval_output(msg.get("content", ""))
            if ev is not None:
                turn_id = len(turns) + 1
                turns.append(
                    (
                        turn_id,
                        ev,
                        pending_code,
                        finish_reason,
                        (msg.get("extra") or {}).get("phase_timings") or {},
                        pending_cu,
                    )
                )
    return turns


def extract_best(trajectory_path: Path) -> dict:
    """Flat, readable summary of a trajectory: the best genuine eval + a per-turn history.

    "Best" = the fastest genuine round (correct, compiled, not a decoy). Falls back to the
    last real eval when nothing genuine was found. Never raises: returns a minimal record on failure.
    """
    empty = {"solved": False, "num_turns": 0, "best_turn": None, "correctness": False, "speedup": None, "score": 0.0}
    try:
        data = json.loads(Path(trajectory_path).read_text())
    except Exception:
        return empty
    turns = _walk_turns(data)
    if not turns:
        return empty
    best = None
    for t in turns:
        if _genuine(t[1]) and (best is None or (_speedup(t[1]) or 0.0) > (_speedup(best[1]) or 0.0)):
            best = t
    chosen = best or turns[-1]
    entry = {"solved": best is not None, "num_turns": len(turns), "best_turn": chosen[0]}
    entry.update(flat_eval(chosen[1], chosen[0], chosen[3], chosen[4]))
    entry["per_turn"] = [
        {
            "turn": tid,
            "correctness": bool(ev.get("correctness")),
            "compilation": bool(ev.get("compiled")),
            "decoy": bool(ev.get("decoy_kernel")),
            "anti_hacking_failed_layers": _failed_layers(ev),
            "speedup": _r(_speedup(ev)),
            "score": (_r(_speedup(ev)) if _genuine(ev) else 0.0),
            "speedup_stats": ev.get("speedup_stats"),
            "phase_timings": tim or None,
        }
        for (tid, ev, _code, _fr, tim, _cu) in turns
    ]
    entry["best_code"] = chosen[2]
    return entry


def _extract_archive(archive: bytes, dest: Path) -> None:
    """Untar a fetched pod archive into dest. Members are flat (`./task.py`) and dest is the owning
    directory (`kb_ws/` or `workdir_meta/`), so extraction lands in one dir instead of spilling
    into the instance root. Members are path-checked (extractall's default permits `../`/absolute
    traversal)."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        safe = [m for m in tf.getmembers() if not m.name.startswith("/") and ".." not in Path(m.name).parts]
        tf.extractall(dest, members=safe)


def export_problem_dir(
    problem_statement: str,
    output_dir: Path,
    instance_id: str,
    trajectory_path: Path,
    archive: bytes | None = None,
    meta_archive: bytes | None = None,
) -> None:
    """Write kernelgym-style per-instance artifacts next to the trajectory.

    reference.py (the original problem), kernel_turn_N.py (each round's submitted kernel.py),
    kernel_turn_N.cu (that round's native CUDA/CUTLASS/C++ source, when the round carried one),
    eval_turn_N.json (each round's flat metrics) and, when `archive` is given, the whole pod
    workspace (incl. .git history — carries multi-file C++ kernels the flat {code, output} records
    alone cannot) is unpacked into `kb_ws/`; `meta_archive` (run artifacts: oc_done / oc_run.log
    / eval records) into `workdir_meta/`. Best-effort: never raises into the run.
    """
    dst = output_dir / instance_id
    dst.mkdir(parents=True, exist_ok=True)
    try:
        (dst / "reference.py").write_text(problem_statement)
    except Exception:
        pass
    try:
        data = json.loads(Path(trajectory_path).read_text())
    except Exception:
        return
    for tid, ev, code, fr, tim, cu in _walk_turns(data):
        if code:
            (dst / f"kernel_turn_{tid}.py").write_text(code.rstrip() + "\n")
        if cu:
            (dst / f"kernel_turn_{tid}.cu").write_text(cu.rstrip() + "\n")
        (dst / f"eval_turn_{tid}.json").write_text(
            json.dumps(flat_eval(ev, tid, fr, tim), indent=2, ensure_ascii=False)
        )
    try:
        # Human-readable mirror of trajectory.json (block literals for multiline
        # strings). Best-effort: renders to the same output dir as the trajectory.
        (dst / "trajectory.readable.yaml").write_text(render_trajectory_yaml(trajectory_path), encoding="utf-8")
    except Exception:
        pass
    try:
        if archive:
            _extract_archive(archive, dst / "kb_ws")
    except Exception as e:
        logger.warning(f"kb_ws archive extraction failed for {instance_id}: {type(e).__name__}: {e}")
    try:
        if meta_archive:
            _extract_archive(meta_archive, dst / "workdir_meta")
    except Exception as e:
        logger.warning(f"workdir_meta archive extraction failed for {instance_id}: {type(e).__name__}: {e}")


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    backend: KernelBackend,
    progress_manager: RunBatchProgressManager,
    total_instances: int,
    sandbox_client: SandboxClient,
    max_infra_retries: int = 2,
) -> None:
    """Create a sandbox session with a fresh workspace for one instance and run the configured agent.

    A dead sandbox session (pod evicted → SandboxInfraError) is not a model failure: re-run the
    whole problem on a brand-new session, bounded by `max_infra_retries`. Only after exhausting
    those re-runs is the instance recorded as failed (exit_status "SandboxInfraError"), so infra
    flakiness never gets scored as an incorrect kernel.
    """
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    trajectory_path = instance_dir / "trajectory.json"
    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Creating sandbox session")
    COLLECTOR.record_instance_start(instance_id)

    timer = PhaseTimer()
    exit_status = None
    env = None
    agent = None  # env 创建失败时 except 分支仍需安全访问 getattr(agent, ...)
    instance_failed = False
    try:
        for infra_attempt in range(1, max_infra_retries + 2):
            stage = "sandbox_setup"
            try:
                env_cfg = config.get("environment", {})
                if env_cfg.get("session_type", "cpu_gpu") != "cpu_gpu":
                    raise ValueError("This release supports controller cpu_gpu sessions only")
                env = KernelSandboxEnvironment(
                    **env_cfg
                    | {
                        "base_url": sandbox_client.config.base_url,
                        "image": sandbox_client.config.image,
                        "gpu_pool_id": sandbox_client.gpu_pool_id,
                        "problem_source": instance["code"],
                        "workspace_name": instance_id,
                        "backend": backend.value,
                        "modeling": instance.get("modeling", "") or "",
                        "config": instance.get("config", "") or "",
                    }
                )
                stage = "agent_setup"
                timer.mark("sandbox_setup")
                logger.info(f"{instance_id} stage sandbox_setup: {timer.timings['sandbox_setup']}s")
                # Per-instance api_base templating: a trajectory-capturing proxy serializes requests that share
                # a trajectory id, so each problem needs its own. If model_kwargs.api_base contains the literal
                # "{instance_id}", substitute it here (thread-safe copy; no-op for configs without the marker).
                _mcfg = config.get("model", {})
                _mk = _mcfg.get("model_kwargs")
                if isinstance(_mk, dict) and "{instance_id}" in str(_mk.get("api_base", "")):
                    _mk = dict(_mk)
                    _mk["api_base"] = _mk["api_base"].format(instance_id=instance_id)
                    _mcfg = dict(_mcfg, model_kwargs=_mk)
                tol = instance.get("tolerance") or {}
                agent = ProgressControllerAgent(
                    get_model(config=_mcfg),
                    env,
                    progress_manager=progress_manager,
                    instance_id=instance_id,
                    output_path=trajectory_path,
                    **config.get("agent", {}) | {"eval_atol": tol.get("max_atol"), "eval_rtol": tol.get("max_rtol")},
                )
                stage = "agent_run"
                exit_status = agent.run(
                    instance["code"],
                    modeling=instance.get("modeling", "") or "",
                    config=instance.get("config", "") or "",
                ).get("exit_status")
                break
            except SandboxInfraError as e:
                if env is not None:
                    env.cleanup()
                    env = None
                if infra_attempt >= max_infra_retries + 1:
                    logger.error(f"{instance_id} exhausted {max_infra_retries} sandbox infra re-runs: {e}")
                    raise
                logger.warning(
                    f"{instance_id} sandbox infra failure ({e}); re-running problem on a fresh "
                    f"session, attempt {infra_attempt}/{max_infra_retries}"
                )
                progress_manager.update_instance_status(
                    instance_id, f"Infra re-run {infra_attempt}/{max_infra_retries}"
                )
                time.sleep(min(30 * infra_attempt, 120))
        timer.mark("agent_run")
        logger.info(f"{instance_id} stage agent_run: {timer.timings['agent_run']}s (exit_status={exit_status})")
    except (Exception, KeyboardInterrupt) as e:
        timer.mark("agent_run")
        # token 超限不算异常, 按"正常停止"处理: 它是多轮累积把 input+completion 顶过模型 max context
        # 窗口触发的终止信号(非质量问题), 崩溃前的轮次都是有效产出。不记 error/traceback、不计 instance
        # error, 直接落到下方正常收尾(extract_best 救回 best_turn/speedup/solved + record_instance_done)。
        if type(e).__name__ == "BadRequestError" and (
            "Please reduce the number of tokens in the input messages or the completion "
            "to fit within the limit." in str(e)
        ):
            exit_status = "ContextWindowExceeded"
            logger.info(f"{instance_id} stopped: context window exceeded, treated as normal stop")
        else:
            instance_failed = True
            logger.error(f"Error processing {instance_id}: {e}", exc_info=True)
            logger.info(f"{instance_id} stage timings (failed): {timer.timings}")
            exit_status = type(e).__name__
            if stage == "sandbox_setup":
                COLLECTOR.record_error("session_create", e)
            record = {
                "instance_id": instance_id,
                "exit_status": exit_status,
                "solved": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            update_preds_file(output_dir / "preds.json", instance_id, record)
            write_eval_result_file(output_dir / "preds.json", total_instances)
            # 异常/中断路径也导出已产出的轮次产物：agent 中途被杀（LLM 超时等）时
            # trajectory 里仍有已完成轮的 eval/kernel 原文，export 能从消息重建
            # reference.py / kernel_turn_N.py / eval_turn_N.json（best-effort，不 raise）。
            export_problem_dir(
                instance["code"],
                output_dir,
                instance_id,
                trajectory_path,
                getattr(agent, "workspace_archive", None),
                getattr(agent, "workdir_meta_archive", None),
            )
            return
    finally:
        if env is not None:
            env.cleanup()
        progress_manager.on_instance_end(instance_id, exit_status)
        update_progress_file(
            output_dir / "progress.json", progress_manager.n_completed, total_instances, instance_id, exit_status
        )
        if instance_failed:
            COLLECTOR.record_instance_error(exit_status, instance_id=instance_id)

    result = {"instance_id": instance_id, "exit_status": exit_status}
    try:
        result.update(extract_best(trajectory_path))
    except Exception as e:  # summary extraction must never fail the run
        logger.warning(f"extract_best failed for {instance_id}: {e}")
    result["trajectory"] = str(trajectory_path)
    try:
        export_problem_dir(
            instance["code"],
            output_dir,
            instance_id,
            trajectory_path,
            getattr(agent, "workspace_archive", None),
            getattr(agent, "workdir_meta_archive", None),
        )
    except Exception as e:  # per-instance export must never fail the run
        logger.warning(f"export_problem_dir failed for {instance_id}: {e}")
    update_preds_file(output_dir / "preds.json", instance_id, result)
    write_eval_result_file(output_dir / "preds.json", total_instances)
    COLLECTOR.record_instance_done(
        exit_status,
        solved=bool(result.get("solved")),
        instance_id=instance_id,
        correctness=bool(result.get("correctness")),
    )
    timer.mark("cleanup_export")
    logger.info(
        f"{instance_id} stage timings: {timer.timings} "
        f"total={round(sum(timer.timings.values()), 3)}s speedup={result.get('speedup')}"
    )


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option(
        "lite", "--subset", help="SWEBench subset to use or path to a dataset",
        rich_help_panel="Data selection"),
    split: str = typer.Option(
        "dev", "--split", help="Dataset split",
        rich_help_panel="Data selection"),
    slice_spec: str = typer.Option(
        "", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)",
        rich_help_panel="Data selection"),
    filter_spec: str = typer.Option(
        "", "--filter", help="Filter instance IDs by regex",
        rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(
        False, "--shuffle", help="Shuffle instances",
        rich_help_panel="Data selection"),
    output: str = typer.Option(
        "", "-o", "--output", help="Output directory",
        rich_help_panel="Basic"),
    workers: int = typer.Option(
        1, "-w", "--workers", help="Number of worker threads for parallel processing",
        rich_help_panel="Basic"),
    model: str | None = typer.Option(
        None, "-m", "--model", help="Model to use",
        rich_help_panel="Basic"),
    model_class: str | None = typer.Option(
        None, "--model-class",
        help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')",
        rich_help_panel="Advanced"),
    backend: KernelBackend = typer.Option(
        KernelBackend.cuda, "--backend", help="Candidate kernel backend",
        rich_help_panel="Advanced"),
    compiled_reference: bool = typer.Option(
        False, "--compiled-reference", help="Also time torch.compile(reference) during eval",
        rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(
        False, "--redo-existing", help="Redo existing instances",
        rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option(
        [str(DEFAULT_CONFIG_FILE)], "-c", "--config",
        help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    sandbox_server_url: str = typer.Option(
        SandboxClientConfig.model_fields["base_url"].default, "--sandbox-server-url",
        help="Base URL of the shared GPU worker pool API", rich_help_panel="Advanced"),
    sandbox_image: str = typer.Option(
        SandboxClientConfig.model_fields["image"].default, "--sandbox-image",
        help="Container image to use for the GPU worker pool", rich_help_panel="Advanced"),
    pool_replicas: int | None = typer.Option(
        None, "--pool-replicas",
        help="GPU pool worker replicas; None (default) = workers (1:1, one worker per instance). "
             "Set < workers for oversubscription (CPU pod evals queue on the master)",
        rich_help_panel="Advanced"),
    wandb: bool = typer.Option(
        False, "--wandb/--no-wandb",
        help="Stream live trajectory-quality metrics (timings/tokens/tool calls/errors) to Weights & Biases",
        rich_help_panel="Metrics"),
    wandb_project: str = typer.Option(
        "kernelbench", "--wandb-project", help="wandb project name", rich_help_panel="Metrics"),
    wandb_run_name: str = typer.Option(
        "", "--wandb-run-name", help="wandb run name (default: auto-generated)", rich_help_panel="Metrics"),
    wandb_entity: str = typer.Option(
        "", "--wandb-entity", help="wandb entity/team (default: your default entity)", rich_help_panel="Metrics"),
    wandb_interval: float = typer.Option(
        120.0, "--wandb-interval", min=1.0, help="Seconds between wandb metric flushes (default: 120)",
        rich_help_panel="Metrics"),
    wandb_window: float = typer.Option(
        3600.0, "--wandb-window", min=1.0, help="Rolling interval-metric window in seconds (default: 3600)",
        rich_help_panel="Metrics"),
    wandb_mode: str = typer.Option(
        "online", "--wandb-mode", help="wandb mode: online / offline / disabled", rich_help_panel="Metrics"),
    max_infra_retries: int = typer.Option(
        2, "--max-infra-retries",
        help="Max times to re-run a whole problem on a fresh session when its sandbox session dies "
             "(pod evicted). Exhausting these records the instance as 'SandboxInfraError', not a "
             "wrong kernel.",
        rich_help_panel="Advanced"),
) -> None:
    """Run KernelBench evaluation."""
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    from datasets import DatasetDict, load_dataset, load_from_disk

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    if Path(dataset_path, "state.json").exists():
        dataset = load_from_disk(dataset_path)
    elif Path(dataset_path).is_file() and Path(dataset_path).suffix in {".jsonl", ".parquet"}:
        loader = "json" if Path(dataset_path).suffix == ".jsonl" else "parquet"
        dataset = load_dataset(loader, data_files=dataset_path, split=split)
    else:
        dataset = load_dataset(dataset_path, split=split)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]
    instances = [normalize_instance(instance) for instance in dataset]
    logger.info(f"Loaded {len(instances)} instances")

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    total_instances = len(instances)
    # 旁路元数据（供 SFT 清洗按 instance_id 查 op_type）：写盘失败只告警，不阻断推理。
    try:
        write_instance_meta(output_path, instances)
    except Exception as e:
        logger.warning(f"write_instance_meta failed: {e}")
    if not redo_existing and (output_path / "preds.json").exists():
        preds = json.loads((output_path / "preds.json").read_text())
        done = {k for k, v in preds.items() if v.get("correctness") is not None}
        logger.info(f"Skipping {len(done)} existing instances ({len(preds) - len(done)} incomplete, will retry)")
        instances = [i for i in instances if i["instance_id"] not in done]
    logger.info(f"Running on {len(instances)} instances...")
    # 开跑即落一份 README（planned 标注进行中）：中途崩溃/中断时目录仍有导览，
    # 结束时的 finally 会刷新为最终版（统计以下方实际数据为准）。
    try:
        render_run_readme(output_path, planned=total_instances)
    except Exception as e:
        logger.warning(f"render_run_readme (start) failed: {e}")
    if not instances:
        return
    workers = min(workers, len(instances))  # no point in more GPU workers/threads than instances

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({"model": {"model_name": model or UNSET, "model_class": model_class or UNSET}})
    if compiled_reference:
        configs.append({"environment": {"env": {"KERNEL_EVAL_COMPILED_REFERENCE": "1"}}})
    config = recursive_merge(*configs)
    if config.get("environment", {}).get("session_type", "cpu_gpu") != "cpu_gpu":
        raise typer.BadParameter("The controller release supports cpu_gpu sessions only")
    agent_config = config.setdefault("agent", {})
    if not agent_config.get("eval_server_token"):
        agent_config["eval_server_token"] = os.environ.get("KODA_EVAL_SERVER_TOKEN") or secrets.token_urlsafe(32)


    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    if wandb:
        api_base = str((config.get("model", {}).get("model_kwargs", {}) or {}).get("api_base", ""))
        COLLECTOR.start(
            project=wandb_project, name=wandb_run_name, entity=wandb_entity,
            interval=wandb_interval, window=wandb_window, mode=wandb_mode, trajectory_dir=str(output_path),
            output_dir=str(output_path),
            run_timeout=float(config.get("agent", {}).get("run_timeout", 0) or 0),
            total_instances=total_instances,
            api_base=api_base,
            trajectory_format="auto",
            round_mode=True,
            config={
                "subset": subset, "split": split, "slice": slice_spec, "agent_type": "controller",
                "backend": backend.value, "workers": workers, "num_instances": total_instances,
                "model": config.get("model", {}).get("model_name"),
                "metrics_source": "round",
            },
        )

    session_type = config.get("environment", {}).get("session_type", "cpu_gpu")
    sandbox_client = None
    owns_pool = False  # 只删自己建的池；复用 config 指定的已有池时不删 (池生命周期归调用方)
    if session_type == "cpu_gpu":
        # Route the pool to a cluster by GPU resource type (environment.resource_request/limit in the
        # config). Unset = SandboxClient omits `resources` and the master uses its default pool cluster.
        env_cfg = config.get("environment", {})
        try:
            sandbox_client = SandboxClient(
                base_url=sandbox_server_url, image=sandbox_image,
                resource_request=env_cfg.get("resource_request", {}),
                resource_limit=env_cfg.get("resource_limit", {}),
            )
            if env_cfg.get("gpu_pool_id"):
                # config 指定了已有 pool → 复用: 跳过 ensure_created, 末尾也不删。
                # 池镜像决定 GPU worker 镜像 (session 的 image 只覆盖 CPU pod), 不一致 = CPU 新/GPU 旧
                # 静默跑完 → 复用前 GET 校验: 池不存在直接 SandboxAPIError, 镜像不符直接中止。
                sandbox_client.gpu_pool_id = env_cfg["gpu_pool_id"]
                pool_image = sandbox_client.get()["image"]
                if pool_image != sandbox_image:
                    raise ValueError(
                        f"GPU pool {sandbox_client.gpu_pool_id} image {pool_image!r} != --sandbox-image "
                        f"{sandbox_image!r}; 换镜像匹配的池, 或清空 gpu_pool_id 让本次自建自删"
                    )
                logger.info(
                    f"Reusing existing GPU pool {sandbox_client.gpu_pool_id} (image={pool_image}) "
                    f"on {sandbox_server_url}"
                )
            else:
                # GPU pool replicas: 显式 --pool-replicas 覆盖 (M:N 共享池, eval 在 master 排队);
                # None = workers (1:1, 每实例独占 worker, 无排队)。
                replicas = pool_replicas if pool_replicas is not None else workers
                logger.info(f"Creating GPU pool on {sandbox_server_url} with image {sandbox_image} "
                            f"and replicas {replicas} (workers={workers})")
                sandbox_client.ensure_created(image=sandbox_image, replicas=replicas)
                owns_pool = True
        except Exception as e:
            logger.error(f"Failed to create GPU pool on {sandbox_server_url}: {e}")
            COLLECTOR.record_error("pool_create", e)
            COLLECTOR.stop()
            raise
        logger.info(f"GPU pool {sandbox_client.gpu_pool_id} is ready")
        COLLECTOR.bind_gpu_pool(sandbox_server_url, sandbox_client.gpu_pool_id)

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    try:
        with Live(progress_manager.render_group, refresh_per_second=4):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_instance, instance, output_path, config, backend,
                        progress_manager, total_instances, sandbox_client, max_infra_retries
                    ): instance["instance_id"]
                    for instance in instances
                }
                try:
                    process_futures(futures)
                except KeyboardInterrupt:
                    logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                    for future in futures:
                        if not future.running() and not future.done():
                            future.cancel()
                    process_futures(futures)
    finally:
        COLLECTOR.stop()
        # 批量收尾：写批次 README 导览（任何退出路径都落）——本目录会被没跑过该批的
        # 人/agent review，README 是唯一入口说明；仅覆盖同目录，异常不阻断收尾。
        try:
            render_run_readme(output_path)
        except Exception as e:
            logger.warning(f"render_run_readme failed: {e}")
        if sandbox_client is not None and owns_pool:
            logger.info(f"Deleting GPU pool {sandbox_client.gpu_pool_id}")
            try:
                sandbox_client.delete()
            except Exception as e:
                logger.warning(f"Failed to delete GPU pool {sandbox_client.gpu_pool_id}: {e}")


if __name__ == "__main__":
    app()
