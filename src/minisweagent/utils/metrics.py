"""Optional controller-only round, sandbox and evaluation metrics."""

from __future__ import annotations

import collections
import json
import math
import re
import statistics
import threading
import time
from pathlib import Path

import requests

from minisweagent.agents.utils.timing import extract_tool_result
from minisweagent.utils.log import logger


def _mean(xs: list[float]) -> float:
    """Arithmetic mean of a value list; 0.0 when empty."""
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def _median(xs: list[float]) -> float:
    """Median of a value list; 0.0 when empty."""
    return round(statistics.median(xs), 4) if xs else 0.0


def _pct(xs: list[float], q: float) -> float:
    """q-th percentile (0-100), numpy-style linear interpolation between closest ranks; 0.0 on empty."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return round(float(s[0]), 4)
    pos = (len(s) - 1) * (q / 100)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (pos - lo), 4)


def _pcts(xs: list[float], prefix: str) -> dict:
    """`<prefix>_p50/p80/p90/p95/p99` for a value list (tail view: exposes skew hidden by the mean)."""
    return {f"{prefix}_p{q}": _pct(xs, q) for q in (50, 80, 90, 95, 99)}


def _time_stats(xs: list[float], prefix: str) -> dict:
    """mean/min/max + p50/p80/p90/p95/p99 (all suffixed ``_s``) for a list of seconds; ``{}`` if empty.

    The time analogue of ``_token_metrics``: every ``time/<unit>/<dim>`` (unit=round/turn/instance,
    dim=llm/env/total) shares this shape so the dashboards line up with the ``token/`` panels.
    """
    if not xs:
        return {}
    payload = {
        f"{prefix}_mean_s": _mean(xs),
        f"{prefix}_min_s": round(min(xs), 4),
        f"{prefix}_max_s": round(max(xs), 4),
    }
    return payload | {f"{prefix}_p{q}_s": _pct(xs, q) for q in (50, 80, 90, 95, 99)}


def _token_metrics(units: list[dict], prefix: str) -> dict:
    """Full token/char metric set (mean/max/percentiles + reasoning ratio) for a list of units.

    A "unit" is either one turn (``token/turn``) or one instance's summed tokens (``token/instance``);
    the two share metric names so the dashboards line up. ``{}`` when there are no units.
    """
    if not units:
        return {}
    comp = [u["completion_tokens"] for u in units]
    prm = [u["prompt_tokens"] for u in units]
    payload = {
        f"{prefix}/completion_tokens_mean": _mean(comp),
        f"{prefix}/completion_tokens_max": max(comp),
        f"{prefix}/prompt_tokens_mean": _mean(prm),
        f"{prefix}/prompt_tokens_max": max(prm),
        f"{prefix}/cot_chars_mean": _mean([u["cot_chars"] for u in units]),
        f"{prefix}/content_chars_mean": _mean([u["content_chars"] for u in units]),
    }
    known_reasoning = [u for u in units if u["reasoning_tokens"] is not None]
    if known_reasoning:
        reas = [u["reasoning_tokens"] for u in known_reasoning]
        completion = sum(u["completion_tokens"] for u in known_reasoning)
        payload |= {
            f"{prefix}/reasoning_tokens_mean": _mean(reas),
            f"{prefix}/reasoning_tokens_max": max(reas),
            f"{prefix}/reasoning_ratio": round(sum(reas) / completion, 4) if completion else 0.0,
        } | _pcts(reas, f"{prefix}/reasoning_tokens")
    return payload | _pcts(comp, f"{prefix}/completion_tokens") | _pcts(prm, f"{prefix}/prompt_tokens")


def _sum_by_instance(turns: list[dict]) -> list[dict]:
    """Per-instance token/char aggregates — units for ``token/instance``.

    Generated fields (completion/reasoning tokens, cot/content chars) are **summed** over the
    instance's turns = total the model produced for that task, each turn counted once (they come
    from each turn's own response, not the request history). ``prompt_tokens`` is **NOT summed**:
    every request carries the full cumulative conversation, so the per-turn prompt already contains
    all earlier turns — summing would re-count the context many times over. We take the peak
    (``max``, = the final/largest context), i.e. how big the context grew for that instance.
    """
    acc: dict[str, dict] = {}
    add_keys = ("completion_tokens", "reasoning_tokens", "cot_chars", "content_chars")
    for t in turns:
        u = acc.setdefault(t.get("instance_id", ""), dict.fromkeys(add_keys, 0) | {"prompt_tokens": 0})
        for k in add_keys:
            # An unreported reasoning count makes the instance's total unknown.
            u[k] = u[k] + t[k] if u[k] is not None and t[k] is not None else None
        u["prompt_tokens"] = max(u["prompt_tokens"], t["prompt_tokens"])  # peak context, never summed
    return list(acc.values())


def _usage(message: dict) -> dict:
    """Token usage from an agent message's ``extra.response.usage``; ``{}`` when absent."""
    return (((message.get("extra") or {}).get("response") or {}).get("usage")) or {}


def _brief_time_stats(values: list[float], prefix: str) -> dict:
    """Mean and tail latency for new timing details; retain legacy distributions elsewhere."""
    if not values:
        return {}
    return {f"{prefix}_mean_s": _mean(values), f"{prefix}_p95_s": _pct(values, 95)}


def _tool_time_metrics(rounds: list[dict]) -> dict:
    """One distribution per measured tool stage, without round or instance copies."""
    samples: dict[str, list[float]] = {}
    for row in rounds:
        for tool in row.get("tool_phases", []):
            name = _label(tool.get("name"))
            for stage, seconds in (tool.get("stage_timings") or {}).items():
                if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
                    samples.setdefault(f"time/{name}_{_label(stage)}", []).append(seconds)
    payload = {}
    for name, values in samples.items():
        payload |= _brief_time_stats(values, name)
    return payload


def _sandbox_metrics(timings: list[dict]) -> dict:
    """Request latency and GPU retries, counted independently from agent rounds."""
    payload = {}
    executions = {
        "cpu": [timing["cpu_execution"] for timing in timings if timing.get("cpu_execution")],
        "gpu": [record for timing in timings for record in timing.get("gpu_executions") or []],
    }
    for device, records in executions.items():
        stages = ("total", "exec") if device == "cpu" else ("total", "queue", "precheck", "exec")
        for stage in stages:
            values = [
                record[f"{stage}_ms"] / 1000
                for record in records
                if type(record.get(f"{stage}_ms")) in (int, float)
                and math.isfinite(record[f"{stage}_ms"])
                and record[f"{stage}_ms"] >= 0
            ]
            prefix = f"sandbox/{device}_{stage}"
            payload |= _brief_time_stats(values, prefix)
            if values:
                payload[f"{prefix}_max_s"] = round(max(values), 4)
            if device == "gpu" and stage == "queue" and values:
                payload[f"{prefix}_p99_s"] = _pct(values, 99)
        if device == "gpu":
            attempts = [
                record["attempts"]
                for record in records
                if type(record.get("attempts")) is int and record["attempts"] >= 1
            ]
            payload["sandbox/gpu_retries_total"] = sum(value - 1 for value in attempts)
            if attempts:
                payload["sandbox/gpu_retries_per_command"] = _mean([float(value - 1) for value in attempts])
    return payload


def split_sandbox_display(payload: dict) -> tuple[dict, dict]:
    """Keep mean/p95/max latencies in history and snapshot diagnostics in the run summary."""
    history, summary = {}, {}
    for key, value in payload.items():
        metric = key.removeprefix("interval/")
        summary_only = (
            metric.startswith("sandbox/workers_")
            or metric == "sandbox/gpu_commands_per_instance"
            or (
                metric.startswith("sandbox/")
                and metric.endswith("_s")
                and not metric.endswith(("_mean_s", "_p95_s", "_max_s"))
            )
        )
        (summary if summary_only else history)[key] = value
    return history, summary


def _with_defaults(defaults: dict, payload: dict) -> dict:
    """Start every step from zero defaults, then overlay the collected values."""
    defaults.update({key: 0.0 for key, value in payload.items() if type(value) in (int, float)})
    return defaults | payload


def pool_usage_metrics(usage: dict, baseline: dict, prefix: str) -> dict:
    """Worker-time utilization and capacity-weighted Jain fairness, including retired workers."""
    workers = usage.get("workers")
    if not isinstance(workers, dict):
        return {}
    previous = baseline.get("workers") or {}
    measured = []
    for name in workers.keys() | previous.keys():
        current = workers.get(name) or {}
        before = previous.get(name) or {}
        values = [current.get(field, 0) - before.get(field, 0) for field in ("busy_ms", "capacity_ms")]
        if any(not math.isfinite(value) or value < 0 for value in values):
            return {}
        busy, capacity = values
        if capacity:
            measured.append((busy, capacity))
        elif busy:
            return {}
    capacity = sum(value[1] for value in measured)
    if not capacity:
        return {}
    busy = sum(value[0] for value in measured)
    payload = {f"{prefix}/gpu_pool_utilization": round(min(1.0, busy / capacity), 4)}
    if busy:
        denominator = capacity * sum(value[0] ** 2 / value[1] for value in measured)
        payload[f"{prefix}/gpu_pool_balance"] = round(min(1.0, busy**2 / denominator), 4)
    return payload


def _label(value: object) -> str:
    """Only bounded codes/class names become metric keys; messages/IDs never do."""
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value) else "unknown"


def _tool_error(output: dict) -> str:
    extra = output.get("extra") or {}
    if extra.get("error_code") or extra.get("exception_type") or output.get("exception_info"):
        return _label(extra.get("error_code") or extra.get("exception_type") or output["exception_info"])
    result = extract_tool_result(output)
    if result.get("error_code"):
        return {"COMPILATION_ERROR": "compilation_failed", "CORRECTNESS_ERROR": "correctness_failed"}.get(
            result["error_code"], _label(result["error_code"])
        )
    status = result.get("status")
    if status and status not in ("completed", "success", "ok", "passed"):
        return _label(status)
    if result.get("compiled") is False:
        return "compilation_failed"
    if result.get("correctness") is False:
        return "correctness_failed"
    if output.get("returncode") not in (None, 0):
        return "nonzero_exit"
    return "tool_error" if result.get("error_message") else ""


def _round_sample(message: dict, phase_timings: dict, outputs: list[dict], instance_id: str = "") -> dict:
    """Distill one agent round (a tool pair) into the scalar sample the collector aggregates."""
    usage = _usage(message)
    response = (message.get("extra") or {}).get("response") or {}
    finish = ((response.get("choices") or [{}])[0]).get("finish_reason")
    tools = phase_timings.get("tools") or []
    # A tool counts as "called" only when it actually ran: skipped placeholders start with "[skipped]".
    ran = [o for o in outputs if not (o.get("output") or "").lstrip().startswith("[skipped]")]
    return {
        "instance_id": instance_id,
        "inference_s": phase_timings.get("llm") or 0.0,
        "env_s": sum(t.get("total") or 0.0 for t in tools),
        "completion_tokens": usage.get("completion_tokens") or 0,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "cot_chars": len(message.get("reasoning_content") or ""),
        "content_chars": len(message.get("content") or ""),
        "tool_calls": len(ran),
        "tool_times": {tool["name"]: tool.get("total") or 0.0 for tool in tools if tool.get("name")},
        "sandbox_errors": sum(bool((output.get("exception_info") or "").strip()) for output in outputs),
        "sandbox_timings": [t["sandbox"] for t in tools if t.get("sandbox") and not t.get("skipped")],
        "tool_phases": [t for t in tools if not t.get("skipped")],
        "tool_errors": [(o.get("tool", "unknown"), error) for o in ran if (error := _tool_error(o))],
        "tools_skipped": len(outputs) - len(ran),
        "truncated": 1 if finish == "length" else 0,
    }


def _distributions(rounds: list[dict]) -> dict:
    """Fixed counts include zero; distributions require actual samples."""
    payload = {}
    errors = collections.Counter((name, error) for r in rounds for name, error in r.get("tool_errors", []))
    compilation = sum(n for (_, reason), n in errors.items() if reason == "compilation_failed")
    correctness = sum(n for (_, reason), n in errors.items() if reason == "correctness_failed")
    failed = sum(errors.values()) - compilation - correctness
    calls = sum(r["tool_calls"] for r in rounds)
    payload |= {
        "tools/calls_total": calls,
        "errors/tools-failed_total": failed,
        "errors/candidate-compilation_failed_total": compilation,
        "errors/candidate-correctness_failed_total": correctness,
        "tools/skipped_total": sum(r.get("tools_skipped", 0) for r in rounds),
        "errors/tools-failure_rate": round(failed / calls, 4) if calls else 0.0,
    }
    for (name, reason), n in errors.items():
        if reason in ("unknown", "tool_error", "UNKNOWN_ERROR", "PROFILE_ERROR"):
            continue
        category = "candidate" if reason in ("compilation_failed", "correctness_failed") else "tool"
        payload[f"errors/{category}-{_label(name)}-{reason}_total"] = n
    if not rounds:
        return payload
    col = lambda key: [r[key] for r in rounds]  # noqa: E731
    inf, env = col("inference_s"), col("env_s")
    round_total = [a + b for a, b in zip(inf, env)]
    payload |= {
        "tools/calls_per_round": _mean(col("tool_calls")),
        "errors/truncated_rate": round(sum(col("truncated")) / len(rounds), 4),
        "errors/sandbox_rate": round(sum(col("sandbox_errors")) / len(rounds), 4),
    }
    for name, values in (("llm", inf), ("env", env), ("total", round_total)):
        payload |= _time_stats(values, f"time/round/{name}")
    # token/round: each round is a unit; token/instance: each instance's summed tokens is a unit (aligned metrics).
    payload |= _token_metrics(rounds, "token/round")
    payload |= _token_metrics(_sum_by_instance(rounds), "token/instance")
    tool_times: dict[str, list[float]] = {}
    for row in rounds:
        for name, seconds in row["tool_times"].items():
            tool_times.setdefault(name, []).append(seconds)
    for name, values in tool_times.items():
        payload[f"tool/{name}_mean_s"] = _mean(values)
        payload[f"tool/{name}_max_s"] = round(max(values), 4)
    # 每实例工具调用量: sum this instance's per-round tool calls, then min/max/mean across instances.
    calls_by_instance: dict[str, int] = {}
    for r in rounds:
        iid = r.get("instance_id", "")
        calls_by_instance[iid] = calls_by_instance.get(iid, 0) + r["tool_calls"]
    per_instance_calls = list(calls_by_instance.values())
    payload |= {
        "tools/calls_per_instance_mean": _mean([float(v) for v in per_instance_calls]),
        "tools/calls_per_instance_min": min(per_instance_calls),
        "tools/calls_per_instance_max": max(per_instance_calls),
    }
    # 数据级别耗时: per-instance sums (llm / env / total), then min/max/mean/percentiles across instances.
    inst: dict[str, list[float]] = {}
    for r, tot in zip(rounds, round_total):
        acc = inst.setdefault(r.get("instance_id", ""), [0.0, 0.0, 0.0])
        acc[0] += r["inference_s"]
        acc[1] += r["env_s"]
        acc[2] += tot
    payload |= _time_stats([v[0] for v in inst.values()], "time/instance/llm")
    payload |= _time_stats([v[1] for v in inst.values()], "time/instance/env")
    payload |= _time_stats([v[2] for v in inst.values()], "time/instance/total")
    return payload


def aggregate(
    rounds: list[dict],
    *,
    instances_started: int = 0,
    instances_done: int = 0,
    instances_solved: int = 0,
    other_errors: int = 0,
    format_errors: int = 0,
    format_error_seconds: float = 0.0,
    rounds_last_interval: int = 0,
    window_rounds: list[dict] | None = None,
) -> dict:
    """Fold the accumulated round samples into a flat, wandb-loggable metrics dict.

    The ``time/`` ``token/round/`` ``token/instance/`` ``tools/`` distributions are cumulative
    over the whole run. The same distributions over the rounds in the rolling window are emitted
    under the existing ``interval/`` prefix. New tool-stage details are cumulative only.
    """
    payload = {
        "progress/rounds_total": len(rounds),
        "progress/rounds_last_interval": rounds_last_interval,
        "progress/instances_started": instances_started,
        "progress/instances_done": instances_done,
        "progress/instances_solved": instances_solved,
        "progress/solved_rate": round(instances_solved / instances_done, 4) if instances_done else 0.0,
        "errors/other_total": other_errors,
        "errors/sandbox_total": sum(row["sandbox_errors"] for row in rounds),
        # A format error is raised while parsing the reply, so the round never completes and produces no
        # round sample: without these two the wasted (reasoning-heavy) model call is invisible here.
        "errors/format_total": format_errors,
        "errors/format_seconds_total": round(format_error_seconds, 3),
        "errors/truncated_total": sum(r["truncated"] for r in rounds),
    }
    payload |= _distributions(rounds) | _tool_time_metrics(rounds)
    if window_rounds:
        for key, value in _distributions(window_rounds).items():
            if key.startswith("errors/") and key not in ("errors/sandbox_rate", "errors/truncated_rate"):
                continue
            payload[f"interval/{key}"] = value
    return payload


_LOOP_OBS_HEADER = re.compile(r"^### Round \d+/\d+: (\w+)")

_EVAL_DECODER = json.JSONDecoder()


def instance_status_metrics(
    exit_statuses: dict,
    *,
    instances_planned: int = 0,
    instances_started: int = 0,
    instances_done: int = 0,
    instances_solved: int = 0,
    other_errors: int = 0,
) -> dict:
    """Instance-level progress + per-exit-status breakdown; emitted in BOTH collection modes.

    ``status/<exit_status>_total`` makes the timeout / empty-exit rate visible live (e.g. spotting
    "TimeoutError 80%" mid-run). Runner exit statuses are available in both collection modes.

    ``instances_planned`` is the fixed dataset size for this run (known up front), so
    ``progress/instances_total`` and ``progress/completion_rate`` (done / planned) are stable
    denominators, unlike the live ``instances_started`` / trajectory-file counts.
    """
    timeouts = exit_statuses.get("TimeoutError", 0)
    payload = {
        "progress/instances_total": instances_planned,
        "progress/completion_rate": round(instances_done / instances_planned, 4) if instances_planned else 0.0,
        "progress/instances_started": instances_started,
        "progress/instances_done": instances_done,
        "progress/instances_solved": instances_solved,
        "progress/solved_rate": round(instances_solved / instances_done, 4) if instances_done else 0.0,
        "errors/timeout_total": timeouts,
        "errors/other_total": other_errors,
        "errors/timeout_rate": round(timeouts / instances_done, 4) if instances_done else 0.0,
    }
    payload |= {f"status/{name or 'unknown'}_total": n for name, n in exit_statuses.items()}
    return payload


def submit_metrics(counts: list[int]) -> dict:
    """Submissions (= eval turns). `submit_eval/total` across all instances + per-instance count distribution."""
    payload = {"submit_eval/total": sum(counts)}
    if counts:
        vals = [float(c) for c in counts]
        payload |= {
            "submit_eval/per_instance_mean": _mean(vals),
            "submit_eval/per_instance_min": min(counts),
            "submit_eval/per_instance_max": max(counts),
            "submit_eval/per_instance_p50": _pct(vals, 50),
            "submit_eval/per_instance_p80": _pct(vals, 80),
            "submit_eval/per_instance_p90": _pct(vals, 90),
        }
    return payload


def _tt_heatmap_plotly(timings: dict, mode: str, budget_s: float):
    """2D-histogram heatmap of per-instance timing-vs-speedup (density) for wandb.

    ``mode="first"``: x = time to first correct, y = first correct speedup, **y capped at 3.0**.
    ``mode="best"``: x = time to best speedup, y = best speedup, **y capped at 10.0**. Both caps
    are FIXED per mode so runs compare on identical scales (suited to full ~200-instance runs
    where a scatter's points just pile up; density is the point). Both axes are binned (x over
    ``[0, budget_s]``, y over ``[0, cap]``); cell colour = number of instances landing there,
    hovering a cell lists them. Unsolved instances land in the far bottom-right cell (budget end
    × y=0) so they do not pollute the solved cluster. Title carries batch metrics. Styling mirrors
    :func:`_heatmap_plotly` (same colorscale/hover/scaleanchor conventions).
    """
    import wandb

    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    cap = 3.0 if mode == "first" else 10.0
    nx = ny = 10
    xstep = budget_s / nx
    ystep = cap / ny
    solved_sp: list[float] = []
    cells: dict[tuple[int, int], list[str]] = {}
    for name, t in timings.items():
        x = t.get("ttfc") if mode == "first" else t.get("ttb")
        y = t.get("first_sp") if mode == "first" else t.get("best_sp")
        if x is None or y is None:  # unsolved -> far bottom-right cell
            xi, yi = nx - 1, 0
        else:
            solved_sp.append(y)
            xi = min(int(x / xstep), nx - 1)
            yi = min(int(min(y, cap) / ystep), ny - 1)
        cells.setdefault((xi, yi), []).append(name.split("__")[0])
    if not cells:
        return None
    # Plotly Heatmap draws the FIRST z row at the top; reverse rows so y=0 (slow / unsolved)
    # sits at the bottom -> unsolved ends up bottom-right.
    mat = [[0] * nx for _ in range(ny)]
    hover = [["" for _ in range(nx)] for _ in range(ny)]
    for (xi, yi), names in cells.items():
        row = ny - 1 - yi
        mat[row][xi] = len(names)
        shown = names[:12]
        hover[row][xi] = (
            f"{len(names)} 题: " + "<br>".join(shown) + (f"<br>+{len(names) - 12} more" if len(names) > 12 else "")
        )
    xticks = [f"{i * xstep:.0f}" for i in range(nx)]
    yticks = [f"{(ny - 1 - i) * ystep:.2f}" for i in range(ny)]  # top = cap, bottom = 0
    ttimes = (
        [t.get("ttfc") for t in timings.values() if t.get("ttfc") is not None]
        if mode == "first"
        else [t.get("ttb") for t in timings.values() if t.get("ttb") is not None]
    )
    tmed = round(statistics.median(ttimes)) if ttimes else None
    smed = round(statistics.median(solved_sp), 2) if solved_sp else None
    total = len(timings)
    unsolved = sum(1 for t in timings.values() if not t.get("solved"))
    if mode == "first":
        title = (
            f"time-to-first-correct density — solved {total - unsolved}/{total}"
            + (f" · median {tmed}s · first-sp median {smed}x" if tmed is not None else "")
            + f" · unsolved {unsolved}"
        )
        xlab, ylab = "time to first correct (s)", f"first correct speedup (capped {cap:g})"
    else:
        title = (
            f"time-to-best density — solved {total - unsolved}/{total}"
            + (f" · median {tmed}s · best median {smed}x" if tmed is not None else "")
            + f" · unsolved {unsolved}"
        )
        xlab, ylab = "time to best speedup (s)", f"best speedup (capped {cap:g})"
    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=xticks,
            y=yticks,
            colorscale="Viridis",
            zmin=0,
            zauto=False,
            text=[[str(c) if c else "" for c in row] for row in mat],
            texttemplate="%{text}",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        xaxis=dict(tickangle=-45, showgrid=False, scaleanchor="y"),
        yaxis=dict(showgrid=False, scaleanchor="x"),
        margin=dict(l=80, r=40, t=60, b=80),
    )
    return wandb.Plotly(fig)


class _EvalTiming:
    """Per-eval (time, status, speedup) rows from the react db or the loop eval_turn files.

    The two run formats record timing differently, mirroring how the rest of the wandb metrics
    stay format-agnostic:
      * react / in-pod-opencode runs archive ``workdir_meta/opencode.db``: each ``part`` row for
        ``tool="eval"`` carries ``state.output`` with the wall-clock ``[timing] since_start_s=N``.
      * loop agents (loop / loopcuda / loopopencode) run no opencode: fall back to
        ``eval_turn_N.json`` + cumulative ``phase_timings`` (llm+tools per round) — the same
        time source the trajectory/round aggregation already uses for loop.
    """

    @staticmethod
    def _evalturn_rows(d: Path) -> list:
        """Loop fallback: (cumulative phase_timings, status, speedup) per eval_turn_N.json.

        Cumulative timing needs ROUND order, so files are sorted by their numeric turn (a plain
        string sort would put ``eval_turn_10.json`` before ``eval_turn_2.json`` and corrupt the
        running total).
        """
        rows: list[tuple[int, str, float]] = []
        acc = 0.0
        files = sorted(d.glob("eval_turn_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
        for p in files:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            pt = data.get("phase_timings") or {}
            acc += float(pt.get("llm") or 0.0)
            for t in pt.get("tools") or []:
                acc += float(t.get("total") or 0.0)
            status = data.get("status")
            if not status:
                continue
            sp = data.get("speedup")
            spf = float(sp) if isinstance(sp, (int, float)) and sp == sp else None
            rows.append((int(acc), status, spf))
        return rows


class EvalTurnScanner:
    """Incremental per-instance submit counts + best-speedup eval turn over an output dir.

    Only instance dirs whose signature changed re-read their ``eval_turn_N.json`` contents, so a
    10k-instance output dir costs O(dir listings) + O(changed eval_turn files) per flush instead
    of re-reading every ``eval_turn_N.json`` every time. The signature is ``(dir mtime_ns, file
    count)``: counting the eval_turn files per dir is cheap (directory listing) and catches new
    rounds even when the fs mtime granularity does not advance, while unchanged dirs never touch
    file contents. ``eval_turn_N.json`` are append-only per instance (one per eval round, written
    once); an instance that stops holding a ``trajectory.json`` drops its previous contribution.

    ``submit_counts()`` / ``best_turns()`` return the SAME values a full re-scan of the current
    output dir would (``count_submits`` / ``best_turn_metrics``).
    """

    def __init__(self) -> None:
        """Start with no seen instances and empty submit/best-turn state."""
        self._seen: dict[str, tuple[int, int, int, int]] = {}
        # instance -> (dir mtime_ns, eval_turn count, rec mtime_ns, rec size)
        self._submits: dict[str, int] = {}  # instance -> number of eval_turn files
        self._best_turns: dict[str, int] = {}  # instance -> first turn at its best speedup
        self._timings: dict[str, dict] = {}  # instance -> {ttfc, first_sp, ttb, best_sp, solved}

    def _records_sig(self, d: Path) -> tuple[int, int]:
        """Controller output has no external session database."""
        return (0, 0)

    def _changed(self, d: Path, name: str, count: int) -> bool:
        """Whether the instance dir signature (dir mtime_ns, eval_turn count, db sig) changed."""
        try:
            st = d.stat()
        except OSError:
            return False  # transient stat failure: leave as-is, retry next flush
        sig = (st.st_mtime_ns, count, *self._records_sig(d))
        if self._seen.get(name) != sig:
            self._seen[name] = sig
            return True
        return False

    @staticmethod
    def _records_timings(d: Path) -> dict:
        """Read timing from controller per-round eval receipts."""
        rows = _EvalTiming._evalturn_rows(d)
        if not rows:
            return {}
        completed = [(ss, sp) for ss, status, sp in rows if status == "completed" and sp is not None]
        first = completed[0] if completed else None
        best_sp = max((sp for _, sp in completed), default=None)
        ttb = None
        if best_sp is not None:
            ttb = next(ss for ss, sp in completed if sp == best_sp)
        return {
            "ttfc": first[0] if first else None,
            "first_sp": first[1] if first else None,
            "ttb": ttb,
            "best_sp": best_sp,
            "solved": first is not None,
        }

    def _extract(self, d: Path, files: list[Path]) -> None:
        """Re-scan one instance dir (new or changed): refresh submit count, best turn and timings."""
        self._submits[d.name] = len(files)
        speedups: list[tuple[int, float]] = []  # (eval turn number, speedup)
        for p in sorted(files):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            sp = data.get("speedup")
            if isinstance(sp, int | float) and sp == sp:  # not None / not nan
                turn = data.get("turn_id")
                if not isinstance(turn, int) or turn < 1:
                    turn = int(p.stem.rsplit("_", 1)[-1])  # fall back to filename N
                speedups.append((turn, sp))
        if not speedups:
            self._best_turns.pop(d.name, None)
        else:
            best = max(sp for _, sp in speedups)
            self._best_turns[d.name] = next(t for t, sp in speedups if sp == best)  # first turn at best
        # 无论是否解出都记 timings（unsolved = solved:False，散点按哨兵展示）；
        # db 缺失时 _records_timings 返回 {}，等同无数据。
        self._timings[d.name] = self._records_timings(d)

    def scan(self, root: Path) -> None:
        """Re-scan instance dirs whose signature changed; non-instance dirs are ignored."""
        if not root.is_dir():
            self._seen = {}
            self._submits = {}
            self._best_turns = {}
            self._timings = {}
            return
        for d in root.iterdir():
            if not d.is_dir():
                continue
            if not (d / "trajectory.json").exists():  # not (yet) an instance
                if d.name in self._seen:
                    self._seen.pop(d.name)
                    self._submits.pop(d.name, None)
                    self._best_turns.pop(d.name, None)
                    self._timings.pop(d.name, None)
                continue
            files = list(d.glob("eval_turn_*.json"))
            if self._changed(d, d.name, len(files)):
                self._extract(d, files)

    def submit_counts(self) -> list[int]:
        """Per-instance eval_turn file counts (feed ``submit_metrics``)."""
        return list(self._submits.values())

    def best_turns(self) -> list[int]:
        """Per-instance first eval turn at its best speedup (feed ``best_turn_metrics``)."""
        return list(self._best_turns.values())

    def timings(self) -> dict:
        """Per-instance correct/best landing times (feed the ttfc/ttb scatter charts)."""
        return dict(self._timings)


def fastn_distribution(speedups: list[float]) -> dict[str, int]:
    """``fast{threshold}`` counts: instances with speedup >= threshold, per threshold bucket.

    Thresholds run 1.0→2.0 (step 0.1) then 2.2→3.0 (step 0.2), plus ``fast3+`` (>=3.0) — the same
    distribution ``eval_result.json`` writes (kernelbench ``_speedup_distribution``), so the wandb
    dashboard shows the identical curve live. Pure counts (absolute), matching eval_result semantics.
    """
    thresholds = [round(1.0 + i * 0.1, 2) for i in range(11)] + [round(2.0 + i * 0.2, 2) for i in range(1, 6)]
    dist = {f"fast{threshold}": 0 for threshold in thresholds}
    dist["fast3+"] = 0
    for s in speedups:
        for threshold in thresholds:
            if s >= threshold:
                dist[f"fast{threshold}"] += 1
        if s >= 3.0:
            dist["fast3+"] += 1
    return dist


def score_metrics(preds: dict) -> dict:
    """Scoreboard from preds.json; missing measurements are not reported as zero.

    Funnel ``compile_rate >= correctness_rate >= solved_rate`` shows where kernels drop off; speedup
    is emitted for THREE baselines — ``score/speedup_*`` (eager PyTorch, weakest/largest), ``score/
    speedup_compiled_*`` (torch.compile ref) and ``score/speedup_primary_*`` (primary/strictest) —
    each with mean/median/max + ``fast_{1,2}{,_compiled,_primary}`` (fraction ≥1x / ≥2x over ALL
    instances) + the ``score/fastn_eager|_compiled|_primary`` distribution (counts ≥ each threshold,
    see ``fastn_distribution``; ``_flush`` renders each as a vertical bar chart); plus anti-cheat
    counts (decoy kernels / anti-hacking).
    """
    vals = [v for v in preds.values() if isinstance(v, dict)]
    n = len(vals)
    if not n:
        return {}
    rate = lambda pred: round(sum(1 for v in vals if pred(v)) / n, 4)  # noqa: E731
    payload = {
        "score/total": n,
        "score/solved": sum(1 for v in vals if v.get("solved")),
        "score/solved_rate": rate(lambda v: v.get("solved")),
        "score/correctness_rate": rate(lambda v: v.get("correctness")),
        "score/compile_rate": rate(lambda v: v.get("compilation")),
        "score/decoy_kernels": sum(1 for v in vals if v.get("is_decoy_kernel")),
        "score/anti_hacking_failed": sum(1 for v in vals if v.get("anti_hacking_passed") is False),
        "score/cheat_total": sum(1 for v in vals if v.get("is_decoy_kernel") or v.get("anti_hacking_passed") is False),
    }
    # Three speedup baselines side by side (eager PyTorch is the weakest/largest, primary the strictest).
    # ``score/speedup_*`` + ``score/fast_{1,2}`` stay on eager (names unchanged); the others get suffixes.
    # The fastn charts carry an explicit baseline suffix on all three (fastn_eager / fastn_compiled / fastn_primary).
    for field, sp_prefix, suffix, fastn_prefix in (
        ("eager_ref_speedup", "score/speedup", "", "score/fastn_eager"),
        ("compiled_ref_speedup", "score/speedup_compiled", "_compiled", "score/fastn_compiled"),
        ("speedup", "score/speedup_primary", "_primary", "score/fastn_primary"),
    ):
        payload[f"score/fast_1{suffix}"] = rate(lambda v, f=field: (v.get(f) or 0) >= 1)
        payload[f"score/fast_2{suffix}"] = rate(lambda v, f=field: (v.get(f) or 0) >= 2)
        sp = [v[field] for v in vals if isinstance(v.get(field), int | float)]
        if sp:
            payload |= {
                f"{sp_prefix}_mean": _mean(sp),
                f"{sp_prefix}_median": _median(sp),
                f"{sp_prefix}_max": round(max(sp), 4),
            }
            # raw distribution; never logged as-is — _flush pops it and renders a bar chart
            payload[f"{fastn_prefix}"] = fastn_distribution(sp)
    # Each task contributes its selected eval's measurement quality, not a pooled CV.
    stats = [v["speedup_stats"] for v in vals if isinstance(v.get("speedup_stats"), dict)]
    for field in ("num_pairs", "cv"):
        samples = [
            s[field] for s in stats if type(s.get(field)) in (int, float) and math.isfinite(s[field]) and s[field] >= 0
        ]
        prefix = f"score/speedup_{field}"
        if samples:
            payload |= {
                f"{prefix}_mean": _mean(samples),
                f"{prefix}_min": round(min(samples), 4),
                f"{prefix}_max": round(max(samples), 4),
            }
    return payload


def read_preds(output_dir: Path) -> dict:
    """Load ``<output_dir>/preds.json`` (best-effort; ``{}`` if missing/unreadable/caught mid-write)."""
    try:
        return json.loads((output_dir / "preds.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


class MetricsCollector:
    """Process-wide, thread-safe accumulator with a background wandb flusher.

    All ``record_*`` methods are no-ops until ``start()`` enables the collector, so the
    agents can call them unconditionally without paying for them on non-wandb runs.
    """

    def __init__(self) -> None:
        """Create a disabled collector (no-op until ``start()`` enables it)."""
        self._lock = threading.Lock()
        self._enabled = False
        self._run = None
        self._flusher: threading.Thread | None = None
        self._stop = threading.Event()
        self._interval = 120.0
        self._window = 3600.0
        self._step = 0
        self._trajectory_dir = ""
        self._output_dir = ""
        self._run_timeout = 0.0
        self._total_instances = 0
        self._reset()

    def _reset(self) -> None:
        """Clear all accumulated state back to a fresh, disabled collector."""
        self._rounds: list[dict] = []
        self._instances_started = 0
        self._instances_done = 0
        self._instances_solved = 0
        self._other_errors = 0
        self._format_errors = 0
        self._format_error_seconds = 0.0
        self._last_flush_round = 0
        self._last_turn_total = 0
        self._run_started = time.monotonic()
        self._instance_starts: dict[str, float] = {}
        self._instance_results: list[dict] = []
        self._sandbox_rounds: list[dict] = []
        self._gpu_pool: tuple[str, str] | None = None
        self._pool_snapshot: dict = {}
        self._pool_samples: collections.deque[tuple[float, dict]] = collections.deque()
        self._metric_defaults = {
            f"{prefix}/{metric}": 0.0
            for prefix in ("sandbox", "interval/sandbox")
            for metric in (
                "gpu_pool_utilization",
                "gpu_pool_balance",
                *(
                    f"{stage}_{stat}_s"
                    for stage in ("gpu_total", "gpu_queue", "gpu_precheck", "gpu_exec", "cpu_total", "cpu_exec")
                    for stat in ("mean", "p95", "max")
                ),
            )
        }
        self._metric_defaults.update(
            dict.fromkeys(
                (
                    "interval/progress/solved_rate",
                    "sandbox/gpu_retries_per_command",
                    "time/instance/wall_mean_s",
                    "time/instance/wall_p95_s",
                ),
                0.0,
            )
        )
        self._exit_statuses: collections.Counter = collections.Counter()
        self._eval_scanner: EvalTurnScanner | None = None
        self._eval_root: Path | None = None
        self._preds_sig: tuple[int, int] | None = None
        self._preds_score: dict = {}
        self._request_metrics: collections.Counter = collections.Counter(
            {
                "errors/pool_create_failed_total": 0,
                "errors/session_create_failed_total": 0,
                "errors/workspace_setup_failed_total": 0,
                "errors/llm-requests_failed_total": 0,
                "errors/llm-retries_total": 0,
                "errors/llm-empty-cot_retries_total": 0,
            }
        )
        self._round_mode = True

    @property
    def enabled(self) -> bool:
        """Whether metrics collection is currently enabled."""
        return self._enabled

    @property
    def trajectory_dir(self) -> str:
        """Directory holding per-instance trajectory files, if any."""
        return self._trajectory_dir

    def record_round(self, message: dict, phase_timings: dict, outputs: list[dict], instance_id: str = "") -> None:
        """Record a single agent round sample for later aggregation."""
        if not self._enabled:
            return
        sample = _round_sample(message, phase_timings, outputs, instance_id)
        with self._lock:
            sample["observed_at"] = time.monotonic()
            self._rounds.append(sample)

    def record_format_error(self, seconds: float = 0.0) -> None:
        """Record a model call rejected by the parser. Such a round raises before it can be sampled, so
        without this the wasted call (a full, reasoning-heavy LLM latency) never shows up in any metric."""
        if not self._enabled:
            return
        with self._lock:
            self._format_errors += 1
            self._format_error_seconds += seconds

    def record_sandbox_timings(self, timings: dict, instance_id: str = "") -> None:
        """Record GPU RPC timings outside in-process round samples."""
        if not self._enabled or not timings.get("gpu_executions"):
            return
        with self._lock:
            self._sandbox_rounds.append(
                {
                    "instance_id": instance_id,
                    "sandbox_timings": [timings],
                    "observed_at": time.monotonic(),
                }
            )

    def bind_gpu_pool(self, base_url: str, pool_id: str) -> None:
        """Attach the newly created pool without delaying collector startup or losing lifecycle errors."""
        if self._enabled:
            self._gpu_pool = (base_url.rstrip("/"), pool_id)

    def _poll_pool(self) -> None:
        if self._gpu_pool is None:
            return
        base_url, pool_id = self._gpu_pool
        response = requests.get(f"{base_url}/v1/gpu_pool/{pool_id}", timeout=10)
        response.raise_for_status()
        self._pool_snapshot = response.json()

    def record_llm_attempt(self, error: BaseException | None, *, retried: bool, final: bool) -> None:
        """One completed model-client attempt; SDK-internal retries are not visible here."""
        if not self._enabled:
            return
        with self._lock:
            self._request_metrics["errors/llm-retries_total"] += int(retried)
            self._request_metrics["errors/llm-requests_failed_total"] += int(final)
            if error is not None:
                response = getattr(error, "response", None)
                status = getattr(error, "status_code", None) or (response.status_code if response is not None else None)
                reason = f"HTTP{status}" if isinstance(status, int) else _label(type(error).__name__)
                self._request_metrics[f"errors/llm-{reason}_total"] += 1

    def record_empty_cot_retry(self) -> None:
        """One discarded model call whose reply carried no reasoning text (empty COT). The reply never
        reaches the agent, so its cost/latency would otherwise vanish from the round samples."""
        if not self._enabled:
            return
        with self._lock:
            self._request_metrics["errors/llm-empty-cot_retries_total"] += 1

    def record_error(self, stage: str, error: Exception) -> None:
        """Record a lifecycle failure at its existing exception handler."""
        if not self._enabled:
            return
        reason = type(error).__name__
        response = getattr(error, "response", None)
        if response is not None:
            reason = f"HTTP{response.status_code}"
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if isinstance(payload, dict) and payload.get("error_code"):
                reason = payload["error_code"]
        with self._lock:
            self._request_metrics[f"errors/{_label(stage)}_failed_total"] += 1
            if reason != "SandboxAPIError" and _label(reason) != "unknown":
                self._request_metrics[f"errors/{_label(stage)}-{_label(reason)}_total"] += 1

    def record_instance_start(self, instance_id: str = "") -> None:
        """Mark an instance as started."""
        if not self._enabled:
            return
        with self._lock:
            self._instances_started += 1
            self._instance_starts[instance_id] = time.monotonic()

    def record_instance_done(
        self,
        exit_status: str = "",
        solved: bool = False,
        instance_id: str = "",
        *,
        failed: bool = False,
        correctness: bool = False,
    ) -> None:
        """Record a finished instance's correctness, solved state, and exit status."""
        if not self._enabled:
            return
        with self._lock:
            self._instances_done += 1
            self._instances_solved += int(bool(solved))
            self._other_errors += int(failed)
            self._exit_statuses[exit_status or "unknown"] += 1
            started = self._instance_starts.pop(instance_id, None)
            now = time.monotonic()
            self._instance_results.append(
                {
                    "instance_id": instance_id,
                    "solved": bool(solved),
                    "observed_at": now,
                    "correctness": bool(correctness),
                    "wall_s": now - started if started is not None else None,
                }
            )

    def record_instance_error(self, exit_status: str = "", instance_id: str = "") -> None:
        """Record an errored instance sample (exit status) for aggregation."""
        self.record_instance_done(exit_status, instance_id=instance_id, failed=True)

    def start(
        self,
        *,
        project: str,
        name: str = "",
        entity: str = "",
        interval: float = 120.0,
        window: float = 3600.0,
        mode: str = "online",
        config: dict | None = None,
        trajectory_dir: str = "",
        output_dir: str = "",
        run_timeout: float = 0.0,
        total_instances: int = 0,
        api_base: str = "",
        trajectory_format: str = "",
        round_mode: bool = False,
    ) -> None:
        """Start optional W&B reporting from in-process controller rounds."""
        try:
            import wandb
        except ImportError as e:
            raise RuntimeError("wandb is not installed. Install it with `pip install wandb` to use --wandb.") from e
        self._reset()
        self._interval = max(1.0, interval)
        self._window = window
        self._step = 0
        self._trajectory_dir = trajectory_dir
        self._output_dir = output_dir
        self._format = "controller"
        self._round_mode = True
        if round_mode:
            rounds = [_round_sample({}, {}, [])]
            self._metric_defaults.update(dict.fromkeys(aggregate(rounds, window_rounds=rounds), 0.0))
        self._run_timeout = run_timeout
        self._total_instances = total_instances
        self._stop.clear()
        self._run = wandb.init(
            project=project,
            name=name or None,
            entity=entity or None,
            config=(config or {}) | {"wandb_interval": self._interval, "wandb_window": self._window},
            mode=mode,
            reinit=True,
        )
        self._enabled = True
        self._flush()
        self._flusher = threading.Thread(target=self._loop, name="wandb-metrics", daemon=True)
        self._flusher.start()

    def _loop(self) -> None:
        """Read integrated pool counters at the reporting cadence."""
        while not self._stop.wait(self._interval):
            self._flush()

    def _pool_window_metrics(self, usage: dict, now: float) -> dict:
        """Subtract the latest counter snapshot at or before the rolling boundary."""
        if not isinstance(usage.get("workers"), dict):
            return {}
        if self._pool_samples:
            previous = self._pool_samples[-1][1].get("workers", {})
            if any(
                usage["workers"].get(name, {}).get(field, 0) < worker.get(field, 0)
                for name, worker in previous.items()
                for field in ("busy_ms", "capacity_ms")
            ):
                self._pool_samples.clear()
                self._pool_samples.append((now, usage))
                return {}
        else:
            self._pool_samples.append((self._run_started, {}))
        self._pool_samples.append((now, usage))
        cutoff = max(self._run_started, now - self._window)
        while len(self._pool_samples) > 1 and self._pool_samples[1][0] <= cutoff:
            self._pool_samples.popleft()
        return pool_usage_metrics(usage, self._pool_samples[0][1], "interval/sandbox")

    def _sandbox_snapshot(self, now: float) -> dict:
        with self._lock:
            rounds = list(self._rounds) + list(self._sandbox_rounds)
            results = list(self._instance_results)
        cutoff = now - self._window
        window = [row for row in rounds if cutoff < row["observed_at"] <= now]
        finished = [row for row in results if cutoff < row["observed_at"] <= now]
        prefix = "sandbox"
        interval_prefix = "interval/sandbox"
        elapsed_s = now - self._run_started
        window_s = min(self._window, elapsed_s)
        commands = [
            (row.get("instance_id", ""), execution)
            for row in rounds
            for timing in row.get("sandbox_timings", [])
            for execution in timing.get("gpu_executions") or []
        ]
        window_commands = sum(
            len(timing.get("gpu_executions") or []) for row in window for timing in row.get("sandbox_timings", [])
        )
        payload = {
            f"{prefix}/gpu_commands_total": len(commands),
            f"{interval_prefix}/gpu_commands_total": window_commands,
            "progress/instances_done_per_hour": round(len(results) * 3600 / elapsed_s, 4) if elapsed_s else 0.0,
            "interval/progress/instances_done_per_hour": round(len(finished) * 3600 / window_s, 4) if window_s else 0.0,
            "progress/correctness_rate": _mean([result["correctness"] for result in results]),
            "interval/progress/correctness_rate": _mean([result["correctness"] for result in finished]),
        }
        counts = collections.Counter(instance_id for instance_id, _ in commands)
        if results:
            payload["sandbox/gpu_commands_per_instance"] = _mean(
                [float(counts[result["instance_id"]]) for result in results]
            )
        if finished:
            payload["interval/progress/solved_rate"] = round(
                sum(result["solved"] for result in finished) / len(finished), 4
            )
            payload["interval/sandbox/gpu_commands_per_instance"] = _mean(
                [float(counts[result["instance_id"]]) for result in finished]
            )
        payload |= _brief_time_stats(
            [result["wall_s"] for result in results if result["wall_s"] is not None],
            "time/instance/wall",
        )
        payload |= _sandbox_metrics([timing for row in rounds for timing in row.get("sandbox_timings", [])])
        payload |= {
            f"interval/{key}": value
            for key, value in _sandbox_metrics(
                [timing for row in window for timing in row.get("sandbox_timings", [])]
            ).items()
            if not key.startswith("sandbox/gpu_retries")
        }
        snapshot = self._pool_snapshot
        if snapshot:
            workers = snapshot.get("workers") or {}
            payload |= {f"{prefix}/workers_{state}": workers.get(state, 0) for state in ("idle", "busy", "pending")}
            payload[f"{prefix}/workers_total"] = sum(workers.get(state, 0) for state in ("idle", "busy", "pending"))
            usage = snapshot.get("usage") or {}
            payload |= pool_usage_metrics(usage, {}, prefix)
            payload |= self._pool_window_metrics(usage, now)
        return payload

    def _snapshot(self, now: float | None = None) -> dict:
        """Aggregate in-process controller rounds and their output receipts."""
        now = time.monotonic() if now is None else now
        with self._lock:
            rounds = list(self._rounds)
            window = [row for row in rounds if now - self._window < row["observed_at"] <= now]
            meta = {
                "instances_started": self._instances_started,
                "instances_done": self._instances_done,
                "instances_solved": self._instances_solved,
                "format_errors": self._format_errors,
                "format_error_seconds": self._format_error_seconds,
                "rounds_last_interval": len(rounds) - self._last_flush_round,
            }
            self._last_flush_round = len(rounds)
            status = self._instance_status_snapshot()
        return (
            aggregate(rounds, window_rounds=window, **meta)
            | status
            | self._output_snapshot()
            | self._sandbox_snapshot(now)
        )

    def _output_snapshot(self, live_best_turns: list[int] | None = None) -> dict:
        """submit_eval/ (eval_turn file counts) + score/ (preds.json scoreboard) + best_turn/
        histogram from the run's output dir; {} if unset. ``live_best_turns`` (opencode mode) is the
        trajectory-derived per-instance best-speedup eval round, available before instances complete."""
        if not self._output_dir:
            return {}
        root = Path(self._output_dir)
        if self._eval_scanner is None or self._eval_root != root:
            self._eval_scanner = EvalTurnScanner()
            self._eval_root = root
        self._eval_scanner.scan(root)  # incremental: only changed instance dirs re-read
        best_turns = live_best_turns if live_best_turns else self._eval_scanner.best_turns()
        return (
            submit_metrics(self._eval_scanner.submit_counts())
            | self._preds_snapshot(root)
            | ({"best_turn/best_speedup_turn": best_turns} if best_turns else {})
        )

    def _preds_snapshot(self, root: Path) -> dict:
        """score/ scoreboard from ``<root>/preds.json``, re-read only when the file changed.

        At 10k instances preds.json grows to tens of MB, so re-parsing it every flush would dominate
        the flush cost on a network FS; the signature (mtime_ns, size) makes it incremental.
        """
        p = root / "preds.json"
        try:
            sig = (p.stat().st_mtime_ns, p.stat().st_size)
        except OSError:
            self._preds_sig = None
            self._preds_score = {}
            return {}
        if sig != self._preds_sig:
            self._preds_sig = sig
            self._preds_score = score_metrics(read_preds(root))
        return self._preds_score

    def _instance_status_snapshot(self) -> dict:
        """Instance progress, exit statuses and in-flight ages. Caller must hold the lock."""
        ages = [time.monotonic() - started for started in self._instance_starts.values()]
        payload = dict(self._request_metrics) | instance_status_metrics(
            dict(self._exit_statuses),
            instances_planned=self._total_instances,
            instances_started=self._instances_started,
            instances_done=self._instances_done,
            instances_solved=self._instances_solved,
            other_errors=self._other_errors,
        )
        payload |= {
            "progress/inflight": len(ages),
            "time/overall/inflight_max_age_s": round(max(ages), 2) if ages else 0.0,
        }
        if self._run_timeout > 0:
            payload["progress/inflight_near_timeout"] = sum(age >= 0.8 * self._run_timeout for age in ages)
        return payload

    # Re-render tool-pair charts every N flushes (~6 min at the default 120s interval).
    # Rendering 3 heatmaps + sequence tables + best-turn bars costs ~1-2s and is CPU-bound;
    # ingest stays incremental (only changed files re-read) but the charts are redrawn on this
    # fixed cadence.
    PAIR_RENDER_INTERVAL = 3

    def _flush(self, *, force_render: bool = False) -> None:
        if self._run is None:
            return
        try:
            try:
                self._pool_snapshot = {}
                self._poll_pool()
            finally:
                self._log_snapshot(force_render=force_render)
        except Exception as e:
            logger.warning(f"wandb metrics flush failed (step {self._step}): {e}")

    def _log_snapshot(self, *, force_render: bool = False) -> None:
        """Compose the current snapshot, render charts (on cadence or when ``force_render``), and log it to wandb."""
        if self._run is None:
            return
        self._step += 1
        # Cadence: re-render charts only on every PAIR_RENDER_INTERVAL-th flush, but the final
        # stop() flush always renders so the run's last data point reaches wandb (otherwise a
        # run finishing off-cadence keeps a chart that is stale by up to 2 intervals).
        render = force_render or self._step % self.PAIR_RENDER_INTERVAL == 0
        payload, summary = split_sandbox_display(self._snapshot())
        payload = _with_defaults(self._metric_defaults, payload)
        if summary:
            self._run.summary.update(summary)
        hist = payload.pop("best_turn/best_speedup_turn", None)
        if hist and render:  # raw best-turn list -> vertical bar chart (wandb.Histogram renders horizontal)
            try:
                payload["best_turn/best_speedup_turn"] = self._render_best_turn_bars(hist, self._total_instances)
            except Exception as e:  # render failure must never kill the flusher
                logger.warning(f"best_turn histogram render failed (step {self._step}): {e}")
        # fastn distributions (score/fastn_{eager,compiled,primary}) -> vertical bar charts; the raw
        # dict is never logged as-is (wandb would not render it as a chart).
        for key in ("score/fastn_eager", "score/fastn_compiled", "score/fastn_primary"):
            dist = payload.pop(key, None)
            if dist and render:
                median = payload.get(
                    {
                        "score/fastn_eager": "score/speedup_median",
                        "score/fastn_compiled": "score/speedup_compiled_median",
                        "score/fastn_primary": "score/speedup_primary_median",
                    }[key]
                )
                try:
                    payload[key] = self._render_fastn_bars(dist, key, median=median)
                except Exception as e:
                    logger.warning(f"fastn bar chart render failed (step {self._step}): {e}")
        # off-cadence steps: do NOT log the raw list — wandb would plot it as a scalar
        # time series along the step axis (wall-time), which is what we saw on the panel.
        if render and self._eval_scanner is not None:
            try:
                payload |= self._timing_artifacts()
            except Exception as e:  # render failure must never kill the flusher
                logger.warning(f"traj-timing render failed (step {self._step}): {e}")
        self._run.log(payload, step=self._step)

    @staticmethod
    def _render_best_turn_bars(best_turns: list[int], total: int = 0):
        """Vertical bar chart of which eval turn holds each traj's best speedup.

        x = eval turn (1-based round), y = number of trajs whose best lands there. A traj
        contributes exactly one bar at the FIRST turn that reached its best speedup. Rendered
        as Plotly (a matplotlib PNG uploaded as wandb.Image gets a wall-time axis on the panel;
        Plotly charts do not). The x-axis title carries the progress ``n/total`` (instances with a
        best turn so far / planned) so the chart reads as live progress AND the final completion
        count even when the panel edge clips the top chart title.
        """
        import wandb

        try:
            import plotly.graph_objects as go
        except ImportError:
            return best_turns
        from collections import Counter

        counts = Counter(best_turns)
        turns = sorted(counts)
        fig = go.Figure(
            data=go.Bar(
                x=[str(t) for t in turns],
                y=[counts[t] for t in turns],
                text=[counts[t] for t in turns],
                textposition="outside",
                marker_color="#4C72B0",
                hovertemplate="turn %{x}: %{y} trajs<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title=f"eval turn of best speedup ({len(best_turns)}/{total})"
            if total
            else "eval turn of best speedup",
            yaxis_title="trajectories",
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
            margin=dict(l=80, r=40, t=60, b=60),
        )
        return wandb.Plotly(fig)

    @staticmethod
    def _render_fastn_bars(dist: dict[str, int], name: str, median: float | None = None):
        """Interactive bar chart of a fastn distribution (x = speedup threshold, y = #instances ≥ it).

        One bar per threshold bucket, in distribution order (1.0→2.0 step 0.1, 2.2→3.0 step 0.2,
        then ``3+``). Plotly so the panel gets a category x-axis; the flat ``fast{threshold}``
        scalars would otherwise each plot as their own line along the step axis. When ``median``
        (the baseline's median speedup, already logged as ``score/speedup*_median``) is given it
        rides on the x-axis title so the chart always shows the center of the distribution.
        """
        import wandb

        try:
            import plotly.graph_objects as go
        except ImportError:
            return dist
        labels = [k.removeprefix("fast") for k in dist]
        hover = [f"speedup >= {label}: {cnt} instances" for label, cnt in zip(labels, dist.values())]
        fig = go.Figure(
            data=go.Bar(
                x=labels,
                y=list(dist.values()),
                text=list(dist.values()),
                textposition="outside",
                marker_color="#4C72B0",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            )
        )
        fig.update_layout(
            title=name,
            xaxis_title=f"speedup threshold · median {median:g}x" if median is not None else "speedup threshold",
            yaxis_title="instances",
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
            margin=dict(l=80, r=40, t=60, b=60),
        )
        return wandb.Plotly(fig)

    def _timing_artifacts(self) -> dict:
        """Render the two per-instance timing scatter charts (first-correct / best).

        Fed by ``EvalTurnScanner.timings()`` — the SAME instance-dir incremental scan that already
        feeds best_turn (reuses its ``workdir_meta/oc_eval_records.jsonl`` parse, no extra walk).
        Rendered on the same cadence as the tool-pair heatmaps. Returns wandb keys
        ``traj_timing/first_correct`` / ``traj_timing/best`` (``{}`` when no instance data yet).
        """
        timings = self._eval_scanner.timings() if self._eval_scanner else {}
        if not timings:
            return {}
        budget = self._run_timeout or 5400.0
        arts: dict = {}
        for mode, key in (("first", "first_correct"), ("best", "best")):
            fig = _tt_heatmap_plotly(timings, mode, budget)
            if fig is not None:
                arts[f"traj_timing/{key}"] = fig
        return arts

    def stop(self) -> None:
        """Stop the flusher, log a final snapshot, and finish the wandb run."""
        if not self._enabled:
            return
        self._stop.set()
        if self._flusher is not None:
            self._flusher.join(timeout=self._interval + 5)
        self._flush(force_render=True)  # final snapshot after all workers finished, charts always fresh
        self._enabled = False
        run, self._run = self._run, None
        try:  # shutdown of an optional observability sink must never mask the run's real exit
            run.finish()
        except Exception as e:
            logger.warning(f"wandb run.finish() failed: {e}")


COLLECTOR = MetricsCollector()
