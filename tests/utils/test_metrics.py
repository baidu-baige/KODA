"""Tests for the live wandb metrics collector (aggregation only; no wandb needed)."""

import json
import time

import pytest

from minisweagent.agents.utils.timing import PhaseTimer, build_phase_timings

from minisweagent.utils.metrics import (EvalTurnScanner, MetricsCollector, _with_defaults, _mean, _median, _pct, _round_sample, _sandbox_metrics, aggregate, fastn_distribution, instance_status_metrics, pool_usage_metrics, read_preds, score_metrics, split_sandbox_display, submit_metrics)

def _proxy_turn(*, completion=0, reasoning=0, prompt=0, cot="", content="", tools=(), finish="stop", llm_duration_ms=None, request_started_at=None):
    """One llm-proxy schema-v2 turn: response.body.value is the OpenAI chat-completion JSON."""
    turn = {
        "response": {
            "body": {
                "format": "json",
                "value": {
                    "usage": {"completion_tokens": completion, "reasoning_tokens": reasoning, "prompt_tokens": prompt},
                    "choices": [{
                        "finish_reason": finish,
                        "message": {
                            "content": content, "reasoning_content": cot,
                            "tool_calls": [{"function": {"name": n}} for n in tools],
                        },
                    }],
                },
            }
        }
    }
    if llm_duration_ms is not None:
        turn["llm_duration_ms"] = llm_duration_ms
    if request_started_at is not None:
        turn["request_started_at"] = request_started_at
    return turn

def _write_trajectory(directory, instance_id, turns):
    (directory / f"{instance_id}.json").write_text(
        json.dumps({"schema_version": 2, "trajectory_id": instance_id, "turns": turns}), encoding="utf-8"
    )

def _message(*, completion=0, reasoning=0, cot=""):
    return {
        "content": "final answer",
        "reasoning_content": cot,
        "extra": {
            "response": {
                "usage": {
                    "completion_tokens": completion,
                    "completion_tokens_details": {"reasoning_tokens": reasoning},
                }
            }
        },
    }

def _phase(llm, tools):
    return {"llm": llm, "tools": [{"name": n, "total": t} for n, t in tools]}

def test_round_sample_extracts_tokens_tools_and_errors():
    sample = _round_sample(
        _message(completion=1000, reasoning=400, cot="thinking..."),
        _phase(2.5, [("eval", 10.0), ("profile", 4.0)]),
        [
            {"tool": "eval", "output": '{"status": "completed"}', "exception_info": ""},
            {"tool": "profile", "output": "[skipped] profile: eval not passed", "exception_info": ""},
        ],
    )
    assert sample["inference_s"] == 2.5
    assert sample["env_s"] == 14.0
    assert sample["completion_tokens"] == 1000
    assert sample["reasoning_tokens"] == 400
    assert sample["cot_chars"] == len("thinking...")
    # profile is a "[skipped]" placeholder -> only eval counts as a real tool call
    assert sample["tool_calls"] == 1
    assert [t["total"] for t in sample["tool_phases"]] == [10.0, 4.0]
    assert sample["tool_errors"] == []

def test_round_sample_records_tool_exception():
    sample = _round_sample(
        _message(),
        _phase(1.0, [("eval", 0.0)]),
        [{"tool": "eval", "output": "", "exception_info": "ConnectionError"}],
    )
    assert sample["tool_errors"] == [("eval", "ConnectionError")]
    assert sample["tool_calls"] == 1

def test_aggregate_empty_reports_zero_counts_without_inventing_measurements():
    payload = aggregate([], instances_started=3, window_rounds=[])
    assert payload["progress/instances_started"] == 3
    assert payload["errors/sandbox_total"] == payload["errors/other_total"] == 0
    assert payload["tools/calls_total"] == 0
    assert not any(key.startswith(("time/", "interval/", "sandbox/")) for key in payload)

def test_aggregate_surfaces_format_errors_that_produced_no_round():
    # a rejected reply raises before the round is sampled, so its wasted model call would otherwise be
    # invisible: a run that burns half its wall clock on re-prompts must not look identical to a clean one.
    payload = aggregate([], format_errors=4, format_error_seconds=123.4567)
    assert (payload["errors/format_total"], payload["errors/format_seconds_total"]) == (4, 123.457)
    assert aggregate([])["errors/format_total"] == 0

def test_aggregate_folds_rounds_into_means_ratios_and_per_tool():
    rounds = [
        _round_sample(
            _message(completion=1000, reasoning=500),
            _phase(2.0, [("eval", 10.0), ("profile", 6.0)]),
            [
                {"tool": "eval", "output": "{}", "exception_info": ""},
                {"tool": "profile", "output": "{}", "exception_info": "sandbox blew up"},
            ],
        ),
        _round_sample(
            _message(completion=3000, reasoning=500),
            _phase(4.0, [("eval", 20.0), ("profile", 2.0)]),
            [
                {"tool": "eval", "output": "{}", "exception_info": ""},
                {"tool": "profile", "output": "[skipped]", "exception_info": ""},
            ],
        ),
    ]
    payload = aggregate(rounds, instances_started=2, instances_done=1, instances_solved=1, other_errors=1)
    assert payload["time/round/llm_mean_s"] == 3.0
    assert payload["time/round/llm_max_s"] == 4.0
    assert payload["time/round/env_mean_s"] == 19.0  # (16 + 22) / 2
    assert payload["time/round/total_mean_s"] == 22.0
    # token/round: reasoning_ratio is token-weighted: (500+500) / (1000+3000)
    assert payload["token/round/reasoning_ratio"] == 0.25
    assert payload["token/round/completion_tokens_mean"] == 2000.0
    assert payload["token/round/completion_tokens_max"] == 3000  # spot the biggest-output round
    assert payload["token/round/reasoning_tokens_max"] == 500
    # token/instance: both rounds share instance "" -> summed completion = 1000+3000 = 4000
    assert payload["token/instance/completion_tokens_max"] == 4000
    # round 1 ran both tools; round 2 skipped profile -> 3 real calls total
    assert payload["tools/calls_total"] == 3
    assert payload["errors/sandbox_total"] == 1
    assert payload["errors/sandbox_rate"] == 0.5
    assert payload["errors/other_total"] == 1
    assert payload["progress/instances_solved"] == 1
    assert payload["progress/solved_rate"] == 1.0  # 1 solved / 1 done
    assert payload["tool/eval_mean_s"] == 15.0
    assert payload["tool/eval_max_s"] == 20.0
    assert payload["tool/profile_mean_s"] == 4.0

def test_aggregate_interval_metrics_scope_to_recent_window():
    old = _round_sample(
        _message(completion=100), _phase(1.0, [("eval", 1.0)]),
        [{"tool": "eval", "output": "{}", "exception_info": ""}],
    )
    recent = _round_sample(
        _message(completion=100), _phase(9.0, [("eval", 5.0)]),
        [{"tool": "eval", "output": "{}", "exception_info": "sandbox blew up"}],
    )
    payload = aggregate([old, recent], window_rounds=[recent], rounds_last_interval=1)
    # cumulative mean blends both rounds; the interval view sees only the recent (slow) round,
    # so a late-run regression is visible instead of being diluted by the running mean.
    assert payload["time/round/llm_mean_s"] == 5.0
    assert payload["interval/time/round/llm_mean_s"] == 9.0
    assert payload["interval/tools/calls_total"] == 1
    assert payload["interval/errors/sandbox_rate"] == 1.0
    assert payload["errors/sandbox_total"] == 1  # cumulative, spans both rounds
    assert payload["progress/rounds_last_interval"] == 1

def test_aggregate_without_window_emits_no_interval_keys():
    payload = aggregate(
        [_round_sample(_message(completion=10), _phase(1.0, [("eval", 1.0)]),
                       [{"tool": "eval", "output": "{}", "exception_info": ""}])]
    )
    assert not any(k.startswith("interval/") for k in payload)

def test_sandbox_timings_count_requests_preserve_zero_and_omit_missing_fields():
    timings = [{"cpu_execution": {"total_ms": 9000, "exec_ms": 8500}, "gpu_executions": [
        {"total_ms": 4000, "queue_ms": 0, "precheck_ms": 100, "exec_ms": 3000},
        {"total_ms": 6000, "queue_ms": 200, "precheck_ms": None, "exec_ms": 4000},
    ]}, {"gpu_executions": [{"total_ms": 2000, "exec_ms": 1000}]}]
    metrics = _sandbox_metrics(timings)
    assert metrics["sandbox/cpu_total_mean_s"] == 9
    assert metrics["sandbox/cpu_exec_mean_s"] == 8.5
    assert metrics["sandbox/gpu_total_mean_s"] == 4
    assert metrics["sandbox/gpu_exec_mean_s"] == pytest.approx(2.6667)
    assert metrics["sandbox/gpu_queue_mean_s"] == 0.1
    assert metrics["sandbox/gpu_queue_p99_s"] == 0.198
    assert metrics["sandbox/gpu_precheck_mean_s"] == 0.1
    assert {key: value for key, value in metrics.items() if key.endswith("_max_s")} == {
        "sandbox/cpu_total_max_s": 9, "sandbox/cpu_exec_max_s": 8.5,
        "sandbox/gpu_total_max_s": 6, "sandbox/gpu_queue_max_s": 0.2,
        "sandbox/gpu_precheck_max_s": 0.1, "sandbox/gpu_exec_max_s": 4,
    }
    assert not any("cpu_queue" in key or "cpu_precheck" in key for key in metrics)
    assert "sandbox/cpu_total_mean_s" not in _sandbox_metrics(timings[1:])
    rounds = [_round_sample(_message(), {"llm": 1, "tools": [
        {"name": "eval", "total": total, "sandbox": timing}
    ]}, []) for total, timing in zip((10, 3, 20), [*timings, None])]
    payload = aggregate(rounds)
    assert payload["time/round/env_mean_s"] == 11
    assert payload["time/round/total_mean_s"] == 12
    assert not any("sandbox/" in key for key in payload)

@pytest.mark.parametrize(("tool",), [("compile",), ("eval",), ("profile",)])
def test_tool_stages_are_flat_and_not_repeated_by_grain_or_window(tool):
    rounds = [_round_sample({}, {"tools": [
        {"name": tool, "total": value, "stage_timings": {"correctness": value / 2}}
        for value in values]}, [], instance_id) for instance_id, values in (("a", [2, 4]), ("a", [8]), ("b", [10]))]
    rounds.append(_round_sample({}, {"tools": [
        {"name": tool, "stage_timings": {"correctness": float("nan"), "invalid": -1}},
        {"name": "skipped_tool", "skipped": True, "stage_timings": {"correctness": 999}},
    ]}, [], "missing"))
    metrics = aggregate(rounds, window_rounds=rounds[1:])
    assert metrics[f"time/{tool}_correctness_mean_s"] == 3
    assert metrics[f"time/{tool}_correctness_p95_s"] == 4.85
    assert {key for key in metrics if "correctness_" in key and key.startswith("time/")} == {
        f"time/{tool}_correctness_mean_s", f"time/{tool}_correctness_p95_s"}
    assert not any("/env/" in key or "skipped_tool_correctness" in key or "_invalid_" in key for key in metrics)
    assert not any(key.startswith("interval/time/") and "correctness" in key for key in metrics)

def test_sandbox_display_keeps_mean_p95_and_max_for_all_request_stages():
    metrics = {"sandbox/workers_total": 2, "sandbox/workers_busy": 1, "sandbox/workers_pending": 1,
               "time/eval_correctness_mean_s": 2, "time/profile_nsys_sample_p95_s": 5,
               "progress/instances_done_per_hour": 3, "sandbox/gpu_commands_total": 9}
    latencies = {"gpu_total": (4, 10, 12), "gpu_queue": (1, 3, 4), "gpu_precheck": (0.1, 0.2, 0.3),
                 "gpu_exec": (2, 8, 9), "cpu_total": (13, 20, 25), "cpu_exec": (12, 19, 24)}
    defaults = MetricsCollector()._metric_defaults
    for scope in ("sandbox", "interval/sandbox"):
        assert all(defaults[f"{scope}/{stage}_max_s"] == 0 for stage in latencies)
        metrics |= {f"{scope}/{stage}_{stat}_s": value for stage, values in latencies.items()
                    for stat, value in zip(("mean", "p95", "max"), values)}
        metrics |= {f"{scope}/gpu_commands_per_instance": 3, f"{scope}/gpu_queue_p99_s": 5}
    history, summary = split_sandbox_display(metrics)
    assert history | summary == metrics
    assert not history.keys() & summary.keys()
    assert history["time/eval_correctness_mean_s"] == 2
    assert history["time/profile_nsys_sample_p95_s"] == 5
    assert history["sandbox/gpu_commands_total"] == 9
    assert summary["sandbox/workers_busy"] == 1
    for scope in ("sandbox", "interval/sandbox"):
        assert {key for key in history if key.startswith(f"{scope}/") and key.endswith("_s")} == {
            f"{scope}/{stage}_{stat}_s" for stage in latencies for stat in ("mean", "p95", "max")}
        assert history[f"{scope}/gpu_precheck_mean_s"] == 0.1
        assert history[f"{scope}/gpu_precheck_p95_s"] == 0.2
        assert history[f"{scope}/cpu_exec_p95_s"] == 19
        assert history[f"{scope}/gpu_precheck_max_s"] == 0.3
        assert history[f"{scope}/cpu_exec_max_s"] == 24
        assert summary[f"{scope}/gpu_queue_p99_s"] == 5
        assert summary[f"{scope}/gpu_commands_per_instance"] == 3
        assert history[f"{scope}/gpu_total_mean_s"] == 4

def test_profile_stage_timings_reach_history_alongside_eval_timings():
    timer = PhaseTimer()
    timer.timings = {"llm": 1, "eval": 3, "profile": 7}
    outputs = [{"tool": tool, "returncode": 0, "output": "tool log\n" + json.dumps({
        "status": "completed", "stage_timings": timings,
    })} for tool, timings in (("eval", {"correctness": 2}), ("profile", {"nsys_sample": 4, "ncu_details": 2}))]
    phase = build_phase_timings(timer, [(output["tool"], output) for output in outputs])
    history, summary = split_sandbox_display(aggregate([_round_sample(_message(), phase, outputs)]))
    assert history["time/eval_correctness_mean_s"] == 2
    assert history["time/profile_nsys_sample_mean_s"] == 4
    assert history["time/profile_ncu_details_p95_s"] == 2
    assert history["tool/profile_mean_s"] == 7
    assert history["time/round/env_mean_s"] == 10
    assert not summary

def test_gpu_server_retries_do_not_infer_final_failures():
    metrics = _sandbox_metrics([{"gpu_executions": [{"attempts": 3}, {"attempts": 1}, {"attempts": 0}]}])
    assert metrics["sandbox/gpu_retries_total"] == 2
    assert metrics["sandbox/gpu_retries_per_command"] == 1
    assert not any("failed" in key for key in metrics)

def test_aggregate_round_total_and_per_instance_total_min_max_mean():
    rounds = [
        _round_sample(_message(), _phase(2.0, [("eval", 10.0)]), [{"tool": "eval", "output": "{}"}], "inst_a"),
        _round_sample(_message(), _phase(4.0, [("eval", 6.0)]), [{"tool": "eval", "output": "{}"}], "inst_a"),
        _round_sample(_message(), _phase(1.0, [("eval", 1.0)]), [{"tool": "eval", "output": "{}"}], "inst_b"),
    ]
    payload = aggregate(rounds)
    # per-round totals = [12, 10, 2]
    assert payload["time/round/total_mean_s"] == 8.0
    assert payload["time/round/total_min_s"] == 2.0
    assert payload["time/round/total_max_s"] == 12.0
    # per-instance totals: inst_a = 12 + 10 = 22, inst_b = 2
    assert payload["time/instance/total_mean_s"] == 12.0
    assert payload["time/instance/total_min_s"] == 2.0
    assert payload["time/instance/total_max_s"] == 22.0
    # per-instance tool calls: inst_a ran eval twice (2), inst_b once (1) -> min 1, max 2, mean 1.5
    assert payload["tools/calls_per_instance_mean"] == 1.5
    assert payload["tools/calls_per_instance_min"] == 1
    assert payload["tools/calls_per_instance_max"] == 2

def test_time_overviews_preserve_merged_names_without_duplicate_overall_views():
    rounds = [_round_sample({}, _phase(llm, [("eval", 6)]), [], instance_id)
              for instance_id, llm in (("a", 2), ("a", 4), ("b", 8), ("c", 16))]
    metrics = aggregate(rounds, window_rounds=rounds[1:3])
    expected = {f"{name}_{stat}_s" for name in ("llm", "env", "total")
                for stat in ("mean", "min", "max", "p50", "p80", "p90", "p95", "p99")}
    for scope in ("time", "interval/time"):
        for unit in ("round", "instance"):
            prefix = f"{scope}/{unit}/"
            assert {key.removeprefix(prefix) for key in metrics if key.startswith(prefix)} == expected
        assert not any(key.startswith(f"{scope}/overall/") for key in metrics)
    assert metrics["time/round/llm_mean_s"] == 7.5
    assert metrics["time/instance/llm_mean_s"] == 10
    assert metrics["interval/time/round/llm_mean_s"] == metrics["interval/time/instance/llm_mean_s"] == 6
    assert metrics["tool/eval_mean_s"] == metrics["interval/tool/eval_mean_s"] == 6

def _protocol_turn(path, response):
    return {
        "request": {"path": path},
        "response": {"body": {"format": "json", "value": response}},
        "llm_duration_ms": 2000,
        "request_started_at": 100.0,
    }

def test_instance_status_metrics_breaks_down_exit_status_and_timeout_rate():
    payload = instance_status_metrics(
        {"completed": 30, "TimeoutError": 60, "OPENCODE_EMPTY_EXIT": 10},
        instances_planned=200, instances_started=100, instances_done=100, instances_solved=80,
    )
    assert payload["progress/instances_total"] == 200  # fixed dataset size, not the live started/done count
    assert payload["progress/completion_rate"] == 0.5   # 100 done / 200 planned
    assert payload["progress/solved_rate"] == 0.8
    assert payload["errors/timeout_total"] == 60
    assert payload["errors/timeout_rate"] == 0.6  # 60 timeouts / 100 done
    assert payload["status/completed_total"] == 30
    assert payload["status/TimeoutError_total"] == 60
    assert payload["status/OPENCODE_EMPTY_EXIT_total"] == 10
    # planned unknown (0) -> completion_rate stays 0 instead of dividing by zero
    assert instance_status_metrics({}, instances_done=5)["progress/completion_rate"] == 0.0

def test_pct_linear_interpolation_and_edges():
    assert _pct([10, 20, 30, 40], 50) == 25.0   # (20 + 30) / 2
    assert _pct([10, 20, 30, 40], 90) == 37.0    # 30 + (40-30)*0.7
    assert _pct([10, 20, 30, 40], 95) == 38.5
    assert _pct([5], 95) == 5.0                  # single value
    assert _pct([], 50) == 0.0                   # empty

def test_submit_metrics_total_and_per_instance_distribution():
    payload = submit_metrics([8, 2, 5, 0])
    assert payload["submit_eval/total"] == 15
    assert payload["submit_eval/per_instance_mean"] == 3.75
    assert payload["submit_eval/per_instance_min"] == 0
    assert payload["submit_eval/per_instance_max"] == 8
    assert payload["submit_eval/per_instance_p50"] == _pct([0, 2, 5, 8], 50)
    assert payload["submit_eval/per_instance_p90"] == _pct([0, 2, 5, 8], 90)
    empty = submit_metrics([])
    assert empty == {"submit_eval/total": 0}  # no per-instance keys when there are no instances

def test_eval_turn_scanner_timings_extracts_first_and_best_landing(tmp_path):
    d=tmp_path/"task";d.mkdir()
    (d/"trajectory.json").write_text("{}")
    for i,(seconds,status,speedup) in enumerate([(2,"failed",None),(5,"completed",1.0),(3,"completed",2.0)],1):
        (d/f"eval_turn_{i}.json").write_text(json.dumps({"phase_timings":{"llm":seconds,"tools":[]},"status":status,"speedup":speedup,"turn_id":i}))
    scanner=EvalTurnScanner();scanner.scan(tmp_path)
    row=scanner.timings()["task"]
    assert row["ttfc"]==7 and row["ttb"]==10
    assert row["first_sp"]==1.0 and row["best_sp"]==2.0

def test_eval_turn_scanner_timings_loop_fallback_uses_cumulative_phase_timings(tmp_path):
    # loop mode: no opencode.db -> timing = cumulative phase_timings over eval_turn files,
    # and the files MUST be consumed in numeric round order (a string sort would put
    # eval_turn_10.json before eval_turn_2.json and corrupt the running total).
    d = tmp_path / "loop_inst"
    d.mkdir()
    (d / "trajectory.json").write_text("{}", encoding="utf-8")

    def eturn(n, llm, status, sp):
        (d / f"eval_turn_{n}.json").write_text(json.dumps({
            "turn_id": n, "status": status, "speedup": sp,
            "phase_timings": {"llm": llm, "tools": []}}), encoding="utf-8")

    eturn(10, 5, "correctness_failed", None)  # 5s; must sort AFTER turn 4 (numeric, not lexical)
    eturn(2, 40, "correctness_failed", None)
    eturn(3, 60, "completed", 1.5)            # first correct at cum 100
    eturn(4, 80, "completed", 2.5)            # best at cum 180
    sc = EvalTurnScanner()
    sc.scan(tmp_path)
    t = sc.timings()["loop_inst"]
    assert t["ttfc"] == 100 and t["first_sp"] == 1.5
    assert t["ttb"] == 180 and t["best_sp"] == 2.5 and t["solved"] is True

def _instance(tmp_path, name, speedups):
    """Create one instance dir with trajectory.json + eval_turn_N.json (N = 1-based round)."""
    d = tmp_path / name
    d.mkdir()
    (d / "trajectory.json").write_text("{}", encoding="utf-8")
    for i, sp in enumerate(speedups, start=1):
        (d / f"eval_turn_{i}.json").write_text(
            json.dumps({"turn_id": i, "speedup": sp}), encoding="utf-8")
    return d

def test_score_metrics_funnel_speedup_and_cheat():
    preds = {
        "a": {"solved": True, "correctness": True, "compilation": True,
              "eager_ref_speedup": 3.0, "compiled_ref_speedup": 2.5, "speedup": 2.0},
        "b": {"solved": False, "correctness": False, "compilation": True,
              "eager_ref_speedup": None, "compiled_ref_speedup": None, "speedup": None},
        "c": {"solved": True, "correctness": True, "compilation": True, "is_decoy_kernel": True,
              "eager_ref_speedup": 1.5, "compiled_ref_speedup": 1.2, "speedup": 0.9},
        "d": {"solved": False, "correctness": True, "compilation": True, "anti_hacking_passed": False,
              "eager_ref_speedup": 0.8, "compiled_ref_speedup": 0.7, "speedup": 0.5},
    }
    p = score_metrics(preds)
    assert p["score/total"] == 4
    assert p["score/solved"] == 2
    assert p["score/solved_rate"] == 0.5
    assert p["score/correctness_rate"] == 0.75          # a, c, d correct
    assert p["score/compile_rate"] == 1.0
    # eager baseline (names unchanged): a(3.0), c(1.5) >= 1x; d 0.8 <1; b None
    assert p["score/fast_1"] == 0.5
    assert p["score/fast_2"] == 0.25                     # only a >= 2x
    assert p["score/speedup_mean"] == _mean([3.0, 1.5, 0.8])
    assert p["score/speedup_max"] == 3.0
    # compiled-ref baseline: values [2.5, 1.2, 0.7]; a,c >=1x, a >=2x
    assert p["score/fast_1_compiled"] == 0.5
    assert p["score/fast_2_compiled"] == 0.25
    assert p["score/speedup_compiled_mean"] == _mean([2.5, 1.2, 0.7])
    assert p["score/speedup_compiled_max"] == 2.5
    # primary baseline (strictest): values [2.0, 0.9, 0.5]; only a >=1x and >=2x
    assert p["score/fast_1_primary"] == 0.25
    assert p["score/fast_2_primary"] == 0.25
    assert p["score/speedup_primary_mean"] == _mean([2.0, 0.9, 0.5])
    assert p["score/speedup_primary_median"] == _median([2.0, 0.9, 0.5])
    assert p["score/decoy_kernels"] == 1
    assert p["score/anti_hacking_failed"] == 1
    assert p["score/cheat_total"] == 2                   # c (decoy) + d (anti-hacking)

def test_score_metrics_fastn_curve_counts_above_each_threshold():
    """score/fastn_{eager,compiled,primary}: raw distributions, counts of instances >= each threshold."""
    preds = {f"i{i}": {"speedup": s, "eager_ref_speedup": s * 2, "compiled_ref_speedup": s / 2}
             for i, s in enumerate((0.5, 0.9, 1.0, 1.1, 2.5))}
    p = score_metrics(preds)
    d = p["score/fastn_primary"]
    # primary (speedup): 0.5,0.9,1.0,1.1,2.5 -> >=1.0: 3, >=1.1: 2, >=2.0: 1, >=3.0: 0
    assert d["fast1.0"] == 3
    assert d["fast1.1"] == 2
    assert d["fast2.0"] == 1
    assert d["fast3+"] == 0
    # eager: *2 -> 1.0,1.8,2.0,2.2,5.0 -> >=2.0: 3, >=3.0: 1
    assert p["score/fastn_eager"]["fast2.0"] == 3
    assert p["score/fastn_eager"]["fast3+"] == 1
    # compiled: /2 -> 0.25,0.45,0.5,0.55,1.25 -> >=1.0: 1
    assert p["score/fastn_compiled"]["fast1.0"] == 1
    # curve is monotone non-increasing in the threshold
    keys = [round(1.0 + i * 0.1, 2) for i in range(11)] + [round(2.0 + i * 0.2, 2) for i in range(1, 6)]
    counts = [d[f"fast{t}"] for t in keys]
    assert counts == sorted(counts, reverse=True)  # fast3+ bucket is >=3.0, after the 3.0 threshold
    assert d["fast3.0"] == 0 and d["fast3+"] == 0
    assert d == fastn_distribution([0.5, 0.9, 1.0, 1.1, 2.5])  # primary dist equals the direct distribution
    # the fastn keys are raw dicts (never scalar curves); _flush renders them as bar charts
    assert not any(len(k.split("/")) >= 3 and k.split("/")[1].startswith("fastn_") for k in p)

def test_fastn_distribution_matches_eval_result_semantics():
    d = fastn_distribution([1.2, 1.2, 2.4, 4.0, 0.8])
    assert d["fast1.0"] == 4
    assert d["fast1.2"] == 4  # >= 1.2: the two 1.2s count too
    assert d["fast2.0"] == 2
    assert d["fast2.4"] == 2  # >= 2.4: 2.4 and 4.0
    assert d["fast3.0"] == 1
    assert d["fast3+"] == 1
    assert len(d) == 17  # 11 (1.0..2.0) + 5 (2.2..3.0) + fast3+

def test_read_preds_best_effort(tmp_path):
    (tmp_path / "preds.json").write_text('{"x": {"solved": true}}', encoding="utf-8")
    assert read_preds(tmp_path) == {"x": {"solved": True}}
    assert read_preds(tmp_path / "nope") == {}  # missing dir -> {}

def _write_opencode_trajectory(instance_dir, msgs):
    """opencode-format trajectory: <instance>/trajectory.json with messages[]."""
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / "trajectory.json").write_text(
        json.dumps({"trajectory_format": "opencode", "messages": msgs}), encoding="utf-8"
    )

def _opencode_assistant(text, tools=()):
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [{"id": f"{t}:{i}", "function": {"name": t}} for i, t in enumerate(tools)],
    }

def _loop_assistant(text, reasoning, actions):
    """loop-format assistant message: COT in reasoning_content, tools under extra.actions."""
    return {"role": "assistant", "content": text, "reasoning_content": reasoning,
            "extra": {"actions": actions}}

def _write_loop_trajectory(instance_dir, messages):
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / "trajectory.json").write_text(
        json.dumps({"trajectory_format": "mini-swe-agent-1.1", "messages": messages}), encoding="utf-8")

def _loop_traj_with_eval(instance_dir, rounds):
    """Write a loop-format trajectory + matching eval_turn_N.json for (round, speedup) pairs."""
    instance_dir.mkdir(parents=True, exist_ok=True)
    msgs = []
    for rnd, sp in rounds:
        msgs.append(_loop_assistant(f"a{rnd}", f"r{rnd}", [{"command": "c", "kernel_py": "p"}]))
        msgs.append({"role": "user",
                     "content": f"### Round {rnd}/30: eval (returncode 0)\n<output>\n"
                                f'{{"compiled": true, "correctness": true, "speedup": {sp}}}\n</output>'})
        (instance_dir / f"eval_turn_{rnd}.json").write_text(
            json.dumps({"turn_id": rnd, "speedup": sp}), encoding="utf-8")
    (instance_dir / "trajectory.json").write_text(
        json.dumps({"trajectory_format": "mini-swe-agent-1.1", "messages": msgs}), encoding="utf-8")

def test_output_snapshot_prefers_live_best_turns(tmp_path):
    """best_turn/ uses the passed live list when given; else falls back to eval_turn files."""
    d = tmp_path / "out"
    _loop_traj_with_eval(d / "inst_a", [(1, 1.0), (2, 5.0)])
    c = MetricsCollector()
    c._output_dir = str(d)
    assert c._output_snapshot()["best_turn/best_speedup_turn"] == [2]  # eval_turn files
    assert c._output_snapshot(live_best_turns=[7])["best_turn/best_speedup_turn"] == [7]  # live wins
    assert c._output_snapshot(live_best_turns=[])["best_turn/best_speedup_turn"] == [2]  # empty -> fallback

def test_snapshot_prefers_in_process_rounds_over_opencode_scan(tmp_path):
    """A direct-connect loop run (round_mode, output_dir set) must aggregate the in-process rounds —
    which carry real token/time — instead of the output-dir scan, whose loop parser zeroes tokens and
    emits no time/*. Guards the regression where any non-proxy api_base silently dropped
    token/time/instance metrics for loop agents."""
    c = MetricsCollector()
    c._enabled = True
    c._round_mode = True
    c._format = "opencode"
    c._output_dir = str(tmp_path)  # set (as --output always is) but must NOT pre-empt round aggregation
    c.record_round(
        {"content": "answer", "reasoning_content": "cot",
         "extra": {"response": {"usage": {
             "completion_tokens": 100, "prompt_tokens": 2000,
             "completion_tokens_details": {"reasoning_tokens": 40}}}}},
        {"llm": 3.0, "tools": [{"name": "eval", "total": 7.0}]},
        [{"tool": "eval", "output": "ok"}],
        "inst_a",
    )
    payload = c._snapshot()
    assert payload["time/instance/llm_mean_s"] == 3.0
    assert payload["time/instance/env_mean_s"] == 7.0
    assert payload["time/instance/total_mean_s"] == 10.0
    assert payload["token/instance/completion_tokens_mean"] == 100
    assert payload["token/round/reasoning_tokens_mean"] == 40

def test_snapshot_round_mode_before_first_round_does_not_use_opencode_scan(tmp_path):
    """At loop-run start no round has landed yet; round_mode must still take the round path (empty
    aggregate), never the opencode scan — otherwise early flushes would be scored as react."""
    _write_loop_trajectory(tmp_path / "inst_a", [
        _loop_assistant("t1", "r1", [{"command": "c", "kernel_py": "p"}]),
    ])  # a trajectory file already exists on disk, but no round recorded in-process yet
    c = MetricsCollector()
    c._enabled = True
    c._round_mode = True
    c._format = "opencode"
    c._output_dir = str(tmp_path)
    payload = c._snapshot()
    # round path (even with empty rounds) emits progress/rounds_total and never runs the trajectory
    # scanner; the scan path would instead emit progress/turns_total.
    assert payload["progress/rounds_total"] == 0
    assert "progress/turns_total" not in payload
    assert not hasattr(c, "_traj_scanner")

def test_score_metrics_speedup_measurement_quality_skips_missing_and_invalid_values():
    payload = score_metrics({
        "a": {"speedup": 1.5, "speedup_stats": {"num_pairs": 10, "cv": 0.0}},
        "b": {"speedup": 2.5, "speedup_stats": {"num_pairs": 30, "cv": 0.06}},
        "one_pair": {"speedup_stats": {"num_pairs": 1, "cv": None}},
        "old": {}, "missing": {"speedup_stats": None},
        "bad": {"speedup_stats": {"num_pairs": True, "cv": float("nan")}},
        "negative": {"speedup_stats": {"num_pairs": -1, "cv": -1}},
    })
    assert payload["score/speedup_num_pairs_mean"] == 13.6667
    assert payload["score/speedup_num_pairs_min"] == 1
    assert payload["score/speedup_num_pairs_max"] == 30
    assert payload["score/speedup_cv_mean"] == 0.03
    assert payload["score/speedup_cv_min"] == 0
    assert payload["score/speedup_cv_max"] == 0.06
    assert payload["score/fastn_primary"]["fast1.0"] == 2
    assert payload["score/fastn_primary"]["fast2.0"] == 1
    old = score_metrics({"old": {}})
    assert "score/speedup_cv_mean" not in old and "score/speedup_num_pairs_mean" not in old
    assert score_metrics({}) == {}
    assert not any(k.endswith("_samples") for k in payload | old)

def test_phase_metrics_omit_setup_preserve_stages_and_separate_tool_failures():
    timer = PhaseTimer()
    timer.timings = {"llm": 2, "setup": 3, "eval": 5, "profile": 0}
    outputs = [{"tool": "eval", "returncode": 0, "output": json.dumps({
        "compiled": True, "correctness": False, "status": "completed",
        "stage_timings": {"load_reference": 0, "correctness": 2, "total": 4,
                          "invalid": "2", "negative": -1, "nan": float("nan")},
    })}, {"tool": "profile", "returncode": 0, "output": "[skipped] incorrect kernel"}]
    phase = build_phase_timings(timer, [(output["tool"], output) for output in outputs])
    assert "setup" not in phase
    metrics = aggregate([_round_sample(_message(), phase, outputs)])
    assert metrics["time/round/env_mean_s"] == metrics["tool/eval_mean_s"] == 5
    assert metrics["time/round/total_mean_s"] == 7
    assert metrics["time/eval_total_mean_s"] == 4
    assert metrics["time/eval_load_reference_mean_s"] == 0
    assert metrics["time/eval_correctness_p95_s"] == 2
    assert metrics["errors/candidate-correctness_failed_total"] == metrics["tools/skipped_total"] == 1
    assert metrics["errors/tools-failed_total"] == metrics["errors/sandbox_total"] == 0
    assert not any("/env/" in key or "_invalid_" in key or "_negative_" in key or "_nan_" in key for key in metrics)

def test_collector_preserves_request_errors_across_round_and_incremental_proxy_snapshots(tmp_path):
    collector = MetricsCollector()
    collector._enabled = True
    collector._round_mode = True
    collector.record_format_error(2.5)
    collector.record_llm_attempt(RuntimeError("request failed"), retried=True, final=True)
    collector.record_error("session_create", ValueError("session failed"))
    rounds = collector._snapshot()
    assert rounds["errors/format_total"] == 1
    assert rounds["errors/format_seconds_total"] == 2.5

    assert rounds["errors/llm-retries_total"] == 1
    assert rounds["errors/llm-requests_failed_total"] == 1
    assert rounds["errors/llm-RuntimeError_total"] == 1
    assert rounds["errors/session_create_failed_total"] == 1
    assert rounds["errors/session_create-ValueError_total"] == 1
    collector._reset()
    assert collector._snapshot()["errors/llm-retries_total"] == 0
    assert collector._snapshot()["errors/format_total"] == 0
    assert not hasattr(collector, "_traj_scanner")

@pytest.mark.parametrize(("busy", "utilization", "balance"), [
    ([30, 30, 30, 30], 0.3, 1.0),
    ([100, 100, 0, 0], 0.5, 0.5),
    ([0, 0, 0, 0], 0.0, None),
])
def test_pool_usage_distinguishes_balance_from_utilization(busy, utilization, balance):
    usage = {"workers": {str(index): {"busy_ms": value, "capacity_ms": 100}
                         for index, value in enumerate(busy)}}
    metrics = pool_usage_metrics(usage, {}, "sandbox")
    assert metrics["sandbox/gpu_pool_utilization"] == utilization
    assert metrics.get("sandbox/gpu_pool_balance") == balance
    assert "sandbox/utilization" not in metrics and "sandbox/pool_balance" not in metrics

def test_pool_usage_handles_resizing_window_deltas_and_counter_resets():
    first = {"workers": {"worker-a": {"busy_ms": 100, "capacity_ms": 100}}}
    last = {"workers": {"worker-a": {"busy_ms": 200, "capacity_ms": 200},
                        "worker-b": {"busy_ms": 100, "capacity_ms": 100}}}
    for baseline in ({}, first):
        assert pool_usage_metrics(last, baseline, "pool") == {"pool/gpu_pool_utilization": 1.0, "pool/gpu_pool_balance": 1.0}
    assert pool_usage_metrics(first, last, "pool") == {}
    assert pool_usage_metrics({}, {}, "pool") == {}
    assert pool_usage_metrics({"workers": {"a": {"busy_ms": 0, "capacity_ms": 0}}}, {}, "pool") == {}

def test_loop_with_proxy_directory_keeps_round_source_and_does_not_double_count(tmp_path):
    _write_trajectory(tmp_path, "instance", [_proxy_turn(completion=999)])
    collector = MetricsCollector()
    collector._enabled = True
    collector._round_mode = True
    collector._format = "proxy"
    collector._trajectory_dir = str(tmp_path)
    collector.record_round(_message(completion=10), _phase(3, [("eval", 5)]), [], "instance")
    metrics = collector._snapshot()
    assert metrics["progress/rounds_total"] == 1
    assert metrics["token/round/completion_tokens_mean"] == 10
    assert metrics["interval/time/instance/llm_mean_s"] == 3
    assert "progress/turns_total" not in metrics
    assert "token/turn/completion_tokens_mean" not in metrics

@pytest.mark.parametrize("source", ["round"])
def test_sandbox_metrics_use_existing_names_and_one_source_per_agent(tmp_path, source):
    collector = MetricsCollector()
    collector._enabled = True
    collector._round_mode = source == "round"
    collector._output_dir = str(tmp_path)
    if source == "proxy":
        _write_trajectory(tmp_path, "done", [_proxy_turn(completion=100)])
        collector._format = "proxy"
        collector._trajectory_dir = str(tmp_path)
    empty = collector._snapshot()
    assert empty["progress/correctness_rate"] == empty["interval/progress/correctness_rate"] == 0
    timings = {"gpu_executions": [{"total_ms": 10000, "exec_ms": 8000, "queue_ms": 0, "attempts": 3}]}
    if source != "round":
        collector.record_sandbox_timings(timings, "done")
    else:
        collector.record_round(_message(), {"tools": [{"name": "eval", "total": 10, "sandbox": timings}]}, [], "done")
    collector.record_instance_done(instance_id="done", correctness=True, solved=False)
    metrics = collector._snapshot()
    if source == "proxy":
        assert metrics["token/turn/completion_tokens_mean"] == 100
        assert "token/round/completion_tokens_mean" not in metrics
    assert metrics["sandbox/gpu_commands_total"] == 1
    assert metrics["interval/sandbox/gpu_commands_total"] == 1
    assert metrics["sandbox/gpu_exec_mean_s"] == 8
    assert metrics["interval/sandbox/gpu_exec_mean_s"] == 8
    assert metrics["sandbox/gpu_exec_max_s"] == metrics["interval/sandbox/gpu_exec_max_s"] == 8
    assert metrics["progress/correctness_rate"] == metrics["interval/progress/correctness_rate"] == 1
    assert metrics["progress/solved_rate"] == metrics["interval/progress/solved_rate"] == 0
    assert metrics["sandbox/gpu_queue_mean_s"] == 0
    assert "sandbox/gpu_precheck_mean_s" not in metrics
    assert metrics["sandbox/gpu_retries_total"] == 2
    assert metrics["sandbox/gpu_retries_per_command"] == 2
    assert not any("/env/sandbox/" in key for key in metrics)
    next_metrics = collector._snapshot()
    assert next_metrics["sandbox/gpu_commands_total"] == 1
    assert next_metrics["interval/sandbox/gpu_commands_total"] == 1
    assert next_metrics["interval/sandbox/gpu_exec_mean_s"] == 8
    expired = collector._snapshot(now=time.monotonic() + collector._window)
    assert expired["sandbox/gpu_commands_total"] == 1
    assert expired["interval/sandbox/gpu_commands_total"] == 0
    assert "interval/sandbox/gpu_exec_mean_s" not in expired
    assert "interval/sandbox/gpu_exec_max_s" not in expired
    assert expired["progress/correctness_rate"] == 1
    assert expired["interval/progress/correctness_rate"] == 0

def test_gpu_probe_and_round_timings_count_once_without_creating_fake_rounds():
    collector = MetricsCollector()
    collector._enabled = True
    collector._round_mode = True
    collector.record_sandbox_timings({"gpu_executions": [{"exec_ms": 3000, "attempts": 1}]}, "instance")
    first = collector._snapshot()
    assert first["interval/sandbox/gpu_commands_total"] == 1
    assert "time/round/env/sandbox/gpu_exec_mean_s" not in first
    collector.record_round(_message(), {"tools": [{"name": "eval", "total": 8, "sandbox": {
        "gpu_executions": [{"exec_ms": 8000, "attempts": 1}]
    }}]}, [], "instance")
    collector.record_instance_done(solved=True, instance_id="instance")
    metrics = collector._snapshot()
    assert metrics["sandbox/gpu_commands_total"] == 2
    assert metrics["interval/sandbox/gpu_commands_total"] == 2
    assert metrics["sandbox/gpu_commands_per_instance"] == 2
    assert metrics["sandbox/gpu_exec_mean_s"] == 5.5
    assert not any("/env/sandbox/" in key for key in metrics)
    assert metrics["time/round/env_mean_s"] == 8
    assert metrics["progress/rounds_total"] == 1
    collector.record_sandbox_timings({"gpu_executions": [{"exec_ms": 1000, "attempts": 1}]}, "instance")
    assert collector._snapshot()["interval/sandbox/gpu_commands_total"] == 3
    assert collector._snapshot()["interval/sandbox/gpu_commands_total"] == 3

def test_trajectory_efficiency_uses_finished_cohort_and_cumulative_wall():
    collector = MetricsCollector()
    collector._enabled = True
    collector._round_mode = True
    now = time.monotonic()
    collector._run_started = now - 7200
    collector.record_instance_start("done")
    collector._instance_starts["done"] = now - 5400
    for instance_id, count in (("done", 2), ("pending", 10)):
        collector.record_round(_message(), {"tools": [{"name": "eval", "sandbox": {
            "gpu_executions": [{"exec_ms": 1000, "total_ms": 2000, "attempts": 1}] * count,
        }}]}, [], instance_id)
    collector.record_instance_done("Finished", solved=True, correctness=True, instance_id="done")
    metrics = collector._snapshot()
    assert metrics["progress/instances_done_per_hour"] == 0.5
    assert metrics["interval/progress/instances_done_per_hour"] == 1.0
    assert not any(key.endswith("/instances_per_hour") for key in metrics)
    assert metrics["progress/solved_rate"] == metrics["interval/progress/solved_rate"] == 1.0
    assert metrics["time/instance/wall_mean_s"] == pytest.approx(5400, abs=0.1)
    assert metrics["time/instance/wall_p95_s"] == pytest.approx(5400, abs=0.1)
    assert not any(key.startswith("interval/time/instance/wall_") for key in metrics)
    assert metrics["sandbox/gpu_commands_per_instance"] == 2
    assert metrics["interval/sandbox/gpu_commands_per_instance"] == 2
    assert "sandbox/gpu_commands_per_minute" not in metrics
    next_metrics = collector._snapshot()
    assert next_metrics["interval/progress/instances_done_per_hour"] == 1
    assert next_metrics["interval/progress/solved_rate"] == 1
    assert "interval/time/instance/wall_mean_s" not in next_metrics
    collector.record_instance_start("error")
    collector.record_instance_error("TimeoutError", instance_id="error")
    failed = collector._snapshot()
    assert failed["progress/instances_done"] == 2
    assert failed["interval/progress/solved_rate"] == 0.5
    assert failed["progress/correctness_rate"] == failed["interval/progress/correctness_rate"] == 0.5
    assert "time/instance/wall_mean_s" in failed
    assert not any(key.startswith("interval/time/instance/wall_") for key in failed)

def test_existing_inflight_and_error_metrics_survive_completed_trajectory_tracking():
    collector = MetricsCollector()
    collector._enabled = True
    collector._run_timeout = 100
    collector.record_instance_start("slow")
    collector._instance_starts["slow"] -= 90
    active = collector._snapshot()
    assert active["progress/inflight"] == active["progress/inflight_near_timeout"] == 1
    assert active["time/overall/inflight_max_age_s"] == pytest.approx(90, abs=0.1)
    collector.record_instance_error("TimeoutError", instance_id="slow")
    finished = collector._snapshot()
    assert finished["errors/other_total"] == finished["errors/timeout_total"] == 1
    assert finished["progress/instances_done"] == 1
    assert finished["progress/inflight"] == finished["time/overall/inflight_max_age_s"] == 0

def test_pool_snapshots_keep_gauges_and_do_not_zero_fill_legacy_usage():
    collector = MetricsCollector()
    collector._pool_snapshot = {"workers": {"idle": 1, "busy": 1, "pending": 2}, "replicas": 4}
    legacy = collector._snapshot()
    assert legacy["sandbox/workers_total"] == 4
    assert "sandbox/gpu_pool_utilization" not in legacy
    collector._pool_snapshot["usage"] = {"workers": {
        "worker-a": {"busy_ms": 50, "capacity_ms": 100},
        "worker-b": {"busy_ms": 50, "capacity_ms": 100},
    }}
    sampled = collector._snapshot()
    assert sampled["sandbox/gpu_pool_utilization"] == 0.5
    assert sampled["sandbox/gpu_pool_balance"] == 1
    assert sampled["interval/sandbox/gpu_pool_utilization"] == 0.5
    unchanged = collector._snapshot()
    assert unchanged["interval/sandbox/gpu_pool_utilization"] == 0.5

def test_rolling_round_and_request_samples_expire_without_changing_cumulative_metrics():
    collector = MetricsCollector()
    collector._enabled = collector._round_mode = True
    collector._run_started = 0
    collector._window = 100
    for observed_at, seconds in ((10, 2), (70, 8), (100, 20)):
        collector.record_round(_message(completion=seconds), _phase(seconds, []), [], "instance")
        collector._rounds[-1]["observed_at"] = observed_at
        collector.record_sandbox_timings({
            "cpu_execution": {"exec_ms": seconds * 1000}, "gpu_executions": [{"exec_ms": seconds * 1000}],
        }, "instance")
        collector._sandbox_rounds[-1]["observed_at"] = observed_at
    first = collector._snapshot(now=100)
    repeated = collector._snapshot(now=105)
    for metrics in (first, repeated):
        assert metrics["interval/time/round/llm_mean_s"] == 10
        assert metrics["interval/sandbox/cpu_exec_mean_s"] == 10
        assert "interval/progress/rounds_total" not in metrics
        assert "interval/progress/instances_done" not in metrics
    assert first["progress/rounds_last_interval"] == 3
    assert repeated["progress/rounds_last_interval"] == 0
    boundary = collector._snapshot(now=110)
    assert boundary["interval/time/round/llm_mean_s"] == 14
    assert boundary["interval/time/round/llm_p95_s"] == 19.4
    assert boundary["interval/time/instance/llm_mean_s"] == 28
    assert boundary["interval/sandbox/cpu_exec_mean_s"] == 14
    assert boundary["interval/sandbox/cpu_exec_max_s"] == 20
    assert boundary["time/round/llm_mean_s"] == boundary["sandbox/cpu_exec_mean_s"] == 10
    empty = collector._snapshot(now=200)
    assert "interval/time/round/llm_mean_s" not in empty
    assert empty["time/round/llm_mean_s"] == 10

def test_rolling_completion_rate_uses_actual_warmup_duration_and_full_finished_cohort():
    collector = MetricsCollector()
    collector._enabled = True
    collector._run_started = 0
    collector._window = 100
    collector._instance_results = [
        {"instance_id": "done", "observed_at": 10, "solved": True, "correctness": True, "wall_s": 200},
    ]
    warmup = collector._snapshot(now=50)
    assert warmup["interval/progress/instances_done_per_hour"] == 72
    assert warmup["interval/progress/solved_rate"] == 1
    assert warmup["progress/correctness_rate"] == warmup["interval/progress/correctness_rate"] == 1
    assert collector._snapshot(now=100)["interval/progress/instances_done_per_hour"] == 36
    expired = collector._snapshot(now=110)
    assert expired["interval/progress/instances_done_per_hour"] == 0
    assert "interval/progress/solved_rate" not in expired
    assert expired["time/instance/wall_mean_s"] == 200
    assert expired["progress/correctness_rate"] == 1
    assert expired["interval/progress/correctness_rate"] == 0

def test_rolling_pool_counters_use_boundary_snapshot_and_recover_after_reset():
    collector = MetricsCollector()
    collector._run_started = 0
    collector._window = 100
    metrics = {}
    for observed_at, busy_ms in ((50, 0), (100, 50000), (150, 100000), (175, 100000)):
        collector._pool_snapshot = {"usage": {"workers": {
            "worker": {"busy_ms": busy_ms, "capacity_ms": observed_at * 1000},
        }}}
        metrics[observed_at] = collector._snapshot(now=observed_at)
    assert metrics[50]["interval/sandbox/gpu_pool_utilization"] == 0
    assert metrics[100]["interval/sandbox/gpu_pool_utilization"] == 0.5
    assert metrics[150]["interval/sandbox/gpu_pool_utilization"] == 1
    assert metrics[175]["interval/sandbox/gpu_pool_utilization"] == 0.8
    assert metrics[175]["sandbox/gpu_pool_utilization"] == 0.5714
    assert metrics[175]["interval/sandbox/gpu_pool_balance"] == 1
    assert collector._pool_samples[0][0] == 50
    assert len(collector._pool_samples) == 4
    collector._pool_snapshot = {}
    assert "interval/sandbox/gpu_pool_utilization" not in collector._snapshot(now=180)
    collector._pool_snapshot = {"usage": {"workers": {"new": {"busy_ms": 1000, "capacity_ms": 2000}}}}
    assert "interval/sandbox/gpu_pool_utilization" not in collector._snapshot(now=200)
    collector._pool_snapshot = {"usage": {"workers": {"new": {"busy_ms": 3000, "capacity_ms": 10000}}}}
    assert collector._snapshot(now=208)["interval/sandbox/gpu_pool_utilization"] == 0.25

def test_defaults_are_overridden_per_step_without_retaining_values_or_charts():
    defaults = {"sandbox/gpu_pool_utilization": 0.0}
    assert _with_defaults(defaults, {"sandbox/gpu_pool_utilization": 0.5, "sandbox/gpu_pool_balance": 1.0}) == {
        "sandbox/gpu_pool_utilization": 0.5, "sandbox/gpu_pool_balance": 1.0,
    }
    assert defaults == {"sandbox/gpu_pool_utilization": 0.0, "sandbox/gpu_pool_balance": 0.0}
    carried = _with_defaults(defaults, {
        "progress/rounds_total": 3, "interval/tools/calls_total": 2, "interval/time/round/llm_mean_s": 5,
        "interval/progress/solved_rate": 1,
    })
    assert carried["sandbox/gpu_pool_utilization"] == carried["sandbox/gpu_pool_balance"] == 0
    assert carried["progress/rounds_total"] == 3
    expired = _with_defaults(defaults, {"progress/rounds_total": 3})
    assert expired["progress/rounds_total"] == 3
    assert expired["interval/tools/calls_total"] == 0
    assert expired["interval/time/round/llm_mean_s"] == expired["interval/progress/solved_rate"] == 0
    assert "best_turn/best_speedup_turn" in _with_defaults(defaults, {"best_turn/best_speedup_turn": [1, 2]})
    assert "best_turn/best_speedup_turn" not in _with_defaults(defaults, {})

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
