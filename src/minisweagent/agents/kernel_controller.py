"""Controller-managed kernel optimization with immutable candidates and correctness gates."""

from __future__ import annotations

import hashlib

import json

import re

from dataclasses import dataclass

from typing import Any

from minisweagent import Environment, Model

from minisweagent.agents.base import AgentRuntime

from minisweagent.agents.controller_runtime import (
    ToolRuntime,
    ToolRuntimeConfig,
    _env_version_note,
    _read_env_info,
    _strip_tool_output,
    model_specs_markdown,
)

from minisweagent.agents.utils.kernel_workspace import read_workspace_file

from minisweagent.agents.utils.controller_tools import run_controller_tool

from minisweagent.agents.utils.timing import PhaseTimer, build_phase_timings


from minisweagent.utils.metrics import COLLECTOR

_ACTIONS = {
    "IMPLEMENT",
    "PATCH",
    "DIAGNOSE",
    "REVERT",
    "BRANCH",
    "REQUEST_ENV_INFO",
    "ABORT_UNSUPPORTED",
}

_ENVIRONMENT_FIELDS = (
    "target_environment_id",
    "shape_signature",
    "dtype_signature",
    "layout_signature",
    "backend",
    "gpu",
    "device_id",
    "architecture",
    "compile_arch",
    "env_versions",
)


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Find the first JSON object containing a controller action."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("action") in _ACTIONS:
            return value
    return None


def _decision_from_message(message: dict, *, round_number: int, baseline: str | None) -> dict[str, Any]:
    """Read a structured envelope while remaining compatible with legacy models."""
    content = message.get("content", "")
    decision = _first_json_object(content)
    if decision is None:
        decision = {
            "action": "IMPLEMENT" if round_number == 1 else "PATCH",
            "baseline_candidate": baseline,
            "hypothesis": "Legacy source response; evaluate the submitted candidate.",
            "expected_effect": "Unknown until measured.",
            "correctness_risk": "Source-specific correctness risk is unknown.",
            "validation_plan": ["compile", "correctness", "timing"],
            "files_changed": [],
            "revert_condition": "Reject on any correctness failure or measured regression.",
        }
    action = str(decision.get("action", "")).upper()
    if action not in _ACTIONS:
        raise ValueError(f"unknown controller action {action!r}")
    decision["action"] = action
    decision.setdefault("baseline_candidate", baseline)
    decision.setdefault("hypothesis", "No hypothesis supplied.")
    decision.setdefault("expected_effect", "Unknown until measured.")
    decision.setdefault("correctness_risk", "Unspecified.")
    decision.setdefault("validation_plan", ["compile", "correctness"])
    decision.setdefault("files_changed", [])
    decision.setdefault("revert_condition", "Reject invalid or slower candidates.")
    return decision


def _json_from_output(output: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", output or ""):
        try:
            value, _ = decoder.raw_decode(output, match.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _evaluation_summary(output: dict[str, Any]) -> dict[str, Any]:
    """Extract a stable, small summary from eval output."""
    parsed = _json_from_output(output.get("output", ""))
    if not parsed:
        return {
            "status": "unparseable",
            "compiled": False,
            "correctness": False,
            "score": None,
        }
    completed = parsed.get("status") == "completed"
    # The opencode envelope deliberately omits the legacy ``correctness`` bool:
    # ``status=completed`` is its correctness verdict.  Legacy eval keeps the
    # explicit bool and therefore still requires it.
    correctness = completed and (bool(parsed.get("correctness")) or "correctness" not in parsed)
    score = parsed.get("speedup", parsed.get("score"))
    return {
        "status": parsed.get("status"),
        "compiled": bool(parsed.get("compiled", completed)),
        "correctness": correctness,
        "score": score if isinstance(score, (int, float)) else None,
        "speedup": parsed.get("speedup"),
        "kernel_wall_time": parsed.get("kernel_wall_time"),
        "raw": parsed,
    }


def _profile_allowed(eval_output: dict) -> bool:
    """Profile only a candidate that passed the evaluator's correctness gate."""
    summary = _evaluation_summary(eval_output)
    return summary["correctness"] is True


@dataclass(frozen=True)
class CandidateSnapshot:
    """Immutable snapshot of one submitted kernel candidate in the controller loop.

    A candidate is never mutated after evaluation; ``source_digest`` lets the
    controller detect identical submissions and ``target_environment_key`` keeps
    candidates from different target environments from being compared.
    """

    candidate_id: str
    parent_id: str | None
    source: dict[str, str]
    source_digest: str
    decision: dict[str, Any]
    round_number: int
    target_environment_key: str
    shape_signature: str


class _CandidateControllerMixin:
    """State and selection logic shared by all kernel loop routes."""

    def _init_candidate_controller(self) -> None:
        self._candidates: dict[str, CandidateSnapshot] = {}
        self._evaluations: dict[tuple[str, str], dict[str, Any]] = {}
        # One controller instance optimizes one target environment.  The key is
        # retained on candidates so externally merged training trajectories
        # cannot accidentally compare candidates from different devices.
        self._best_by_target: dict[str, str] = {}
        self._active_candidate_id: str | None = None
        self._last_candidate_id: str | None = None
        self._candidate_sequence = 0
        self._latest_decision: dict[str, Any] = {}
        self._latest_outputs: list[dict[str, Any]] = []
        self._target_environment_key = "unknown"
        self._shape_signature = ""
        self._known_facts: list[dict[str, Any]] = []
        self._rejected_hypotheses: list[dict[str, Any]] = []

    def _environment_context(self) -> dict[str, Any]:
        cfg = getattr(self.env, "config", None)
        values = {}
        for field in _ENVIRONMENT_FIELDS:
            values[field] = getattr(cfg, field, None) if cfg is not None else None
        values["device_name"] = getattr(cfg, "gpu", None) if cfg is not None else None
        values["backend"] = values["backend"] or "unknown"
        return values

    def _set_target_environment_key(self) -> None:
        values = self._environment_context()
        encoded = json.dumps(values, sort_keys=True, default=str)
        self._target_environment_key = hashlib.sha256(encoded.encode()).hexdigest()[:16]
        self._shape_signature = str(values.get("shape_signature") or "")

    def _source_from_action(self, action: dict[str, Any]) -> dict[str, str]:
        if "kernel_py" in action:
            source = {"kernel.py": action["kernel_py"]}
            if action.get("kernel_cu") is not None:
                source["kernel.cu"] = action["kernel_cu"]
            return source
        if "command" in action:
            return {"kernel.py": action["command"]}
        return {}

    def _new_candidate(self, source: dict[str, str], decision: dict[str, Any]) -> CandidateSnapshot:
        if not source:
            raise ValueError("model response did not contain source files")
        self._candidate_sequence += 1
        candidate_id = f"candidate_{self._candidate_sequence:04d}"
        parent_id = decision.get("baseline_candidate") or self._active_candidate_id
        digest_input = "\n".join(f"{name}\0{source[name]}" for name in sorted(source))
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        candidate = CandidateSnapshot(
            candidate_id=candidate_id,
            parent_id=parent_id if parent_id in self._candidates else None,
            source=dict(source),
            source_digest=digest,
            decision=dict(decision),
            round_number=self.round,
            target_environment_key=self._target_environment_key,
            shape_signature=self._shape_signature,
        )
        self._candidates[candidate_id] = candidate
        self._active_candidate_id = candidate_id
        self._last_candidate_id = candidate_id
        return candidate

    def _best_candidate_id(self) -> str | None:
        return self._best_by_target.get(self._target_environment_key)

    def _best_candidate(self) -> CandidateSnapshot | None:
        candidate_id = self._best_candidate_id()
        return self._candidates.get(candidate_id) if candidate_id else None

    def _candidate_score(self, candidate_id: str) -> float | None:
        evaluation = self._evaluations.get((candidate_id, self._target_environment_key), {})
        summary = evaluation.get("summary", {})
        score = summary.get("score")
        return float(score) if isinstance(score, (int, float)) else None

    def _record_evaluation(
        self,
        candidate: CandidateSnapshot,
        eval_output: dict[str, Any],
        outputs: list[dict[str, Any]],
    ) -> None:
        summary = _evaluation_summary(eval_output)
        key = (candidate.candidate_id, self._target_environment_key)
        self._evaluations[key] = {
            "summary": summary,
            "outputs": outputs,
            "round": self.round,
        }
        accepted = False
        if summary["correctness"]:
            current_best = self._best_candidate_id()
            current_score = self._candidate_score(current_best) if current_best else None
            new_score = summary["score"]
            min_improvement = float(getattr(self.config, "min_relative_improvement", 0.0))
            improves = current_best is None or (
                isinstance(new_score, (int, float))
                and (current_score is None or float(new_score) > current_score * (1.0 + min_improvement))
            )
            if improves:
                self._best_by_target[self._target_environment_key] = candidate.candidate_id
                accepted = True
                self._known_facts.append(
                    {
                        "fact": "candidate passed the configured correctness gate",
                        "candidate_id": candidate.candidate_id,
                        "target_environment_key": self._target_environment_key,
                        "score": new_score,
                    }
                )
        else:
            self._rejected_hypotheses.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "target_environment_key": self._target_environment_key,
                    "reason": summary["status"],
                    "hypothesis": candidate.decision.get("hypothesis", ""),
                }
            )
        if not accepted:
            self._active_candidate_id = self._best_candidate_id()
            self._restore_best()

    def _restore_best(self) -> CandidateSnapshot | None:
        candidate = self._best_candidate()
        if candidate is None:
            return None
        if hasattr(self.env, "write_candidate"):
            if "kernel.cu" in candidate.source:
                self.env.write_candidate(candidate.source["kernel.py"], candidate.source["kernel.cu"])
            else:
                self.env.write_candidate(candidate.source["kernel.py"])
        return candidate

    def _submission_source(self) -> str:
        candidate = self._best_candidate() or self._candidates.get(self._last_candidate_id or "")
        if candidate is None:
            return ""
        return candidate.source.get("kernel.py", next(iter(candidate.source.values()), ""))

    def _controller_state_message(self) -> dict[str, Any]:
        best = self._best_candidate()
        best_summary = (
            self._evaluations.get((best.candidate_id, self._target_environment_key), {}).get("summary", {})
            if best
            else {}
        )
        state = {
            "controller_state": {
                "optimization_scope": "single_target_environment",
                "shape_scope": "single_fixed_shape",
                "round": self.round,
                "target_environment_key": self._target_environment_key,
                "shape_signature": self._shape_signature,
                "active_candidate": self._active_candidate_id,
                "best_candidate": best.candidate_id if best else None,
                "best_score": best_summary.get("score"),
                "candidate_count": len(self._candidates),
                "known_facts": self._known_facts[-8:],
                "rejected_hypotheses": self._rejected_hypotheses[-8:],
                "latest_decision": self._latest_decision,
            }
        }
        return {
            "role": "user",
            "content": "CONTROLLER_STATE:\n" + json.dumps(state, ensure_ascii=False, indent=2),
            "extra": {"controller_state": state},
        }

    def _prepare_decision_and_candidate(self, message: dict) -> tuple[dict[str, Any], CandidateSnapshot | None]:
        decision = _decision_from_message(
            message,
            round_number=self.round,
            baseline=self._active_candidate_id or self._best_candidate_id(),
        )
        self._latest_decision = decision
        if decision["action"] == "REVERT":
            self._restore_best()
            return decision, None
        if decision["action"] == "DIAGNOSE":
            # 诊断轮: 不产候选、不跑评估 pipeline —— 诊断结论本身是状态, 记进 known_facts,
            # 下一轮经 CONTROLLER_STATE 带上 (无代码动作, model 侧 _NO_SOURCE_ACTIONS 放行)。
            hypothesis = str(decision.get("hypothesis", "")).strip()
            if hypothesis:
                self._known_facts.append({"fact": "DIAGNOSE: " + hypothesis, "round": self.round})
            return decision, None
        if decision["action"] in {"REQUEST_ENV_INFO", "ABORT_UNSUPPORTED"}:
            return decision, None
        action = message.get("extra", {}).get("actions", [{}])[0]
        candidate = self._new_candidate(self._source_from_action(action), decision)
        return decision, candidate

    def _finish_round(
        self,
        message: dict,
        outputs: list[dict[str, Any]],
        timer: PhaseTimer,
        eval_output: dict[str, Any] | None = None,
    ) -> list[dict]:
        phase_timings = build_phase_timings(timer, [(o["tool"], o) for o in outputs])
        if eval_output is not None:
            eval_output.setdefault("extra", {})["phase_timings"] = phase_timings
        self._latest_outputs = outputs
        self._log_round_timings(phase_timings)
        COLLECTOR.record_round(message, phase_timings, outputs, self.instance_id)
        observed = [_strip_tool_output(output) for output in outputs]
        messages = self.model.format_observation_messages(message, observed, self.get_template_vars(round=self.round))
        if getattr(self.config, "record_candidate_state", True):
            messages.append(self._controller_state_message())
        # Count only rounds that sent a candidate to eval (eval_output is not None); no-code rounds
        # (DIAGNOSE/REVERT/REQUEST_ENV_INFO) don't submit and don't count toward max_submit_eval.
        if eval_output is not None:
            self._eval_submits = getattr(self, "_eval_submits", 0) + 1
        submit_limit = 0 < getattr(self.config, "max_submit_eval", 0) <= getattr(self, "_eval_submits", 0)
        round_limit = self.round >= self.config.num_rounds
        aborted = self._latest_decision.get("action") == "ABORT_UNSUPPORTED"
        if round_limit or aborted or submit_limit:
            # Distinct exit_status per cause. ABORT_UNSUPPORTED is the model's explicit "cannot support
            # under this backend/device contract" decision, not budget exhaustion — keep it separate so
            # stats don't fold it into RoundLimitReached. Genuine round-limit still wins if it coincides.
            if round_limit:
                status = "RoundLimitReached"
            elif aborted:
                status = "AbortUnsupported"
            else:
                status = "EvalSubmitLimitReached"
            messages.append(
                {
                    "role": "exit",
                    "content": status,
                    "extra": {
                        "exit_status": status,
                        "submission": self._submission_source(),
                        "best_candidate": self._best_candidate_id(),
                    },
                }
            )
        return self.add_messages(*messages)

    def _special_action_outputs(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        action = decision["action"]
        if action == "REVERT":
            return [
                {
                    "tool": "controller",
                    "output": json.dumps(
                        {
                            "status": "reverted",
                            "candidate_id": self._best_candidate_id(),
                            "message": "Restored the best valid candidate; no performance evaluation was run.",
                        }
                    ),
                    "returncode": 0,
                    "exception_info": "",
                    "extra": {},
                }
            ]
        if action == "DIAGNOSE":
            # 诊断轮输出: 回显已记录的诊断结论, 提示下轮基于它继续 (无 eval/profile)。
            return [
                {
                    "tool": "controller",
                    "output": json.dumps(
                        {
                            "status": "diagnosed",
                            "message": "Diagnosis recorded; no kernel was evaluated this round. "
                            "Next round, act on the recorded DIAGNOSE fact.",
                            "hypothesis": decision.get("hypothesis", ""),
                        }
                    ),
                    "returncode": 0,
                    "exception_info": "",
                    "extra": {},
                }
            ]
        if action == "REQUEST_ENV_INFO":
            return [
                {
                    "tool": "controller",
                    "output": json.dumps({"status": "environment_info", "environment": self._environment_context()}),
                    "returncode": 0,
                    "exception_info": "",
                    "extra": {},
                }
            ]
        return [
            {
                "tool": "controller",
                "output": json.dumps(
                    {
                        "status": "unsupported",
                        "message": decision.get("hypothesis", "The requested path is unsupported."),
                        "environment": self._environment_context(),
                    }
                ),
                "returncode": 0,
                "exception_info": "",
                "extra": {},
            }
        ]


class ControllerConfig(ToolRuntimeConfig):
    """Opencode configuration with controller selection settings."""

    target_environment_id: str = ""
    """Stable identity of the device/environment optimized by this run."""

    shape_signature: str = ""
    """Canonical fixed input shape signature used by this run."""

    dtype_signature: str = ""
    """Input/output dtype signature for the fixed-shape task."""

    layout_signature: str = ""
    """Input/output layout and stride signature for the fixed-shape task."""

    min_relative_improvement: float = 0.01
    record_candidate_state: bool = True


class ToolControllerAgent(_CandidateControllerMixin, ToolRuntime):
    """Controller-managed opencode pipeline.

    The existing envelope tools remain the execution backend.  This class adds
    immutable candidates and applies compile/eval/profile gates using the
    structured result returned by those tools.
    """

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        instance_id: str = "",
        **kwargs,
    ):
        AgentRuntime.__init__(
            self,
            model,
            env,
            config_class=ControllerConfig,
            instance_id=instance_id,
            **kwargs,
        )
        self._env_versions = ""
        self._init_candidate_controller()

    def run(self, task: str = "", **kwargs) -> dict:
        """Run the controller-managed loop: reset candidate state, then delegate."""
        self._init_candidate_controller()
        self._set_target_environment_key()
        template_kwargs = {
            "shape_signature": self.config.shape_signature,
            "compile_arch": self.config.compile_arch,
            "dtype_signature": self.config.dtype_signature,
            "layout_signature": self.config.layout_signature,
        }
        # 与 legacy 对齐的模板变量: CUTLASS 骨架 / 环境探测注入 (test yaml 的 system_template
        # 引用了这些 Jinja 变量 —— 缺失时 {% if %} 块隐去, 但变量本身必须传入避免 StrictUndefined)。
        cfg = getattr(self.env, "config", None)
        env_info = _read_env_info(self.env, getattr(self.config, "kernelcompile_dir", None))
        template_kwargs.update(
            {
                "task_id": getattr(cfg, "workspace_name", "") or "task",
                "backend": getattr(cfg, "backend", "") or self.config.backend,
                "gpu": getattr(cfg, "gpu", None) or "the CUDA device selected by the evaluator",
                "reference_code": read_workspace_file(self.env, "task.py"),
                "model_specs": model_specs_markdown(cfg),
                "env_versions": env_info["versions"] or self.config.env_versions,
                "triton_api_hints": env_info["triton_api_hints"],
                "gpu_specs": env_info["gpu_specs"],
                "cutlass_version": env_info.get("cutlass_version", ""),
                "cutlass_example": env_info.get("cutlass_example", ""),
                "handwritten_example": env_info.get("handwritten_example", ""),
                "cutlass3x_example": env_info.get("cutlass3x_example", ""),
            }
        )
        self._env_versions = template_kwargs["env_versions"]
        template_kwargs.update(kwargs)
        return super().run(task, **template_kwargs)

    def _write_round(self, action: dict) -> None:
        source = self._source_from_action(action)
        if "kernel.py" in source:
            self.env.write_candidate(source["kernel.py"])

    def step(self) -> list[dict]:
        """One controller round: query LLM, run the action, record the evaluation."""
        timer = PhaseTimer()
        message = self.query()
        timer.mark("llm")
        self.round += 1
        decision, candidate = self._prepare_decision_and_candidate(message)
        if candidate is None:
            return self._finish_round(
                message,
                self._special_action_outputs(decision),
                timer,
            )
        self._write_round(message.get("extra", {}).get("actions", [{}])[0])
        timer.mark("setup")
        outputs: list[dict[str, Any]] = []
        gate_ok = True
        env_note = ""
        for tool in self.config.pipeline_tools:
            if not gate_ok:
                outputs.append(
                    self._skipped(
                        tool,
                        f"[skipped] {tool}: prerequisite failed (pipeline={self.config.pipeline_tools})",
                    )
                )
                timer.mark(tool)
                continue
            output = run_controller_tool(
                tool,
                cfg=self.config,
                session=self.env,
                workspace=self._workspace(),
            )
            if tool == "compile":
                gate_ok = bool(output.get("extra", {}).get("compiled"))
            elif tool == "eval":
                gate_ok = _profile_allowed(output)
                env_note = _env_version_note(self._env_versions, output.get("output", ""))
            outputs.append(output)
            timer.mark(tool)
        eval_output = next(
            (output for output in outputs if output["tool"] == "eval"),
            outputs[0] if outputs else None,
        )
        self._record_evaluation(candidate, eval_output or {}, outputs)
        if env_note:
            eval_output.setdefault("extra", {})["controller_env_note"] = env_note
        return self._finish_round(message, outputs, timer, eval_output)


class KernelControllerAgent(ToolControllerAgent):
    """Opencode variant that writes both ``kernel.py`` and ``kernel.cu``."""

    def _write_round(self, action: dict) -> None:
        source = self._source_from_action(action)
        if "kernel.py" in source:
            self.env.write_candidate(source["kernel.py"], source.get("kernel.cu"))
