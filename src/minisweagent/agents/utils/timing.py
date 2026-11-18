"""Wall-clock timing for the agent loop's per-round phases.

Mirrors kerneleval's ``StageTimer`` but lives at the agent layer: it measures the
round-level phases the eval pipeline's own timer cannot see -- the model latency
and the sandbox queue/transport wrapped around each tool execution. Each
:meth:`mark` closes the interval since the previous mark and attributes its
elapsed seconds to ``name`` (summing if a name is marked more than once).

:func:`build_phase_timings` assembles the marks into the canonical layout shared
by the trajectory and preds.json::

    {"llm": <s>, "tools": [{"name", "total", "stage_timings", "sandbox"}, ...]}

where each tool's ``stage_timings`` is the eval pipeline's own per-stage
breakdown recovered from the tool output (``{}`` for non-eval tools) and
``sandbox`` carries the execution response's server timings.
"""

from __future__ import annotations

import json
import time

_DECODER = json.JSONDecoder()


class PhaseTimer:
    """Accumulate wall-clock seconds between successive marks, keyed by phase name."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._last = time.perf_counter()

    def mark(self, name: str) -> None:
        """Update the interval since the previous mark and add it to the named total."""
        now = time.perf_counter()
        self.timings[name] = round(self.timings.get(name, 0.0) + (now - self._last), 3)
        self._last = now


def extract_stage_timings(output: dict) -> dict:
    """Recover the eval pipeline's ``stage_timings`` from a tool's execute output, else ``{}``.

    The eval result JSON (which carries ``stage_timings``) is embedded in the tool's
    stdout; scan for the first JSON object that contains that key.
    """
    text = output.get("output") or ""
    i = text.find("{")
    while i >= 0:
        try:
            obj, _ = _DECODER.raw_decode(text, i)
        except Exception:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and "stage_timings" in obj:
            return obj.get("stage_timings") or {}
        i = text.find("{", i + 1)
    return {}


def extract_tool_result(output: dict) -> dict:
    """Read a structured tool result, ignoring log prefixes and unrelated JSON objects."""
    text = output.get("output") or ""
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = _DECODER.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.keys() & {"status", "compiled", "correctness", "error_code"}:
            return obj
    return {}


def build_phase_timings(timer: PhaseTimer, tool_outputs: list[tuple[str, dict]]) -> dict:
    """Assemble the canonical phase_timings: LLM latency plus one record per tool run.

    ``tool_outputs`` is the round's ordered ``(tool_name, execute_output)`` list.
    """
    return {
        "llm": timer.timings.get("llm"),
        "tools": [
            {
                "name": name,
                "total": timer.timings.get(name),
                "stage_timings": extract_stage_timings(output),
                "sandbox": output.get("extra", {}).get("sandbox_timings", {}),
                "skipped": (output.get("output") or "").lstrip().startswith("[skipped]"),
            }
            for name, output in tool_outputs
        ],
    }
