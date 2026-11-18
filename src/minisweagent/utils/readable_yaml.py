"""Render a trajectory dict as human-readable YAML (trajectory.readable.yaml).

Keeps the original key/value structure but renders multiline strings as real
newlines + indentation via block literals. Round-trip verified against the
original JSON (structure + values + key order) on the full 255-sample dataset.

Two PyYAML quirks are handled (verified empirically, do not "simplify"):

1. Values ending in a newline get ``allow_block=False`` in
   ``analyze_scalar``, so a representer asking for ``|`` style is silently
   demoted back to double-quoted by ``choose_scalar_style`` -- the emitter
   must force ``|`` directly.
2. ``yaml.dump`` defaults ``sort_keys=True``, so ``kw.setdefault`` in
   ``__init__`` is a no-op; ``sort_keys`` must be assigned unconditionally.
3. ``\\r`` in a value is written raw inside a block literal and the YAML
   scanner treats it as a CRLF line end (breaks the whole document / loses
   the char on reload) -- such values fall back to double-quoted style.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from yaml.emitter import Emitter
from yaml.representer import SafeRepresenter


class _ForcingEmitter(Emitter):
    """Emitter that keeps asked-for block-literal style for multiline values."""

    def choose_scalar_style(self) -> str:
        """Force ``|`` style when the pending event requests it for a multiline value."""
        evt = self.event
        if (
            evt.style == "|"
            and not self.flow_level
            and not self.simple_key_context
            and isinstance(evt.value, str)
            and ("\n" in evt.value or "\r" in evt.value)
        ):
            return "|"
        return super().choose_scalar_style()


class _BlockDumper(_ForcingEmitter, yaml.SafeDumper):
    """Safe dumper rendering multiline strings as block literals and keeping key order."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Force unicode, insertion-ordered, non-flow output before delegating."""
        kwargs["allow_unicode"] = True
        kwargs["sort_keys"] = False  # yaml.dump defaults True; setdefault is overridden
        kwargs["default_flow_style"] = False
        kwargs["width"] = 120
        super().__init__(*args, **kwargs)

    def represent_str_multi(self, data: str) -> yaml.nodes.ScalarNode:
        """Represent a string as block literal, or double-quoted when it contains ``\\r``."""
        if "\n" in data or "\r" in data:
            if "\r" in data:
                # Raw CR inside a block-literal breaks the YAML scanner (it is
                # treated as a CRLF line end). Double-quoted escapes it losslessly.
                return self.represent_scalar("tag:yaml.org,2002:str", data, style='"')
            return self.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return SafeRepresenter.represent_str(self, data)


_BlockDumper.add_representer(str, _BlockDumper.represent_str_multi)


def _unescape_json_text(value: Any) -> Any:
    """Render-time only: turn JSON escape sequences into real whitespace/quote chars.

    Deep-copies the value; JSON-encoded strings (tool ``arguments`` / ``content`` in
    the react format) keep their JSON syntax (braces, quoted keys) but ``\\n`` / ``\\t``
    / ``\\"`` render as real newlines/tabs/quotes, so the mirror shows a formatted
    JSON-ish text instead of one escaped line. Never touches ``trajectory.json``.
    """
    if isinstance(value, dict):
        return {k: _unescape_json_text(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unescape_json_text(v) for v in value]
    if isinstance(value, str) and any(s in value for s in ("\\n", "\\t", "\\\"")):
        return value.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
    return value


def _pretty_tool_output(value: Any) -> Any:
    """Render-time only: pretty-print compact JSON in tool results.

    Tool ``content`` can be a shell ``{"returncode": 0, "output": "<json>"}``
    (eval/profile) or a bare compact JSON object (compile: ``{"sources":…}``).
    When the tool serialised it as a compact single line there are no ``\\n``
    escapes for unescaping to restore — so re-format such JSON (still JSON
    syntax, never YAML) at render time. Already-multiline JSON payloads and
    plain text (compile logs etc.) pass through as before; the JSON file is
    untouched. Parse-then-unescape order is required: unescaping ``\\"`` first
    would break the outer JSON string boundaries (``"output": "{"...}``).
    """
    if not isinstance(value, str):
        return value
    if "\n" in value:
        # 已有真换行: 裸多行 JSON(profile)或纯文本日志 → 原样/unescape 原样
        try:
            already = json.loads(value)
        except ValueError:
            return _unescape_json_text(value)
        if isinstance(already, (dict, list)):
            return value  # 工具侧已 pretty,不重复重排
        return _unescape_json_text(value)
    try:
        obj = json.loads(value)  # 无真换行 ⇒ 字面是合法 JSON(\\n 仅是转义序列)
    except ValueError:
        return _unescape_json_text(value)
    if isinstance(obj, dict) and isinstance(obj.get("output"), str):
        try:
            out = json.loads(obj["output"])
        except ValueError:
            return _unescape_json_text(value)
        if not isinstance(out, (dict, list)):
            return _unescape_json_text(value)
        obj["output"] = json.dumps(out, indent=2, ensure_ascii=False)
    elif not isinstance(obj, (dict, list)):
        return _unescape_json_text(value)
    return _unescape_json_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _display_messages(data: dict) -> dict:
    """Render-time copy of a trajectory dict with escape sequences unescaped.

    Tool-call ``arguments`` (a JSON string in the react format) and ``tool`` result
    ``content`` (``{"returncode", "output"}`` JSON) keep their JSON syntax but get
    real newlines/indentation, so the readable mirror is not one escaped line.
    Loop observations (``user`` messages starting "### Round …") carry the same
    escaped JSON in their ``<output>`` block and are unescaped the same way;
    compact tool results get their embedded JSON re-formatted (see
    ``_pretty_tool_output``).
    """
    out = dict(data)
    msgs = out.get("messages")
    if not isinstance(msgs, list):
        return out
    tidy_msgs = []
    for m in msgs:
        tm = dict(m)
        tcs = tm.get("tool_calls")
        if isinstance(tcs, list):
            tm["tool_calls"] = [
                {
                    **tc,
                    "function": {
                        **tc["function"],
                        "arguments": _unescape_json_text(tc["function"].get("arguments")),
                    },
                }
                if isinstance(tc, dict) and isinstance(tc.get("function"), dict) else tc
                for tc in tcs
            ]
        if tm.get("role") == "tool":
            tm["content"] = _pretty_tool_output(tm.get("content"))
        if (
            tm.get("role") == "user"
            and isinstance(tm.get("content"), str)
            and tm["content"].lstrip().startswith("### Round")
        ):
            tm["content"] = _unescape_json_text(tm["content"])
        tidy_msgs.append(tm)
    out["messages"] = tidy_msgs
    return out


def render_readable_yaml(data: dict) -> str:
    """Serialize ``data`` as readable YAML (block literals for multiline strings)."""
    return yaml.dump(data, Dumper=_BlockDumper)


def render_trajectory_yaml(trajectory_path: str | Path) -> str:
    """Read a trajectory.json and return its readable-YAML rendering.

    Tool-call arguments / tool results are unescaped at render time (see
    ``_display_messages``) — the JSON file itself is left byte-identical.
    """
    data = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    return render_readable_yaml(_display_messages(data))