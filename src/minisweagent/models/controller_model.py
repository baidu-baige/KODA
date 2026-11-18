"""Multi-file controller source extraction and validated patch reconstruction."""

import ast
import re
from typing import Literal, NamedTuple

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel, LitellmTextbasedModelConfig

# Controller-managed envelope loop (kernel_controller.py): actions that deliberately carry NO source.
# For these the envelope JSON alone is the whole round — the mandatory fenced blocks must not be
# demanded. The legacy loop's prompt never emits an envelope, so this only affects the controller route.
_NO_SOURCE_ACTIONS = {"DIAGNOSE", "REVERT", "REQUEST_ENV_INFO", "ABORT_UNSUPPORTED"}


def _envelope_action(content: str) -> str:
    """Action name from the first JSON object in `content` that carries one; '' if none found."""
    import json

    for match in re.finditer(r"\{", content or ""):
        try:
            value, _ = json.JSONDecoder().raw_decode(content, match.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            action = str(value.get("action", "")).upper()
            if action:
                return action
    return ""


# A fenced block ```<lang> ... ``` — capture the language token and the body.
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.\-]*)[^\n]*\n(.*?)\n```", re.DOTALL)
# An UNTAGGED fence is deliberately NOT python: the prompt makes every round open with a fenced
# `SOLUTION LOG`, and the model routinely writes it as a bare ```. Counting that as kernel.py made the
# log text either overwrite the diff baseline or land on disk as a broken kernel.py.
_PY_LANGS = {"python", "py"}
_CU_LANGS = {"cpp", "c++", "cxx", "cc", "c", "cu", "cuda"}

# --- aider-style apply-patch (diff edit_mode) -----------------------------------------------------
# Mirrors opencode packages/core/src/patch.ts: one patch can carry several `*** Update File:` hunks,
# each a list of `@@` chunks of ` `(context) / `-`(remove) / `+`(add) lines. We locate each chunk's
# context with a fuzzy `_seek` (exact → rstrip → trim) so trivial whitespace drift doesn't reject it,
# after narrowing the search window with the chunk's `@@ <text>` anchors.
_PATCH_BEGIN = "*** Begin Patch"
_PATCH_END = "*** End Patch"
_UPDATE_FILE = "*** Update File:"
_EOF_MARKER = "*** End of File"
# `@@ -12,7 +12,7 @@` — unified-diff line numbers, not anchor text we could locate in the file.
_UNIFIED_HEADER = re.compile(r"^[-+]\d")


class PatchError(Exception):
    """A patch that cannot be parsed or applied — surfaced to the model as a FormatError re-prompt."""


class Chunk(NamedTuple):
    """One `@@` hunk. `anchors` are the texts of the `@@ <text>` lines introducing it (outer → inner,
    usually empty), which narrow where `old` is searched for — and are the ONLY locator an
    insert-only hunk has. `old`/`new` are the context+removed / context+added lines. `at_eof` marks a
    hunk closed by `*** End of File`, i.e. one anchored at the end of the file. `kinds` is the
    ' '/'-'/'+' prefix of each emitted line in order; it lets `_reindent` tell a kept context line
    from an added one and so re-base `new` onto the file's own indentation."""

    anchors: list[str]
    old: list[str]
    new: list[str]
    at_eof: bool = False
    kinds: str = ""


def _parse_patch_block(lines: list[str], begin: int, end: int) -> dict[str, list[Chunk]]:
    """Parse ONE `*** Begin Patch ... *** End Patch` span (open interval begin<i<end) into {path: chunks}."""
    files: dict[str, list[Chunk]] = {}
    cur: str | None = None
    anchors: list[str] = []
    i = begin + 1
    while i < end:
        line = lines[i]
        if line.startswith(_UPDATE_FILE):
            cur = line[len(_UPDATE_FILE) :].strip()
            if not cur:
                raise PatchError("empty path after '*** Update File:'")
            files.setdefault(cur, [])
            anchors = []
            i += 1
            continue
        if line.startswith("@@"):
            if cur is None:
                raise PatchError("'@@' chunk before any '*** Update File:'")
            header = line[2:].strip().removesuffix("@@").strip()
            if _UNIFIED_HEADER.match(header):  # `@@ -12,7 +12,7 @@ void g(){`: drop the line numbers,
                header = header.partition("@@")[2].strip()  # but keep any function context after them
            if header:
                anchors.append(header)
            old_lines: list[str] = []
            new_lines: list[str] = []
            kinds = ""
            at_eof = False
            i += 1
            while i < end:
                body = lines[i]
                if body.strip() == _EOF_MARKER:
                    at_eof = True
                elif body.startswith("@@") or body.startswith("***"):
                    break
                elif body.startswith(" "):
                    old_lines.append(body[1:])
                    new_lines.append(body[1:])
                    kinds += " "
                elif body.startswith("-"):
                    old_lines.append(body[1:])
                    kinds += "-"
                elif body.startswith("+"):
                    new_lines.append(body[1:])
                    kinds += "+"
                elif body == "":  # tolerate a bare blank line used as a context line
                    old_lines.append("")
                    new_lines.append("")
                    kinds += " "
                else:
                    raise PatchError(f"invalid chunk line (needs ' ', '-' or '+' prefix): {body!r}")
                i += 1
            if not old_lines and not new_lines:  # a bare `@@ <text>` line: pure locator for the next hunk
                continue
            if old_lines == new_lines:  # a no-op hunk: drop just this hunk. Rejecting the whole patch
                anchors = []  # also threw away the genuine hunks (and blocks) the model sent beside it
                continue
            files[cur].append(Chunk(anchors, old_lines, new_lines, at_eof, kinds))
            anchors = []
            continue
        if line.strip() == "" or line.startswith(("--- ", "+++ ")):  # blank, or a stray git-diff header
            i += 1
            continue
        raise PatchError(f"unexpected line inside patch: {line!r}")
    if not files:
        raise PatchError("patch contained no '*** Update File:' section")
    files = {path: chunks for path, chunks in files.items() if chunks}
    if not files:  # every hunk was a no-op, so applying this would silently re-run the previous kernel
        raise PatchError("no '@@' hunk in this patch carries an effective '+' / '-' line, so it would change nothing")
    return files


def _is_template_quote(lines: list[str], start: int, end: int) -> bool:
    """True when every hunk line between the markers is one of the `<...>` placeholders from the prompt's
    own patch template, i.e. the model restated the format instead of sending a patch. Such a block cannot
    apply, so parsing it as real would lose the whole round including any genuine patch sent beside it.
    The fence around it is no help: the prompt lets a real patch live in a fenced block too."""
    body = [
        stripped for line in lines[start + 1 : end] if line[:1] in (" ", "-", "+") and (stripped := line[1:].strip())
    ]
    return bool(body) and all(line.startswith("<") and line.endswith(">") for line in body)


def _parse_patch(text: str) -> list[dict[str, list[Chunk]]]:
    """Parse ALL `*** Begin Patch ... *** End Patch` blocks, in order, into a list of {path: chunks}.
    A single message may carry several blocks (same file at several spots, and/or multiple files); every
    block is returned so the caller can apply them sequentially — the earlier parser silently dropped all
    but the first block, leaving kernel.py/kernel.cu half-edited. Blocks that only quote the prompt's own
    template are dropped (`_is_template_quote`), so the list may be empty even though markers were found."""
    lines = text.split("\n")
    blocks: list[dict[str, list[Chunk]]] = []
    found = 0
    i = 0
    while i < len(lines):
        # The marker may carry leaked tool-call scaffolding BEFORE it (`...type="string"<|sep|>*** Begin
        # Patch`), so it need not be the whole line — but it must END the line, or prose merely mentioning
        # it would open a block and turn a plain wrong answer into a confusing parse error.
        if not lines[i].rstrip().endswith(_PATCH_BEGIN):
            i += 1
            continue
        end = next((j for j in range(i + 1, len(lines)) if _PATCH_END in lines[j]), -1)
        if end == -1:
            raise PatchError("'*** Begin Patch' without a matching '*** End Patch'")
        found += 1
        if not _is_template_quote(lines, i, end):
            blocks.append(_parse_patch_block(lines, i, end))
        i = end + 1
    if not found:
        raise PatchError("missing '*** Begin Patch' / '*** End Patch' markers")
    return blocks


_COMPARATORS = (
    lambda a, b: a == b,
    lambda a, b: a.rstrip() == b.rstrip(),
    lambda a, b: a.strip() == b.strip(),
)


def _matches(hay: list[str], pattern: list[str], start: int) -> list[int]:
    """Every index >= start where `pattern` matches `hay`, from the FIRST comparator level that matches
    at all (exact → rstrip → trim). Levels are never mixed, so two indices here mean the pattern is
    genuinely ambiguous at that strictness rather than merely loose-matching a second time. Empty
    pattern never matches."""
    if not pattern:
        return []
    for cmp in _COMPARATORS:
        if hits := [
            off
            for off in range(start, len(hay) - len(pattern) + 1)
            if all(cmp(hay[off + k], p) for k, p in enumerate(pattern))
        ]:
            return hits
    return []


def _seek(hay: list[str], pattern: list[str], start: int) -> int:
    """First index >= start where `pattern` matches `hay` (see `_matches`); -1 if never found."""
    return hits[0] if (hits := _matches(hay, pattern, start)) else -1


def _seek_anchors(hay: list[str], anchors: list[str], start: int) -> tuple[int, int, int]:
    """Resolve the `@@ <text>` anchors to a search window. Returns (window_start, anchor_line, hits):
    the window `old` is then searched from, the index of the innermost anchor line (-1 when there is
    no anchor or it does not match — anchors only *narrow* context matching, so a stale one falls
    back to `start` rather than failing the hunk), and how many places that innermost anchor matched.
    An anchor matching several places has pinned nothing down, so the caller must not credit it with
    disambiguating the hunk."""
    found, hits = -1, 0
    for anchor in anchors:
        at = _matches(hay, [anchor], start)
        if not at:
            return start, -1, 0
        start, found, hits = at[0], at[0], len(at)
    return start, found, hits


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _overlaps(replacements: list[tuple[int, int, list[str]]], at: int, length: int) -> bool:
    """Whether `[at, at+length)` touches a span another chunk already claimed. `max(..., 1)` gives a
    zero-width insert a one-line footprint, so an insert landing inside a replaced span still counts."""
    return any(at < start + max(remove, 1) and start < at + max(length, 1) for start, remove, _ in replacements)


def _reindent(matched: list[str], chunk: Chunk) -> list[str]:
    """`chunk.new`, re-based onto the indentation of the `matched` file lines the hunk replaces. The
    trim comparator in `_seek` happily matches a hunk whose lines lost their leading whitespace;
    writing `new` back verbatim then silently emits mis-indented code (fatal in kernel.py). So a kept
    context line is taken from the file verbatim, and every added line is moved from the indentation
    the model wrote to the one the file actually uses — which keeps the added block's own relative
    indentation intact."""
    if not chunk.kinds:
        return chunk.new
    out: list[str] = []
    src = dst = ""  # the indentation basis of the nearest line above that the file could confirm
    oi = ni = 0
    for kind in chunk.kinds:
        if kind == "+":
            line = chunk.new[ni]
            out.append(dst + line[len(src) :] if line.strip() and line.startswith(src) else line)
            ni += 1
            continue
        src, dst = _indent(chunk.old[oi]), _indent(matched[oi])
        if kind == " ":
            out.append(matched[oi])
            ni += 1
        oi += 1
    return out


def _apply_patch_file(original: str, chunks: list[Chunk]) -> str:
    """Apply an ordered list of chunks onto `original`: each is located by its `@@` anchors plus its
    own `old` lines (`_matches`), insert-only chunks land right after their anchor and `*** End of File`
    pins a chunk to the tail. Raises PatchError if a chunk cannot be placed, or if its context matches
    several spots and no anchor sits directly above the one to edit."""
    lines = original.split("\n")
    replacements: list[tuple[int, int, list[str]]] = []
    cursor = 0
    for chunk in chunks:
        window, anchor_line, anchor_hits = _seek_anchors(lines, chunk.anchors, cursor)
        if not chunk.old:  # insert-only: there is no context to match, so the anchor IS the location
            if chunk.at_eof:
                at = len(lines)
            elif anchor_hits > 1:
                raise PatchError(
                    f"an insert-only hunk's '@@ {chunk.anchors[-1]}' anchor matches {anchor_hits} places, "
                    "so it does not say where to insert — anchor it on a line that appears only ONCE, or "
                    "add ' ' context lines around the insertion point"
                )
            elif anchor_line != -1:
                at = anchor_line + 1
            else:
                if not chunk.anchors:
                    why = "it carries no '@@' anchor"
                elif _seek(lines, chunk.anchors[-1:], 0) != -1:
                    why = (
                        f"its '@@ {chunk.anchors[-1]}' anchor only appears BEFORE the previous hunk "
                        "— send the hunks in file order, top down, one '@@' per region"
                    )
                else:
                    why = f"its '@@ {chunk.anchors[-1]}' anchor is not in the file"
                raise PatchError(
                    f"an insert-only hunk (no ' ' context and no '-' line) cannot be placed: {why}. "
                    "Put the line to insert AFTER on the '@@' line, or add ' ' context lines, or close "
                    "the hunk with '*** End of File' to append at the end"
                )
            replacements.append((at, 0, chunk.new))
            cursor = max(cursor, at)
            continue
        # Models routinely repeat the `@@` anchor line as one of the hunk's OWN context lines. The anchor
        # then sits inside `old` instead of above it, so searching from `anchor_line` starts past the
        # hunk's real first line and drops the only match — back the window up by that offset.
        if chunk.anchors and anchor_line != -1:
            inside = next((k for k, line in enumerate(chunk.old) if line.strip() == chunk.anchors[-1].strip()), 0)
            window = max(cursor, anchor_line - inside)
        if chunk.at_eof:  # `*** End of File`: the hunk is anchored at the END, so only the tail may match
            window = max(window, len(lines) - len(chunk.old))
        hits = _matches(lines, chunk.old, window)
        # A trailing blank line the model appended to the hunk. Only trimmable when that blank IS the
        # last emitted line (`kinds[-1] == ' '`): dropping `old[-1]` otherwise desyncs `kinds` from
        # `old`/`new` and `_reindent` would index past the end.
        if not hits and chunk.kinds[-1:] == " " and chunk.old[-1] == "":
            trimmed = chunk._replace(old=chunk.old[:-1], new=chunk.new[:-1], kinds=chunk.kinds[:-1])
            if hits := _matches(lines, trimmed.old, window):
                chunk = trimmed
        if not hits:
            # The hunk may simply have been sent out of file order. Placing it above `window` is safe as
            # long as it is unique and claims a span no other chunk touches, because `replacements` are
            # applied back to front — only overlap, never emission order, could corrupt the result.
            earlier = _matches(lines, chunk.old, 0)
            if len(earlier) == 1 and not _overlaps(replacements, earlier[0], len(chunk.old)):
                hits = earlier
            elif earlier:
                raise PatchError(
                    f"its context lines do not appear at or after line {window + 1}, where the previous "
                    f"hunk and the '@@' anchors leave off — they match {len(earlier)} place(s) ABOVE that, "
                    f"the first at line {earlier[0] + 1}. Send the hunks in file order, top down, and give "
                    "each one enough context to be unique:\n" + "\n".join(chunk.old)
                )
            else:
                raise PatchError(
                    "could not locate the context lines — they appear NOWHERE in the current "
                    "file, so the hunk was written against a different version of it:\n" + "\n".join(chunk.old)
                )
        # An anchor only speaks for the region it is attached to: if it sits directly above the first
        # candidate it has named the spot, otherwise it merely narrowed the search and picking the first
        # of several candidates would be a guess (codex guesses here; aider refuses, and so do we).
        if len(hits) > 1 and not (anchor_hits == 1 and hits[0] - anchor_line <= 1):
            if anchor_hits > 1:
                why = f"its '@@ {chunk.anchors[-1]}' anchor is not unique either ({anchor_hits} matches)"
            elif anchor_hits == 1:
                why = (
                    f"its '@@ {chunk.anchors[-1]}' anchor sits {hits[0] - anchor_line} lines above the "
                    "first of them, so it only narrows the search instead of naming one"
                )
            elif chunk.anchors:
                why = f"its '@@ {chunk.anchors[-1]}' anchor does not match at all"
            else:
                why = "it carries no '@@' anchor"
            raise PatchError(
                f"these context lines match {len(hits)} different places in the file and {why}, so the "
                "hunk does not say which one to edit — give more ' ' context lines until they match only "
                "ONCE, or introduce the hunk with a '@@ <the unique line DIRECTLY above it>' anchor:\n"
                + "\n".join(chunk.old)
            )
        found = hits[0]
        replacements.append((found, len(chunk.old), _reindent(lines[found : found + len(chunk.old)], chunk)))
        cursor = max(cursor, found + len(chunk.old))
    updated = list(lines)
    # Back to front so earlier indices stay valid; ties (several inserts at one point) break on the
    # emission index so the applied order matches the order the model wrote them in.
    for idx in sorted(range(len(replacements)), key=lambda k: (replacements[k][0], k), reverse=True):
        start, remove, insert = replacements[idx]
        updated[start : start + remove] = insert
    return "\n".join(updated)


class ControllerModelConfig(LitellmTextbasedModelConfig):
    """Config for the multi-file kernel loop model, selecting how each round re-emits the kernel files."""

    edit_mode: Literal["full", "diff"] = "full"
    """`full`: every round re-emits whole kernel.py + kernel.cu fenced blocks (default; also what the
    shared gpu_only route uses). `diff`: from round 2 the model may send an apply-patch instead, which
    the controller re-applies onto the previous FULL files to cut output tokens (per-round latency)."""


class ControllerModel(LitellmTextbasedModel):
    """Parse the TWO files the multi-backend loop requires every round out of one response:
    `kernel.py` (a ```python block defining `ModelNew`) AND `kernel.cu` (a ```cpp / ```cuda block,
    the CUDA / CUTLASS / C++ kernel source). BOTH are mandatory — a response missing either block
    raises FormatError so the model is re-prompted. Returns one action carrying `kernel_py` and
    `kernel_cu` (`command` mirrors `kernel_py` for submission bookkeeping).

    In `edit_mode: diff` the model may instead send an aider-style `*** Begin Patch` against the
    files it last produced; the reconstructed FULL files are what flow downstream (write_candidate,
    compile, eval) — the patch is purely a transport optimisation to shrink the model's output. A
    patch may be combined with a full fenced rewrite of the OTHER file; see `_parse_diff`.
    """

    submission_marker = "ModelNew"

    def __init__(self, **kwargs):
        # LitellmTextbasedModel.__init__ hardcodes its own config_class, so go straight to the base
        # to slot in our edit_mode-carrying config (pydantic drops unknown kwargs, so it must be a
        # real field on a dedicated config class, not a loose kwarg).
        LitellmModel.__init__(self, config_class=ControllerModelConfig, **kwargs)
        self._prev_py: str | None = None
        self._prev_cu: str | None = None

    def _fail(self, error: str, finish_reason) -> None:
        content = Template(self.config.format_error_template, undefined=StrictUndefined).render(
            error=error, actions=[], finish_reason=finish_reason
        )
        raise FormatError({"role": "user", "content": content, "extra": {"interrupt_type": "FormatError"}})

    def _parse_actions(self, response) -> list[dict]:
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        # controller-managed envelope loop: actions that carry NO source (REVERT / REQUEST_ENV_INFO /
        # ABORT_UNSUPPORTED / DIAGNOSE) are allowed to skip the mandatory fenced blocks — the envelope
        # alone is the whole round. The legacy loop's prompt never emits an envelope, so this path is
        # inert there. Returning an empty action list means "no candidate this round"; the agent routes
        # the action to its special-case outputs instead of building a candidate.
        envelope_action = _envelope_action(content)
        if envelope_action in _NO_SOURCE_ACTIONS:
            return [{}]
        if self.config.edit_mode == "diff" and _PATCH_BEGIN in content:
            return self._parse_diff(content, finish_reason)
        return self._parse_full(content, finish_reason)

    def _fenced_sources(self, content: str) -> tuple[str | None, str | None]:
        """(kernel_py, kernel_cu) from the fenced blocks — None for a file the response does not carry.
        Blocks that merely QUOTE a patch are skipped: the prompt's own `*** Begin Patch` template is
        fenced, so a model echoing it must not have that fence mistaken for source."""
        blocks = [(m.group(1).lower(), m.group(2).strip()) for m in _FENCE.finditer(content)]
        blocks = [(lang, b) for lang, b in blocks if b and _PATCH_BEGIN not in b]
        py_blocks = [b for lang, b in blocks if lang in _PY_LANGS]
        cu_blocks = [b for lang, b in blocks if lang in _CU_LANGS]
        # kernel.py: the last python block defining ModelNew (fall back to the last python block).
        # kernel.cu: the last native block.
        return (
            next((b for b in reversed(py_blocks) if self.submission_marker in b), py_blocks[-1] if py_blocks else None),
            cu_blocks[-1] if cu_blocks else None,
        )

    def _parse_full(self, content: str, finish_reason) -> list[dict]:
        """Extract whole kernel.py + kernel.cu from fenced blocks; both mandatory. Records the pair as
        the new baseline so a later diff can patch against it."""
        kernel_py, kernel_cu = self._fenced_sources(content)
        if not kernel_py and not kernel_cu:
            self._fail(
                "Your response contained neither a ```python block (kernel.py) nor a ```cpp "
                "block (kernel.cu). Every fence must carry its language tag — an untagged ``` "
                "block is read as prose (the SOLUTION LOG), never as source.",
                finish_reason,
            )
        if not kernel_py:
            self._fail(
                "Missing the ```python block: kernel.py must define `ModelNew` and call into "
                "the compiled `cuda_extension`. Tag the fence `python` — an untagged ``` block "
                "is read as prose (the SOLUTION LOG), never as source.",
                finish_reason,
            )
        if not kernel_cu:
            self._fail(
                "Missing the ```cpp block: every round must include kernel.cu — your CUDA / "
                "CUTLASS / C++ kernel source (with its REGISTER_BINDING). A pure-PyTorch or "
                "Triton-only kernel.py is not accepted here.",
                finish_reason,
            )
        # Same two gates the diff path has: a full rewrite can just as well be unparseable or a verbatim
        # restatement of the previous round, and here there is nothing downstream that would catch it.
        try:
            ast.parse(kernel_py)
        except SyntaxError as e:
            self._fail(
                f"kernel.py is not valid Python: {e.msg} (line {e.lineno}). Re-send it as a "
                "complete ```python block that defines `ModelNew`.",
                finish_reason,
            )
        if (kernel_py, kernel_cu) == (self._prev_py, self._prev_cu):
            self._fail(
                "Both files are byte-identical to the ones you sent last round, so this round "
                "would re-run the previous kernel and measure the same number again. Send a "
                "change, or state that you are done.",
                finish_reason,
            )
        self._prev_py, self._prev_cu = kernel_py, kernel_cu
        return [{"command": kernel_py, "kernel_py": kernel_py, "kernel_cu": kernel_cu}]

    def _parse_diff(self, content: str, finish_reason) -> list[dict]:
        """Apply an apply-patch onto the previous FULL files, reconstruct both, and record the result
        as the new baseline. Any parse/apply failure re-prompts (fall back to full fenced blocks).

        A response MAY carry a patch for one file and a full fenced rewrite of the other — the rewrite
        is the base the patch is then applied onto, instead of being silently dropped. When BOTH files
        arrive as full blocks the response already says what to run, so any `*** Begin Patch` text in it
        is a quote and the full blocks win — unless those blocks merely restate the baseline, which means
        the change was in the patch and would be thrown away.
        """
        full_py, full_cu = self._fenced_sources(content)
        if full_py and full_cu:
            if (full_py, full_cu) == (self._prev_py, self._prev_cu):
                self._fail(
                    "Your response restated BOTH files unchanged and put the actual change in a "
                    "patch. A response carrying both complete files is read as a full rewrite, so "
                    "the patch was ignored and this round would re-run the previous kernel. Send "
                    "the patch ALONE, or fold the change into the full blocks.",
                    finish_reason,
                )
            return self._parse_full(content, finish_reason)
        kernel_py, kernel_cu = full_py or self._prev_py, full_cu or self._prev_cu
        if kernel_py is None or kernel_cu is None:
            self._fail(
                "You sent a patch (*** Begin Patch) but there is no previous kernel to patch. "
                "Round 1 must contain the FULL kernel.py and kernel.cu as fenced code blocks.",
                finish_reason,
            )
        try:
            blocks = _parse_patch(content)
        except PatchError as e:
            self._fail(
                f"Your patch could not be parsed: {e}. Re-send the change either as a valid "
                "'*** Begin Patch ... *** End Patch' block or as full fenced code blocks.",
                finish_reason,
            )
        if not blocks:  # every block only quoted this prompt's `<...>` template, so nothing was patched
            return self._parse_full(content, finish_reason)
        for files in blocks:
            for path, chunks in files.items():
                target = path.lower()
                is_py = target.endswith(".py") or "kernel.py" in target
                is_cu = (
                    target.endswith((".cu", ".cuh", ".cpp", ".cxx", ".cc", ".c", ".h", ".hpp")) or "kernel.cu" in target
                )
                if not (is_py or is_cu):
                    self._fail(
                        f"Patch targets unknown file '{path}'. Only kernel.py and kernel.cu are editable.",
                        finish_reason,
                    )
                try:
                    if is_py:
                        kernel_py = _apply_patch_file(kernel_py, chunks)
                    else:
                        kernel_cu = _apply_patch_file(kernel_cu, chunks)
                except PatchError as e:
                    self._fail(
                        f"Patch for '{path}' did not apply: {e}. Context lines must match the current "
                        "file; if unsure, re-send that file as a full fenced code block.",
                        finish_reason,
                    )
        try:
            ast.parse(kernel_py)
        except SyntaxError as e:
            self._fail(
                f"kernel.py is not valid Python: {e.msg} (line {e.lineno}). If a patch hunk stated "
                "the indentation of the lines it touched differently from the file, the rebuilt file "
                "ends up mis-indented — re-send kernel.py as a full ```python block.",
                finish_reason,
            )
        if (kernel_py, kernel_cu) == (self._prev_py, self._prev_cu):
            self._fail(
                "Your patch applied but left both files byte-identical, so this round would "
                "re-run the previous kernel. Send a patch that actually changes something.",
                finish_reason,
            )
        self._prev_py, self._prev_cu = kernel_py, kernel_cu
        return [{"command": kernel_py, "kernel_py": kernel_py, "kernel_cu": kernel_cu}]
