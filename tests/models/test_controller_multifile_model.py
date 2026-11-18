"""Tests for ControllerModel full/diff parsing (see models/kernel_loop_model.py)."""

from types import SimpleNamespace

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.controller_model import (
    Chunk,
    ControllerModel,
    PatchError,
    _apply_patch_file,
    _parse_patch,
    _seek,
)


def _resp(content: str, finish_reason: str = "stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)])


PY_V1 = "import torch\n\nclass ModelNew(torch.nn.Module):\n    def forward(self, x):\n        return x + 1\n"
CU_V1 = "#include <torch/extension.h>\n\nvoid launch() {}\nREGISTER_BINDING(launch)\n"


def _full_response(py: str, cu: str) -> str:
    return f"prose\n\n```python\n{py}\n```\n\nmore prose\n\n```cpp\n{cu}\n```\n"


# --- low-level patch primitives -----------------------------------------------------------------

def test_parse_patch_collects_chunks_per_file():
    patch = (
        "*** Begin Patch\n"
        "*** Update File: kernel.py\n"
        "@@\n"
        " def forward(self, x):\n"
        "-        return x + 1\n"
        "+        return x + 2\n"
        "*** Update File: kernel.cu\n"
        "@@\n"
        "-void launch() {}\n"
        "+void launch() { /* fused */ }\n"
        "*** End Patch\n"
    )
    blocks = _parse_patch(patch)
    assert len(blocks) == 1
    files = blocks[0]
    assert set(files) == {"kernel.py", "kernel.cu"}
    anchors, old, new, at_eof, kinds = files["kernel.py"][0]
    assert (anchors, at_eof, kinds) == ([], False, " -+")
    assert old == ["def forward(self, x):", "        return x + 1"]
    assert new == ["def forward(self, x):", "        return x + 2"]
    assert files["kernel.cu"][0] == Chunk([], ["void launch() {}"], ["void launch() { /* fused */ }"], False, "-+")


def test_parse_patch_collects_every_block_not_just_the_first():
    # regression: the model emits TWO independent Begin/End blocks in one message; the old parser kept
    # only the first, silently dropping the rest and leaving files half-edited.
    patch = (
        "*** Begin Patch\n"
        "*** Update File: kernel.cu\n"
        "@@\n"
        "-void launch() {}\n"
        "+void launch() { /* a */ }\n"
        "*** End Patch\n"
        "some prose between blocks\n"
        "*** Begin Patch\n"
        "*** Update File: kernel.py\n"
        "@@\n"
        "-        return x + 1\n"
        "+        return x + 2\n"
        "*** End Patch\n"
    )
    blocks = _parse_patch(patch)
    assert [set(b) for b in blocks] == [{"kernel.cu"}, {"kernel.py"}]
    assert blocks[0]["kernel.cu"][0] == Chunk([], ["void launch() {}"], ["void launch() { /* a */ }"], False, "-+")
    assert blocks[1]["kernel.py"][0] == Chunk([], ["        return x + 1"], ["        return x + 2"], False, "-+")


def test_parse_patch_unmatched_begin_raises():
    with pytest.raises(PatchError) as exc:
        _parse_patch("*** Begin Patch\n*** Update File: kernel.py\n@@\n-a\n+b\n")
    assert "matching '*** End Patch'" in str(exc.value)


@pytest.mark.parametrize(
    ("patch", "needle"),
    [
        ("no markers here", "Begin Patch"),
        ("*** Begin Patch\n@@\n x\n*** End Patch\n", "before any"),
        ("*** Begin Patch\n*** Update File: kernel.py\n@@\n?bad line\n*** End Patch\n", "invalid chunk line"),
        ("*** Begin Patch\n*** End Patch\n", "no '*** Update File:'"),
        # a section with no hunk at all used to apply as a silent no-op and burn the round
        ("*** Begin Patch\n*** Update File: kernel.py\n*** End Patch\n", "no '@@' hunk"),
        ("*** Begin Patch\n*** Update File: kernel.py\n@@ def forward\n*** End Patch\n", "no '@@' hunk"),
        # a hunk that only repeats context (or removes and re-adds the same line) is a silent no-op, and
        # a patch made of nothing but such hunks has nothing left to apply
        ("*** Begin Patch\n*** Update File: kernel.py\n@@\n class ModelNew:\n*** End Patch\n", "change nothing"),
        ("*** Begin Patch\n*** Update File: kernel.py\n@@\n-a\n+a\n*** End Patch\n", "change nothing"),
    ],
)
def test_parse_patch_rejects_malformed(patch, needle):
    with pytest.raises(PatchError) as exc:
        _parse_patch(patch)
    assert needle in str(exc.value)


def test_parse_patch_tolerates_a_stray_git_diff_file_header():
    # a model mixing git-diff habits into apply-patch used to hard-fail the whole round
    assert _parse_patch(
        "*** Begin Patch\n*** Update File: kernel.cu\n--- a/kernel.cu\n+++ b/kernel.cu\n"
        "@@\n-void launch() {}\n+void launch() { }\n*** End Patch\n"
    )[0]["kernel.cu"][0].old == ["void launch() {}"]
    # ...but a real removal of a `--count;` line must NOT be mistaken for one of those headers
    assert _parse_patch("*** Begin Patch\n*** Update File: kernel.cu\n@@\n---count;\n+++count;\n"
                        "*** End Patch\n")[0]["kernel.cu"][0].old == ["--count;"]


def test_parse_patch_reads_the_at_header_as_an_anchor_and_ignores_line_numbers():
    patch = (
        "*** Begin Patch\n"
        "*** Update File: kernel.cu\n"
        "@@ void g(){\n"
        "-    int i = 0;\n"
        "+    int i = 7;\n"
        "@@ -12,3 +12,3 @@\n"
        "-int done;\n"
        "+int done2;\n"
        "@@ -20,3 +20,3 @@ void h(){\n"
        "-int last;\n"
        "+int last2;\n"
        "*** End Patch\n"
    )
    chunks = _parse_patch(patch)[0]["kernel.cu"]
    # unified-diff line numbers are not an anchor, but a function context printed after them is
    assert [c.anchors for c in chunks] == [["void g(){"], [], ["void h(){"]]


def test_parse_patch_stacks_bare_at_locator_lines_onto_the_next_hunk():
    patch = (
        "*** Begin Patch\n"
        "*** Update File: kernel.py\n"
        "@@ class ModelNew\n"
        "@@     def forward(self, x):\n"
        "-        return x + 1\n"
        "+        return x + 2\n"
        "*** End Patch\n"
    )
    chunks = _parse_patch(patch)[0]["kernel.py"]
    # the two locator-only `@@` lines used to become an empty chunk each; now they anchor the real hunk
    assert len(chunks) == 1
    assert chunks[0].anchors == ["class ModelNew", "def forward(self, x):"]  # matched by `_seek`'s trim pass


def test_seek_falls_back_through_looser_comparisons():
    hay = ["alpha", "    beta  ", "gamma"]
    assert _seek(hay, ["alpha"], 0) == 0  # exact
    assert _seek(hay, ["    beta"], 0) == 1  # rstrip match
    assert _seek(hay, ["beta"], 0) == 1  # trim match
    assert _seek(hay, ["missing"], 0) == -1
    assert _seek(hay, [], 0) == -1


def test_apply_patch_file_replaces_and_locates_fuzzily():
    original = "line0\n  target()  \nline2\n"
    # context has different surrounding whitespace than the file -> only trim comparison matches
    result = _apply_patch_file(original, [Chunk([], ["target()"], ["fixed()"])])
    assert "fixed()" in result and "target()" not in result
    assert result.splitlines()[0] == "line0" and result.splitlines()[-1] == "line2"


def test_apply_patch_file_raises_when_context_missing():
    with pytest.raises(PatchError):
        _apply_patch_file("a\nb\nc\n", [Chunk([], ["does not exist"], ["x"])])


def test_apply_patch_file_anchor_disambiguates_repeated_context():
    # regression: `@@ void g(){` used to be dropped, so this one-line hunk hit the FIRST `int i = 0;`
    # (inside f()) and silently edited the wrong function.
    original = "void f(){\n    int i = 0;\n}\nvoid g(){\n    int i = 0;\n}\n"
    patched = _apply_patch_file(original, [Chunk(["void g(){"], ["    int i = 0;"], ["    int i = 7;"])])
    assert patched == "void f(){\n    int i = 0;\n}\nvoid g(){\n    int i = 7;\n}\n"


def test_apply_patch_file_insert_only_lands_after_its_anchor():
    # regression: an insert-only hunk used to be appended at END OF FILE regardless of its anchor,
    # which puts an #include after the bindings and breaks the compile.
    original = "#include <torch/extension.h>\n#define TILE 16\nvoid launch(){}\n"
    patched = _apply_patch_file(original, [Chunk(["#include <torch/extension.h>"], [], ["#include <cuda_fp16.h>"])])
    assert patched.splitlines()[:2] == ["#include <torch/extension.h>", "#include <cuda_fp16.h>"]


def test_apply_patch_file_insert_only_needs_a_locator():
    with pytest.raises(PatchError) as exc:
        _apply_patch_file("a\nb\n", [Chunk([], [], ["// where?"])])
    assert "insert-only hunk" in str(exc.value)
    with pytest.raises(PatchError) as exc:
        _apply_patch_file("a\nb\n", [Chunk(["not in the file"], [], ["// where?"])])
    assert "anchor is not in the file" in str(exc.value)
    # `*** End of File` is the explicit way to ask for an append
    assert _apply_patch_file("a\nb\n", [Chunk([], [], ["// tail"], True)]) == "a\nb\n\n// tail"


def test_apply_patch_file_keeps_the_emitted_order_of_several_inserts_at_one_point():
    # regression: equal insertion points were applied back-to-front, reversing the added lines.
    chunks = [Chunk([], [], ["// FIRST"], True), Chunk([], [], ["// SECOND"], True)]
    assert _apply_patch_file("a\n", chunks).splitlines()[-2:] == ["// FIRST", "// SECOND"]


def test_apply_patch_file_tells_an_out_of_order_anchor_from_a_missing_one():
    # the anchor IS in the file, just before the previous hunk -> the message must say so, not lie
    chunks = [Chunk([], ["REGISTER_BINDING(launch)"], ["REGISTER_BINDING(l2)"]),
              Chunk(["#include <torch/extension.h>"], [], ["#include <cuda_fp16.h>"])]
    with pytest.raises(PatchError) as exc:
        _apply_patch_file(CU_V1, chunks)
    assert "only appears BEFORE the previous hunk" in str(exc.value)


def _patched_py(patch_body: str) -> str:
    return _apply_patch_file(PY_V1, _parse_patch(
        f"*** Begin Patch\n*** Update File: kernel.py\n@@\n{patch_body}*** End Patch\n")[0]["kernel.py"])


@pytest.mark.parametrize(
    ("patch_body", "expected_tail"),
    [
        # regression: the trim comparator matches a hunk the model sent WITHOUT its leading whitespace;
        # writing `new` back verbatim then emitted `return x + 2` at column 0 — invalid Python, and the
        # failure only surfaced as a confusing IndentationError from the model's own file.
        ("-return x + 1\n+return x + 2\n", ["    def forward(self, x):", "        return x + 2"]),
        # mixed drift: unindented context line, correctly indented removal -> both must come out right
        (" def forward(self, x):\n-        return x + 1\n+        return x + 2\n",
         ["    def forward(self, x):", "        return x + 2"]),
        # an exact match must be left exactly alone
        ("-        return x + 1\n+        return x + 2\n",
         ["    def forward(self, x):", "        return x + 2"]),
        # an added block keeps its own relative indentation while being re-based onto the file's
        ("-return x + 1\n+if x is None:\n+    return 0\n+return x + 2\n",
         ["    def forward(self, x):", "        if x is None:", "            return 0", "        return x + 2"]),
    ],
)
def test_apply_patch_file_rebases_new_lines_onto_the_files_own_indentation(patch_body, expected_tail):
    assert _patched_py(patch_body).splitlines()[-len(expected_tail):] == expected_tail


def test_apply_patch_file_trailing_blank_tolerance_does_not_desync_the_hunk():
    # regression: the trailing-blank retry dropped `old[-1]` and `kinds[-1]`, but when the LAST emitted
    # line is a `+` those are different lines -> `_reindent` walked off the end of `old` with an
    # IndexError, which is NOT a PatchError and so killed the whole run instead of re-prompting.
    with pytest.raises(PatchError) as exc:
        _patched_py("     def forward(self, x):\n\n+        y = x * 2\n")
    assert "could not locate" in str(exc.value)
    # the tolerance itself must still work when the blank really is the hunk's last line
    assert _patched_py("-        return x + 1\n+        return x + 2\n \n").splitlines()[-1] == "        return x + 2"


def test_apply_patch_file_end_of_file_marker_pins_a_context_hunk_to_the_tail():
    # regression: `at_eof` was only honoured for insert-only hunks, so this appended after the FIRST
    # `}` (inside f()) instead of at the real end of the file.
    original = "void f(){\n}\nvoid g(){\n}"
    chunks = _parse_patch("*** Begin Patch\n*** Update File: kernel.cu\n@@\n }\n+// tail\n*** End of File\n"
                          "*** End Patch\n")[0]["kernel.cu"]
    assert _apply_patch_file(original, chunks) == "void f(){\n}\nvoid g(){\n}\n// tail"


def test_apply_patch_file_rejects_an_ambiguous_hunk_instead_of_editing_the_first_match():
    # aider-style ambiguity check: without an anchor these context lines fit two regions, and silently
    # taking the first one edited the wrong function.
    original = "void f(){\n  int i = 0;\n}\nvoid g(){\n  int i = 0;\n}"
    body = "-  int i = 0;\n+  int i = 1;\n"
    with pytest.raises(PatchError) as exc:
        _apply_patch_file(original, _parse_patch(
            f"*** Begin Patch\n*** Update File: kernel.cu\n@@\n{body}*** End Patch\n")[0]["kernel.cu"])
    assert "2 different places" in str(exc.value)
    # an anchor resolves it, and the edit must land in g()
    assert _apply_patch_file(original, _parse_patch(
        f"*** Begin Patch\n*** Update File: kernel.cu\n@@ void g(){{\n{body}*** End Patch\n"
    )[0]["kernel.cu"]) == "void f(){\n  int i = 0;\n}\nvoid g(){\n  int i = 1;\n}"


def test_apply_patch_file_a_non_unique_anchor_does_not_count_as_disambiguation():
    # regression: the ambiguity check was skipped whenever an anchor *resolved*, but an anchor matching
    # several places resolves to the FIRST one — which re-opened the very hole the check closes and
    # silently edited f() instead of g().
    body = "-    sum += x[i];\n+    sum += x[i] * 2.0f;\n"
    loop = "  for (int i = 0; i < n; ++i) {"
    original = f"void f(){{\n{loop}\n    sum += x[i];\n  }}\n}}\nvoid g(){{\n{loop}\n    sum += x[i];\n  }}\n}}"

    def patch(anchor: str, hunk: str = body):
        return _apply_patch_file(original, _parse_patch(
            f"*** Begin Patch\n*** Update File: kernel.cu\n@@{anchor}\n{hunk}*** End Patch\n")[0]["kernel.cu"])

    with pytest.raises(PatchError) as exc:
        patch(loop)
    assert "not unique either (2 matches)" in str(exc.value)
    # an insert-only hunk has nothing BUT the anchor, so a non-unique one must be rejected too
    with pytest.raises(PatchError) as exc:
        patch(loop, "+    // unrolled\n")
    assert "anchor matches 2 places" in str(exc.value)
    assert patch(" void g(){").splitlines()[-3] == "    sum += x[i] * 2.0f;"  # a unique anchor still works


def test_apply_patch_file_a_distant_anchor_does_not_pick_between_several_candidates():
    # regression: a UNIQUE anchor switched the ambiguity check off entirely, so a hunk whose context
    # repeats inside the anchored region (very common in CUDA: `}`, `__syncthreads();`, `sum += ...`)
    # was applied to the first candidate. An anchor only names a spot when it sits directly above it.
    body = "   for (int i = 0; i < n; ++i) {\n-    sum += x[i];\n+    sum += x[i] * 2.0f;\n"
    loop = "  for (int i = 0; i < n; ++i) {"
    original = (f"void f(){{\n  // first pass\n{loop}\n    sum += x[i];\n  }}\n"
                f"  // second pass\n{loop}\n    sum += x[i];\n  }}\n}}")

    def patch(anchor: str, hunk: str = body):
        return _apply_patch_file(original, _parse_patch(
            f"*** Begin Patch\n*** Update File: kernel.cu\n@@{anchor}\n{hunk}*** End Patch\n")[0]["kernel.cu"])

    with pytest.raises(PatchError) as exc:
        patch(" void f(){")
    assert "only narrows the search" in str(exc.value)
    # directly above the hunk the anchor DOES name it: the first pass changes, the second must not
    assert patch(" // first pass").splitlines()[3:8:4] == ["    sum += x[i] * 2.0f;", "    sum += x[i];"]
    assert patch(" // second pass").splitlines()[3:8:4] == ["    sum += x[i];", "    sum += x[i] * 2.0f;"]


def test_parse_patch_drops_a_quoted_template_but_keeps_real_hunks():
    # the prompt's patch template is fenced and full of `<...>` placeholders; a model that restates it
    # while ALSO sending a real patch used to lose the whole round on the un-appliable template block.
    template = (
        "*** Begin Patch\n*** Update File: kernel.cu\n"
        "@@ <the enclosing function / a nearby unique line, to pin down WHERE this hunk goes>\n"
        " <a few UNCHANGED context lines, each prefixed by a single space>\n"
        "-<the exact line(s) to remove, prefixed by ->\n"
        "+<the new line(s) to add, prefixed by +>\n"
        "*** End Patch\n"
    )
    real = ("*** Begin Patch\n*** Update File: kernel.cu\n@@\n-void launch() {}\n"
            "+void launch() { /*x*/ }\n*** End Patch\n")
    assert len(_parse_patch(template + real)) == 1
    assert _parse_patch(template) == []  # nothing to apply, but no bogus PatchError either
    # real C++ that merely contains angle brackets must not be mistaken for a placeholder
    for hunk in ["+#include <cuda.h>\n", "+  k<<<grid, block>>>(p);\n", "+  Gemm<\n+    float, RowMajor>\n"]:
        assert len(_parse_patch(f"*** Begin Patch\n*** Update File: kernel.cu\n@@ void f(){{\n{hunk}"
                                "*** End Patch\n")) == 1
    with pytest.raises(PatchError):  # a response with no markers at all is still an error
        _parse_patch("no patch here")


def test_parse_patch_skips_a_no_op_hunk_instead_of_losing_the_whole_patch():
    # models send a wrong hunk and then the corrected one in the same message ("resending the correct
    # one"); rejecting the whole patch over the dud threw the good hunk away with it.
    blocks = _parse_patch(
        "*** Begin Patch\n*** Update File: kernel.cu\n"
        "@@\n-void launch() {}\n+void launch() {}\n"          # no-op: dropped
        "@@\n-REGISTER_BINDING(launch)\n+REGISTER_BINDING(go)\n"
        "*** End Patch\n"
    )
    assert blocks[0]["kernel.cu"] == [Chunk([], ["REGISTER_BINDING(launch)"], ["REGISTER_BINDING(go)"], False, "-+")]
    # a dropped hunk must not leak its `@@` anchor onto the next one
    assert _parse_patch("*** Begin Patch\n*** Update File: kernel.cu\n@@ void launch() {}\n x\n"
                        "@@\n-a\n+b\n*** End Patch\n")[0]["kernel.cu"][0].anchors == []
    # a file left with no surviving hunk disappears, but a sibling file's hunks still apply
    assert set(_parse_patch("*** Begin Patch\n*** Update File: kernel.py\n@@\n-a\n+a\n"
                            "*** Update File: kernel.cu\n@@\n-b\n+c\n*** End Patch\n")[0]) == {"kernel.cu"}


def test_parse_patch_accepts_a_marker_line_carrying_leaked_tool_call_scaffolding():
    # reasoning models leak their own harness syntax onto the marker line, which the whole-line
    # comparison rejected as "missing '*** Begin Patch'" even though a valid patch followed
    blocks = _parse_patch(
        'ready.<|open|>call tool="patch"<|sep|>*** Begin Patch\n'
        "*** Update File: kernel.cu\n@@\n-void launch() {}\n+void launch() { /*x*/ }\n"
        "*** End Patch<|close|>\n"
    )
    assert blocks[0]["kernel.cu"][0].new == ["void launch() { /*x*/ }"]


def test_apply_patch_file_anchor_repeated_as_its_own_context_line_still_locates_the_hunk():
    # the commonest anchor mistake: `@@ <line>` where <line> is ALSO a context line of the hunk. The
    # window started AT the anchor, i.e. below the hunk's first line, and dropped the only match.
    original = "int a;\nvoid f() {\n  int x = 1;\n  return;\n}\n"
    chunk = Chunk(["void f() {"], ["int a;", "void f() {", "  int x = 1;"],
                  ["int a;", "void f() {", "  int x = 2;"], False, "  -+")
    assert _apply_patch_file(original, [chunk]) == "int a;\nvoid f() {\n  int x = 2;\n  return;\n}\n"


def test_apply_patch_file_rescues_a_unique_hunk_sent_out_of_file_order():
    original = "a = 1\nb = 2\nc = 3\nd = 4\n"
    late, early = Chunk([], ["c = 3"], ["c = 30"], False, "-+"), Chunk([], ["a = 1"], ["a = 10"], False, "-+")
    # emitted bottom-up: replacements are applied back to front, so a non-overlapping earlier hunk is safe
    assert _apply_patch_file(original, [late, early]) == "a = 10\nb = 2\nc = 30\nd = 4\n"
    # ...but an ambiguous one above the cursor is still refused, and says so precisely
    with pytest.raises(PatchError) as exc:
        _apply_patch_file("x\ny\nx\nz\n", [Chunk([], ["z"], ["z!"], False, "-+"), Chunk([], ["x"], ["x!"], False, "-+")])
    assert "in file order" in str(exc.value) and "2 place(s) ABOVE" in str(exc.value)
    # context that is nowhere in the file gets a different message: re-checking the order cannot help
    with pytest.raises(PatchError) as exc:
        _apply_patch_file(original, [Chunk([], ["nope"], ["nope!"], False, "-+")])
    assert "NOWHERE" in str(exc.value)


def test_apply_patch_file_out_of_order_rescue_refuses_to_overlap_an_earlier_hunk():
    original = "a\nb\nc\n"
    first = Chunk([], ["a", "b"], ["a", "B"], False, " -+")
    with pytest.raises(PatchError) as exc:  # "b" is unique but already claimed by `first`
        _apply_patch_file(original, [first, Chunk([], ["b"], ["b2"], False, "-+")])
    assert "in file order" in str(exc.value)


# --- model-level full/diff behaviour ------------------------------------------------------------

def test_full_mode_parses_both_files_and_records_baseline():
    model = ControllerModel(model_name="dummy")
    assert model.config.edit_mode == "full"
    action = model._parse_actions(_resp(_full_response(PY_V1, CU_V1)))[0]
    assert action["kernel_py"].strip() == PY_V1.strip()
    assert action["kernel_cu"].strip() == CU_V1.strip()
    assert action["command"] == action["kernel_py"]
    # baseline is recorded even in full mode so a later switch/diff can patch it
    assert model._prev_py == action["kernel_py"] and model._prev_cu == action["kernel_cu"]


def test_full_mode_missing_cu_block_reprompts():
    model = ControllerModel(model_name="dummy")
    with pytest.raises(FormatError):
        model._parse_actions(_resp(f"```python\n{PY_V1}\n```\n"))


def test_diff_mode_applies_patch_and_leaves_other_file_untouched():
    model = ControllerModel(model_name="dummy", edit_mode="diff")
    assert model.config.edit_mode == "diff"
    # round 1: full files establish the baseline
    model._parse_actions(_resp(_full_response(PY_V1, CU_V1)))
    patch = (
        "reasoning...\n\n"
        "*** Begin Patch\n"
        "*** Update File: kernel.py\n"
        "@@\n"
        " def forward(self, x):\n"
        "-        return x + 1\n"
        "+        return x + 2\n"
        "*** End Patch\n"
    )
    action = model._parse_actions(_resp(patch))[0]
    assert "return x + 2" in action["kernel_py"] and "return x + 1" not in action["kernel_py"]
    assert action["kernel_cu"] == CU_V1.strip()  # cu untouched, still the baseline
    # new baseline advanced so the next diff patches the patched file
    assert model._prev_py == action["kernel_py"]


def test_diff_mode_without_baseline_reprompts():
    model = ControllerModel(model_name="dummy", edit_mode="diff")
    patch = "*** Begin Patch\n*** Update File: kernel.py\n@@\n-a\n+b\n*** End Patch\n"
    with pytest.raises(FormatError):
        model._parse_actions(_resp(patch))


def test_diff_mode_unlocatable_context_reprompts_without_advancing_state():
    model = ControllerModel(model_name="dummy", edit_mode="diff")
    model._parse_actions(_resp(_full_response(PY_V1, CU_V1)))
    bad = "*** Begin Patch\n*** Update File: kernel.py\n@@\n-nonexistent line\n+whatever\n*** End Patch\n"
    with pytest.raises(FormatError):
        model._parse_actions(_resp(bad))
    assert model._prev_py == PY_V1.strip()  # baseline not corrupted by a failed patch


def test_diff_mode_full_rewrite_still_accepted_and_resets_baseline():
    model = ControllerModel(model_name="dummy", edit_mode="diff")
    model._parse_actions(_resp(_full_response(PY_V1, CU_V1)))
    py_v2 = PY_V1.replace("x + 1", "x + 99")
    action = model._parse_actions(_resp(_full_response(py_v2, CU_V1)))[0]
    assert "x + 99" in action["kernel_py"]
    assert model._prev_py == py_v2.strip()


def test_full_mode_ignores_a_patch_and_reprompts():
    # in full mode a *** Begin Patch with no fenced blocks is not diff-parsed -> missing-file reprompt
    model = ControllerModel(model_name="dummy")
    patch = "*** Begin Patch\n*** Update File: kernel.py\n@@\n-a\n+b\n*** End Patch\n"
    with pytest.raises(FormatError):
        model._parse_actions(_resp(patch))


def test_diff_mode_applies_two_blocks_touching_both_files():
    # regression for the dropped-block bug: block 1 edits kernel.cu, block 2 edits kernel.py; BOTH must land.
    model = ControllerModel(model_name="dummy", edit_mode="diff")
    model._parse_actions(_resp(_full_response(PY_V1, CU_V1)))
    patch = (
        "*** Begin Patch\n"
        "*** Update File: kernel.cu\n"
        "@@\n"
        "-void launch() {}\n"
        "+void launch() { /* fused */ }\n"
        "*** End Patch\n"
        "*** Begin Patch\n"
        "*** Update File: kernel.py\n"
        "@@\n"
        "-        return x + 1\n"
        "+        return x + 2\n"
        "*** End Patch\n"
    )
    action = model._parse_actions(_resp(patch))[0]
    assert "/* fused */" in action["kernel_cu"] and "void launch() {}" not in action["kernel_cu"]
    assert "return x + 2" in action["kernel_py"] and "return x + 1" not in action["kernel_py"]
    assert model._prev_cu == action["kernel_cu"] and model._prev_py == action["kernel_py"]


def test_diff_mode_two_blocks_on_same_file_thread_state():
    # two blocks edit different regions of kernel.cu; the second must apply onto the first's result.
    model = ControllerModel(model_name="dummy", edit_mode="diff")
    model._parse_actions(_resp(_full_response(PY_V1, CU_V1)))
    patch = (
        "*** Begin Patch\n"
        "*** Update File: kernel.cu\n"
        "@@\n"
        "-void launch() {}\n"
        "+void launch() { /*x*/ }\n"
        "*** End Patch\n"
        "*** Begin Patch\n"
        "*** Update File: kernel.cu\n"
        "@@\n"
        "-REGISTER_BINDING(launch)\n"
        "+REGISTER_BINDING(launch2)\n"
        "*** End Patch\n"
    )
    action = model._parse_actions(_resp(patch))[0]
    assert "/*x*/" in action["kernel_cu"] and "REGISTER_BINDING(launch2)" in action["kernel_cu"]
    assert action["kernel_py"] == PY_V1.strip()  # py untouched


def test_diff_mode_second_block_failure_does_not_advance_baseline():
    # if any block cannot locate its context the whole round re-prompts and the baseline stays intact.
    model = ControllerModel(model_name="dummy", edit_mode="diff")
    model._parse_actions(_resp(_full_response(PY_V1, CU_V1)))
    patch = (
        "*** Begin Patch\n"
        "*** Update File: kernel.py\n"
        "@@\n"
        "-        return x + 1\n"
        "+        return x + 2\n"
        "*** End Patch\n"
        "*** Begin Patch\n"
        "*** Update File: kernel.cu\n"
        "@@\n"
        "-this line does not exist\n"
        "+whatever\n"
        "*** End Patch\n"
    )
    with pytest.raises(FormatError):
        model._parse_actions(_resp(patch))
    assert model._prev_py == PY_V1.strip() and model._prev_cu == CU_V1.strip()


def _model_at_baseline(**kwargs) -> ControllerModel:
    model = ControllerModel(model_name="dummy", edit_mode="diff", **kwargs)
    model._parse_actions(_resp(_full_response(PY_V1, CU_V1)))
    return model


def test_diff_mode_keeps_a_full_rewrite_of_the_other_file_alongside_a_patch():
    # regression: a patch for one file plus a FULL fenced rewrite of the other silently dropped the
    # rewrite (the same class of bug as the dropped second patch block) and evaluated a stale kernel.cu.
    model = _model_at_baseline()
    action = model._parse_actions(_resp(
        "*** Begin Patch\n*** Update File: kernel.py\n@@\n-        return x + 1\n+        return x + 2\n"
        "*** End Patch\n"
        "and kernel.cu is rewritten from scratch:\n```cpp\n// BRAND NEW\nvoid launch() {}\n```\n"
    ))[0]
    assert "return x + 2" in action["kernel_py"]
    assert "// BRAND NEW" in action["kernel_cu"]
    assert model._prev_cu == action["kernel_cu"]


def test_diff_mode_patch_applies_on_top_of_a_full_rewrite_of_the_same_file():
    model = _model_at_baseline()
    action = model._parse_actions(_resp(
        "```cpp\n// V2\nvoid launch() {}\nREGISTER_BINDING(launch)\n```\n"
        "*** Begin Patch\n*** Update File: kernel.cu\n@@\n-REGISTER_BINDING(launch)\n"
        "+REGISTER_BINDING(launch2)\n*** End Patch\n"
    ))[0]
    assert "// V2" in action["kernel_cu"] and "REGISTER_BINDING(launch2)" in action["kernel_cu"]
    assert action["kernel_py"] == PY_V1.strip()  # py still the baseline


def test_diff_mode_full_blocks_win_over_a_quoted_patch_template():
    # regression: the system prompt's own `*** Begin Patch` template is fenced, so a model that
    # echoes it while sending BOTH full files used to have the ILLUSTRATIVE patch applied instead.
    model = _model_at_baseline()
    py_v2 = PY_V1.replace("x + 1", "x + 5")
    action = model._parse_actions(_resp(
        "for reference the patch format is:\n```\n*** Begin Patch\n*** Update File: kernel.py\n@@\n"
        "-        return x + 1\n+        return x + 999\n*** End Patch\n```\n"
        "this round I rewrite both files:\n" + _full_response(py_v2, CU_V1)
    ))[0]
    assert "x + 5" in action["kernel_py"] and "x + 999" not in action["kernel_py"]


def test_diff_mode_rejects_both_files_restated_unchanged_beside_a_patch():
    # full blocks beating a patch is right when they carry the round's real content, but if they merely
    # restate the baseline the change lived in the ignored patch -> the round would silently re-run the
    # previous kernel, so say so instead.
    model = _model_at_baseline(format_error_template="{{ error }}")
    with pytest.raises(FormatError) as exc:
        model._parse_actions(_resp(
            "*** Begin Patch\n*** Update File: kernel.py\n@@\n-        return x + 1\n+        return x + 2\n"
            "*** End Patch\nand for reference, the current files:\n" + _full_response(PY_V1, CU_V1)
        ))
    assert "restated BOTH files unchanged" in exc.value.messages[0]["content"]
    assert model._prev_py == PY_V1.strip()


def test_diff_mode_applies_a_patch_sent_beside_a_quote_of_the_prompt_template():
    # the prompt's own template is fenced and made of `<...>` placeholders; parsing it as a real block
    # failed to locate its context and threw away the genuine patch sent with it.
    model = _model_at_baseline()
    action = model._parse_actions(_resp(
        "recalling the format:\n```\n*** Begin Patch\n*** Update File: kernel.cu\n"
        "@@ <the enclosing function / a nearby unique line, to pin down WHERE this hunk goes>\n"
        " <a few UNCHANGED context lines, each prefixed by a single space>\n"
        "-<the exact line(s) to remove, prefixed by ->\n"
        "+<the new line(s) to add, prefixed by +>\n*** End Patch\n```\nmy actual change:\n"
        "*** Begin Patch\n*** Update File: kernel.cu\n@@\n-void launch() {}\n"
        "+void launch() { /* fused */ }\n*** End Patch\n"
    ))[0]
    assert "/* fused */" in action["kernel_cu"]
    assert action["kernel_py"] == PY_V1.strip()


def test_diff_mode_patch_from_round_one_may_carry_the_missing_file_as_a_full_block():
    model = ControllerModel(model_name="dummy", edit_mode="diff")
    with pytest.raises(FormatError):  # no baseline and no full block for either file
        model._parse_actions(_resp("*** Begin Patch\n*** Update File: kernel.py\n@@\n-a\n+b\n*** End Patch\n"))
    assert model._prev_py is None


def test_diff_mode_rejects_a_patch_that_changes_nothing():
    # two blocks where the second reverts the first: each hunk is valid, the net effect is not
    model = _model_at_baseline(format_error_template="{{ error }}")
    with pytest.raises(FormatError) as exc:
        model._parse_actions(_resp(
            "*** Begin Patch\n*** Update File: kernel.cu\n@@\n-void launch() {}\n+void launch() { /*x*/ }\n"
            "*** End Patch\n"
            "*** Begin Patch\n*** Update File: kernel.cu\n@@\n-void launch() { /*x*/ }\n+void launch() {}\n"
            "*** End Patch\n"
        ))
    assert "byte-identical" in exc.value.messages[0]["content"]
    assert model._prev_cu == CU_V1.strip()


def test_diff_mode_rejects_a_reconstruction_that_is_not_valid_python():
    # `_reindent` cannot recover an added block that is LESS indented than the context line the model
    # de-indented, so the rebase over-indents it. That used to reach write_candidate silently and only
    # surface as a confusing compile error; now the round is re-prompted with the syntax error instead.
    model = _model_at_baseline(format_error_template="{{ error }}")
    with pytest.raises(FormatError) as exc:
        model._parse_actions(_resp(
            "*** Begin Patch\n*** Update File: kernel.py\n@@\n return x + 1\n+\n+    def helper(self):\n"
            "+        return 0\n*** End Patch\n"
        ))
    assert "not valid Python" in exc.value.messages[0]["content"]
    assert model._prev_py == PY_V1.strip()  # baseline must not advance on a rejected reconstruction


def test_an_untagged_fence_is_never_read_as_kernel_py():
    # the prompt makes every round open with a fenced SOLUTION LOG and the model writes it as a bare
    # ```; counting that as python made the log text overwrite the diff baseline (diff mode) or land on
    # disk as a broken kernel.py (full mode).
    log = "```\nBEST-SO-FAR: 1.54x @ round 3 — vectorised loads\n```\n"
    model = ControllerModel(model_name="dummy", format_error_template="{{ error }}")
    with pytest.raises(FormatError) as exc:
        model._parse_actions(_resp(f"{log}\n```cpp\n{CU_V1}\n```\n"))
    assert "Missing the ```python block" in exc.value.messages[0]["content"]
    assert model._prev_cu is None
    # in diff mode the log must simply not be mistaken for a rewrite, so the patch applies to the baseline
    model = _model_at_baseline()
    action = model._parse_actions(_resp(
        f"{log}\n*** Begin Patch\n*** Update File: kernel.cu\n@@\n-void launch() {{}}\n"
        "+void launch() { /*x*/ }\n*** End Patch\n"
    ))[0]
    assert action["kernel_py"] == PY_V1.strip() and "/*x*/" in action["kernel_cu"]


@pytest.mark.parametrize(
    ("content", "needle"),
    [
        (_full_response("class ModelNew:\n  1.54x not python\n", CU_V1), "not valid Python"),
        (_full_response(PY_V1.strip(), CU_V1.strip()), "byte-identical"),
    ],
)
def test_full_mode_gates_a_rewrite_the_same_way_the_diff_path_does(content, needle):
    # a full rewrite reaches write_candidate directly, so nothing downstream would catch unparseable
    # Python or a verbatim restatement of the previous round
    model = _model_at_baseline(format_error_template="{{ error }}")
    with pytest.raises(FormatError) as exc:
        model._parse_actions(_resp(content))
    assert needle in exc.value.messages[0]["content"]
    assert (model._prev_py, model._prev_cu) == (PY_V1.strip(), CU_V1.strip())
