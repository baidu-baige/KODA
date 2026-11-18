import json

from minisweagent.run.benchmarks.kernelbench import normalize_instance, write_instance_meta


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_write_instance_meta_appends_and_last_row_wins(tmp_path):
    # problem_id / code are in the input but must not be written
    first = [
        normalize_instance({"name": "b_gemm", "level": 2, "problem_id": 7, "code": "x", "op_type": "gemm"}),
        normalize_instance({"name": "a_gelu", "level": 1, "problem_id": 3, "code": "y", "op_type": "fusion"}),
    ]
    write_instance_meta(tmp_path, first)
    # second run into the same dir appends; missing op_type column -> null
    write_instance_meta(tmp_path, [normalize_instance({"name": "b_gemm", "level": 2, "code": "z"})])
    rows = _read(tmp_path / "instance_meta.jsonl")
    assert rows == [
        {"instance_id": "level_2_b_gemm", "name": "b_gemm", "level": 2, "op_type": "gemm"},
        {"instance_id": "level_1_a_gelu", "name": "a_gelu", "level": 1, "op_type": "fusion"},
        {"instance_id": "level_2_b_gemm", "name": "b_gemm", "level": 2, "op_type": None},
    ]
    # same lookup as data-forge sft_data/instance_meta.py::load_optype_map
    assert {r["instance_id"]: r["op_type"] for r in rows} == {"level_2_b_gemm": None, "level_1_a_gelu": "fusion"}


def test_write_instance_meta_after_torn_line_starts_fresh_line(tmp_path):
    # a previous append was interrupted mid-line: only that half line may be lost, not the new row
    (tmp_path / "instance_meta.jsonl").write_text('{"instance_id": "level_1_a", "na')
    write_instance_meta(tmp_path, [normalize_instance({"name": "b", "level": 1, "code": "", "op_type": "gemm"})])
    assert (tmp_path / "instance_meta.jsonl").read_text().splitlines() == [
        '{"instance_id": "level_1_a", "na',
        '{"instance_id": "level_1_b", "name": "b", "level": 1, "op_type": "gemm"}',
    ]
