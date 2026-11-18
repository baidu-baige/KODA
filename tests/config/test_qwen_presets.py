"""H/B presets share the Qwen recipe and render the actual compile target."""
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, StrictUndefined, meta

from minisweagent.agents.kernel_controller import ControllerConfig
from minisweagent.config import builtin_config_dir
from minisweagent.models.controller_model import ControllerModelConfig


def load(name):
    return yaml.safe_load((builtin_config_dir / name).read_text())


def test_qwen_presets_differ_only_in_compile_target():
    h = load("controller_h.yaml")
    b = load("controller_b.yaml")
    assert h["agent"].pop("compile_arch") == "9.0"
    assert b["agent"].pop("compile_arch") == "10.0a"
    assert h == b
    agent = h["agent"]
    assert (agent["num_rounds"], agent["max_submit_eval"], agent["wall_time_limit_seconds"], agent["token_limit"]) == (70, 50, 7200, 262144)
    model = h["model"]
    assert model["model_name"] == "openai/ckpt140"
    assert model["model_kwargs"]["temperature"] == 0.6
    assert model["model_kwargs"]["extra_body"]["top_k"] == 20
    assert model["model_kwargs"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.parametrize(("name", "arch", "gpu"), [
    ("controller_h.yaml", "9.0", "H20"),
    ("controller_b.yaml", "10.0a", "B200"),
])
def test_preset_schema_and_prompt_render_for_target(name, arch, gpu):
    config = load(name)
    ControllerConfig(**config["agent"])
    ControllerModelConfig(**config["model"])
    engine = Environment(undefined=StrictUndefined)
    source = config["agent"]["system_template"]
    variables = {k: "" for k in meta.find_undeclared_variables(engine.parse(source))}
    variables.update(config["agent"], compile_arch=arch, gpu_specs=gpu, cutlass3x_example="// target-matched example")
    rendered = engine.from_string(source).render(**variables)
    assert f"configured CUDA compile target is `{arch}`" in rendered
    assert "the compile target is `arch = sm_90a`" not in rendered
    assert "CUTLASS 3.x Hopper GEMM reference" not in rendered
    repo = Path(__file__).resolve().parents[2]
    assert (repo / "configs" / name).read_bytes() == (builtin_config_dir / name).read_bytes()
