"""Public controller-only release boundaries and portable defaults."""
import inspect
import importlib.util
from pathlib import Path

import yaml
import pytest

from minisweagent.config import builtin_config_dir
from minisweagent.models import get_model_class
from minisweagent.models.controller_model import ControllerModel
from minisweagent.run.benchmarks.kernelbench import main, process_instance
from minisweagent.run.benchmarks.utils.sandbox_client import SandboxClientConfig


def test_only_controller_and_its_prompt_are_published():
    assert "agent_type" not in inspect.signature(main).parameters
    assert "agent_type" not in inspect.signature(process_instance).parameters
    assert sorted(p.name for p in builtin_config_dir.rglob("*.yaml")) == ["controller_b.yaml", "controller_h.yaml"]
    config = yaml.safe_load((builtin_config_dir / "controller_h.yaml").read_text())
    assert config["agent"]["eval_server_token"] == ""
    assert get_model_class("openai/example") is ControllerModel
    assert get_model_class("openai/example", config["model"]["model_class"]) is ControllerModel


def test_portable_sandbox_defaults():
    config = SandboxClientConfig()
    assert config.base_url == "http://127.0.0.1:18766"
    assert config.image.startswith("ghcr.io/baidu-baige/")


def test_no_retired_local_imports_or_internal_release_configuration():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/audit_release.py"
    spec = importlib.util.spec_from_file_location("koda_release_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.audit(root)
    assert report["findings"] == []


def test_audit_reports_locations_without_echoing_values(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("koda_release_audit", root / "scripts/audit_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "src/minisweagent/config"
    config.mkdir(parents=True)
    (config / "controller_h.yaml").write_text("{}")
    (config / "controller_b.yaml").write_text("{}")
    value = ".".join(["10", "1", "2", "3"])
    (tmp_path / "bad.txt").write_text("endpoint=" + value)
    report = module.audit(tmp_path)
    assert any(x["file"] == "bad.txt" and x["kind"] == "private_network_address" for x in report["findings"])
    assert value not in str(report)


@pytest.mark.parametrize(
    "value,kind",
    [
        ("-".join(["wh", "test"]), "cloud_mount"),
        ("-".join(["ainf", "cfs"]), "cloud_mount"),
        ("cache_" + "-".join(["ainf", "cfs"]), "cloud_mount"),
        ("https://host." + "baidu" + ".com/api", "internal_registry_or_domain"),
        ("registry." + "baidu" + "-int.com/image", "internal_registry_or_domain"),
        ("/home/" + "users/example/output", "internal_home_path"),
        ("http://" + ".".join(["10", "1", "2", "3"]), "private_network_address"),
        ("http://[" + "fd" + "00::12]/api", "private_network_address"),
        ("http://[::ffff:" + ".".join(["192", "168", "1", "2"]) + "]/api", "private_network_address"),
        ("api_key: " + "synthetic-credential-for-audit", "hardcoded_credential"),
        ('MODEL_API_KEY="' + "synthetic-credential-for-audit" + '"', "hardcoded_credential"),
        ("export MODEL_API_KEY=" + "synthetic-credential-for-audit", "hardcoded_credential"),
        ('{"api_key": "' + "synthetic-credential-for-audit" + '"}', "hardcoded_credential"),
        ("https://example:" + "synthetic-password" + "@example.org", "url_credentials"),
        ("Authorization: Bearer " + "a" * 30, "authorization_literal"),
        ("gh" + "p_" + "a" * 30, "credential"),
        ("AK" + "IA" + "A" * 16, "credential"),
        ("-----BEGIN " + "ENCRYPTED PRIVATE KEY-----", "credential"),
    ],
)
def test_audit_detects_sensitive_literals_and_redacts_values(value, kind):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("koda_release_audit", root / "scripts/audit_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.sensitive_locations("header\n" + value)
    assert any(item["kind"] == kind and item["line"] == 2 for item in report)
    assert value not in str(report)


@pytest.mark.parametrize("value", [
    "api_key: EMPTY", 'api_key: ""', "api_key: REPLACE_MODEL_API_KEY",
    "api_key: str", "export MODEL_API_KEY=${MODEL_API_KEY}",
    'MODEL_API_KEY="${MODEL_API_KEY}"', 'api_key=os.environ["MODEL_API_KEY"]',
    "http://127.0.0.1:18766", "http://[::1]:18766", "http://0.0.0.0:18766",
    "https://github.com/baidu-baige/KODA", "ghcr.io/baidu-baige/koda-agent:example",
])
def test_audit_allows_runtime_configuration_and_public_endpoints(value):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("koda_release_audit", root / "scripts/audit_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.sensitive_locations(value) == []
