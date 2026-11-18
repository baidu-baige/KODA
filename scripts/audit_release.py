"""Check the release tree without printing potentially sensitive matched values."""

from __future__ import annotations

import argparse
import ast
import json
import ipaddress
import re
from pathlib import Path


def sensitive_locations(text: str) -> list[dict]:
    """Find sensitive literals without returning their values."""
    patterns = {
        "cloud_mount": r"(?i)(?:" + "wh" + r"-test|" + "ainf" + r"-cfs)",
        "internal_registry_or_domain": r"(?i)(?:iregistry\.[\w.-]+|[\w.-]+\." + r"(?:baidu" + r"-int\.com|baidu\.com))",
        "internal_home_path": r"/(?:home/users)/[^\s\"'<>]+",
        "private_network_address": r"(?<![\w.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|169\.254(?:\.\d{1,3}){2})(?![\w.])",
        "credential": r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{25,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{32,}|glpat-[A-Za-z0-9_-]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{35}|xox[baprs]-[A-Za-z0-9-]{20,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|kb-cpu-gpu-token-[\w-]+|-----BEGIN (?:RSA |OPENSSH |EC |ENCRYPTED )?PRIVATE KEY-----)",
        "authorization_literal": r"(?i)\bAuthorization[\"']?\s*[:=]\s*[\"']?(?:Bearer|Basic)[ \t]+[A-Za-z0-9_.+/=-]{20,}",
        "url_credentials": r"(?i)[a-z][a-z0-9+.-]*://[^\s/\"'<>:@]+:[^\s/\"'<>@]+@",
    }
    findings = []

    def add(kind: str, offset: int) -> None:
        findings.append({"line": text.count("\n", 0, offset) + 1, "kind": kind})

    for kind, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            add(kind, match.start())
    # Credentials without a vendor-specific prefix still need checking in config
    # literals. Environment references and explicit placeholders are not secrets.
    field = r"(?:[a-z][a-z0-9]*_)*(?:api[_-]?key|token|password|passwd|secret|access[_-]?key)"
    quoted = r"(?i)(?<![\w])[\"']?" + field + r"[\"']?\s*[:=]\s*([\"'])([^\"'\r\n]+)\1"
    plain = r"(?im)^\s*(?:export\s+)?" + field + r"\s*[:=]\s*([A-Za-z0-9_./+~=-]+)\s*(?:#.*)?$"
    for pattern, group in [(quoted, 2), (plain, 1)]:
        for match in re.finditer(pattern, text):
            value = match.group(group)
            placeholder = (
                value.upper() in {"EMPTY", "NONE", "NULL"}
                or value.startswith(("REPLACE_", "YOUR_", "$", "<"))
            )
            type_annotation = group == 1 and value in {"str", "int", "bool", "bytes"}
            if not (placeholder or type_annotation) and (group == 1 or len(value.strip()) >= 8):
                add("hardcoded_credential", match.start())
    # Private/link-local IPv6 (including IPv4-mapped IPv6) is not covered by
    # the IPv4 regex. Loopback and unspecified bind addresses are valid defaults.
    for match in re.finditer(r"(?<![\w:])(?:[A-Fa-f0-9]{0,4}:){2,}[A-Fa-f0-9:.%]+", text):
        try:
            address = ipaddress.ip_address(match.group().split("%", 1)[0])
        except ValueError:
            continue
        address = getattr(address, "ipv4_mapped", None) or address
        if not (address.is_loopback or address.is_unspecified) and (address.is_private or address.is_link_local):
            add("private_network_address", match.start())
    return findings


def audit(root: Path) -> dict:
    skipped = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "bin", ".cache", "build", "dist"}
    findings = []
    files = 0
    modules = {}
    sources = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or skipped.intersection(path.relative_to(root).parts):
            continue
        files += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            findings.append({"file": str(path.relative_to(root)), "kind": "unexpected_binary"})
            continue
        for finding in sensitive_locations(text):
            findings.append({"file": str(path.relative_to(root)), **finding})
        if path.suffix == ".py":
            tree = ast.parse(text, filename=str(path))
            sources.append((path, tree))
            if root / "src" in path.parents:
                name = ".".join(path.relative_to(root / "src").with_suffix("").parts)
                modules[name[:-9] if name.endswith(".__init__") else name] = path
    for path, tree in sources:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("minisweagent")
                and node.module not in modules
            ):
                findings.append(
                    {
                        "file": str(path.relative_to(root)),
                        "line": node.lineno,
                        "kind": "missing_local_import",
                        "module": node.module,
                    }
                )
    for retired in [
        "src/llm-proxy",
        "gpu_opencode",
        "src/minisweagent/agents/kernel_loop.py",
        "src/minisweagent/agents/kernel_opencode.py",
    ]:
        if (root / retired).exists():
            findings.append({"file": retired, "kind": "retired_component_present"})
    configs = sorted(str(p.relative_to(root)) for p in (root / "src/minisweagent/config").rglob("*.yaml"))
    if configs != ["src/minisweagent/config/controller_b.yaml", "src/minisweagent/config/controller_h.yaml"]:
        findings.append({"kind": "unexpected_packaged_configs", "files": configs})
    return {"scanned_files": files, "findings": findings, "ok": not findings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
