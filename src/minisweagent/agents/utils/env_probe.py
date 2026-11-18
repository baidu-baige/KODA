"""Env version / triton API / GPU specs probe shared by kernel-loop and opencode agents.

Both routes want to tell the model (in its very first turn) what versions it is running
against, so it does not waste rounds probing itself. The probe runs a small python script
and parses three tagged lines:

- ``VERSIONS:``  Python / torch / triton / transformers / CUDA versions
- ``GPU_SPECS:`` GPU name / cc / SM count / register file & shared-mem limits / L2 / device memory (device-dependent)
- ``TRITON_HINTS:``  actually-available ``tl`` API signatures (never goes stale)

The probe is executed through a caller-supplied ``execute`` callable so each route can use
its own sandbox channel:

- kernel-loop: ``env.execute(..., device="gpu")`` → real GPU worker (versions + GPU specs)
- opencode:    CPU pod ``_exec`` → same-image CPU pod (versions yes, GPU_SPECS empty)

Results are cached per image key (shared across both routes for the same image), and a
failed probe degrades to empty strings so it never blocks a run.
"""

import re
import time

# Normalize opaque OEM identifiers in GPU descriptions to the public model name.
_ANONYMOUS_B200_NAME = re.compile(r"(?<![\w-])[A-Z]{2}-[A-Z]{2}-[A-Z]{3}\d-[A-Z](?![\w-])")


def normalize_gpu_name(text: str) -> str:
    return _ANONYMOUS_B200_NAME.sub("NVIDIA B200", text)


_PROBE_SCRIPT = (
    "import platform\n"
    "v = [f'Python {platform.python_version()}']\n"
    "for mod in ['torch', 'triton', 'transformers']:\n"
    "    try:\n"
    "        m = __import__(mod)\n"
    "        v.append(f'{mod} {getattr(m, \"__version__\", \"?\")}')\n"
    "    except Exception:\n"
    "        pass\n"
    "try:\n"
    "    import torch\n"
    "    if torch.version.cuda:\n"
    "        v.append(f'CUDA {torch.version.cuda}')\n"
    "except Exception:\n"
    "    pass\n"
    "print('VERSIONS:' + ' / '.join(v))\n"
    "try:\n"
    "    import torch\n"
    "    p = torch.cuda.get_device_properties(0)\n"
    "    regs = getattr(p, 'regs_per_multiprocessor', 0) or 0\n"
    "    maxthr = getattr(p, 'max_threads_per_multi_processor', 0) or 0\n"
    "    smem = getattr(p, 'shared_memory_per_multiprocessor', 0) or 0\n"
    "    optin = getattr(p, 'shared_memory_per_block_optin', 0) or 0\n"
    "    l2 = getattr(p, 'L2_cache_size', 0) or 0\n"
    "    print('GPU_SPECS:' + (\n"
    "        f'{p.name} | compute capability {p.major}.{p.minor} | '\n"
    "        f'{p.multi_processor_count} SM | {regs} regs/SM | {maxthr} max threads/SM | '\n"
    "        f'{round(smem/1024, 1)} KB smem/SM | {round(optin/1024, 1)} KB opt-in smem/block | '\n"
    "        f'{round(l2/1024**2, 1)} MB L2 | {round(p.total_memory/1024**3, 1)} GB device memory'\n"
    "    ))\n"
    "except Exception:\n"
    "    print('GPU_SPECS:')\n"
    "try:\n"
    "    import triton.language as tl\n"
    "    import inspect\n"
    "    avail = []\n"
    "    for name in ['range', 'dot', 'associative_scan', 'trans', 'cumsum', 'sort']:\n"
    "        if hasattr(tl, name):\n"
    "            obj = getattr(tl, name)\n"
    "            try:\n"
    "                params = list(inspect.signature(obj).parameters.keys())\n"
    "                avail.append(name + '(' + ', '.join(params) + ')')\n"
    "            except Exception:\n"
    "                avail.append(name)\n"
    "    missing = [n for n in ['libdevice'] if not hasattr(tl, n)]\n"
    "    import triton\n"
    "    if not hasattr(triton, 'num_programs'):\n"
    "        missing.append('triton.num_programs')\n"
    "    parts = ['Available: ' + ', '.join(avail)]\n"
    "    if missing:\n"
    "        parts.append('NOT available: ' + ', '.join(missing))\n"
    "    print('TRITON_HINTS:' + ' | '.join(parts))\n"
    "except Exception:\n"
    "    print('TRITON_HINTS:')\n"
)

# Public alias so the kernel-loop route can reuse the shared base probe (single source of truth)
# and append its CUTLASS-only blocks instead of keeping a drifting inline copy.
BASE_PROBE_SCRIPT = _PROBE_SCRIPT

_ENV_INFO_CACHE: dict[str, dict] = {}
"""Successful probe results, keyed by image. A 202-instance batch on the same image probes once.
Empty results never enter this table (see _PROBE_FAILED_AT)."""

_PROBE_FAILED_AT: dict[str, float] = {}
"""monotonic timestamp of the last failed probe per image; failures are quieted for _PROBE_RETRY_AFTER_S."""

_PROBE_RETRY_AFTER_S = 300
"""Quiet period after a failed probe. Both extremes are wrong: caching a failure forever loses
version hints for the whole batch on one sandbox blip; retrying every instance wastes GPU slots
if the image genuinely has no python3. 5 minutes is the middle ground — self-heals after blips."""

_PROBE_TIMEOUT_S = 60
"""Independent probe timeout. Must not inherit config.timeout (cpu_gpu is 6000s): the probe holds
a GPU-pool slot the eval itself needs, one stuck `import torch` must not stall the whole batch."""


def _parse(output: str) -> dict:
    """Parse the probe script's tagged output into {versions, triton_api_hints, gpu_specs}."""
    result = {"versions": "", "triton_api_hints": "", "gpu_specs": ""}
    for line in output.strip().splitlines():
        if line.startswith("VERSIONS:"):
            result["versions"] = line[len("VERSIONS:"):].strip()
        elif line.startswith("TRITON_HINTS:"):
            hints = line[len("TRITON_HINTS:"):].strip()
            if hints:
                result["triton_api_hints"] = hints
        elif line.startswith("GPU_SPECS:"):
            specs = line[len("GPU_SPECS:"):].strip()
            if specs:
                result["gpu_specs"] = normalize_gpu_name(specs)
    return result


def probe_env_info(execute, cache_key: str) -> dict:
    """Run the probe script once per image and return parsed env info.

    ``execute(command: str, timeout: int) -> {"returncode": int, "output": str}`` is the caller's
    sandbox channel (loop: env.execute(device="gpu"); opencode: CPU pod _exec). ``cache_key`` is
    the image name the probe ran on (shared cache across routes for the same image).
    """
    if cache_key in _ENV_INFO_CACHE:
        return _ENV_INFO_CACHE[cache_key]
    if time.monotonic() - _PROBE_FAILED_AT.get(cache_key, float("-inf")) < _PROBE_RETRY_AFTER_S:
        return {"versions": "", "triton_api_hints": "", "gpu_specs": ""}

    try:
        out = execute(f"python3 <<'PYEOF'\n{_PROBE_SCRIPT}PYEOF", timeout=_PROBE_TIMEOUT_S)
    except Exception:  # version hints are a prompt nicety; a sandbox blip must not sink the batch
        _PROBE_FAILED_AT[cache_key] = time.monotonic()
        return {"versions": "", "triton_api_hints": "", "gpu_specs": ""}

    if out.get("returncode") != 0:
        _PROBE_FAILED_AT[cache_key] = time.monotonic()
        return {"versions": "", "triton_api_hints": "", "gpu_specs": ""}

    result = _parse(out.get("output", ""))
    if result["versions"] or result["triton_api_hints"] or result["gpu_specs"]:
        _ENV_INFO_CACHE[cache_key] = result
    else:
        _PROBE_FAILED_AT[cache_key] = time.monotonic()
    return result
