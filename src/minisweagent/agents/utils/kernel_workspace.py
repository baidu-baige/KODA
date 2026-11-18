"""Helpers for reading kernel workspace files through an Environment.

The kernel agents never touch the workspace with local filesystem calls: the
workspace may live on a remote Kernel Sandbox CPU pod, and the only portable way
to reach it is the environment's execute(). With LocalEnvironment these commands
run against the local workspace directory, so the same agents work in both setups.
"""

import shlex

from minisweagent import Environment


def read_workspace_file(env: Environment, path: str) -> str:
    """Read a workspace-relative file via the environment; raise RuntimeError if unreadable."""
    output = env.execute({"command": f"cat {shlex.quote(path)}"})
    if output["returncode"] != 0:
        raise RuntimeError(f"Could not read workspace file {path!r}: {output['output']}")
    return output["output"]
