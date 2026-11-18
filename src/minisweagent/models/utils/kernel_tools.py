"""Controller diagnostics execute in CPU sessions; tools request GPU work through the Master."""
TOOL_DEVICES = {"bash": "cpu", "eval": "cpu", "profile": "cpu", "diff": "cpu"}
