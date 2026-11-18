class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class LimitsTokenExceeded(LimitsExceeded):
    """Raised when the agent has exceeded its cumulative token limit."""


class TimeExceeded(LimitsExceeded):
    """Raised when the agent has exceeded its wall-clock time limit."""


class UserInterruption(InterruptAgentFlow):
    """Raised when the user interrupts the agent."""


class FormatError(InterruptAgentFlow):
    """Raised when the LM's output is not in the expected format."""


class SandboxInfraError(Exception):
    """Raised when a sandbox session is unrecoverable (e.g. its pod was evicted).

    Not an InterruptAgentFlow: it must propagate past the agent loop up to the batch
    runner, which re-runs the whole problem on a fresh session rather than retrying the
    dead session's command_exec.
    """
