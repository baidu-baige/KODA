"""Conversation, budget and serialization runtime used by the controller."""

import json
import logging
import time
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, LimitsTokenExceeded, TimeExceeded
from minisweagent.utils.log import logger
from minisweagent.utils.metrics import COLLECTOR
from minisweagent.utils.readable_yaml import render_trajectory_yaml
from minisweagent.utils.serialize import recursive_merge


class AgentRuntimeConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    token_limit: int = 262144
    """Stop agent once the current conversation size (the latest request's total_tokens = full prompt +
    that reply's completion) exceeds this. This tracks the *current* context size, NOT a cumulative sum
    across calls (the prompt already carries all history, so the latest total IS the trajectory size).
    0 means no limit."""
    is_dynamic_completion: bool = True
    """Dynamically shrink each request's max_completion_tokens to fit the remaining token_limit budget.
    When on (and token_limit>0 and the model has a base cap B = model_kwargs.max_completion_tokens), every
    call sends max_completion_tokens = min(B, token_limit - n_tokens - dynamic_completion_margin_tokens):
    full cap early, tapering to the leftover room as the trajectory grows, so a reply never pushes context
    past token_limit. No-op when the model has no configured max_completion_tokens."""
    dynamic_completion_margin_tokens: int = 10240
    """Safety headroom (tokens) subtracted when computing the dynamic cap, to absorb the next prompt's growth."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class AgentRuntime:
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type = AgentRuntimeConfig,
        instance_id: str = "",
        **kwargs,
    ):
        """See the `AgentRuntimeConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.instance_id = instance_id
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_tokens = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "n_tokens": self.n_tokens,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def _record_format_error(self, e: FormatError, seconds: float) -> None:
        """Surface a rejected model call. It is raised while parsing the reply, so the step never finishes
        and never records a round sample — the model call is paid for (a full, reasoning-heavy latency and
        its tokens) but would otherwise leave no trace outside the final exit status."""
        COLLECTOR.record_format_error(seconds)
        self.logger.warning(
            f"{self.instance_id or 'agent'} format error {self.n_consecutive_format_errors}/"
            f"{self.config.max_consecutive_format_errors} after {round(seconds, 3)}s (no round consumed): "
            + " ".join(str(m.get("content", "")) for m in e.messages)[:400]
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            started = time.time()
            try:
                self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                self.n_consecutive_format_errors += 1
                self._record_format_error(e, time.time() - started)
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.token_limit <= self.n_tokens:
            raise LimitsTokenExceeded(
                {
                    "role": "exit",
                    "content": "LimitsTokenExceeded",
                    "extra": {"exit_status": "LimitsTokenExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        extra_kwargs: dict = {}
        if self.config.is_dynamic_completion and self.config.token_limit:
            # base cap = the model's statically configured max_completion_tokens; shrink it to the room left
            model_kwargs = getattr(getattr(self.model, "config", None), "model_kwargs", None) or {}
            base = model_kwargs.get("max_completion_tokens")
            if base:
                room = self.config.token_limit - self.n_tokens - self.config.dynamic_completion_margin_tokens
                if room <= 0:
                    # No room left for a reply: the next prompt (already past n_tokens, which lags one turn)
                    # plus any completion would overrun the context window. Stop now instead of sending a
                    # doomed request; extract_best still salvages the best turn.
                    logger.info(
                        f"{self.instance_id or 'kernel'} room<=0, LimitsTokenExceeded: step={self.n_calls} "
                        f"token_limit={self.config.token_limit} "
                        f"n_tokens={self.n_tokens} margin={self.config.dynamic_completion_margin_tokens} room={room}"
                    )
                    raise LimitsTokenExceeded(
                        {
                            "role": "exit",
                            "content": "LimitsTokenExceeded",
                            "extra": {"exit_status": "LimitsTokenExceeded", "submission": ""},
                        }
                    )
                extra_kwargs["max_completion_tokens"] = min(base, room)
                if base > room:
                    logger.info(
                        f"{self.instance_id or 'kernel'} max_completion_tokens capped to room: step={self.n_calls} "
                        f"token_limit={self.config.token_limit} n_tokens={self.n_tokens} "
                        f"margin={self.config.dynamic_completion_margin_tokens} base={base} -> {room}"
                    )
        self.n_calls += 1
        message = self.model.query(self.messages, **extra_kwargs)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.n_tokens = self._call_total_tokens(message)
        self.add_messages(message)
        return message

    @staticmethod
    def _call_total_tokens(message: dict) -> int:
        """This call's total tokens (prompt + completion) from ``extra.response.usage.total_tokens``;
        0 when usage is absent (e.g. streaming without include_usage) so the token limit stays inert."""
        return (message.get("extra", {}).get("response", {}).get("usage") or {}).get("total_tokens", 0) or 0

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                    "total_tokens": self.n_tokens,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            # trajectory.json 每一步保存（含异常/中断路径的 finally）都同步
            # 刷新同目录 readable yaml 镜像：运行中途崩掉时仍有最新完成轮的
            # 人读产物（run 结束后的 export 只做幂等收尾，两者内容一致）。
            if path.name == "trajectory.json":
                try:
                    (path.parent / "trajectory.readable.yaml").write_text(
                        render_trajectory_yaml(path), encoding="utf-8"
                    )
                except Exception:
                    pass
        return data
