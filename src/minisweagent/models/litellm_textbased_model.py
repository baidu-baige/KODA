import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_text import format_observation_messages, parse_regex_actions


class LitellmTextbasedModelConfig(LitellmModelConfig):
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    """Regex to extract the action from the LM's output."""
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )
    """Template used when the LM's output is not in the expected format."""


class LitellmTextbasedModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=LitellmTextbasedModelConfig, **kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        completion_kwargs = self.config.model_kwargs | kwargs
        # Streaming keeps bytes flowing so proxies don't drop long generations on their non-stream
        # response timeout. But litellm reuses a process-wide sync HTTP client, and concurrent
        # long-lived streams race on its socket fds ([Errno 9] Bad file descriptor). Give each
        # streaming call its own client so the connection pool is never shared across threads.
        client = None
        # The openai provider uses per-config-cached (thread-safe) openai.OpenAI clients and
        # rejects an HTTPHandler (no .api_key attribute); only HTTPHandler-based providers need
        # their own per-call client to avoid the shared-socket race ([Errno 9]).
        if (completion_kwargs.get("stream") and "client" not in completion_kwargs
                and not self.config.model_name.startswith("openai/")):
            client = HTTPHandler(timeout=completion_kwargs.get("timeout"))
            completion_kwargs["client"] = client
        try:
            response = litellm.completion(model=self.config.model_name, messages=messages, **completion_kwargs)
            # Reassemble the chunks into a normal ModelResponse for parsing/cost.
            if completion_kwargs.get("stream"):
                response = litellm.stream_chunk_builder(list(response), messages=messages)
            return response
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
        finally:
            if client is not None:
                client.close()

    def _parse_actions(self, response: dict) -> list[dict]:
        """Parse actions from the model response. Raises FormatError if not exactly one action."""
        content = response.choices[0].message.content or ""
        return parse_regex_actions(
            content,
            action_regex=self.config.action_regex,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response.choices[0].finish_reason},
        )

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into observation messages."""
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
