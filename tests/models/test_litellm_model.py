from unittest.mock import MagicMock, patch

import pytest
from litellm.types.utils import Choices, CompletionTokensDetailsWrapper, Message, ModelResponse, Usage

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class TestLitellmModelConfig:
    def test_default_format_error_template(self):
        assert LitellmModelConfig(model_name="test").format_error_template == "{{ error }}"


def _mock_litellm_response(tool_calls):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = tool_calls
    mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    mock_response.model_dump.return_value = {}
    return mock_response


class TestLitellmModel:
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_bash_tool(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        model.query([{"role": "user", "content": "test"}])

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["tools"] == [BASH_TOOL]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_valid_tool_call(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        result = model.query([{"role": "user", "content": "list files"}])
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_no_tool_calls_raises(self, mock_cost, mock_completion):
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_finish_reason_threaded_into_format_error_template(self, mock_cost, mock_completion):
        """The response finish_reason is exposed to format_error_template via template_kwargs, so a
        config can report a max_tokens truncation instead of the misleading "no tool call" error."""
        response = _mock_litellm_response(None)
        response.choices[0].finish_reason = "length"
        mock_completion.return_value = response
        mock_cost.return_value = 0.001

        model = LitellmModel(
            model_name="gpt-4",
            format_error_template="{% if finish_reason == 'length' %}cut off{% else %}{{ error }}{% endif %}",
        )
        with pytest.raises(FormatError) as exc:
            model.query([{"role": "user", "content": "test"}])
        assert exc.value.messages[0]["content"] == "cut off"

    def test_format_observation_messages(self):
        model = LitellmModel(model_name="gpt-4", observation_template="{{ output.output }}")
        message = {"extra": {"actions": [{"command": "echo test", "tool_call_id": "call_1"}]}}
        outputs = [{"output": "test output", "returncode": 0}]
        result = model.format_observation_messages(message, outputs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"

    def test_format_observation_messages_no_actions(self):
        model = LitellmModel(model_name="gpt-4")
        result = model.format_observation_messages({"extra": {}}, [])
        assert result == []


_TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}


def _cot_response(cot, thinking_blocks=None, reasoning_tokens=None):
    message = Message(content="ok", role="assistant", reasoning_content=cot, tool_calls=[_TOOL_CALL])
    if thinking_blocks:
        message = Message(content="ok", role="assistant", tool_calls=[_TOOL_CALL], thinking_blocks=thinking_blocks)
    usage = None
    if reasoning_tokens is not None:
        usage = Usage(
            completion_tokens=100,
            prompt_tokens=10,
            total_tokens=110,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=reasoning_tokens),
        )
    return ModelResponse(choices=[Choices(message=message)], model="gpt-4", usage=usage)


class TestEmptyCotRetry:
    """A turn whose reply has no visible reasoning is re-rolled; the discarded replies must not reach
    the agent but must still be charged, otherwise the paid-for latency/cost silently disappears."""

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_retries_until_cot_arrives_and_sums_cost(self, mock_cost, mock_completion):
        mock_completion.side_effect = [_cot_response(None), _cot_response("  "), _cot_response("plan")]
        mock_cost.return_value = 0.01

        result = LitellmModel(model_name="gpt-4", empty_cot_retries=2).query([{"role": "user", "content": "x"}])

        assert mock_completion.call_count == 3
        assert result["reasoning_content"] == "plan"
        assert result["extra"]["cost"] == 0.03
        assert result["extra"]["actions"] == [{"command": "ls", "tool_call_id": "call_1"}]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_gives_up_after_budget_and_keeps_last_reply(self, mock_cost, mock_completion):
        mock_completion.return_value = _cot_response(None)
        mock_cost.return_value = 0.01

        result = LitellmModel(model_name="gpt-4", empty_cot_retries=2).query([{"role": "user", "content": "x"}])

        assert mock_completion.call_count == 3
        assert not result.get("reasoning_content")

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_disabled_by_default_and_by_zero(self, mock_cost, mock_completion):
        mock_completion.return_value = _cot_response(None)
        mock_cost.return_value = 0.01

        LitellmModel(model_name="gpt-4").query([{"role": "user", "content": "x"}])

        assert mock_completion.call_count == 1

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_thinking_blocks_count_as_cot(self, mock_cost, mock_completion):
        blocks = [{"type": "thinking", "thinking": "plan"}]
        mock_completion.return_value = _cot_response(None, thinking_blocks=blocks)
        mock_cost.return_value = 0.01

        result = LitellmModel(model_name="gpt-4", empty_cot_retries=2).query([{"role": "user", "content": "x"}])

        assert mock_completion.call_count == 1
        assert result["thinking_blocks"] == blocks

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_redacted_thinking_block_counts_as_cot(self, mock_cost, mock_completion):
        """A redacted block carries no readable text, but the model did reason — re-rolling it would
        only burn calls."""
        blocks = [{"type": "redacted_thinking", "data": "encrypted"}]
        mock_completion.return_value = _cot_response(None, thinking_blocks=blocks)
        mock_cost.return_value = 0.01

        LitellmModel(model_name="gpt-4", empty_cot_retries=2).query([{"role": "user", "content": "x"}])

        assert mock_completion.call_count == 1

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_reported_reasoning_tokens_count_as_cot(self, mock_cost, mock_completion):
        """A backend that strips the reasoning text still reports reasoning_tokens; treating that as
        empty would retry every single round for nothing."""
        mock_completion.return_value = _cot_response(None, reasoning_tokens=80)
        mock_cost.return_value = 0.01

        LitellmModel(model_name="gpt-4", empty_cot_retries=2).query([{"role": "user", "content": "x"}])

        assert mock_completion.call_count == 1

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_zero_reported_reasoning_still_retries(self, mock_cost, mock_completion):
        mock_completion.return_value = _cot_response(None, reasoning_tokens=0)
        mock_cost.return_value = 0.01

        LitellmModel(model_name="gpt-4", empty_cot_retries=1).query([{"role": "user", "content": "x"}])

        assert mock_completion.call_count == 2
