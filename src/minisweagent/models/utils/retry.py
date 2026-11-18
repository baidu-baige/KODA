"""Retry utility for model queries."""

import logging
import os

from tenacity import Retrying, before_sleep_log, stop_after_attempt, wait_exponential

from minisweagent.utils.metrics import COLLECTOR


def retry(*, logger: logging.Logger, abort_exceptions: list[type[Exception]]) -> Retrying:
    """Thin wrapper around tenacity.Retrying to make use of global config etc.

    Args:
        logger: Logger to use for reporting retries
        abort_exceptions: Exceptions to abort on.

    Returns:
        A tenacity.Retrying object.
    """
    max_attempts = int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))

    def should_retry(state) -> bool:
        error = state.outcome.exception() if state.outcome.failed else None
        retryable = error is not None and not isinstance(error, tuple(abort_exceptions))
        COLLECTOR.record_llm_attempt(
            error, retried=state.attempt_number > 1,
            final=error is not None and (not retryable or state.attempt_number >= max_attempts),
        )
        return retryable

    return Retrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=should_retry,
    )
