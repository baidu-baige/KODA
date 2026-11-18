import pytest

from minisweagent.environments.kernel_sandbox import _is_infra_fatal


class _Resp:
    """Minimal stand-in for requests.Response carrying just what _is_infra_fatal reads."""

    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


@pytest.mark.parametrize(
    ("resp", "expected"),
    [
        # 404 = session id gone server-side -> re-run on a fresh session (regardless of body)
        (_Resp(404, None), True),
        (_Resp(404, {"unrelated": "html-ish"}), True),
        # explicit pod-not-found infra error
        (_Resp(500, {"error_code": "INFRASTRUCTURE_ERROR", "message": 'pods "x-session-abc" not found'}), True),
        # transient 5xx / other infra variants must NOT be treated as fatal (they hit the retry path)
        (_Resp(503, {"error_code": "INFRASTRUCTURE_ERROR", "message": "scheduling timeout"}), False),
        (_Resp(500, None), False),
        (_Resp(400, {"error_code": "BAD_REQUEST"}), False),
    ],
)
def test_is_infra_fatal(resp, expected):
    assert _is_infra_fatal(resp) is expected
