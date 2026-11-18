"""Client for the shared GPU Sandbox pool API."""

import logging
import time

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SandboxAPIError(Exception):
    """Raised when the Sandbox pool API returns an error response."""


class SandboxClientConfig(BaseModel):
    """Configuration for connecting to the Sandbox pool API."""

    base_url: str = "http://127.0.0.1:18766"
    image: str = "ghcr.io/baidu-baige/koda-sandbox:py312-cuda129-torch28-2026110922"
    timeout: int = 60
    resource_request: dict = {}
    """k8s requests sent on pool create. These OVERRIDE the same-named resources in the selected
    GPU pool cluster's deployment template — i.e. they set the GPU worker (GPU pod) container's
    requests (names not listed keep the template default). A GPU resource name additionally routes
    the pool to the cluster that owns it (gpu_pool_clusters.resource_names). Empty (default) = omit
    `resources` so the master uses its default cluster and that template's default resources."""
    resource_limit: dict = {}
    """k8s limits sent on pool create (quantities stringified on the wire); override the GPU pod
    template's same-named limits the same way. Empty = omit."""


class SandboxClient:
    """Client for creating, inspecting, and deleting the shared Sandbox pool."""

    def __init__(self, **kwargs):
        """Build the client config and start out without an owned pool."""
        self.config = SandboxClientConfig(**kwargs)
        self.gpu_pool_id = ""

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Send an HTTP request to the sandbox master and return the parsed JSON body."""
        url = f"{self.config.base_url}{path}"
        try:
            response = requests.request(method, url, timeout=self.config.timeout, **kwargs)
            if not response.ok:
                logger.error(f"{method} {url} returned {response.status_code}: {response.text}")
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            logger.error(f"{method} {url} failed: {e}")
            raise SandboxAPIError(f"{method} {url} failed: {e}") from e

    def create(self, image: str, replicas: int) -> dict:
        """POST /v1/gpu_pool — create the pool. Returns 202 without waiting for Workers to become ready.

        When resource_request/limit are configured they are sent as `resources`; the master applies
        them to the GPU worker (GPU pod) deployment template (overriding same-named template defaults)
        and routes the pool to the matching cluster. Left empty, `resources` is omitted and the master
        uses its default cluster with that template's default resources.
        """
        body = {"image": image, "replicas": replicas}
        if self.config.resource_request or self.config.resource_limit:
            resources = {}
            if self.config.resource_request:
                resources["requests"] = {k: str(v) for k, v in self.config.resource_request.items()}
            if self.config.resource_limit:
                resources["limits"] = {k: str(v) for k, v in self.config.resource_limit.items()}
            body["resources"] = resources
        pool = self._request("POST", "/v1/gpu_pool", json=body)
        self.gpu_pool_id = pool["gpu_pool_id"]
        return pool

    def get(self) -> dict:
        """GET /v1/gpu_pool/{gpu_pool_id} — current pool status (workers, pods)."""
        return self._request("GET", f"/v1/gpu_pool/{self.gpu_pool_id}")

    def delete(self) -> dict:
        """DELETE /v1/gpu_pool/{gpu_pool_id} — delete the pool. All Sessions must be deleted first."""
        return self._request("DELETE", f"/v1/gpu_pool/{self.gpu_pool_id}")

    def ensure_created(self, image: str, replicas: int, ready_timeout: int = 600, poll_interval: int = 10) -> dict:
        """Create the pool once, then poll until at least one Worker is idle.

        The pool is created exactly once: every POST provisions a *new* pool, so retrying
        creation would leak GPU deployments. A half-started pool is deleted before raising.
        """
        status = self.create(image=image, replicas=replicas)
        deadline = time.monotonic() + ready_timeout
        while time.monotonic() < deadline:
            if status.get("workers", {}).get("idle", 0) > 0:
                return status
            time.sleep(poll_interval)
            status = self.get()
        self.delete()
        raise SandboxAPIError(f"GPU pool {self.gpu_pool_id} had no idle worker within {ready_timeout}s: {status}")
