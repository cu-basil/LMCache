# SPDX-License-Identifier: Apache-2.0
"""Async GMS HTTP client for PESTO LMCache integration.

Design principles
-----------------
* **Fail-open**: every public method catches all exceptions, logs them at
  WARNING level, and returns a safe default.  Errors are NEVER raised into
  the LMCache data path.
* **Async-safe**: uses ``httpx.AsyncClient``; never blocks the event loop.
* **Circuit-breaker**: after ``_CB_FAILURE_THRESHOLD`` consecutive failures
  the client opens the circuit for ``_CB_OPEN_SECS`` seconds, skipping
  requests to relieve back-pressure.  The circuit half-opens on the next
  scheduled attempt.
* **Lazy import**: this module is imported only when ``pesto_enabled=True``
  so there is zero import cost when PESTO is off.
"""

# Standard
from typing import Optional
import asyncio
import time

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger

# Local
from pesto_gms.api_models import (
    BatchedLookupRequest,
    BatchedLookupResponse,
    CommitRemoteWrite,
    Heartbeat,
    HeartbeatResponse,
    ReportAccess,
    ReportBlockLocation,
    ReportLocationEvict,
    ReportPrefetchOutcome,
    ReportQueueState,
    ReserveRemoteWrite,
    ReserveRemoteWriteResponse,
)

logger = init_logger(__name__)

# Circuit-breaker thresholds
_CB_FAILURE_THRESHOLD: int = 5
_CB_OPEN_SECS: float = 15.0

# Default per-request timeout (seconds)
_DEFAULT_TIMEOUT: float = 2.0


class _CircuitBreaker:
    """Simple consecutive-failure circuit-breaker."""

    def __init__(
        self,
        threshold: int = _CB_FAILURE_THRESHOLD,
        open_secs: float = _CB_OPEN_SECS,
    ) -> None:
        self._threshold = threshold
        self._open_secs = open_secs
        self._failures = 0
        self._opened_at: float = 0.0
        self._open = False

    def is_open(self) -> bool:
        """Return True if the circuit is open (requests should be skipped)."""
        if not self._open:
            return False
        if time.monotonic() - self._opened_at >= self._open_secs:
            # Half-open: let one request through to probe the GMS.
            self._open = False
            self._failures = 0
            logger.info("PESTO circuit-breaker half-open — probing GMS")
        return self._open

    def record_success(self) -> None:
        """Reset failure counter on a successful GMS round-trip."""
        if self._failures > 0 or self._open:
            logger.info("PESTO circuit-breaker closed — GMS healthy")
        self._failures = 0
        self._open = False

    def record_failure(self) -> None:
        """Increment failure counter and open the circuit if threshold hit."""
        self._failures += 1
        if self._failures >= self._threshold:
            self._open = True
            self._opened_at = time.monotonic()
            logger.warning(
                "PESTO circuit-breaker OPEN after %d consecutive GMS failures; "
                "will retry in %.0fs",
                self._failures,
                self._open_secs,
            )


class GmsMetadataClient:
    """Async client for all head→GMS RPCs defined in ``pesto_gms.api_models``.

    Args:
        gms_url: Base URL of the GMS server, e.g. ``"http://localhost:8500"``.
        timeout_secs: Per-request HTTP timeout in seconds.
        client: Optional pre-built ``httpx.AsyncClient`` (dependency injection
            for tests, e.g. an in-process ASGI transport). When supplied, the
            caller owns its lifecycle and ``close()`` will not close it.
    """

    def __init__(
        self,
        gms_url: str,
        timeout_secs: float = _DEFAULT_TIMEOUT,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base_url = gms_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_secs)
        self._client: Optional[httpx.AsyncClient] = client
        self._owns_client: bool = client is None
        self._cb = _CircuitBreaker()

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialise the shared async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
            self._owns_client = True
        return self._client

    async def _post(self, path: str, payload: dict) -> Optional[dict]:
        """POST *payload* to *path*.

        Returns the parsed JSON dict on success, or ``None`` on any error.
        Never raises.
        """
        if self._cb.is_open():
            logger.debug("PESTO circuit-breaker open — skipping GMS call to %s", path)
            return None
        try:
            client = self._get_client()
            response = await client.post(path, json=payload)
            response.raise_for_status()
            self._cb.record_success()
            return response.json()
        except Exception as exc:
            self._cb.record_failure()
            logger.warning("GMS call POST %s failed: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # RPC implementations
    # ------------------------------------------------------------------

    async def report_queue_state(self, req: ReportQueueState) -> None:
        """Report vLLM queue metrics to GMS (fire-and-forget)."""
        await self._post(
            "/api/v1/report/queue-state",
            req.model_dump(),
        )

    async def report_block_location(self, req: ReportBlockLocation) -> None:
        """Announce new block locations to GMS (fire-and-forget)."""
        await self._post(
            "/api/v1/report/block-location",
            req.model_dump(),
        )

    async def report_location_evict(self, req: ReportLocationEvict) -> None:
        """Notify GMS that blocks were evicted from a tier (fire-and-forget)."""
        await self._post(
            "/api/v1/report/location-evict",
            req.model_dump(),
        )

    async def report_prefetch_outcome(self, req: ReportPrefetchOutcome) -> None:
        """Report prefetch outcomes for a completed plan (fire-and-forget)."""
        await self._post(
            "/api/v1/report/prefetch-outcome",
            req.model_dump(),
        )

    async def reserve_remote_write(
        self, req: ReserveRemoteWrite
    ) -> Optional[ReserveRemoteWriteResponse]:
        """Request admission before uploading a block to MinIO.

        Returns:
            A ``ReserveRemoteWriteResponse`` on success, or ``None`` on error
            (caller should fall back to a plain PUT).
        """
        raw = await self._post("/api/v1/write/reserve", req.model_dump())
        if raw is None:
            return None
        try:
            return ReserveRemoteWriteResponse.model_validate(raw)
        except Exception as exc:
            logger.warning("Failed to parse ReserveRemoteWriteResponse: %s", exc)
            return None

    async def commit_remote_write(self, req: CommitRemoteWrite) -> None:
        """Confirm a completed MinIO upload against a reservation (fire-and-forget)."""
        await self._post(
            "/api/v1/write/commit",
            req.model_dump(),
        )

    async def report_access(self, req: ReportAccess) -> None:
        """Report cache access events to GMS (fire-and-forget)."""
        await self._post(
            "/api/v1/report/access",
            req.model_dump(),
        )

    async def heartbeat(self, req: Heartbeat) -> Optional[HeartbeatResponse]:
        """Send a liveness ping to GMS.

        Returns:
            A ``HeartbeatResponse`` on success, or ``None`` on error.
        """
        raw = await self._post("/api/v1/heartbeat", req.model_dump())
        if raw is None:
            return None
        try:
            return HeartbeatResponse.model_validate(raw)
        except Exception as exc:
            logger.warning("Failed to parse HeartbeatResponse: %s", exc)
            return None

    async def batched_lookup(
        self, req: BatchedLookupRequest
    ) -> Optional[BatchedLookupResponse]:
        """Look up all known locations for a batch of block keys.

        Returns:
            A ``BatchedLookupResponse`` on success, or ``None`` on error
            (caller should fall back to normal cache lookup).
        """
        raw = await self._post("/api/v1/blocks/batched-lookup", req.model_dump())
        if raw is None:
            return None
        try:
            return BatchedLookupResponse.model_validate(raw)
        except Exception as exc:
            logger.warning("Failed to parse BatchedLookupResponse: %s", exc)
            return None

    async def close(self) -> None:
        """Close the underlying HTTP connection pool (only if self-owned)."""
        if (
            self._client is not None
            and not self._client.is_closed
            and self._owns_client
        ):
            await self._client.aclose()
        self._client = None


def create_gms_client(
    config_or_url: "str | object",
    timeout_secs: float = _DEFAULT_TIMEOUT,
) -> GmsMetadataClient:
    """Factory helper — accepts either a bare URL string or an LMCacheEngineConfig.

    Args:
        config_or_url: Either a GMS URL string or an ``LMCacheEngineConfig``
            whose ``extra_config["pesto_gms_url"]`` is used.
        timeout_secs: Per-request HTTP timeout in seconds.

    Returns:
        A configured :class:`GmsMetadataClient`.

    Raises:
        ValueError: If no GMS URL can be found.
    """
    if isinstance(config_or_url, str):
        return GmsMetadataClient(config_or_url, timeout_secs=timeout_secs)
    # Treat as LMCacheEngineConfig
    cfg = config_or_url
    url = None
    if cfg.extra_config is not None:
        url = cfg.extra_config.get("pesto_gms_url")
    if not url:
        raise ValueError(
            "pesto_gms_url must be set in extra_config when pesto_enabled=True"
        )
    timeout = float(
        cfg.extra_config.get("pesto_read_deadline_ms", timeout_secs * 1000) / 1000
        if cfg.extra_config
        else timeout_secs
    )
    return GmsMetadataClient(url, timeout_secs=timeout)


async def fire_and_forget(coro: "asyncio.Coroutine") -> None:  # type: ignore[type-arg]
    """Schedule *coro* as a background task, swallowing all exceptions."""
    try:
        await coro
    except Exception as exc:
        logger.debug("PESTO fire-and-forget task raised: %s", exc)
