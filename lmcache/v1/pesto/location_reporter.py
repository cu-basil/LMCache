# SPDX-License-Identifier: Apache-2.0
"""Heartbeat / queue-state reporter for PESTO.

Design
------
Block *location* reporting (local_cpu/local_disk put/evict events pushed
event-by-event, with a short TTL as a staleness fallback) used to live here
too, but was removed: it required hooking every LMCache code path that
mutates cache state, was fire-and-forget with no retry (a dropped RPC left
GMS believing a stale location forever), and needed a TTL band-aid to bound
the damage. GMS now learns local_cpu/local_disk locations exclusively by
polling the LMCache controller's ``/controller/blocks`` endpoint (see
``pesto_gms/app.py::_poll_once``), which reads the controller's own
already-maintained registry -- no per-event hook needed, and a full-snapshot
diff each poll self-corrects instead of accumulating stale state. MinIO
locations are unaffected: ``lmcache_minio_plugin``'s ``MinIOConnector``
reports those directly as durable, non-expiring canonical locations and was
never part of this class.

Heartbeat (``Heartbeat``) and queue-state (``ReportQueueState``) reporting
are unrelated to block locations -- they feed GMS's per-head liveness and
queue-delay estimation -- and are unaffected by the above; both are still
driven by periodic background loops here.

All GMS failures are caught by
:class:`~lmcache.v1.pesto.metadata_client.GmsMetadataClient`
and never surface here.
"""

# Standard
from typing import Callable, Optional
import asyncio
import time

# First Party
from lmcache.logging import init_logger

# Local
from lmcache.v1.pesto.metadata_client import GmsMetadataClient

logger = init_logger(__name__)

# Seconds between heartbeat / queue-state reports
_HEARTBEAT_INTERVAL_SECS: float = 5.0
_QUEUE_STATE_INTERVAL_SECS: float = 2.0


class LocationReporter:
    """Async background reporter wiring LMCache head status to GMS.

    Args:
        client: A live :class:`~lmcache.v1.pesto.metadata_client.GmsMetadataClient`.
        head_id: Stable identifier for this LMCache head (e.g. ``"pesto-0"``).
        heartbeat_interval_secs: Period between ``Heartbeat`` RPCs.
        queue_state_interval_secs: Period between ``ReportQueueState`` RPCs.
        get_queue_state_fn: Optional sync callback ``() -> dict`` returning
            keyword arguments for ``ReportQueueState`` (excluding ``head_id``
            and ``ts_ms``).  If ``None``, queue-state reporting is disabled.
        local_cpu_budget_bytes: Approximate local-CPU budget for heartbeats.
        local_disk_budget_bytes: Approximate local-disk budget for heartbeats.
        endpoint: This head's HTTP endpoint (reported in heartbeat).
    """

    def __init__(
        self,
        client: GmsMetadataClient,
        head_id: str,
        heartbeat_interval_secs: float = _HEARTBEAT_INTERVAL_SECS,
        queue_state_interval_secs: float = _QUEUE_STATE_INTERVAL_SECS,
        get_queue_state_fn: Optional[Callable[[], dict]] = None,
        local_cpu_budget_bytes: int = 0,
        local_disk_budget_bytes: int = 0,
        endpoint: str = "http://localhost:8000",
    ) -> None:
        self._client = client
        self._head_id = head_id
        self._hb_interval = heartbeat_interval_secs
        self._qs_interval = queue_state_interval_secs
        self._get_queue_state_fn = get_queue_state_fn
        self._local_cpu_budget_bytes = local_cpu_budget_bytes
        self._local_disk_budget_bytes = local_disk_budget_bytes
        self._endpoint = endpoint

        self._task_hb: Optional[asyncio.Task] = None
        self._task_qs: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Spawn background tasks on *loop*.  Must be called from the same loop."""
        if self._running:
            return
        self._running = True
        self._task_hb = loop.create_task(
            self._heartbeat_loop(), name="pesto-heartbeat"
        )
        if self._get_queue_state_fn is not None:
            self._task_qs = loop.create_task(
                self._queue_state_loop(), name="pesto-queue-state"
            )
        logger.info(
            "PESTO LocationReporter started for head_id=%s", self._head_id
        )

    async def stop(self) -> None:
        """Cancel background tasks and close the GMS client."""
        self._running = False
        for task in (self._task_hb, self._task_qs):
            if task is not None and not task.done():
                task.cancel()
        await self._client.close()
        logger.info("PESTO LocationReporter stopped")

    # ------------------------------------------------------------------
    # Background coroutines
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send periodic Heartbeat RPCs to GMS."""
        from pesto_gms.api_models import Heartbeat

        while self._running:
            try:
                hb = Heartbeat(
                    head_id=self._head_id,
                    healthy=True,
                    endpoint=self._endpoint,
                    local_cpu_budget_bytes=self._local_cpu_budget_bytes,
                    local_disk_budget_bytes=self._local_disk_budget_bytes,
                    ts_ms=time.time() * 1000,
                )
                await self._client.heartbeat(hb)
            except Exception as exc:
                logger.debug("PESTO heartbeat error (non-fatal): %s", exc)
            await asyncio.sleep(self._hb_interval)

    async def _queue_state_loop(self) -> None:
        """Periodically read vLLM queue metrics and report to GMS."""
        from pesto_gms.api_models import ReportQueueState

        while self._running:
            try:
                if self._get_queue_state_fn is not None:
                    kwargs = self._get_queue_state_fn()
                    qs = ReportQueueState(
                        head_id=self._head_id,
                        ts_ms=time.time() * 1000,
                        **kwargs,
                    )
                    await self._client.report_queue_state(qs)
            except Exception as exc:
                logger.debug("PESTO queue state report error (non-fatal): %s", exc)
            await asyncio.sleep(self._qs_interval)
