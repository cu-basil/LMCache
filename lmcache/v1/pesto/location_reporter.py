# SPDX-License-Identifier: Apache-2.0
"""Non-blocking location and queue-state reporter for PESTO.

Design
------
Hot-path code (``LocalCPUBackend.submit_put_task`` / ``remove``) calls the
synchronous ``enqueue_put`` / ``enqueue_evict`` helpers, which drop events into
an ``asyncio.Queue`` without blocking.  A background async task drains the
queue and batches the reports to GMS.

Queue-state reporting (``ReportQueueState``) and ``Heartbeat`` are driven by
a periodic background loop.  Callers may inject a ``get_queue_state_fn``
callback; if absent those reports are skipped.

All GMS failures are caught by :class:`~lmcache.v1.pesto.metadata_client.GmsMetadataClient`
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

# How many events to drain per batch before yielding back to the event loop
_DRAIN_BATCH = 32

# Seconds between heartbeat / queue-state reports
_HEARTBEAT_INTERVAL_SECS: float = 5.0
_QUEUE_STATE_INTERVAL_SECS: float = 2.0

# TTL in milliseconds used for ephemeral local-CPU location records sent to GMS.
# After this deadline GMS will consider the block no longer guaranteed to be present
# (the head may have evicted it without sending an explicit evict notice).
_LOCAL_CPU_HOLDER_TTL_MS: int = 5_000


class _LocationEvent:
    """Internal event: one insert or evict on a named tier."""

    __slots__ = ("op", "key_str", "tier")

    def __init__(self, op: str, key_str: str, tier: str) -> None:
        self.op = op          # "put" | "evict"
        self.key_str = key_str
        self.tier = tier


class LocationReporter:
    """Async background reporter wiring LMCache storage events to GMS.

    Args:
        client: A live :class:`~lmcache.v1.pesto.metadata_client.GmsMetadataClient`.
        head_id: Stable identifier for this LMCache head (e.g. ``"pesto-0"``).
        namespace: PESTO namespace for key translation (e.g. ``"default"``).
        tokenizer_id: Tokenizer identifier for ``BlockKey`` reconstruction.
        chat_template_id: Chat-template identifier for ``BlockKey`` reconstruction.
        holder_ttl_ms: TTL for ephemeral location records (milliseconds).
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
        namespace: str,
        tokenizer_id: str,
        chat_template_id: str,
        holder_ttl_ms: int = _LOCAL_CPU_HOLDER_TTL_MS,
        heartbeat_interval_secs: float = _HEARTBEAT_INTERVAL_SECS,
        queue_state_interval_secs: float = _QUEUE_STATE_INTERVAL_SECS,
        get_queue_state_fn: Optional[Callable[[], dict]] = None,
        local_cpu_budget_bytes: int = 0,
        local_disk_budget_bytes: int = 0,
        endpoint: str = "http://localhost:8000",
    ) -> None:
        self._client = client
        self._head_id = head_id
        self._namespace = namespace
        self._tokenizer_id = tokenizer_id
        self._chat_template_id = chat_template_id
        self._holder_ttl_ms = holder_ttl_ms
        self._hb_interval = heartbeat_interval_secs
        self._qs_interval = queue_state_interval_secs
        self._get_queue_state_fn = get_queue_state_fn
        self._local_cpu_budget_bytes = local_cpu_budget_bytes
        self._local_disk_budget_bytes = local_disk_budget_bytes
        self._endpoint = endpoint

        # Bounded queue — drop oldest events if the reporter falls behind.
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=4096)

        self._task_drain: Optional[asyncio.Task] = None
        self._task_hb: Optional[asyncio.Task] = None
        self._task_qs: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Synchronous hot-path helpers (safe to call from any thread)
    # ------------------------------------------------------------------

    def enqueue_put(self, key_str: str, tier: str) -> None:
        """Schedule a block-insert event without blocking.

        Safe to call from non-async code.  Drops the event silently if the
        internal queue is full to avoid back-pressure on the data path.
        """
        try:
            self._queue.put_nowait(_LocationEvent("put", key_str, tier))
        except asyncio.QueueFull:
            logger.debug("PESTO location reporter queue full — dropping put event")

    def enqueue_evict(self, key_str: str, tier: str) -> None:
        """Schedule a block-evict event without blocking.

        Safe to call from non-async code.  Drops the event silently if the
        internal queue is full.
        """
        try:
            self._queue.put_nowait(_LocationEvent("evict", key_str, tier))
        except asyncio.QueueFull:
            logger.debug("PESTO location reporter queue full — dropping evict event")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Spawn background tasks on *loop*.  Must be called from the same loop."""
        if self._running:
            return
        self._running = True
        self._task_drain = loop.create_task(
            self._drain_loop(), name="pesto-location-drain"
        )
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
        for task in (self._task_drain, self._task_hb, self._task_qs):
            if task is not None and not task.done():
                task.cancel()
        await self._client.close()
        logger.info("PESTO LocationReporter stopped")

    # ------------------------------------------------------------------
    # Background coroutines
    # ------------------------------------------------------------------

    async def _drain_loop(self) -> None:
        """Drain the event queue and batch-report locations to GMS."""
        from pesto_gms.api_models import (  # lazy import
            ReportBlockLocation,
            ReportLocationEvict,
        )
        from pesto_gms.schemas import BlockLocation

        while self._running:
            # Collect up to _DRAIN_BATCH events (blocking on the first one)
            events: list[_LocationEvent] = []
            try:
                ev = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                events.append(ev)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Drain remaining without blocking
            for _ in range(_DRAIN_BATCH - 1):
                try:
                    events.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            puts = [e for e in events if e.op == "put"]
            evicts = [e for e in events if e.op == "evict"]

            try:
                if puts:
                    await self._report_puts(puts, BlockLocation, ReportBlockLocation)
                if evicts:
                    await self._report_evicts(evicts, ReportLocationEvict)
            except Exception as exc:
                logger.debug("PESTO location drain error (non-fatal): %s", exc)

    async def _report_puts(
        self,
        events: list,
        BlockLocation: type,
        ReportBlockLocation: type,
    ) -> None:
        """Translate put events and call GMS ReportBlockLocation."""
        from pesto_gms.keys import cache_engine_string_to_block_key

        now_ms = int(time.time() * 1000)
        locations = []
        for ev in events:
            try:
                bk = cache_engine_string_to_block_key(
                    ev.key_str,
                    self._namespace,
                    self._tokenizer_id,
                    self._chat_template_id,
                )
                transport = "local" if ev.tier in ("local_cpu", "local_disk") else "s3"
                locations.append(
                    BlockLocation(
                        block_key=bk,
                        holder_id=self._head_id,
                        tier=ev.tier,
                        transport=transport,
                        durable=(ev.tier == "minio"),
                        expires_at_ms=now_ms + self._holder_ttl_ms,
                        fetch_latency_ewma_ms=0.0,
                    )
                )
            except Exception as exc:
                logger.debug(
                    "PESTO: failed to translate key %s to BlockKey: %s",
                    ev.key_str,
                    exc,
                )

        if locations:
            req = ReportBlockLocation(head_id=self._head_id, locations=locations)
            await self._client.report_block_location(req)

    async def _report_evicts(self, events: list, ReportLocationEvict: type) -> None:
        """Translate evict events and call GMS ReportLocationEvict per tier."""
        from pesto_gms.keys import cache_engine_string_to_block_key
        from pesto_gms.schemas import BlockKey

        by_tier: dict[str, list[BlockKey]] = {}
        for ev in events:
            try:
                bk = cache_engine_string_to_block_key(
                    ev.key_str,
                    self._namespace,
                    self._tokenizer_id,
                    self._chat_template_id,
                )
                by_tier.setdefault(ev.tier, []).append(bk)
            except Exception as exc:
                logger.debug(
                    "PESTO: failed to translate evict key %s: %s", ev.key_str, exc
                )

        for tier, keys in by_tier.items():
            req = ReportLocationEvict(
                head_id=self._head_id, block_keys=keys, tier=tier
            )
            await self._client.report_location_evict(req)

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
