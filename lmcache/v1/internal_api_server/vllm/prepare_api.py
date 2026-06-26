# SPDX-License-Identifier: Apache-2.0
"""PESTO Prepare endpoint (Patch Point 4).

``POST /cache/prepare`` — accepts a :class:`pesto_gms.api_models.PrepareRequest`
and stages ``BlockAction`` entries whose source tier is locally reachable
(``minio``, ``local_disk``) into local_cpu memory, bound to the ``request_id``.
The prepare plan is stored in the module-level
:class:`~lmcache.v1.pesto.prepare_store.PrepareStore` so that
``async_lookup_and_prefetch`` can find the pre-staged blocks.

``POST /cache/cancel-prepare`` — cancels a pending plan.

Design notes
------------
* The endpoint is auto-discovered by the ``APIRegistry`` because it lives in
  ``internal_api_server/vllm/`` — no registry changes needed.
* When ``pesto_enabled`` is False the endpoint is still present but returns
  ``{"accepted": False, "reserved_bytes": 0}`` immediately (fail-open, no
  staging work done).  This preserves the registry contract for integration tests.
* Staging is tier-directed (CR-3):

  - ``local_hit`` → skip (block already in local_cpu).
  - ``recompute`` → skip (vLLM recomputes; nothing to stage).
  - ``fetch`` with ``source_tier`` in ``{"minio", "local_disk"}`` AND
    (``source_holder_id`` is absent OR equals this head's id) → stage via
    ``sm.get(key)``, which routes through the local_disk → remote hierarchy.
  - ``fetch`` from a peer tier (``peer_cpu``, ``peer_disk``) OR from a
    *different* head's holder → **skip with a clear log**; P2P is deferred
    to CR-4/P-7.  We do NOT call sm.get for these because that would fetch
    from the wrong source and report an incorrect hit.

* The plan is stored in ``PrepareStore`` **before** the staging task is
  submitted so that the read-path can find it immediately, even if staging has
  not completed yet.
* ``reserved_bytes`` is always ``0`` in the immediate response because actual
  bytes are staged asynchronously in a thread-pool worker.  The gateway should
  not rely on this value for admission decisions; use ``accepted=True`` to know
  the plan was accepted.
* Staging stops when the deadline is hit or ``local_budget_bytes`` is exhausted.
* Any per-block parsing/fetch error is caught and logged; staging never raises.
* This head's id is resolved via :func:`~lmcache.v1.pesto.identity.resolve_identity`
  so it matches the identity contract from CR-2.
"""

# Standard
import concurrent.futures
import time
from typing import Optional

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import JSONResponse

# First Party
from lmcache.logging import init_logger
from lmcache.utils import parse_cache_key

router = APIRouter()
logger = init_logger(__name__)

# Tier names that this MVP supports for source-directed fetch.
_SUPPORTED_FETCH_TIERS = frozenset({"minio", "local_disk"})
# Peer tiers that the MVP intentionally skips (P2P deferred to CR-4/P-7).
_PEER_TIERS = frozenset({"peer_cpu", "peer_disk"})

# Module-level thread pool for background staging.  Using a dedicated pool
# (rather than the event loop's default executor) ensures the staging tasks
# are never awaited by the ASGI server on shutdown, so the prepare endpoint
# truly returns before staging completes.
_STAGING_POOL: concurrent.futures.ThreadPoolExecutor = (
    concurrent.futures.ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="pesto-prepare-bg",
    )
)


def _stage_blocks_background_sync(
    sm: object,
    prep_req: object,
    this_head_id: str,
    deadline_epoch_s: float,
    budget_bytes: int,
) -> None:
    """Synchronous staging worker: fetch eligible blocks from disk/MinIO into local_cpu.

    This is the async-prepare contract implementation (WS2 + WS4 shared contract):
    the function runs in a thread-pool worker submitted via ``run_in_executor``,
    so staging proceeds concurrently with gateway forwarding, racing the prefill
    deadline rather than adding serial gateway latency.

    All errors are caught and logged; this function never raises.

    Args:
        sm: The LMCache storage manager (exposes a synchronous ``get(key)`` API).
        prep_req: The validated :class:`~pesto_gms.api_models.PrepareRequest`.
        this_head_id: Resolved head id for peer-source filtering.
        deadline_epoch_s: Wall-clock deadline (``time.time()`` epoch seconds).
            Staging stops when ``time.time() >= deadline_epoch_s``.
        budget_bytes: Maximum bytes to stage (0 = unlimited).
    """
    reserved_bytes: int = 0

    for action in prep_req.block_actions:  # type: ignore[attr-defined]
        # Skip non-fetch actions.
        if action.action in ("local_hit", "recompute"):
            continue

        # Respect the deadline.
        if deadline_epoch_s > 0 and time.time() >= deadline_epoch_s:
            logger.debug(
                "PESTO prepare bg: deadline exceeded for request_id=%s after %d bytes",
                prep_req.request_id,  # type: ignore[attr-defined]
                reserved_bytes,
            )
            break

        # Respect the local budget.
        if budget_bytes > 0 and reserved_bytes >= budget_bytes:
            logger.debug(
                "PESTO prepare bg: budget_bytes=%d reached for request_id=%s",
                budget_bytes,
                prep_req.request_id,  # type: ignore[attr-defined]
            )
            break

        # Tier / holder dispatch — same rules as the synchronous path.
        source_tier: str = action.source_tier or ""
        source_holder: Optional[str] = action.source_holder_id or None

        if source_tier in _PEER_TIERS or (
            source_holder is not None and source_holder != this_head_id
        ):
            logger.info(
                "PESTO prepare bg: skipping peer-source action "
                "request_id=%s tier=%r holder=%r this_head=%r (P2P deferred to CR-4)",
                prep_req.request_id,  # type: ignore[attr-defined]
                source_tier,
                source_holder,
                this_head_id,
            )
            continue

        if source_tier not in _SUPPORTED_FETCH_TIERS and source_tier:
            logger.info(
                "PESTO prepare bg: unknown source_tier=%r for request_id=%s — skipping",
                source_tier,
                prep_req.request_id,  # type: ignore[attr-defined]
            )
            continue

        # Parse key and fetch — direct synchronous call (in thread-pool worker).
        try:
            from pesto_gms.keys import block_key_to_cache_engine_string  # noqa: PLC0415

            key_str = block_key_to_cache_engine_string(action.block_key)
            key = parse_cache_key(key_str)

            memory_obj: Optional[object] = sm.get(key)  # type: ignore[attr-defined]
            if memory_obj is not None:
                try:
                    block_bytes: int = memory_obj.get_size()  # type: ignore[attr-defined]
                except Exception:
                    block_bytes = 0
                reserved_bytes += block_bytes
                try:
                    memory_obj.ref_count_down()  # type: ignore[attr-defined]
                except Exception:
                    pass
                logger.debug(
                    "PESTO prepare bg: staged block tier=%r bytes=%d request_id=%s",
                    source_tier,
                    block_bytes,
                    prep_req.request_id,  # type: ignore[attr-defined]
                )
        except Exception as exc:
            logger.warning(
                "PESTO prepare bg: error fetching block for request_id=%s: %s — "
                "skipping this block",
                prep_req.request_id,  # type: ignore[attr-defined]
                exc,
            )


def _pesto_enabled(request: Request) -> bool:
    """Return True if PESTO is configured and enabled for this engine instance."""
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        engine = getattr(lmcache_adapter, "lmcache_engine", None)
        if engine is None:
            return False
        cfg = getattr(engine, "config", None)
        if cfg is None:
            return False
        return bool(cfg.get_extra_config_value("pesto_enabled", False))
    except Exception:
        return False


def _get_prepare_timeout_ms(request: Request) -> float:
    """Read pesto_prepare_timeout_ms from config, default 75 ms."""
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        engine = getattr(lmcache_adapter, "lmcache_engine", None)
        if engine is None:
            return 75.0
        cfg = getattr(engine, "config", None)
        if cfg is None:
            return 75.0
        return float(cfg.get_extra_config_value("pesto_prepare_timeout_ms", 75.0))
    except Exception:
        return 75.0


def _get_this_head_id(request: Request) -> str:
    """Resolve this head's identity from the engine config.

    Uses :func:`~lmcache.v1.pesto.identity.resolve_identity` so that the
    id matches the namespace contract established by CR-2.

    Returns:
        The head id string, or ``"default"`` if the config is not available.
    """
    try:
        from lmcache.v1.pesto.identity import resolve_identity

        lmcache_adapter = request.app.state.lmcache_adapter
        engine = getattr(lmcache_adapter, "lmcache_engine", None)
        if engine is None:
            return "default"
        cfg = getattr(engine, "config", None)
        if cfg is None:
            return "default"
        ec = cfg.extra_config or {}
        return resolve_identity(ec).head_id
    except Exception:
        return "default"


@router.post(
    "/cache/prepare",
    summary="PESTO prepare: stage blocks into local_cpu for an upcoming request",
    tags=["pesto"],
)
async def prepare(request: Request) -> JSONResponse:
    """Accept a prepare plan, store it, and enqueue background staging.

    Implements the **async-prepare contract** (WS2 + WS4): the endpoint
    returns as soon as the plan is validated and stored in
    :class:`~lmcache.v1.pesto.prepare_store.PrepareStore`.  Actual block
    staging (``sm.get`` fetches) continues in a background asyncio task
    bounded by ``prep_req.deadline_ms``, racing the prefill deadline instead
    of adding serial gateway latency.

    Accepts a JSON body matching ``pesto_gms.api_models.PrepareRequest``.
    Returns a ``pesto_gms.api_models.PrepareResponse``-shaped JSON object.

    The endpoint is always present.  When ``pesto_enabled=False`` it returns
    ``{"request_id": ..., "accepted": false, "reserved_bytes": 0}`` immediately.

    Tier dispatch (CR-3, enforced inside :func:`_stage_blocks_background`):

    * ``local_hit`` / ``recompute`` actions are skipped (no staging needed).
    * ``fetch`` from ``minio`` or ``local_disk`` where ``source_holder_id`` is
      absent or equals this head's id → staged via ``sm.get``.
    * ``fetch`` from peer tiers or a different holder → skipped with a log
      message; P2P fetch is deferred to CR-4 / P-7.

    ``reserved_bytes`` in the response is always ``0`` because staging has not
    yet completed when the response is sent.
    """
    # Parse body.
    try:
        from pesto_gms.api_models import PrepareRequest  # noqa: PLC0415

        body_json = await request.json()
        prep_req = PrepareRequest.model_validate(body_json)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Invalid PrepareRequest body: {exc}"}, status_code=400
        )

    if not _pesto_enabled(request):
        return JSONResponse(
            {
                "request_id": prep_req.request_id,
                "accepted": False,
                "reserved_bytes": 0,
            }
        )

    lmcache_adapter = request.app.state.lmcache_adapter
    engine = getattr(lmcache_adapter, "lmcache_engine", None)
    if engine is None or engine.storage_manager is None:
        return JSONResponse(
            {"error": "LMCache engine not configured."}, status_code=503
        )

    sm = engine.storage_manager
    this_head_id = _get_this_head_id(request)

    # Compute staging deadline from the PrepareRequest wall-clock deadline.
    # Fall back to the per-instance config timeout so staging is always bounded.
    if prep_req.deadline_ms > 0:
        deadline_epoch_s = prep_req.deadline_ms / 1000.0
    else:
        timeout_secs = _get_prepare_timeout_ms(request) / 1000.0
        deadline_epoch_s = time.time() + timeout_secs

    # Store plan NOW so the read-path can find it even if staging is slow.
    try:
        from lmcache.v1.pesto.prepare_store import (  # noqa: PLC0415
            get_global_prepare_store,
        )

        get_global_prepare_store().put(prep_req.request_id, prep_req)
    except Exception as exc:
        logger.warning(
            "PESTO prepare: failed to store plan (non-fatal): %s", exc
        )

    # Submit staging to the module-level thread pool and return immediately.
    # Using _STAGING_POOL (not the event loop's default executor) ensures the
    # ASGI server does not block on this task during shutdown, so the endpoint
    # truly returns before staging completes even in test environments.
    fut = _STAGING_POOL.submit(
        _stage_blocks_background_sync,
        sm,
        prep_req,
        this_head_id,
        deadline_epoch_s,
        prep_req.local_budget_bytes,
    )
    fut.add_done_callback(
        lambda f: f.exception()
        and logger.debug(
            "PESTO prepare bg: staging thread error (non-fatal): %s", f.exception()
        )
    )

    # Return quickly; staging continues in the thread pool.
    return JSONResponse(
        {
            "request_id": prep_req.request_id,
            "accepted": True,
            "reserved_bytes": 0,
        }
    )


@router.post(
    "/cache/cancel-prepare",
    summary="PESTO cancel-prepare: drop a previously reserved prepare plan",
    tags=["pesto"],
)
async def cancel_prepare(request: Request) -> JSONResponse:
    """Cancel a pending prepare plan.

    Accepts a JSON body matching ``pesto_gms.api_models.CancelPrepare``.
    Returns a ``pesto_gms.api_models.AckResponse``-shaped JSON object.
    """
    try:
        from pesto_gms.api_models import CancelPrepare

        body_json = await request.json()
        cancel = CancelPrepare.model_validate(body_json)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Invalid CancelPrepare body: {exc}"}, status_code=400
        )

    removed = False
    try:
        from lmcache.v1.pesto.prepare_store import get_global_prepare_store

        removed = get_global_prepare_store().cancel(cancel.request_id)
    except Exception as exc:
        logger.warning("PESTO cancel_prepare: store error (non-fatal): %s", exc)

    logger.debug(
        "PESTO cancel_prepare: request_id=%s removed=%s reason=%r",
        cancel.request_id,
        removed,
        cancel.reason,
    )
    return JSONResponse({"ok": True})
