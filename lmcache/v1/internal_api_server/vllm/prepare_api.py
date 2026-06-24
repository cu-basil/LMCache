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

* ``reserved_bytes`` reflects *actual* staged bytes, read from the returned
  ``MemoryObj.get_size()`` call.  Staging stops when ``local_budget_bytes``
  is exceeded (if non-zero) or when the deadline is hit.
* Any per-block parsing/fetch error is caught and logged; the endpoint never
  raises and always returns ``accepted=True`` for successful PESTO plans.
* This head's id is resolved via :func:`~lmcache.v1.pesto.identity.resolve_identity`
  so it matches the identity contract from CR-2.
"""

# Standard
import asyncio
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
    """Stage KV blocks from remote/disk into local_cpu, bound to a request_id.

    Accepts a JSON body matching ``pesto_gms.api_models.PrepareRequest``.
    Returns a ``pesto_gms.api_models.PrepareResponse``-shaped JSON object.

    The endpoint is always present.  When ``pesto_enabled=False`` it returns
    ``{"request_id": ..., "accepted": false, "reserved_bytes": 0}`` immediately.

    Tier dispatch (CR-3):

    * ``local_hit`` / ``recompute`` actions are skipped (no staging needed).
    * ``fetch`` from ``minio`` or ``local_disk`` where ``source_holder_id`` is
      absent or equals this head's id → staged via ``sm.get``.
    * ``fetch`` from peer tiers or a different holder → skipped with a log
      message; P2P fetch is deferred to CR-4 / P-7.

    ``reserved_bytes`` counts actual bytes staged, as reported by each
    ``MemoryObj.get_size()`` call.  Staging stops at ``local_budget_bytes``
    (when non-zero) or at the deadline.
    """
    # Parse body.
    try:
        from pesto_gms.api_models import PrepareRequest

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
    timeout_secs = _get_prepare_timeout_ms(request) / 1000.0
    deadline = time.monotonic() + timeout_secs
    this_head_id = _get_this_head_id(request)

    reserved_bytes: int = 0
    budget_bytes: int = prep_req.local_budget_bytes  # 0 means unlimited
    loop = asyncio.get_event_loop()

    for action in prep_req.block_actions:
        # ── skip non-fetch actions ────────────────────────────────────────────
        if action.action in ("local_hit", "recompute"):
            continue

        if time.monotonic() >= deadline:
            logger.debug(
                "PESTO prepare: deadline exceeded for request_id=%s after staging "
                "%d bytes",
                prep_req.request_id,
                reserved_bytes,
            )
            break

        # ── budget cap ───────────────────────────────────────────────────────
        if budget_bytes > 0 and reserved_bytes >= budget_bytes:
            logger.debug(
                "PESTO prepare: local_budget_bytes=%d reached for request_id=%s",
                budget_bytes,
                prep_req.request_id,
            )
            break

        # ── tier / holder dispatch ───────────────────────────────────────────
        source_tier: str = action.source_tier or ""
        source_holder: Optional[str] = action.source_holder_id or None

        if source_tier in _PEER_TIERS or (
            source_holder is not None and source_holder != this_head_id
        ):
            # P2P fetch (peer tier or foreign holder): deferred to CR-4/P-7.
            logger.info(
                "PESTO prepare: skipping peer-source action "
                "request_id=%s tier=%r holder=%r this_head=%r (P2P deferred to CR-4)",
                prep_req.request_id,
                source_tier,
                source_holder,
                this_head_id,
            )
            continue

        if source_tier not in _SUPPORTED_FETCH_TIERS and source_tier:
            # Unknown/unsupported tier: log and skip safely.
            logger.info(
                "PESTO prepare: unknown source_tier=%r for request_id=%s — skipping",
                source_tier,
                prep_req.request_id,
            )
            continue

        # ── parse key and fetch ───────────────────────────────────────────────
        try:
            from pesto_gms.keys import block_key_to_cache_engine_string

            key_str = block_key_to_cache_engine_string(action.block_key)
            key = parse_cache_key(key_str)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            memory_obj: Optional[object] = await asyncio.wait_for(
                loop.run_in_executor(None, sm.get, key),
                timeout=max(remaining, 0.01),
            )
            if memory_obj is not None:
                # Read the actual staged bytes from the MemoryObj.
                try:
                    block_bytes: int = memory_obj.get_size()  # type: ignore[attr-defined]
                except Exception:
                    block_bytes = 0
                reserved_bytes += block_bytes

                # Release the ref-counted object; write-back to local_cpu has
                # already happened inside sm.get.
                try:
                    memory_obj.ref_count_down()  # type: ignore[attr-defined]
                except Exception:
                    pass

                logger.debug(
                    "PESTO prepare: staged block tier=%r bytes=%d request_id=%s",
                    source_tier,
                    block_bytes,
                    prep_req.request_id,
                )

        except asyncio.TimeoutError:
            logger.debug(
                "PESTO prepare: timeout fetching block for request_id=%s",
                prep_req.request_id,
            )
            break
        except Exception as exc:
            logger.warning(
                "PESTO prepare: error fetching block for request_id=%s: %s — "
                "skipping this block",
                prep_req.request_id,
                exc,
            )
            # Fail-open: continue with remaining blocks.

    # Store plan so the read-path can find it (Gap-1 alias consumption).
    try:
        from lmcache.v1.pesto.prepare_store import get_global_prepare_store

        get_global_prepare_store().put(prep_req.request_id, prep_req)
    except Exception as exc:
        logger.warning(
            "PESTO prepare: failed to store plan (non-fatal): %s", exc
        )

    return JSONResponse(
        {
            "request_id": prep_req.request_id,
            "accepted": True,
            "reserved_bytes": reserved_bytes,
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
