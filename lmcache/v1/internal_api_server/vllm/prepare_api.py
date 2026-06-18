# SPDX-License-Identifier: Apache-2.0
"""PESTO Prepare endpoint (Patch Point 4).

``POST /cache/prepare`` — accepts a :class:`pesto_gms.api_models.PrepareRequest`
and stages every ``BlockAction`` from its source tier (minio / local_disk) into
local_cpu memory, bound to the ``request_id``.  The prepare plan is stored in
the module-level :class:`~lmcache.v1.pesto.prepare_store.PrepareStore` so that
``async_lookup_and_prefetch`` can find the pre-staged blocks.

``POST /cache/cancel-prepare`` — cancels a pending plan.

Design notes
------------
* The endpoint is auto-discovered by the ``APIRegistry`` because it lives in
  ``internal_api_server/vllm/`` — no registry changes needed.
* When ``pesto_enabled`` is False the endpoint is still present but returns
  ``{"accepted": False, "reserved_bytes": 0}`` immediately (fail-open, no
  staging work done).  This preserves the registry contract for future
  integration tests.
* Staging is done by calling ``sm.get(key)`` for each "fetch" action, which
  naturally routes through the tier hierarchy (local_disk → remote) and
  write-backs to local_cpu.  Actions with ``action="local_hit"`` are skipped
  (block is already in local_cpu).
* Any per-block error is caught; the endpoint still returns
  ``accepted=True`` with ``reserved_bytes`` counting only successful fetches.
* The prepare deadline (``pesto_prepare_timeout_ms``) is respected via
  ``asyncio.wait_for``.
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
    """
    # Parse body
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

    reserved_bytes = 0
    loop = asyncio.get_event_loop()

    for action in prep_req.block_actions:
        if action.action == "local_hit":
            # Block should already be in local_cpu — nothing to fetch.
            continue

        if time.monotonic() >= deadline:
            logger.debug(
                "PESTO prepare: deadline exceeded for request_id=%s",
                prep_req.request_id,
            )
            break

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
                # sm.get returns a ref-counted object; release it immediately.
                # The write-back to local_cpu has already happened inside sm.get.
                try:
                    memory_obj.ref_count_down()  # type: ignore[attr-defined]
                except Exception:
                    pass
                reserved_bytes += prep_req.local_budget_bytes // max(
                    len(prep_req.block_actions), 1
                )
        except asyncio.TimeoutError:
            logger.debug(
                "PESTO prepare: timeout fetching block for request_id=%s",
                prep_req.request_id,
            )
            break
        except Exception as exc:
            logger.warning(
                "PESTO prepare: error fetching block for request_id=%s: %s",
                prep_req.request_id,
                exc,
            )

    # Store plan so the read-path can find it
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
