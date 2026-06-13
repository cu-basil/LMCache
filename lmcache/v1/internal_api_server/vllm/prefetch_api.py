# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
from typing import List

# Third Party
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse

# First Party
from lmcache.logging import init_logger
from lmcache.utils import parse_cache_key

router = APIRouter()
logger = init_logger(__name__)


class PrefetchKeysRequest(BaseModel):
    key_strs: List[str]


@router.post(
    "/cache/prefetch-keys",
    summary="Prefetch specific cache blocks from remote into LocalCPUBackend",
    tags=["cache-management"],
)
async def prefetch_keys(request: Request, body: PrefetchKeysRequest) -> JSONResponse:
    lmcache_adapter = request.app.state.lmcache_adapter
    engine = getattr(lmcache_adapter, "lmcache_engine", None)

    if engine is None or engine.storage_manager is None:
        return JSONResponse(
            {"error": "LMCache engine not configured."}, status_code=503
        )

    sm = engine.storage_manager
    loop = asyncio.get_event_loop()
    loaded = 0
    failed: List[str] = []

    for key_str in body.key_strs:
        try:
            key = parse_cache_key(key_str)
            # sm.get() fetches from RemoteBackend (MinIO) and automatically
            # write-backs to LocalCPUBackend if the key is not already local.
            memory_obj = await loop.run_in_executor(None, sm.get, key)
            if memory_obj is not None:
                memory_obj.ref_count_down()
                loaded += 1
            else:
                failed.append(key_str)
        except Exception as e:
            logger.warning("Failed to prefetch key %s: %s", key_str, e)
            failed.append(key_str)

    return JSONResponse({"loaded": loaded, "failed": failed})
