# SPDX-License-Identifier: Apache-2.0
"""Expose the controller's per-worker chunk-hash index to external consumers.

GET /controller/blocks returns every (instance_id, worker_id, location,
chunk_hashes) tuple currently tracked in the controller registry.  The
Global Metadata Service polls this endpoint to learn which CPU/disk blocks
are present on each lmcache instance without any modification to the data
path.
"""

# Standard
from typing import List

# Third Party
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class WorkerBlockEntry(BaseModel):
    instance_id: str
    worker_id: int
    location: str
    chunk_hashes: List[int]


class BlockIndexResponse(BaseModel):
    total_blocks: int
    entries: List[WorkerBlockEntry]


@router.get("/controller/blocks")
async def get_block_index(request: Request) -> BlockIndexResponse:
    """Return all chunk-hashes tracked by the controller, grouped by
    (instance_id, worker_id, location).

    This endpoint is read-only and does not modify any controller state.
    It is intended for the PESTO GMS to poll so it can maintain a
    global view of CPU and disk blocks without requiring lmcache to push
    events directly to an external service.
    """
    controller_manager = getattr(
        request.app.state, "lmcache_controller_manager", None
    )
    if controller_manager is None:
        raise HTTPException(status_code=503, detail="Controller manager not available")

    try:
        registry = controller_manager.reg_controller.registry
        entries: List[WorkerBlockEntry] = []
        total = 0

        # Snapshot instance ids to avoid holding locks across iterations
        instance_ids = list(registry.instances.keys())

        for instance_id in instance_ids:
            instance_node = registry.instances.get(instance_id)
            if instance_node is None:
                continue
            worker_ids = instance_node.get_worker_ids()
            for worker_id in worker_ids:
                worker_node = instance_node.get_worker(worker_id)
                if worker_node is None:
                    continue
                # get_kv_keys acquires the worker lock internally
                for location in list(worker_node.kv_store.keys()):
                    hashes = worker_node.get_kv_keys(location)
                    if not hashes:
                        continue
                    hash_list = list(hashes)
                    entries.append(
                        WorkerBlockEntry(
                            instance_id=instance_id,
                            worker_id=worker_id,
                            location=location,
                            chunk_hashes=hash_list,
                        )
                    )
                    total += len(hash_list)

        return BlockIndexResponse(total_blocks=total, entries=entries)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
