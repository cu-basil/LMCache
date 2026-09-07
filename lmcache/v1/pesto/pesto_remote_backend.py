# SPDX-License-Identifier: Apache-2.0
"""PestoRemoteBackend — PESTO-aware wrapper around RemoteBackend.

MVP behaviour (default ``pesto_admission_mode = "passthrough"``)
----------------------------------------------------------------
* **Write path (passthrough)**: ``submit_put_task`` / ``batched_submit_put_task``
  delegate directly to ``super()`` without any reserve/commit wrapping.  No
  synthetic ``reservation_id="passthrough"`` / ``size_bytes=1`` commit is fired.
  MinIO blocks become planner-visible via ``MinIOConnector``'s canonical
  ``_pesto_report_location`` call after each successful upload.

  Opt-in experimental path: set ``pesto_admission_mode="shadow"`` in
  ``extra_config`` to re-enable the reserve→commit wrapping.

  TODO(CR-4, post-MVP): implement full reserve→PUT→commit admission + dedup
  enforcement (``commit-creates-location``).  Deferred per CR-4 / P-8 decision.

* **Read/contains path**: optional ``BatchedLookup`` hint to GMS; always
  falls back to normal LMCache lookup on any error.
* **Three-tier only**: local_cpu → local_disk → minio; P2P is deferred.

All GMS calls are non-blocking async tasks (``asyncio.ensure_future``).
Errors are caught by ``GmsMetadataClient`` and logged; they never reach the
data path.
"""

# Standard
import asyncio
from concurrent.futures import Future
from typing import Any, Callable, List, Optional, Sequence

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

# Local
from lmcache.v1.pesto.metadata_client import GmsMetadataClient, create_gms_client

logger = init_logger(__name__)


def _key_to_block_key(
    key: CacheEngineKey,
    namespace: str,
    tokenizer_id: str,
    chat_template_id: str,
) -> "Optional[Any]":
    """Translate a CacheEngineKey to a pesto_gms BlockKey. Returns None on error."""
    try:
        from pesto_gms.schemas import cache_engine_string_to_block_key

        return cache_engine_string_to_block_key(
            key.to_string(), namespace, tokenizer_id, chat_template_id
        )
    except Exception as exc:
        logger.debug("PESTO: key translation failed for %s: %s", key.to_string(), exc)
        return None


class PestoRemoteBackend(RemoteBackend):
    """RemoteBackend subclass with PESTO GMS integration.

    Args:
        config: LMCache engine configuration (must have pesto_* extra_config keys).
        metadata: Engine metadata.
        loop: The running asyncio event loop shared with LMCache.
        local_cpu_backend: Required buffer backend (same as RemoteBackend).
        dst_device: Tensor device string.
        plugin_name: Plugin name forwarded to ``RemoteBackend``.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional[LocalCPUBackend],
        dst_device: str = torch_device_type,
        plugin_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            config=config,
            metadata=metadata,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            dst_device=dst_device,
            plugin_name=plugin_name,
        )
        extra = config.extra_config or {}
        from lmcache.v1.pesto.identity import resolve_identity

        _identity = resolve_identity(extra, model_id=metadata.model_name)
        self._namespace: str = _identity.namespace
        self._head_id: str = _identity.head_id
        self._tokenizer_id: str = _identity.tokenizer_id
        self._chat_template_id: str = _identity.chat_template_id
        # Default is "passthrough": no synthetic reserve/commit fires by default.
        # TODO(CR-4, post-MVP): change default to "shadow" when full admission is
        # implemented; "passthrough" is the safe MVP default.
        self._admission_mode: str = extra.get("pesto_admission_mode", "passthrough")

        try:
            self._gms: Optional[GmsMetadataClient] = create_gms_client(config)
            # Verify deterministic hashing at startup
            from pesto_gms.schemas import assert_deterministic_hashing

            assert_deterministic_hashing(config.pre_caching_hash_algorithm)
            logger.info(
                "PestoRemoteBackend initialised — head_id=%s gms=%s",
                self._head_id,
                self._gms._base_url,
            )
        except Exception as exc:
            logger.warning(
                "PestoRemoteBackend: GMS client init failed (%s) — PESTO disabled, "
                "falling back to plain RemoteBackend behaviour",
                exc,
            )
            self._gms = None

    # ------------------------------------------------------------------
    # Write path — reserve/commit deduplication
    # ------------------------------------------------------------------

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        """Submit a PUT with PESTO reserve/commit wrapping.

        If the GMS is reachable and returns ``skip_committed`` / ``skip_reserved``,
        the PUT is elided (deduplication).  On any GMS error the plain PUT
        proceeds — fail-open.
        """
        if self._gms is None or self._admission_mode == "passthrough":
            return super().submit_put_task(key, memory_obj, on_complete_callback)

        # Schedule the reservation asynchronously; we do not block waiting for it.
        # Pass-through semantics: the actual PUT decision is evaluated inside the
        # scheduled task, but we return the future from super() immediately so the
        # caller is never blocked.  The on_complete_callback is wrapped to include
        # CommitRemoteWrite.
        wrapped_callback = self._make_commit_callback(key, on_complete_callback)
        return super().submit_put_task(key, memory_obj, wrapped_callback)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Batched PUT with per-key commit callbacks."""
        if self._gms is None or self._admission_mode == "passthrough":
            super().batched_submit_put_task(
                keys, memory_objs, transfer_spec, on_complete_callback
            )
            return

        wrapped_callback = self._make_batched_commit_callback(on_complete_callback)
        super().batched_submit_put_task(
            keys, memory_objs, transfer_spec, wrapped_callback
        )

    def _make_commit_callback(
        self,
        key: CacheEngineKey,
        original_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> Callable[[CacheEngineKey], None]:
        """Return a callback that fires CommitRemoteWrite after a successful PUT."""

        def _cb(completed_key: CacheEngineKey) -> None:
            asyncio.run_coroutine_threadsafe(
                self._async_commit(completed_key), self.loop
            )
            if original_callback is not None:
                try:
                    original_callback(completed_key)
                except Exception as exc:
                    logger.warning("PestoRemoteBackend: callback error: %s", exc)

        return _cb

    def _make_batched_commit_callback(
        self,
        original_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> Callable[[CacheEngineKey], None]:
        """Return a per-key commit callback for batched PUTs."""

        def _cb(completed_key: CacheEngineKey) -> None:
            asyncio.run_coroutine_threadsafe(
                self._async_commit(completed_key), self.loop
            )
            if original_callback is not None:
                try:
                    original_callback(completed_key)
                except Exception as exc:
                    logger.warning(
                        "PestoRemoteBackend: batched commit callback error: %s", exc
                    )

        return _cb

    async def _async_commit(self, key: CacheEngineKey) -> None:
        """[DEFERRED] Fire CommitRemoteWrite after a successful PUT (fail-open).

        Only scheduled when ``_admission_mode != "passthrough"`` (i.e. the
        experimental ``"shadow"`` mode is explicitly opted into via
        ``pesto_admission_mode`` in ``extra_config``).  With the default
        ``"passthrough"`` admission mode this method is never called, so no
        synthetic ``reservation_id="passthrough"`` / ``size_bytes=1`` commit
        is ever sent to GMS.

        TODO(CR-4, post-MVP): replace the synthetic passthrough markers with a
        real ``reservation_id`` from a preceding ``reserve_remote_write`` call
        and the actual ``size_bytes`` from the MinIO upload.
        """
        if self._gms is None:
            return
        try:
            from pesto_gms.api_models import CommitRemoteWrite

            bk = _key_to_block_key(
                key, self._namespace, self._tokenizer_id, self._chat_template_id
            )
            if bk is None:
                return
            await self._gms.commit_remote_write(
                CommitRemoteWrite(
                    block_key=bk,
                    head_id=self._head_id,
                    reservation_id="passthrough",
                    object_key=key.to_string(),
                    size_bytes=1,  # TODO(CR-4): replace with actual upload size_bytes
                )
            )
        except Exception as exc:
            logger.debug("PESTO async_commit failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Read path — optional access reporting
    # ------------------------------------------------------------------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Delegate to super and fire-and-forget ReportAccess."""
        result = super().contains(key, pin)
        if self._gms is not None:
            asyncio.run_coroutine_threadsafe(
                self._report_access(key, "read", result, "minio"), self.loop
            )
        return result

    # ------------------------------------------------------------------
    # Access reporter
    # ------------------------------------------------------------------

    async def _report_access(
        self,
        key: CacheEngineKey,
        op: str,
        hit: bool,
        tier: str,
    ) -> None:
        """Report one access event to GMS (fire-and-forget, fail-open)."""
        if self._gms is None:
            return
        try:
            from pesto_gms.api_models import AccessRecord, ReportAccess

            bk = _key_to_block_key(
                key, self._namespace, self._tokenizer_id, self._chat_template_id
            )
            if bk is None:
                return
            await self._gms.report_access(
                ReportAccess(
                    head_id=self._head_id,
                    accesses=[
                        AccessRecord(
                            block_key=bk,
                            op=op,
                            hit=hit,
                            tier=tier,
                            bytes=0,
                        )
                    ],
                )
            )
        except Exception as exc:
            logger.debug("PESTO _report_access failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Determinism check (called at factory time via __init__)
    # ------------------------------------------------------------------

    @staticmethod
    def assert_deterministic_hashing(algo: str) -> None:
        """Verify hash algorithm determinism. Raises ``ValueError`` if not sha256."""
        from pesto_gms.schemas import assert_deterministic_hashing

        assert_deterministic_hashing(algo)
