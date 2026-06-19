# SPDX-License-Identifier: Apache-2.0
"""Thread-safe in-memory store for PESTO PrepareRequest plans.

The prepare endpoint stages KV blocks for an upcoming request and writes the
``PrepareRequest`` here.  The read-path binding in
``storage_manager.async_lookup_and_prefetch`` consults this store by
``lookup_id`` (== ``request_id``) to prioritise already-staged blocks.

Entries are automatically expired after ``ttl_secs`` to prevent unbounded
memory growth from un-cancelled orphaned requests.
"""

# Standard
from typing import NamedTuple, Optional
import threading
import time

# Third Party
# (pesto_gms is imported read-only; never modified by WS-C)
from pesto_gms.api_models import PrepareRequest

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_DEFAULT_TTL_SECS: float = 60.0


class _Entry(NamedTuple):
    request: PrepareRequest
    inserted_at: float


class PrepareStore:
    """Thread-safe dict keyed by ``request_id`` → :class:`PrepareRequest`.

    Also maintains an alias table that maps vLLM-native request IDs to
    gateway-generated PESTO request IDs.  This is the binding mechanism for
    Gap 1: the gateway stores entries under its UUID; the LMCache vLLM adapter
    registers ``vllm_id → pesto_id`` at request-start time so that
    ``pop(vllm_id)`` can resolve the prepared plan even though the key in
    ``_store`` is the gateway UUID.

    Args:
        ttl_secs: Seconds after which an unread entry is auto-expired.
    """

    def __init__(self, ttl_secs: float = _DEFAULT_TTL_SECS) -> None:
        self._store: dict[str, _Entry] = {}
        # alias: vllm_native_id → pesto_gateway_id
        self._alias: dict[str, str] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_secs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_alias(self, vllm_id: str, pesto_id: str) -> None:
        """Register a vLLM-native request ID as an alias for a PESTO gateway UUID.

        Called by the LMCache vLLM adapter when it observes a ``pesto_request_id``
        in ``sampling_params.extra_args``.  After registration, ``pop(vllm_id)``
        will resolve and consume the PrepareRequest stored under ``pesto_id``.

        This is the Gap-1 binding mechanism: the gateway stores entries under its
        UUID; ``register_alias`` links vLLM's native ID to that UUID so that
        ``prefetch_all_done_callback`` can pop the plan even though the
        ``lookup_id`` passed in is the vLLM-native ID.

        Args:
            vllm_id: vLLM's native request ID (== ``lookup_id`` in storage_manager).
            pesto_id: Gateway-generated UUID (== ``PrepareRequest.request_id``).
        """
        with self._lock:
            self._alias[vllm_id] = pesto_id

    def put(self, request_id: str, req: PrepareRequest) -> None:
        """Store a prepare request, replacing any existing entry for the same id."""
        with self._lock:
            self._store[request_id] = _Entry(req, time.monotonic())

    def get(self, request_id: str) -> Optional[PrepareRequest]:
        """Return the prepare request or ``None`` if absent or expired."""
        with self._lock:
            entry = self._store.get(request_id)
            if entry is None:
                return None
            if time.monotonic() - entry.inserted_at > self._ttl:
                del self._store[request_id]
                return None
            return entry.request

    def pop(self, request_id: str) -> Optional[PrepareRequest]:
        """Remove and return the prepare request, or ``None`` if absent.

        If ``request_id`` is not found directly, falls back to alias lookup:
        ``_alias[request_id]`` is consulted for a gateway-UUID mapping so that
        calling ``pop(vllm_native_id)`` correctly resolves an entry stored under
        the PESTO gateway UUID.  The alias entry is removed on a successful hit.
        """
        with self._lock:
            # Direct lookup first (fast path for same-id or alias-already-resolved)
            entry = self._store.pop(request_id, None)
            if entry is None:
                # Alias fallback: vllm_id → pesto_id
                pesto_id = self._alias.pop(request_id, None)
                if pesto_id is not None:
                    entry = self._store.pop(pesto_id, None)
            if entry is None:
                return None
            if time.monotonic() - entry.inserted_at > self._ttl:
                return None
            return entry.request

    def cancel(self, request_id: str) -> bool:
        """Remove entry for *request_id*.

        Returns:
            True if an entry was present and removed, False otherwise.
        """
        with self._lock:
            return self._store.pop(request_id, None) is not None

    def purge_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        cutoff = time.monotonic() - self._ttl
        with self._lock:
            expired = [k for k, v in self._store.items() if v.inserted_at < cutoff]
            for k in expired:
                del self._store[k]
        if expired:
            logger.debug("PrepareStore: purged %d expired entries", len(expired))
        return len(expired)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton — shared by the prepare endpoint and the read-path.
# Initialised lazily so that non-PESTO imports don't allocate anything.
_global_prepare_store: Optional[PrepareStore] = None
_global_store_lock = threading.Lock()


def get_global_prepare_store() -> PrepareStore:
    """Return the module-level singleton :class:`PrepareStore`.

    Creates the store on first call (lazy initialisation).
    """
    global _global_prepare_store
    if _global_prepare_store is None:
        with _global_store_lock:
            if _global_prepare_store is None:
                _global_prepare_store = PrepareStore()
    return _global_prepare_store
