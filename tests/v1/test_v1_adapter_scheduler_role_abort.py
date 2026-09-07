# SPDX-License-Identifier: Apache-2.0
"""Regression test for the scheduler-role ``request_finished`` crash.

``LMCacheConnectorV1Impl.request_finished`` used to run
``assert self.lmcache_engine is not None`` unconditionally inside its
``FINISHED_ABORTED`` cleanup branch. On the scheduler-role connector,
``lmcache_engine`` is ``None`` by design whenever
``enable_scheduler_bypass_lookup`` is disabled --
``get_or_create_lmcache_engine`` deliberately skips building an engine for
that role+config combination (the real engine lives on the worker-role
connector instead). ``request_finished`` itself is a scheduler-side-only
API per ``KVConnectorBase_V1``, so it always runs against that connector.

Whenever such a deployment aborts an in-flight request (e.g. the client
disconnects, or a proxy/gateway times out and drops the connection), the
assertion fired as an unhandled ``AssertionError`` inside the EngineCore
busy loop and crashed the entire EngineCore process -- not just the
aborted request, every other in-flight and future request on that engine
failed too, until something restarted it.

The fix replaces the assert with an ``is not None`` guard (matching the
one ``get_kv_events`` already uses a few lines below for the same reason)
so the engine stays alive and only the redundant storage-backend abort
notification is skipped.
"""

# Standard
from types import SimpleNamespace
import logging

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from vllm.v1.request import RequestStatus


class _FakeStorageManager:
    """Records ``cancel_request`` calls so tests can assert which requests
    were notified of an abort."""

    def __init__(self) -> None:
        self.canceled: list[str] = []

    def cancel_request(self, req_id: str) -> None:
        self.canceled.append(req_id)


class _FakeEngine:
    def __init__(self) -> None:
        self.storage_manager = _FakeStorageManager()


def _make_aborted_request(req_id: str) -> SimpleNamespace:
    """Build a minimal request object in the ``FINISHED_ABORTED`` state,
    as the scheduler produces it for a cancelled/disconnected request."""
    return SimpleNamespace(
        request_id=req_id,
        status=RequestStatus.FINISHED_ABORTED,
    )


def _make_connector(engine: "_FakeEngine | None") -> LMCacheConnectorV1Impl:
    """Build a scheduler-role connector with just enough state for
    ``request_finished`` to run, without going through full vLLM/LMCache
    startup."""
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    # ``lmcache_engine`` is a read-only property backed by ``self._manager``;
    # inject the fake engine (or ``None``, as on a real scheduler-role
    # connector without bypass lookup) through the manager so the property
    # resolves to it.
    connector._manager = SimpleNamespace(lmcache_engine=engine)  # type: ignore[assignment]
    connector.async_loading = False
    connector.config = SimpleNamespace(  # type: ignore[assignment]
        get_extra_config_value=lambda key, default=None: default
    )
    return connector


def test_request_finished_skips_storage_cleanup_when_engine_absent() -> None:
    """An aborted request on a scheduler-role connector without an engine
    must not raise, and must log at ``debug`` (not ``warning``/``error``,
    since this is expected steady-state for this role+config combination,
    not an anomaly)."""
    captured_records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_records.append(record)

    handler = _ListHandler(level=logging.DEBUG)
    adapter_logger = logging.getLogger("lmcache.integration.vllm.vllm_v1_adapter")
    original_level = adapter_logger.level
    adapter_logger.setLevel(logging.DEBUG)
    adapter_logger.addHandler(handler)
    try:
        connector = _make_connector(engine=None)
        request = _make_aborted_request("req-no-engine")

        # Must not raise -- this used to be an unconditional
        # `assert self.lmcache_engine is not None` that crashed the whole
        # EngineCore process for every connected user.
        saved, return_params = connector.request_finished(request, block_ids=[])

        assert saved is False
        assert return_params is None

        debug_records = [r for r in captured_records if r.levelno == logging.DEBUG]
        assert any("req-no-engine" in r.getMessage() for r in debug_records), (
            "Expected a debug log naming req-no-engine; "
            f"got {[r.getMessage() for r in captured_records]}"
        )
        assert not any(r.levelno >= logging.WARNING for r in captured_records), (
            "Expected steady-state (no engine on scheduler role) not to "
            "log at warning level or above"
        )
    finally:
        adapter_logger.removeHandler(handler)
        adapter_logger.setLevel(original_level)


def test_request_finished_still_notifies_storage_manager_when_engine_present() -> (
    None
):
    """When an engine *is* attached, aborting a request must still notify
    the storage backend, exactly as before this fix."""
    engine = _FakeEngine()
    connector = _make_connector(engine=engine)
    request = _make_aborted_request("req-with-engine")

    connector.request_finished(request, block_ids=[])

    assert engine.storage_manager.canceled == ["req-with-engine"]
