# SPDX-License-Identifier: Apache-2.0
"""PESTO integration subpackage for LMCache.

All behaviour is guarded behind the ``pesto_enabled`` extra-config flag so
that the vendored LMCache default behaviour and its existing tests are
completely unchanged when PESTO is off.

Sub-modules
-----------
metadata_client  -- async GMS HTTP client with circuit-breaker (fail-open)
prepare_store    -- thread-safe in-memory plan store keyed by request_id
location_reporter -- non-blocking location/queue/heartbeat reporter
pesto_remote_backend -- PestoRemoteBackend (subclass of RemoteBackend)
"""
