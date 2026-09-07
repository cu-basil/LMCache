Title: fix(v1): avoid EngineCore crash on aborted requests without a scheduler-role engine

---

**What this PR does / why we need it**:

`LMCacheConnectorV1Impl.request_finished` unconditionally runs
`assert self.lmcache_engine is not None` inside its `FINISHED_ABORTED`
cleanup branch. On the **scheduler-role** connector, `lmcache_engine` is
`None` by design whenever `enable_scheduler_bypass_lookup` is disabled:
`get_or_create_lmcache_engine` deliberately skips building an engine for
that role+config combination (the real engine lives on the worker-role
connector instead). `request_finished` is a scheduler-side-only API per
`KVConnectorBase_V1`, so it always runs against that connector — the
assert is unconditionally reachable, not a rare race.

Any deployment running in this (common) configuration that aborts an
in-flight request — a client disconnect, or a reverse proxy in front of
vLLM that times out and drops the connection mid-generation — hits this
assert as an unhandled `AssertionError` inside the `EngineCore` busy
loop:

```
File ".../vllm/v1/engine/core.py", line 712, in run_engine_core
    raise e
File ".../vllm/v1/engine/core.py", line 726, in run_busy_loop
    self._process_input_queue()
File ".../vllm/v1/engine/core.py", line 771, in _handle_client_request
    self.abort_requests(request)
File ".../vllm/v1/core/sched/scheduler.py", line 1150, in _free_request
    delay_free_blocks, kv_xfer_params = self._connector_finished(request)
File ".../vllm/v1/core/sched/scheduler.py", line 1235, in _connector_finished
    return self.connector.request_finished(request, block_ids)
File ".../vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py", line 168, in request_finished
    return self._lmcache_engine.request_finished(request, block_ids)
File ".../lmcache/integration/vllm/vllm_v1_adapter.py", line 1849, in request_finished
    assert self.lmcache_engine is not None
AssertionError
```

This crashes the **entire `EngineCore` process**, not just the aborted
request — every other in-flight and future request on that engine then
fails too, until something restarts it. Under any load pattern with a
non-trivial abort rate (client timeouts, load shedding, proxy
disconnects), this turns isolated aborts into full outages.

The fix replaces the assert with an `is not None` guard — the same
pattern `get_kv_events` already uses a few lines below for the same
reason — so the engine stays alive and only the (redundant, since there's
no storage backend to notify on this role) abort-notification step is
skipped. Logs at `debug`, not `warning`, since this is expected
steady-state for this role+config combination, not an anomaly.

**Special notes for your reviewers**:

- Scope is intentionally minimal: only the `FINISHED_ABORTED` branch of
  `request_finished` changes. The `assert self.lookup_client is not None`
  a few lines below is untouched — `lookup_client` *is* unconditionally
  created for the scheduler role regardless of `lmcache_engine`
  (`maybe_create_lookup_client`), so that assert is still valid.
- Added `tests/v1/test_v1_adapter_scheduler_role_abort.py`, which
  reproduces the crash against the pre-fix code (verified locally: fails
  with the exact `AssertionError` above) and locks in the fixed behavior
  for both the engine-absent and engine-present paths.

**If applicable**:

- [ ] this PR contains user facing changes - docs added
- [x] this PR contains unit tests
