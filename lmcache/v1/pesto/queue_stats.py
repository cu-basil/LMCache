# SPDX-License-Identifier: Apache-2.0
"""Lightweight vLLM queue-state sampler for PESTO live telemetry.

Provides :func:`make_prometheus_queue_state_fn`, which returns a zero-argument
callable suitable for injection into
``LocationReporter(get_queue_state_fn=...)``.

Metric sourcing
---------------
**Real** (from the vLLM API server's ``/metrics`` endpoint, with the local
``prometheus_client.REGISTRY`` as a fallback). vLLM's API server and LMCache
engine run in separate processes, so the HTTP scrape is required in production.
Metric names changed across vLLM versions, so each field reads the first
present candidate (see :data:`_VLLM_METRIC_CANDIDATES`):

* ``vllm:num_requests_waiting``  → ``num_waiting``
* ``vllm:num_requests_running``  → ``num_running``
* ``vllm:kv_cache_usage_perc`` (V1) / ``vllm:gpu_cache_usage_perc`` (V0)
  → ``gpu_cache_usage_perc`` (0–1)
* ``vllm:num_requests_swapped`` (V0 only; V1 has no swapping) → ``num_swapped``

**Approximated** (no GPU / scheduler handle required):

* ``active_requests``       = ``num_running + num_waiting`` (derived)
* ``queued_prefill_tokens`` = ``num_waiting`` (lower bound) unless a source has
  been registered via :func:`register_queued_prefill_tokens_source` (see
  below), in which case it is the real sum of remaining (uncomputed) prefill
  tokens across requests the vLLM scheduler has examined and allocated for —
  still not a complete queue-wide total (requests still buried behind the
  scheduler's per-step admission budget aren't visible until it reaches them),
  but a real token count rather than a request headcount.
* ``running_decode_blocks`` = 0 (block-level scheduler access unavailable
  GPU-free; post-MVP extension)
* ``estimated_wait_ms``     = 0 unless the ``vllm:request_queue_time_seconds``
  Prometheus histogram is present, in which case it's the mean queue time (ms)
  over all requests that have *already finished* waiting — a lagging,
  historical statistic, not a live prediction for the request being planned
  right now (see :func:`_read_prometheus_histogram_mean_ms`).
* ``request_progress``      = ``{}`` unless a source has been registered via
  :func:`register_request_progress_source` (see below) — ``{pesto_request_id:
  num_output_tokens}`` for requests currently running on this head.

Extension points
-----------------
Pass an ``extra_sampler: Callable[[], dict]`` to
:func:`make_prometheus_queue_state_fn` to override any subset of the above
fields with exact scheduler values.  This is the intended hook when the caller
has a reference to the vLLM scheduler object (e.g. from a running engine).
Any error in ``extra_sampler`` is silently caught (fail-open).

Call :func:`register_request_progress_source` once, from the LMCache
connector (which runs in-process with vLLM's scheduler and is constructed
*after* this sampler), to populate ``request_progress`` on every subsequent
heartbeat. Unlike ``extra_sampler`` this is process-global rather than
per-``make_prometheus_queue_state_fn``-call, since the connector and the
sampler are built at different points in the startup chain with no direct
reference to each other. :func:`register_queued_prefill_tokens_source` is the
same pattern, for ``queued_prefill_tokens``.

Fail-open guarantee
-------------------
Any exception anywhere in the returned callable returns a copy of
:data:`ZEROS_DICT` and never propagates.  This module is safe to import in
GPU-free environments; no vLLM dependency is required at import time.
"""

# Standard
import os
from typing import Callable, Optional
from urllib.request import urlopen

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# ---- Constants ---------------------------------------------------------------

# Safe all-zeros dict matching ReportQueueState non-identity fields.
# Exported so tests can reference the canonical zero-state shape.
ZEROS_DICT: dict = {
    "queued_prefill_tokens": 0,
    "running_decode_blocks": 0,
    "active_requests": 0,
    "gpu_cache_usage_perc": 0.0,
    "num_waiting": 0,
    "num_running": 0,
    "num_swapped": 0,
    "estimated_wait_ms": 0.0,
    "request_progress": {},
}

# ---- Live per-request progress (registered by the connector) --------------

# Set once at connector construction time via register_request_progress_source;
# read on every heartbeat by _sample(). A plain module-level slot is enough
# here -- unlike PrepareStore (concurrent put/pop/cancel from many requests),
# this is a single write-once-then-read-many callback, and CPython's GIL
# already makes a bare attribute assignment/read atomic, so no lock is needed.
_request_progress_source: Optional[Callable[[], dict]] = None


def register_request_progress_source(source: Callable[[], dict]) -> None:
    """Called once by LMCacheConnectorV1 to expose live per-request progress.

    *source* is a zero-argument callable returning
    ``{pesto_request_id: num_output_tokens}`` for requests currently running
    on this head. This bridges the connector (owns the live data, but is
    constructed *after* this module) and the heartbeat sampler built by
    ``create_storage_backends()`` (storage_backend/__init__.py), which has
    no reference to the connector at construction time.
    """
    global _request_progress_source
    _request_progress_source = source


# Set once at connector construction time via
# register_queued_prefill_tokens_source; read on every heartbeat by
# _sample(). Same single-writer/many-reader shape as
# _request_progress_source above, so no lock is needed here either.
_queued_prefill_tokens_source: Optional[Callable[[], int]] = None


def register_queued_prefill_tokens_source(source: Callable[[], int]) -> None:
    """Called once by LMCacheConnectorV1 to expose a real prefill-token count.

    *source* is a zero-argument callable returning the total number of
    remaining (uncomputed) prefill tokens summed across requests the vLLM
    scheduler has examined and allocated resources for. This bridges the
    connector (owns the live per-request ``Request`` objects, but is
    constructed *after* this module) and the heartbeat sampler, the same way
    :func:`register_request_progress_source` does for ``request_progress``.

    When no source is registered, ``queued_prefill_tokens`` falls back to
    ``num_waiting`` (a request headcount, not a token count — see module
    docstring). Re-registering replaces the previous source rather than
    stacking; a raising source degrades only this one field, not the whole
    sample.
    """
    global _queued_prefill_tokens_source
    _queued_prefill_tokens_source = source


# Maps ReportQueueState field → ordered candidate Prometheus metric-family names.
# vLLM exposes these Gauges at /metrics (the colon prefix follows vLLM's naming
# convention). Names changed across vLLM versions and the supported target is
# unresolved (GCP pins 0.11, pyproject wants >=0.18 — see open-questions Q5), so
# we read the FIRST candidate that is present and tolerate the rest being absent:
#   * V1 renamed ``vllm:gpu_cache_usage_perc`` → ``vllm:kv_cache_usage_perc``.
#   * V1 removed request swapping; ``vllm:num_requests_swapped`` only exists on
#     older (V0) vLLM, so on V1 ``num_swapped`` correctly stays 0.
# (See knowledge-base open-questions Q13.)
_VLLM_METRIC_CANDIDATES: dict[str, tuple[str, ...]] = {
    "num_waiting": ("vllm:num_requests_waiting",),
    "num_running": ("vllm:num_requests_running",),
    "gpu_cache_usage_perc": (
        "vllm:kv_cache_usage_perc",  # vLLM V1 name (preferred)
        "vllm:gpu_cache_usage_perc",  # vLLM V0 fallback
    ),
    "num_swapped": ("vllm:num_requests_swapped",),  # V0 only; absent on V1 → 0
}

# vLLM Histogram of time spent in the WAITING phase, keyed by _sum/_count
# samples rather than a single gauge value (see
# _read_prometheus_histogram_mean_ms). Populated from FinishedRequestStats --
# i.e. this is the mean over requests that have *already finished* waiting,
# not a live estimate for the request currently being planned.
_QUEUE_TIME_HISTOGRAM_NAME = "vllm:request_queue_time_seconds"

# ---- Low-level helpers -------------------------------------------------------


def _read_prometheus_gauge(metric_name: str) -> Optional[float]:
    """Read the summed sample value for *metric_name* from the Prometheus REGISTRY.

    Iterates through all registered metric families in the default
    ``prometheus_client.REGISTRY`` and sums all sample values whose family
    name matches *metric_name* (summing across label combinations, e.g.
    multiple model names on a multi-model server).

    Args:
        metric_name: Full Prometheus metric-family name, e.g.
            ``"vllm:num_requests_waiting"``.

    Returns:
        The summed gauge value as a float, or ``None`` if the metric family is
        not registered or cannot be read for any reason.

    Notes:
        This function is fail-open: any exception returns ``None``.
        Prometheus REGISTRY is only imported lazily to avoid a hard dependency
        when vLLM is absent.
    """
    try:
        # Lazy import — zero cost when prometheus_client is absent.
        from prometheus_client import REGISTRY  # type: ignore[import-untyped]

        total: Optional[float] = None
        for metric_family in REGISTRY.collect():
            if metric_family.name != metric_name:
                continue
            for sample in metric_family.samples:
                # Skip derived _total / _created / _sum / _count samples that
                # Counter families emit alongside the gauge-like base sample.
                if sample.name != metric_name:
                    continue
                value = float(sample.value)
                total = value if total is None else total + value
        return total
    except Exception as exc:
        logger.debug("PESTO queue_stats: failed to read %r: %s", metric_name, exc)
        return None


def _read_prometheus_histogram_mean_ms(metric_name: str) -> Optional[float]:
    """Read a Histogram's _sum/_count from REGISTRY, as a mean in milliseconds.

    Unlike a Gauge, a Prometheus Histogram exposes its data as ``_sum`` and
    ``_count`` samples (plus ``_bucket`` samples this function ignores). This
    returns ``_sum / _count`` converted from seconds to milliseconds -- the
    mean value observed across all recorded samples so far.

    Args:
        metric_name: The Histogram's base name, e.g.
            ``"vllm:request_queue_time_seconds"`` (without the ``_sum``/
            ``_count`` suffix).

    Returns:
        The mean in milliseconds, or ``None`` if the metric is absent, has
        recorded zero observations, or cannot be read for any reason.

    Notes:
        This is a lagging, historical statistic -- the mean queue time of
        requests that have *already finished* waiting, not a live prediction
        for the request currently being planned. Used only because vLLM does
        not expose a live per-request queue-time forecast. Fail-open: any
        exception returns ``None``.
    """
    try:
        # Lazy import — zero cost when prometheus_client is absent.
        from prometheus_client import REGISTRY  # type: ignore[import-untyped]

        total_sum: Optional[float] = None
        total_count: Optional[float] = None
        for metric_family in REGISTRY.collect():
            if metric_family.name != metric_name:
                continue
            for sample in metric_family.samples:
                if sample.name == f"{metric_name}_sum":
                    total_sum = (total_sum or 0.0) + float(sample.value)
                elif sample.name == f"{metric_name}_count":
                    total_count = (total_count or 0.0) + float(sample.value)
        if total_sum is None or not total_count:
            return None
        return (total_sum / total_count) * 1000.0
    except Exception as exc:
        logger.debug(
            "PESTO queue_stats: failed to read histogram %r: %s", metric_name, exc
        )
        return None


def _collect_prometheus_stats() -> dict:
    """Read all tracked vLLM gauge metrics from the Prometheus REGISTRY.

    Returns:
        A dict with the subset of :data:`ZEROS_DICT` keys that could be read
        from Prometheus.  Missing metrics stay at their zero defaults.  The
        returned dict always contains all keys from :data:`ZEROS_DICT`.
    """
    result: dict = ZEROS_DICT.copy()
    for field_name, candidates in _VLLM_METRIC_CANDIDATES.items():
        val: Optional[float] = None
        for prom_name in candidates:
            val = _read_prometheus_gauge(prom_name)
            if val is not None:
                break
        if val is None:
            continue
        if field_name == "gpu_cache_usage_perc":
            result[field_name] = float(max(0.0, min(1.0, val)))
        else:
            result[field_name] = int(max(0, round(val)))

    mean_wait_ms = _read_prometheus_histogram_mean_ms(_QUEUE_TIME_HISTOGRAM_NAME)
    if mean_wait_ms is not None:
        result["estimated_wait_ms"] = mean_wait_ms
    return result


def _collect_http_stats(metrics_url: str, timeout_s: float = 0.25) -> dict:
    """Scrape tracked vLLM gauges from a Prometheus text endpoint.

    The API server owns vLLM's Prometheus registry while LMCache runs in the
    engine process. A short localhost HTTP scrape therefore provides the live
    queue state without coupling LMCache to vLLM internals. Failures return an
    empty partial result so callers can retain registry/default values.
    """
    try:
        with urlopen(metrics_url, timeout=timeout_s) as response:  # noqa: S310
            exposition = response.read().decode("utf-8")
    except Exception as exc:
        logger.debug(
            "PESTO queue_stats: failed to scrape %s: %s", metrics_url, exc
        )
        return {}

    totals: dict[str, float] = {}
    _queue_time_sum_name = f"{_QUEUE_TIME_HISTOGRAM_NAME}_sum"
    _queue_time_count_name = f"{_QUEUE_TIME_HISTOGRAM_NAME}_count"
    candidate_names = {
        metric_name
        for candidates in _VLLM_METRIC_CANDIDATES.values()
        for metric_name in candidates
    } | {_queue_time_sum_name, _queue_time_count_name}
    try:
        for line in exposition.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            metric_name = fields[0].split("{", 1)[0]
            if metric_name not in candidate_names:
                continue
            totals[metric_name] = totals.get(metric_name, 0.0) + float(fields[1])
    except (TypeError, ValueError) as exc:
        logger.debug(
            "PESTO queue_stats: invalid exposition from %s: %s", metrics_url, exc
        )
        return {}

    result: dict = {}
    for field_name, candidates in _VLLM_METRIC_CANDIDATES.items():
        value = next(
            (totals[name] for name in candidates if name in totals),
            None,
        )
        if value is None:
            continue
        if field_name == "gpu_cache_usage_perc":
            result[field_name] = float(max(0.0, min(1.0, value)))
        else:
            result[field_name] = int(max(0, round(value)))

    queue_time_sum = totals.get(_queue_time_sum_name)
    queue_time_count = totals.get(_queue_time_count_name)
    if queue_time_sum is not None and queue_time_count:
        result["estimated_wait_ms"] = (queue_time_sum / queue_time_count) * 1000.0
    return result


# ---- Public API --------------------------------------------------------------


def make_prometheus_queue_state_fn(
    extra_sampler: Optional[Callable[[], dict]] = None,
    metrics_url: Optional[str] = None,
) -> Callable[[], dict]:
    """Return a fail-open callable that samples vLLM queue state for PESTO.

    The returned callable scrapes the vLLM API server and falls back to the
    local Prometheus REGISTRY on every invocation. Approximated fields (see
    module docstring) default to zero when exact scheduler data is unavailable.
    Any unhandled exception returns :data:`ZEROS_DICT` instead of raising.

    Args:
        extra_sampler: Optional zero-argument callable returning a partial
            ``dict`` of :class:`~pesto_gms.api_models.ReportQueueState`
            kwargs (excluding ``head_id`` and ``ts_ms``).  Returned keys
            override the corresponding Prometheus-sourced values; missing keys
            fall through to the Prometheus defaults.  Errors in
            ``extra_sampler`` are silently caught (fail-open).  Pass a
            reference to the vLLM scheduler to supply exact
            ``queued_prefill_tokens``, ``running_decode_blocks``, and
            ``estimated_wait_ms`` values that are not accessible via
            Prometheus alone.
        metrics_url: Optional vLLM Prometheus endpoint. Defaults to
            ``PESTO_VLLM_METRICS_URL``. When unset or unreachable, sampling
            falls back to the current process's Prometheus registry.

    Returns:
        A zero-argument callable ``() -> dict`` returning keyword arguments
        for ``ReportQueueState(**kwargs, head_id=..., ts_ms=...)``.  The dict
        always contains all required keys with non-negative values.

    Notes:
        **Real** fields: ``num_waiting``, ``num_running``, ``num_swapped``,
        ``gpu_cache_usage_perc`` — sourced from the vLLM HTTP metrics endpoint
        or, as a fallback, the current process's Prometheus registry.
        ``estimated_wait_ms`` is also real when the
        ``vllm:request_queue_time_seconds`` histogram is present, but it's a
        lagging historical mean (over already-finished requests), not a live
        prediction — see :func:`_read_prometheus_histogram_mean_ms`.

        **Approximated** fields (GPU-free fallbacks, unless overridden via
        ``extra_sampler`` or, for ``queued_prefill_tokens``, a registered
        :func:`register_queued_prefill_tokens_source`):

        * ``active_requests`` = ``num_running + num_waiting``
        * ``queued_prefill_tokens`` = ``num_waiting`` (lower bound)
        * ``running_decode_blocks`` = 0
        * ``estimated_wait_ms`` = 0 (when the histogram has zero observations)

        Both sources are queried lazily at call time, so this function is safe
        to call before the vLLM API server is ready.
    """

    resolved_metrics_url = metrics_url or os.getenv("PESTO_VLLM_METRICS_URL")

    def _sample() -> dict:
        try:
            result = _collect_prometheus_stats()
            if resolved_metrics_url:
                result.update(_collect_http_stats(resolved_metrics_url))

            # Apply caller-supplied overrides (e.g. exact scheduler values).
            overridden_keys: set[str] = set()
            if extra_sampler is not None:
                try:
                    overrides = extra_sampler()
                    if isinstance(overrides, dict):
                        for key, value in overrides.items():
                            if key in result:
                                result[key] = value
                                overridden_keys.add(key)
                except Exception as exc:
                    logger.debug(
                        "PESTO queue_stats extra_sampler error (non-fatal): %s", exc
                    )

            # Derive composite fields after source overrides.
            num_waiting = result["num_waiting"]
            num_running = result["num_running"]
            if "active_requests" not in overridden_keys:
                result["active_requests"] = num_running + num_waiting
            # queued_prefill_tokens: each waiting request has at least one
            # pending prefill token. This is a lower bound; callers with
            # scheduler access can override it explicitly.
            if "queued_prefill_tokens" not in overridden_keys:
                result["queued_prefill_tokens"] = num_waiting

            # If a real source is registered, it supersedes the num_waiting
            # fallback above -- but never a value extra_sampler already set
            # explicitly. A broken/raising source degrades only this field,
            # keeping the fallback already computed, not the whole sample.
            if (
                "queued_prefill_tokens" not in overridden_keys
                and _queued_prefill_tokens_source is not None
            ):
                try:
                    result["queued_prefill_tokens"] = max(
                        0, int(_queued_prefill_tokens_source())
                    )
                except Exception as exc:
                    logger.debug(
                        "PESTO queue_stats queued_prefill_tokens source error "
                        "(non-fatal): %s",
                        exc,
                    )

            # A broken/raising source degrades only this one field, not the
            # whole sample -- same scoping as the extra_sampler guard above.
            try:
                result["request_progress"] = (
                    _request_progress_source() if _request_progress_source else {}
                )
            except Exception as exc:
                logger.debug(
                    "PESTO queue_stats request_progress source error (non-fatal): %s",
                    exc,
                )
                result["request_progress"] = {}

            return result

        except Exception as exc:
            logger.debug("PESTO queue_stats sample failed (fail-open): %s", exc)
            return ZEROS_DICT.copy()

    return _sample
