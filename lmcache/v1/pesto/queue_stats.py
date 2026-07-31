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
* ``queued_prefill_tokens`` = ``num_waiting`` (lower bound; a precise value
  requires the vLLM scheduler's prefill-token queue, which is not accessible
  without a GPU runtime reference — see extension point below)
* ``running_decode_blocks`` = 0 (block-level scheduler access unavailable
  GPU-free; post-MVP extension)
* ``estimated_wait_ms``     = continuous-batching admission-delay estimate
  derived from ``gpu_cache_usage_perc`` / ``num_waiting`` / ``num_swapped``
  (see :func:`_estimate_wait_ms`). Mirrors the admission-delay model used by
  GMS's ``_ContinuousBatchingQueueEstimator``
  (``pesto_gms/policies/defaults.py``) so the head's self-report is a
  genuine second opinion, not a copy — GMS blends the two. The constants
  below are independently-tunable placeholders (see open-questions Q16).

Extension point
---------------
Pass an ``extra_sampler: Callable[[], dict]`` to
:func:`make_prometheus_queue_state_fn` to override any subset of the above
fields with exact scheduler values.  This is the intended hook when the caller
has a reference to the vLLM scheduler object (e.g. from a running engine).
Any error in ``extra_sampler`` is silently caught (fail-open).

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
}

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

# ---- estimated_wait_ms model --------------------------------------------------
# Continuous-batching admission-delay approximation, independently tunable from
# (but structurally mirroring) GMS's _ContinuousBatchingQueueEstimator in
# pesto_gms/policies/defaults.py. Unlike that estimator, this runs per-head with
# no access to a specific incoming request's token count, so the "queued
# prefill wait" term uses an assumed average request size instead of an exact
# one. Placeholder constants pending calibration (open-questions Q16).
_TPOT_MS: float = 62.6  # decode time-per-output-token
_AVG_DECODE_TOKENS: int = 256  # assumed average remaining decode length
_HEADROOM_THRESHOLD: float = 0.85  # gpu_cache_usage_perc below which new requests admit immediately
_AVG_PREFILL_TOKENS: int = 512  # assumed average prompt length of a queued request
_PREFILL_THROUGHPUT_TPM: float = 50.0  # tokens/ms (PM-06 calibration)


def _estimate_wait_ms(
    gpu_cache_usage_perc: float,
    num_waiting: int,
    num_swapped: int,
) -> float:
    """Approximate admission-delay for a new request arriving at this head.

    Headroom regime (KV cache has room and nothing is already queued): a new
    request joins the next batch iteration immediately, so wait ~= 0.
    Memory-bound regime: the new request waits for running sequences to free
    enough KV blocks, approximated as half of an assumed average decode
    length; any requests already waiting ahead of it add their assumed
    prefill cost on top.
    """
    in_headroom = (
        gpu_cache_usage_perc < _HEADROOM_THRESHOLD
        and num_waiting == 0
        and num_swapped == 0
    )
    base_wait_ms = 0.0 if in_headroom else (_AVG_DECODE_TOKENS * 0.5) * _TPOT_MS
    queue_prefill_wait_ms = (
        num_waiting * _AVG_PREFILL_TOKENS / _PREFILL_THROUGHPUT_TPM
    )
    return max(0.0, base_wait_ms + queue_prefill_wait_ms)


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
    candidate_names = {
        metric_name
        for candidates in _VLLM_METRIC_CANDIDATES.values()
        for metric_name in candidates
    }
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

        **Approximated** fields (GPU-free fallbacks, unless overridden via
        ``extra_sampler``):

        * ``active_requests`` = ``num_running + num_waiting``
        * ``queued_prefill_tokens`` = ``num_waiting`` (lower bound)
        * ``running_decode_blocks`` = 0
        * ``estimated_wait_ms`` = continuous-batching admission-delay
          estimate, see :func:`_estimate_wait_ms`

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
            if "estimated_wait_ms" not in overridden_keys:
                result["estimated_wait_ms"] = _estimate_wait_ms(
                    result["gpu_cache_usage_perc"], num_waiting, result["num_swapped"]
                )

            return result

        except Exception as exc:
            logger.debug("PESTO queue_stats sample failed (fail-open): %s", exc)
            return ZEROS_DICT.copy()

    return _sample
