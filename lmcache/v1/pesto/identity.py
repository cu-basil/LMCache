# SPDX-License-Identifier: Apache-2.0
"""PESTO identity resolver — shared key-identity contract for head-side components.

The identity tuple ``(namespace, head_id, tokenizer_id, chat_template_id,
endpoint)`` must match the values the gateway encodes in every ``BlockKey``
it sends to GMS.  Any mismatch means the GMS canonical-lookup will never bind
a gateway-computed key to a head-reported location — cross-head reuse silently
misses.

Contract
--------
* ``namespace``        — SHARED tenant id; must equal ``PESTO_GATEWAY_NAMESPACE``
                         (gateway default ``"default"``).  Set via
                         ``extra_config["pesto_namespace"]``.  **NOT the
                         per-head instance id.**
* ``head_id``          — per-head instance identifier; set via
                         ``extra_config["pesto_gms_instance_id"]``
                         (e.g. ``"pesto-0"``).
* ``tokenizer_id``     — stable tokenizer string; resolves as
                         ``pesto_tokenizer_id`` → ``model_id`` → ``"default"``.
                         The gateway uses ``cfg.effective_tokenizer`` (which
                         also defaults to the model id), so the head must do the
                         same to achieve parity.
* ``chat_template_id`` — set via ``extra_config["pesto_chat_template_id"]``;
                         defaults ``"default"`` on both sides (OK as-is).
* ``endpoint``         — head's own vLLM URL for location reporting; only used
                         by ``LocationReporter``.
"""

# Standard
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PestoIdentity:
    """Immutable snapshot of a head's PESTO identity fields.

    All five fields are required so callers cannot accidentally omit one.
    Use :func:`resolve_identity` to construct from ``extra_config``.

    Attributes:
        namespace: Shared tenant namespace (must match gateway
            ``PESTO_GATEWAY_NAMESPACE``).
        head_id: Per-head instance identifier (unique across the cluster).
        tokenizer_id: Tokenizer string used in ``BlockKey`` construction.
        chat_template_id: Chat-template string used in ``BlockKey`` construction.
        endpoint: This head's vLLM endpoint URL (used for location reporting).
    """

    namespace: str
    head_id: str
    tokenizer_id: str
    chat_template_id: str
    endpoint: str


def resolve_identity(
    extra_config: dict,
    *,
    model_id: Optional[str] = None,
) -> PestoIdentity:
    """Derive a :class:`PestoIdentity` from LMCache ``extra_config``.

    Resolution order for each field:

    * **namespace** — ``extra_config["pesto_namespace"]`` → ``"default"``.
      This is the *shared* tenant namespace, **not** the per-head instance id.
      It must equal the gateway's ``PESTO_GATEWAY_NAMESPACE`` (default ``"default"``).

    * **head_id** — ``extra_config["pesto_gms_instance_id"]`` → ``"default"``.
      This uniquely identifies the head within the cluster.

    * **tokenizer_id** — ``extra_config["pesto_tokenizer_id"]`` →
      ``model_id`` (caller-supplied, e.g. ``metadata.model_name``) →
      ``"default"``.  Falling back to ``model_id`` matches the gateway's
      ``cfg.effective_tokenizer`` (which also defaults to the model id), so
      the two sides produce the same string without any explicit config.

    * **chat_template_id** — ``extra_config["pesto_chat_template_id"]`` →
      ``"default"``.  The empty-string guard (``or "default"``) prevents an
      accidentally blank YAML value from creating a mismatched key.

    * **endpoint** — ``extra_config["pesto_endpoint"]`` →
      ``"http://localhost:8000"``.

    Args:
        extra_config: The ``extra_config`` dict from
            :class:`~lmcache.v1.config.LMCacheEngineConfig` (may be ``None``
            or empty — handled internally).
        model_id: The serving model id from
            :class:`~lmcache.v1.metadata.LMCacheMetadata` (``metadata.model_name``
            or ``metadata.served_model_name``).  Used as the ``tokenizer_id``
            fallback so head identity automatically matches the gateway's
            ``effective_tokenizer`` without requiring an explicit config key.

    Returns:
        A frozen :class:`PestoIdentity` with all fields resolved to non-empty
        strings.
    """
    ec: dict = extra_config or {}

    namespace: str = ec.get("pesto_namespace", "default") or "default"
    head_id: str = ec.get("pesto_gms_instance_id", "default") or "default"

    # tokenizer_id: explicit config → model_id fallback → "default"
    raw_tok: Optional[str] = ec.get("pesto_tokenizer_id") or model_id
    tokenizer_id: str = raw_tok or "default"

    chat_template_id: str = (
        ec.get("pesto_chat_template_id", "default") or "default"
    )
    endpoint: str = ec.get("pesto_endpoint", "http://localhost:8000") or "http://localhost:8000"

    return PestoIdentity(
        namespace=namespace,
        head_id=head_id,
        tokenizer_id=tokenizer_id,
        chat_template_id=chat_template_id,
        endpoint=endpoint,
    )
