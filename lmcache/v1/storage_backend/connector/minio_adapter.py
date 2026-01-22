# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class MinIOConnectorAdapter(ConnectorAdapter):
    """Adapter for MinIO Server connectors."""

    def __init__(self) -> None:
        super().__init__("minio://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .minio_connector import MinIOConnector

        config = context.config
        assert config is not None

        # Parse URL: minio://[access_key:secret_key@]endpoint/bucket
        # Examples:
        # minio://mykey:mysecret@localhost:9000/mybucket
        # minio://localhost:9000/mybucket
        url = context.url.removeprefix("minio://")

        # Extract credentials and endpoint
        if "@" in url:
            creds, rest = url.split("@", 1)
            if ":" in creds:
                access_key, secret_key = creds.split(":", 1)
            else:
                access_key = creds
                secret_key = ""
        else:
            access_key = ""
            secret_key = ""
            rest = url

        # Extract endpoint and bucket
        if "/" in rest:
            endpoint, bucket = rest.split("/", 1)
        else:
            endpoint = rest
            bucket = None

        # Get config from extra_config with defaults (matching S3 adapter pattern)
        extra_config = config.extra_config if config.extra_config is not None else {}

        # Override with extra_config if provided
        access_key = extra_config.get("minio_access_key", access_key)
        secret_key = extra_config.get("minio_secret_key", secret_key)
        bucket = extra_config.get("minio_bucket", bucket)
        self.minio_endpoint = extra_config.get("minio_endpoint", endpoint)
        self.minio_max_inflight_reqs = int(
            extra_config.get("minio_max_inflight_reqs", 64)
        )
        self.minio_secure = bool(extra_config.get("minio_secure", True))
        self.minio_region = extra_config.get("minio_region", None)
        self.minio_file_prefix = extra_config.get("minio_file_prefix", None)

        if not bucket:
            raise ValueError(
                "MinIO bucket must be specified in URL (minio://endpoint/bucket) "
                "or in config (minio_bucket)"
            )

        if not access_key or not secret_key:
            raise ValueError(
                "MinIO credentials must be specified in URL (minio://key:secret@endpoint) "
                "or in config (minio_access_key, minio_secret_key)"
            )

        if context.metadata is None:
            raise ValueError("metadata is required for MinIOConnector")

        logger.info(f"Creating MinIO connector for endpoint: {self.minio_endpoint}")

        return MinIOConnector(
            minio_endpoint=self.minio_endpoint,
            minio_access_key=access_key,
            minio_secret_key=secret_key,
            minio_bucket=bucket,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            minio_file_prefix=self.minio_file_prefix,
            minio_max_inflight_reqs=self.minio_max_inflight_reqs,
            minio_secure=self.minio_secure,
            minio_region=self.minio_region,
        )
