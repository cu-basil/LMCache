# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import os
import threading
from io import BytesIO
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


def create_test_config(minio_endpoint: str = "localhost:9000"):
    """Create a test configuration for MinIOConnector."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        remote_url=f"minio://testkey:testsecret@{minio_endpoint}/testbucket",
        remote_serde="naive",
        lmcache_instance_id="test_instance",
    )
    return config


def create_test_metadata():
    """Create a test metadata for LMCacheEngineMetadata."""
    return LMCacheEngineMetadata(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        fmt="vllm",
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey("vllm", "test_model", 3, 123, hash(key_id), torch.bfloat16)


def create_test_memory_obj(shape=(2, 16, 8, 128), dtype=torch.bfloat16) -> MemoryObj:
    """Create a test MemoryObj using AdHocMemoryAllocator for testing."""
    allocator = AdHocMemoryAllocator(device="cpu")
    memory_obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_T2D)
    return memory_obj


@pytest.fixture
def async_loop():
    """Create an asyncio event loop running in a separate thread for testing."""
    loop = asyncio.new_event_loop()

    # Start the event loop in a separate thread
    # Standard
    import threading

    # First Party
    from lmcache.utils import start_loop_in_thread_with_exceptions

    thread = threading.Thread(
        target=start_loop_in_thread_with_exceptions,
        args=(loop,),
        name="test-async-loop",
    )
    thread.start()

    yield loop

    # Cleanup: stop the loop and wait for thread to finish
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    return LocalCPUBackend(config, memory_allocator=memory_allocator)


@pytest.fixture
def mock_minio_client():
    """Create a mock MinIO client for testing."""
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = True
    return mock_client


@pytest.fixture
def remote_backend_with_minio(async_loop, local_cpu_backend, mock_minio_client):
    """Create a RemoteBackend with MinIOConnector for testing."""
    config = create_test_config()
    metadata = create_test_metadata()

    # Patch the Minio client creation
    with patch("lmcache.v1.storage_backend.connector.minio_connector.Minio") as mock_minio:
        mock_minio.return_value = mock_minio_client

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )
        yield backend, mock_minio_client
        backend.local_cpu_backend.memory_allocator.close()
        backend.close()


class TestMinIOConnectorAdapter:
    """Test cases for MinIOConnectorAdapter URL parsing."""

    def test_adapter_url_parsing_with_credentials(self):
        """Test MinIOConnectorAdapter URL parsing with credentials."""
        from lmcache.v1.storage_backend.connector.minio_adapter import (
            MinIOConnectorAdapter,
        )

        adapter = MinIOConnectorAdapter()
        assert adapter.can_parse("minio://key:secret@localhost:9000/bucket")

    def test_adapter_url_parsing_without_credentials(self):
        """Test MinIOConnectorAdapter URL parsing without credentials in URL."""
        from lmcache.v1.storage_backend.connector.minio_adapter import (
            MinIOConnectorAdapter,
        )

        adapter = MinIOConnectorAdapter()
        assert adapter.can_parse("minio://localhost:9000/bucket")

    def test_adapter_rejects_non_minio_urls(self):
        """Test that MinIOConnectorAdapter rejects non-MinIO URLs."""
        from lmcache.v1.storage_backend.connector.minio_adapter import (
            MinIOConnectorAdapter,
        )

        adapter = MinIOConnectorAdapter()
        assert not adapter.can_parse("s3://bucket.s3.amazonaws.com")
        assert not adapter.can_parse("fs:///path/to/file")


class TestMinIOConnector:
    """Test cases for MinIOConnector via RemoteBackend."""

    def test_init(self, async_loop, local_cpu_backend):
        """Test MinIOConnector initialization via RemoteBackend."""
        config = create_test_config()
        metadata = create_test_metadata()

        with patch("lmcache.v1.storage_backend.connector.minio_connector.Minio") as mock_minio:
            mock_minio_client = MagicMock()
            mock_minio_client.bucket_exists.return_value = True
            mock_minio.return_value = mock_minio_client

            backend = RemoteBackend(
                config=config,
                metadata=metadata,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
            )

            assert backend.dst_device == "cpu"
            assert backend.local_cpu_backend == local_cpu_backend
            assert backend.remote_url == "minio://testkey:testsecret@localhost:9000/testbucket"

            local_cpu_backend.memory_allocator.close()
            backend.close()

    def test_contains_key_not_exists(self, remote_backend_with_minio):
        """Test contains() when key doesn't exist in MinIO."""
        backend, mock_client = remote_backend_with_minio

        # Mock stat_object to raise S3Error for non-existent key
        from unittest.mock import MagicMock
        from minio.error import S3Error

        mock_response = MagicMock()
        mock_client.stat_object.side_effect = S3Error(
            mock_response, "NoSuchKey", "Not found", "/test/key", "request-123", "host-123"
        )

        key = create_test_key(1)
        assert not backend.contains(key)

    def test_get_blocking_key_not_exists(self, remote_backend_with_minio):
        """Test get_blocking() when key doesn't exist in MinIO."""
        backend, mock_client = remote_backend_with_minio

        from unittest.mock import MagicMock
        from minio.error import S3Error

        mock_response = MagicMock()
        mock_client.stat_object.side_effect = S3Error(
            mock_response, "NoSuchKey", "Not found", "/test/key", "request-123", "host-123"
        )

        key = create_test_key(2)
        result = backend.get_blocking(key)

        assert result is None

    def test_put_and_get_roundtrip(self, remote_backend_with_minio):
        """Test put and get roundtrip for MinIOConnector."""
        backend, mock_client = remote_backend_with_minio

        key = create_test_key(3)
        memory_obj = create_test_memory_obj()

        # Mock stat_object to simulate object exists
        from unittest.mock import MagicMock

        stat_result = MagicMock()
        stat_result.size = memory_obj.get_size()
        mock_client.stat_object.return_value = stat_result

        # Mock put_object to succeed
        mock_client.put_object.return_value = None

        # Mock get_object to return the data
        response = MagicMock()
        response.stream.return_value = [
            memory_obj.raw_data
        ]  # Return full data in one chunk
        mock_client.get_object.return_value = response

        # Put data to MinIO
        future = backend.submit_put_task(key, memory_obj)
        # Wait for the async put to complete
        if future:
            future.result(timeout=5.0)

        # Check that key exists
        assert backend.contains(key)

        # Get data back
        result = backend.get_blocking(key)

        assert result is not None
        assert isinstance(result, MemoryObj)
        assert result.metadata.shape == memory_obj.metadata.shape
        assert result.metadata.dtype == memory_obj.metadata.dtype

    def test_batched_put_and_get(self, remote_backend_with_minio):
        """Test batched put and get operations."""
        backend, mock_client = remote_backend_with_minio

        keys = [create_test_key(i) for i in range(3)]
        memory_objs = [create_test_memory_obj() for _ in range(3)]

        # Mock stat_object
        from unittest.mock import MagicMock

        stat_result = MagicMock()
        stat_result.size = memory_objs[0].get_size()
        mock_client.stat_object.return_value = stat_result

        # Mock put_object
        mock_client.put_object.return_value = None

        # Mock get_object
        response = MagicMock()
        response.stream.return_value = [memory_objs[0].raw_data]
        mock_client.get_object.return_value = response

        # Batched put
        futures = [
            backend.submit_put_task(key, memory_obj)
            for key, memory_obj in zip(keys, memory_objs, strict=False)
        ]
        for future in filter(None, futures):
            future.result(timeout=5.0)

        # Check all keys exist
        for key in keys:
            assert backend.contains(key)

        # Batched get
        results = backend.batched_get_blocking(keys)

        assert results is not None
        assert len(results) == 3
        for result, original in zip(results, memory_objs, strict=False):
            assert result is not None
            assert result.metadata.shape == original.metadata.shape
            assert result.metadata.dtype == original.metadata.dtype

    def test_config_extra_config(self, async_loop, local_cpu_backend):
        """Test MinIOConnector with extra configuration."""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            remote_url="minio://localhost:9000/testbucket",
            remote_serde="naive",
            lmcache_instance_id="test_instance",
            extra_config={
                "minio_access_key": "mykey",
                "minio_secret_key": "mysecret",
                "minio_bucket": "mybucket",
                "minio_secure": False,
                "minio_max_inflight_reqs": 32,
            },
        )
        metadata = create_test_metadata()

        with patch("lmcache.v1.storage_backend.connector.minio_connector.Minio") as mock_minio:
            mock_minio_client = MagicMock()
            mock_minio_client.bucket_exists.return_value = True
            mock_minio.return_value = mock_minio_client

            backend = RemoteBackend(
                config=config,
                metadata=metadata,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
            )

            # Verify Minio was called with correct parameters
            mock_minio.assert_called()
            call_kwargs = mock_minio.call_args[1]
            assert call_kwargs["access_key"] == "mykey"
            assert call_kwargs["secret_key"] == "mysecret"
            assert call_kwargs["secure"] is False

            local_cpu_backend.memory_allocator.close()
            backend.close()

    def test_ping_success(self, remote_backend_with_minio):
        """Test ping operation success."""
        backend, mock_client = remote_backend_with_minio

        mock_client.bucket_exists.return_value = True

        # Access the connector directly
        connector = backend.connection

        # We need to run this in the event loop thread
        future = asyncio.run_coroutine_threadsafe(
            connector.ping(), backend.loop
        )
        result = future.result(timeout=5.0)

        assert result == 0

    def test_ping_failure(self, remote_backend_with_minio):
        """Test ping operation failure."""
        backend, mock_client = remote_backend_with_minio

        from unittest.mock import MagicMock
        from minio.error import S3Error

        mock_response = MagicMock()
        mock_client.bucket_exists.side_effect = S3Error(
            mock_response, "InternalError", "Connection failed", "/bucket", "request-123", "host-123"
        )

        connector = backend.connection

        future = asyncio.run_coroutine_threadsafe(
            connector.ping(), backend.loop
        )
        result = future.result(timeout=5.0)

        assert result != 0
