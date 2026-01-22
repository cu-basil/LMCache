# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import IntEnum, auto
from functools import partial
from typing import List, Optional
import asyncio
import ctypes
import io
import mmap
import os
import tempfile

# Third Party
from minio import Minio
from minio.error import S3Error

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


class AdhocSharedMemoryManager:
    """
    A shared memory manager that allocates shared memory buffers
    on demand.
    """

    def __init__(
        self,
        shm_buffers: list[int],
        shm_names: list[str],
        mmaps: list[mmap.mmap],
    ):
        self.shm_buffers = shm_buffers
        self.shm_names = shm_names
        self.mmaps = mmaps

    def allocate(self) -> tuple[str, int]:
        """
        Allocate a shared memory buffer and return its name and a bytearray
        that can be used to access the buffer.
        """
        if not self.shm_buffers:
            raise RuntimeError("No more shared memory buffers available")

        shm = self.shm_buffers.pop()
        shm_name = self.shm_names.pop()
        return shm_name, shm

    def free(
        self,
        shm_name: str,
        shm: int,
    ) -> None:
        """
        Free a shared memory buffer.
        """

        self.shm_buffers.append(shm)
        self.shm_names.append(shm_name)

    def close(self):
        # let python GC clean up mmap inodes
        for mm in self.mmaps:
            mm.close()
        for shm_name in self.shm_names:
            try:
                os.unlink(shm_name)
            except FileNotFoundError:
                pass  # file probably already removed


class MinIOConnector(RemoteConnector):
    """
    MinIO remote connector for S3-compatible object storage
    """

    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        minio_bucket: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        minio_file_prefix: Optional[str],
        minio_max_inflight_reqs: int,
        minio_secure: bool = True,
        minio_region: Optional[str] = None,
    ):
        # Initialize base class to set full_chunk_size and other metadata
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        # Set minio_part_size to full_chunk_size (matching S3 connector pattern)
        self.minio_part_size = self.full_chunk_size

        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_bucket = minio_bucket
        self.minio_prefix = minio_file_prefix
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.minio_max_inflight_reqs = minio_max_inflight_reqs
        self.minio_secure = minio_secure
        self.minio_region = minio_region or "us-east-1"

        # Initialize MinIO client
        logger.info(f"Initializing MinIO client for endpoint: {minio_endpoint}")
        self.minio_client = Minio(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure,
            region=self.minio_region,
        )

        # Verify bucket exists or create it
        self._verify_bucket()

        # Cache for object sizes to avoid repeated HEAD requests
        self.object_size_cache: dict[str, int] = {}

        # Circuit breaker for connection failures (matching S3 connector pattern)
        self.connection_failures = 0
        self.max_connection_failures = 3
        self.connection_disabled = False

        self.inflight_sema = asyncio.Semaphore(minio_max_inflight_reqs)
        self.pq_executor = AsyncPQExecutor(loop)

    def _verify_bucket(self):
        """Verify that the bucket exists."""
        try:
            self.minio_client.bucket_exists(self.minio_bucket)
            logger.info(f"MinIO bucket '{self.minio_bucket}' verified")
        except S3Error as e:
            logger.error(f"Failed to verify MinIO bucket: {e}")
            raise

    def post_init(self):
        logger.info("Post-initializing MinIO connector")

        shm_name_prefix = "minio_shm"
        shms = []
        shm_names = []
        mmaps = []
        for i in range(self.minio_max_inflight_reqs):
            shm_name = f"{shm_name_prefix}_{i}"

            shm = tempfile.NamedTemporaryFile(
                prefix=shm_name, suffix=".part", dir="/dev/shm", delete=False
            )

            os.ftruncate(shm.fileno(), self.full_chunk_size)

            with open(shm.name, "r+b") as f:
                mm = mmap.mmap(f.fileno(), self.full_chunk_size)
                # create a char buffer view over the mmap
                buf = ctypes.c_char.from_buffer(mm)
                addr = ctypes.addressof(buf)

            shms.append(addr)
            shm_names.append(shm.name)
            mmaps.append(mm)

        self.adhoc_shm_manager = AdhocSharedMemoryManager(
            shm_buffers=shms,
            shm_names=shm_names,
            mmaps=mmaps,
        )

    def _format_safe_path(self, key_str: str) -> str:
        """
        Generate a safe path for the MinIO key.
        """
        flat_key_str = key_str.replace("/", "_")
        if self.minio_prefix:
            path = f"{self.minio_prefix}/{flat_key_str}"
        else:
            path = flat_key_str
        return path

    def _get_object_size_sync(self, key_str: str) -> int:
        """
        Get object size from MinIO using stat_object.
        """
        try:
            stat = self.minio_client.stat_object(
                self.minio_bucket,
                self._format_safe_path(key_str),
            )
            return stat.size
        except S3Error as e:
            logger.debug(f"Exception in `_get_object_size_sync`: {e}")
            return 0

    async def _get_object_size_async(self, key_str: str) -> int:
        """
        Asynchronously get object size from MinIO.
        """
        return await self.loop.run_in_executor(
            None,
            self._get_object_size_sync,
            key_str,
        )

    async def exists(self, key: CacheEngineKey) -> bool:
        return self.exists_sync(key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        # Circuit breaker: if connection is disabled, return False
        if self.connection_disabled:
            return False

        key_str = key.to_string()
        if key_str in self.object_size_cache:
            return self.object_size_cache[key_str] > 0
        cache_size = self._get_object_size_sync(key_str)
        if cache_size > 0:
            self.object_size_cache[key_str] = cache_size
            return True
        return False

    def _minio_download_sync(
        self,
        key_str: str,
        recv_path: str,
    ):
        """
        Download a file from MinIO synchronously.
        """
        try:
            response = self.minio_client.get_object(
                self.minio_bucket,
                self._format_safe_path(key_str),
            )
            with open(recv_path, "wb") as f:
                for data in response.stream(amt=1024 * 1024):
                    f.write(data)
            response.close()
            response.release_conn()
        except S3Error as e:
            raise RuntimeError(
                f"Failed to download {key_str} from MinIO: {e}"
            )

    async def _minio_download_async(
        self,
        key_str: str,
        recv_path: str,
    ):
        """
        Download a file from MinIO asynchronously.
        """
        await self.loop.run_in_executor(
            None,
            self._minio_download_sync,
            key_str,
            recv_path,
        )

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # Circuit breaker: if connection is disabled, return None immediately
        if self.connection_disabled:
            logger.debug(
                f"MinIO connection disabled. Skipping download for {key.to_string()}"
            )
            return None

        key_str = key.to_string()

        obj_size = self.object_size_cache.get(key_str, None)

        if obj_size is None:
            obj_size = await self._get_object_size_async(key_str)
            if obj_size <= 0:
                self.object_size_cache[key_str] = 0
                return None
            self.object_size_cache[key_str] = obj_size

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes,
            self.meta_dtypes,
            self.meta_fmt,
        )

        if memory_obj is None:
            return None

        # Check if stored size matches expected size (matching S3 connector pattern)
        if obj_size != memory_obj.get_size():
            logger.error(
                f"Size mismatch for {key_str}: MinIO has {obj_size} bytes, "
                f"but current config expects {memory_obj.get_size()} bytes. "
                f"This usually means the data was stored with different chunk_size "
                f"or model configuration. Please use matching config or clear MinIO."
            )
            memory_obj.ref_count_down()
            return None

        await self.inflight_sema.acquire()

        recv_path, shm = self.adhoc_shm_manager.allocate()

        try:
            await self._minio_download_async(
                key_str=key_str,
                recv_path=recv_path,
            )

            dst_ptr = memory_obj.data_ptr
            ctypes.memmove(dst_ptr, shm, obj_size)

            self.adhoc_shm_manager.free(recv_path, shm)

            # Reset failure counter on success
            self._reset_connection_failures()

            return memory_obj
        except Exception as e:
            error_msg = str(e)

            # Update connection failures and check if it's a connection error
            is_connection_error = self._update_connection_failures(error_msg)

            if not is_connection_error:
                # Log non-connection errors
                logger.error(f"Failed to get {key_str} from MinIO: {e}")

            self.adhoc_shm_manager.free(recv_path, shm)
            memory_obj.ref_count_down()
            return None
        finally:
            self.inflight_sema.release()

    def on_get_done(
        self,
        obj_size: int,
        memory_obj: MemoryObj,
        shm: int,
        recv_path: str,
        fut: asyncio.Future,
    ):
        try:
            if memory_obj is None or shm is None:
                return None

            # Check if the download task had an error
            fut.result()

            dst_ptr = memory_obj.data_ptr
            ctypes.memmove(dst_ptr, shm, obj_size)

            self.adhoc_shm_manager.free(recv_path, shm)
        except Exception as e:
            logger.error(f"on_get_done failed for {recv_path}: {e}")
        finally:
            self.inflight_sema.release()

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        # Circuit breaker: if connection is disabled, return all None
        if self.connection_disabled:
            logger.debug(
                f"MinIO connection disabled. "
                f"Skipping batched download for {len(keys)} keys"
            )
            return [None] * len(keys)

        memory_objs: List[Optional[MemoryObj]] = []
        futures = []
        future_to_memobj_idx = []

        # It is okay for len(keys) > self.minio_max_inflight_reqs
        # but it will be slower.
        if len(keys) > self.minio_max_inflight_reqs:
            logger.warning(
                f"More keys {len(keys)} to get than "
                f"max inflight requests {self.minio_max_inflight_reqs}. "
                "This will cause slower retrieval."
            )

        for idx, key in enumerate(keys):
            key_str = key.to_string()

            obj_size = self.object_size_cache.get(key_str, None)

            if obj_size is None:
                obj_size = await self._get_object_size_async(key_str)
                if obj_size <= 0:
                    self.object_size_cache[key_str] = 0
                    memory_objs.append(None)
                    continue
                self.object_size_cache[key_str] = obj_size

            memory_obj = self.local_cpu_backend.allocate(
                self.meta_shapes,
                self.meta_dtypes,
                self.meta_fmt,
            )

            if not memory_obj:
                memory_objs.append(None)
                continue

            # Check if stored size matches expected size (matching S3 connector pattern)
            if obj_size != memory_obj.get_size():
                logger.error(
                    f"Size mismatch for {key_str}: MinIO has {obj_size} bytes, "
                    f"but current config expects {memory_obj.get_size()} bytes. "
                    f"Skipping this key."
                )
                memory_obj.ref_count_down()
                memory_objs.append(None)
                continue

            memory_objs.append(memory_obj)

            await self.inflight_sema.acquire()

            # freeing is done in on_get_done callback
            recv_path, shm = self.adhoc_shm_manager.allocate()
            fut = asyncio.ensure_future(
                self._minio_download_async(
                    key_str=key_str,
                    recv_path=recv_path,
                )
            )
            fut.add_done_callback(
                partial(self.on_get_done, obj_size, memory_obj, shm, recv_path)
            )
            futures.append(fut)
            future_to_memobj_idx.append(len(memory_objs) - 1)

        # Use return_exceptions to prevent one failure from stopping all downloads
        results = await asyncio.gather(*futures, return_exceptions=True)

        had_success = False

        for future_idx, result in enumerate(results):
            memobj_idx = future_to_memobj_idx[future_idx]

            if isinstance(result, Exception):
                error_msg = str(result)

                is_connection_error = self._update_connection_failures(error_msg)

                if not is_connection_error:
                    # Log non-connection errors
                    logger.error(
                        f"Failed to download key at index {memobj_idx}: {error_msg}"
                    )
                # Release the memory object for failed download
                memobj = memory_objs[memobj_idx]
                if memobj is not None:
                    memobj.ref_count_down()
                    memory_objs[memobj_idx] = None
            else:
                had_success = True

        if had_success:
            self._reset_connection_failures()

        return memory_objs

    def _minio_upload_sync(
        self,
        key_str: str,
        send_path: str,
        file_size: int,
    ):
        """
        Upload a file to MinIO synchronously.
        """
        try:
            with open(send_path, "rb") as f:
                self.minio_client.put_object(
                    self.minio_bucket,
                    self._format_safe_path(key_str),
                    f,
                    length=file_size,
                )
            logger.debug(f"Uploaded {key_str} to MinIO successfully")
        except S3Error as e:
            raise RuntimeError(f"Upload failed in MinIOConnector: {e}")

    async def _minio_upload_async(
        self,
        key_str: str,
        send_path: str,
        file_size: int,
    ):
        """
        Upload a file to MinIO asynchronously.
        """
        await self.loop.run_in_executor(
            None,
            self._minio_upload_sync,
            key_str,
            send_path,
            file_size,
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Store data to MinIO
        """
        # Circuit breaker: if connection is disabled, just log and return
        if self.connection_disabled:
            logger.debug(
                f"MinIO connection disabled due to repeated failures. "
                f"Skipping upload for {key.to_string()}"
            )
            return

        key_str = key.to_string()

        # Note: MinIO can handle objects of any size. The part_size is only used for
        # multipart uploads of large objects. Small chunks are uploaded directly.
        logger.debug(
            f"Uploading {key_str} to MinIO, size: {memory_obj.get_physical_size()} bytes"
        )

        await self.inflight_sema.acquire()
        send_path, shm = self.adhoc_shm_manager.allocate()
        logger.debug("Allocated shared memory for MinIO upload")

        try:
            buffer_ptr = memory_obj.data_ptr
            ctypes.memmove(shm, buffer_ptr, memory_obj.get_physical_size())
            logger.debug("Data copy to MinIO buffer completed")

            await self._minio_upload_async(
                key_str,
                send_path,
                memory_obj.get_physical_size(),
            )

            self.object_size_cache[key_str] = memory_obj.get_physical_size()
            logger.debug(f"Uploaded {key_str} to MinIO successfully")

            # Reset failure counter on success
            self._reset_connection_failures()
        except Exception as e:
            error_msg = str(e)

            # Update connection failures and check if it's a connection error
            is_connection_error = self._update_connection_failures(error_msg)

            if not is_connection_error:
                # Log non-connection errors
                logger.error(f"Failed to upload {key_str} to MinIO: {e}")
        finally:
            self.inflight_sema.release()
            self.adhoc_shm_manager.free(send_path, shm)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        return await self.pq_executor.submit_job(
            self._put,
            key=key,
            memory_obj=memory_obj,
            priority=Priorities.PUT,
        )

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self, lookup_id: str, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        # Circuit breaker: if connection is disabled, return 0
        if self.connection_disabled:
            return 0

        num_hit_counts = 0
        for key in keys:
            key_str = key.to_string()
            cached_size = self.object_size_cache.get(key_str, None)
            if cached_size is not None:
                if cached_size > 0:
                    num_hit_counts += 1
                    continue
                else:
                    return num_hit_counts

            obj_size = await self._get_object_size_async(key_str)
            if not obj_size > 0:
                self.object_size_cache[key_str] = 0
                return num_hit_counts

            self.object_size_cache[key_str] = obj_size
            num_hit_counts += 1

        return num_hit_counts

    async def batched_async_contains(
        self, lookup_id: str, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.PEEK,
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # batched get is already a coroutine
        result = await self.batched_get(keys)
        return [r for r in result if r is not None]

    async def batched_get_non_blocking(
        self, lookup_id: str, keys: List[CacheEngineKey]
    ) -> List[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )

    async def list(self) -> List[str]:
        raise NotImplementedError

    def support_ping(self) -> bool:
        return True

    async def ping(self) -> int:
        """
        Ping the MinIO server.
        Returns 0 if successful, non-zero otherwise.
        """
        try:
            # Simple check by trying to list bucket
            await self.loop.run_in_executor(
                None,
                lambda: self.minio_client.bucket_exists(self.minio_bucket),
            )
            return 0
        except Exception as e:
            logger.warning(f"MinIO ping failed: {e}")
            return 1

    def support_batched_get(self) -> bool:
        return True

    def _update_connection_failures(self, error_msg: str) -> bool:
        """
        Update connection failure counter and check if it's a connection error.
        Returns True if it's a connection error, False otherwise.
        """
        # Check if it's a connection error
        is_connection_error = (
            "CONNECTION_REFUSED" in error_msg
            or "SOCKET" in error_msg
            or "DNS" in error_msg
            or "TIMEOUT" in error_msg
            or "Connection" in error_msg
            or "connection" in error_msg
        )

        if is_connection_error:
            self.connection_failures += 1
            logger.error(
                f"MinIO connection error ({self.connection_failures}/"
                f"{self.max_connection_failures}): {error_msg}"
            )

            if self.connection_failures >= self.max_connection_failures:
                self.connection_disabled = True
                logger.error(
                    f"MinIO connection disabled after "
                    f"{self.max_connection_failures} "
                    f"consecutive failures. "
                    f"All future MinIO operations will be skipped."
                )

        return is_connection_error

    def _reset_connection_failures(self):
        """Reset connection failure counter on successful operation."""
        if self.connection_failures > 0:
            logger.info("MinIO connection recovered")
            self.connection_failures = 0

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
        self.adhoc_shm_manager.close()
