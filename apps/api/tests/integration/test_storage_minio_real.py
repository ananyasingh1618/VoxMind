"""Genuine `S3StorageBackend` verification against a real, live MinIO
instance - not moto (the in-process S3 simulator `tests/unit/test_storage.py`
already uses for fast, credential-free unit coverage of the boto3 request/
response code path), not the `mc` CLI, and not a mock of any kind.

Marked `requires_minio` and `skipif`-guarded exactly like the `requires_redis`
real-broker tests in `test_celery_task_runner.py` - skipped, never faked,
when no real S3-compatible endpoint is reachable.

Start MinIO for this test with:
    docker compose -f docker/docker-compose.yml up -d minio

Then run:
    pytest -m requires_minio tests/integration/test_storage_minio_real.py

The bucket is created here (a one-time real `create_bucket` call, the same
pattern the moto-based unit test already uses) - `S3StorageBackend` itself
deliberately never creates buckets on first use, since that's an
infrastructure/ops decision, not something to do silently at runtime.
"""
from __future__ import annotations

import uuid

import boto3
import pytest
from botocore.exceptions import ClientError

from voxmind.services.storage.s3_backend import S3StorageBackend

MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "voxmind"
MINIO_SECRET_KEY = "voxmind123"
MINIO_BUCKET = "voxmind-audio"


def _minio_is_reachable() -> bool:
    try:
        client = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name="us-east-1",
        )
        client.list_buckets()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.requires_minio,
    pytest.mark.skipif(
        not _minio_is_reachable(),
        reason="No reachable MinIO/S3-compatible endpoint at localhost:9000. Start one with: "
        "docker compose -f docker/docker-compose.yml up -d minio",
    ),
]


@pytest.fixture
def real_bucket():
    """Ensures the real bucket exists via a real, direct boto3 call - the
    one and only place this test file bypasses `S3StorageBackend`, since
    bucket provisioning is deliberately outside that class's own
    responsibility (see module docstring)."""
    client = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
    )
    try:
        client.create_bucket(Bucket=MINIO_BUCKET)
    except ClientError as exc:
        # Already exists from a previous run - a real, expected, idempotent case.
        if exc.response.get("Error", {}).get("Code") not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise
    return MINIO_BUCKET


@pytest.fixture
def storage(real_bucket) -> S3StorageBackend:
    """The real, unmodified `S3StorageBackend` VoxMind's storage factory
    already builds when `STORAGE_BACKEND=s3` - constructed directly here
    only to supply real MinIO credentials without needing an `.env` change
    just to run this one test file."""
    return S3StorageBackend(
        bucket=real_bucket,
        endpoint_url=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        region="us-east-1",
    )


@pytest.mark.asyncio
async def test_connect_authenticate_and_full_round_trip(storage):
    """Points 1-8 of the required real test, in one coherent sequence:
    connect, authenticate, use the real bucket, upload, download, verify
    exact content, delete, confirm deletion."""
    key = f"test/minio-verification/{uuid.uuid4()}.txt"
    content = b"real minio verification content, byte-for-byte - never fabricated"

    await storage.upload(key, content, content_type="text/plain")
    assert await storage.exists(key) is True

    downloaded = await storage.download(key)
    assert downloaded == content  # exact content, not just "something came back"

    await storage.delete(key)
    assert await storage.exists(key) is False


@pytest.mark.asyncio
async def test_missing_object_is_handled_correctly_not_fabricated(storage):
    """Point 9: a real, genuine 404 from MinIO for a key that was never
    uploaded - `exists()` must report False (not raise), and `download()`
    must raise the real botocore error, never return fabricated bytes."""
    key = f"test/minio-verification/never-existed-{uuid.uuid4()}.txt"

    assert await storage.exists(key) is False
    with pytest.raises(ClientError) as exc_info:
        await storage.download(key)
    assert exc_info.value.response["Error"]["Code"] in ("404", "NoSuchKey")


@pytest.mark.asyncio
async def test_upload_without_content_type_still_round_trips(storage):
    """content_type is optional in the real StorageBackend interface -
    proven against the real endpoint, not assumed from the unit test alone."""
    key = f"test/minio-verification/{uuid.uuid4()}-no-content-type.bin"
    content = b"\x00\x01\x02real-binary-bytes\xff\xfe"

    await storage.upload(key, content)
    assert await storage.download(key) == content
    await storage.delete(key)
