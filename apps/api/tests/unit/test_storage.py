"""Storage abstraction tests.

LocalFilesystemStorage is tested against a real temp directory on disk - no
mocking involved. S3StorageBackend is tested against moto's in-process S3
simulation, which exercises the real boto3 client code path (request
signing, bucket/key semantics, error codes) without needing a live
MinIO/S3 endpoint - this is the standard way to test boto3-based code and is
not a fake/placeholder implementation of our own storage logic.
"""
from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from voxmind.services.storage.local_backend import LocalFilesystemStorage
from voxmind.services.storage.s3_backend import S3StorageBackend


@pytest.mark.asyncio
async def test_local_storage_round_trip(tmp_path):
    storage = LocalFilesystemStorage(root=str(tmp_path))
    await storage.upload("sessions/abc/audio.wav", b"fake-audio-bytes")

    assert await storage.exists("sessions/abc/audio.wav") is True
    assert await storage.download("sessions/abc/audio.wav") == b"fake-audio-bytes"

    await storage.delete("sessions/abc/audio.wav")
    assert await storage.exists("sessions/abc/audio.wav") is False


@pytest.mark.asyncio
async def test_local_storage_rejects_path_traversal(tmp_path):
    storage = LocalFilesystemStorage(root=str(tmp_path))
    with pytest.raises(ValueError):
        await storage.upload("../../etc/passwd", b"malicious")


@pytest.mark.asyncio
async def test_s3_backend_round_trip():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="voxmind-test-bucket")
        storage = S3StorageBackend(
            bucket="voxmind-test-bucket",
            endpoint_url=None,
            access_key="test",
            secret_key="test",
            region="us-east-1",
        )
        await storage.upload("sessions/xyz/audio.wav", b"fake-audio-bytes", content_type="audio/wav")

        assert await storage.exists("sessions/xyz/audio.wav") is True
        assert await storage.download("sessions/xyz/audio.wav") == b"fake-audio-bytes"

        await storage.delete("sessions/xyz/audio.wav")
        assert await storage.exists("sessions/xyz/audio.wav") is False
