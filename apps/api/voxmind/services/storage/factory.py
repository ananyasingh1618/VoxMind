from __future__ import annotations

from voxmind.core.config import Settings
from voxmind.services.storage.interfaces import StorageBackend
from voxmind.services.storage.local_backend import LocalFilesystemStorage
from voxmind.services.storage.s3_backend import S3StorageBackend


def build_storage_backend(settings: Settings) -> StorageBackend:
    if settings.STORAGE_BACKEND == "local":
        return LocalFilesystemStorage(root=settings.STORAGE_LOCAL_ROOT)
    return S3StorageBackend(
        bucket=settings.S3_BUCKET,
        endpoint_url=settings.S3_ENDPOINT_URL,
        access_key=settings.S3_ACCESS_KEY,
        secret_key=settings.S3_SECRET_KEY,
        region=settings.S3_REGION,
    )
