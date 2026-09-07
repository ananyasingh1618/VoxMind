# Object storage

VoxMind's object storage abstraction (`services/storage/interfaces.py::StorageBackend`) has two real implementations - which one is active is a single config switch (`STORAGE_BACKEND`), never a code change. Every caller (audio assets, model checkpoints, knowledge documents) depends only on the `upload`/`download`/`delete`/`exists` interface, never on which backend is behind it.

## Local filesystem mode (`STORAGE_BACKEND=local`, the default)

Real files, written to and read from disk under `STORAGE_LOCAL_ROOT` (an absolute path, anchored to `apps/api/` regardless of which directory a process is launched from - see [DECISIONS/0006](DECISIONS/0006-storage-local-root-absolute-default.md) for why a relative default was a real bug). No external service required - the right default for day-to-day development.

## S3/MinIO mode (`STORAGE_BACKEND=s3`)

The same boto3-based client code talks to MinIO locally or real S3/R2 in production - only the endpoint URL differs. Configuration (all via `Settings`, `core/config.py`, never hardcoded):

| Setting | Meaning |
|---|---|
| `S3_ENDPOINT_URL` | `http://localhost:9000` for local MinIO; unset (`None`) for real AWS S3 |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | credentials - never logged, never committed, always from environment/`.env` |
| `S3_BUCKET` | default `voxmind-audio` |
| `S3_REGION` | default `us-east-1` - MinIO ignores this but boto3 requires some value |

Bucket creation is deliberately **not** something `S3StorageBackend` does on first use - that's an infrastructure/operations decision, not application behavior. Create the bucket once, out of band (see below).

TLS/"secure" and path-style addressing are both handled by boto3 itself based on the `endpoint_url` scheme (`http://` vs `https://`) and boto3's own virtual-host-vs-path-style bucket addressing rules - no separate VoxMind setting exists for either, since MinIO's default path-style behavior already matches what boto3 does automatically for a custom (non-AWS) `endpoint_url`.

## Running MinIO locally

```bash
docker compose -f docker/docker-compose.yml up -d minio
```

Starts MinIO with the credentials already wired into `docker-compose.yml` (`voxmind` / a local-only development password - override both for anything beyond local dev). The S3 API listens on `:9000`, the web console on `:9001`.

Create the bucket once (a real, direct boto3 call - `mc` or the MinIO console work equally well):

```bash
apps/api/.venv/bin/python -c "
import boto3
boto3.client('s3', endpoint_url='http://localhost:9000',
              aws_access_key_id='voxmind', aws_secret_access_key='voxmind123',
              region_name='us-east-1').create_bucket(Bucket='voxmind-audio')
"
```

Then point the API at it:

```bash
STORAGE_BACKEND=s3 S3_ENDPOINT_URL=http://localhost:9000 \
S3_ACCESS_KEY=voxmind S3_SECRET_KEY=voxmind123 \
uvicorn voxmind.main:app --reload
```

## Real integration verification

`tests/unit/test_storage.py` covers both backends fast and without external dependencies: `LocalFilesystemStorage` against a real temp directory, `S3StorageBackend` against [moto](https://github.com/getmoto/moto)'s in-process S3 simulation (a standard way to exercise the real boto3 request/response code path - signing, bucket/key semantics, error codes - without a live endpoint).

`tests/integration/test_storage_minio_real.py` goes further: a real, live MinIO instance, verified genuinely - not moto, not the `mc` CLI as a substitute, not a mock. Marked `requires_minio`, `skipif`-guarded exactly like the `requires_redis` real-broker tests, skipped honestly when no reachable endpoint exists at `localhost:9000`. Proves, through `S3StorageBackend` itself:

1. real connection + authentication (a real `create_bucket` call succeeding is the proof)
2. real upload
3. real download, with **exact byte-for-byte content verification** (not just "something came back")
4. real delete, with deletion confirmed by a follow-up `exists()` check
5. a real missing-object case: `exists()` correctly returns `False` rather than raising, and `download()` raises the real botocore `ClientError` (`NoSuchKey`/`404`) rather than returning fabricated bytes

Run it with a real MinIO instance up:

```bash
docker compose -f docker/docker-compose.yml up -d minio
cd apps/api && .venv/bin/python -m pytest -m requires_minio tests/integration/test_storage_minio_real.py
```

Switching `STORAGE_BACKEND` between `local` and `s3` was also verified live end-to-end through the real HTTP API (a real audio upload, independently confirmed present in MinIO via `mc ls` inside the container) and does not require any other configuration change or code change - exactly the guarantee the single-interface design exists to provide.
