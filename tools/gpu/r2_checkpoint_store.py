"""Upload one swap-pool checkpoint archive to Cloudflare R2 (S3 API).

Credentials follow the bridge pattern used by training-image/receipt_upload.py:
they are read from the environment only and are never printed, logged, written
into receipts or passed to training children.

  R2_ACCOUNT_ID or R2_ENDPOINT   account endpoint (https://<id>.r2.cloudflarestorage.com)
  R2_ACCESS_KEY_ID               scoped token key id (bucket read/write only)
  R2_SECRET_ACCESS_KEY           scoped token secret

The bucket and key prefix come from the reviewed node manifest, never from a
run. Errors carry a public reason code and the exception type only.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tarfile
import tempfile

CREDENTIAL_ENV = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
ENDPOINT_ENV = ("R2_ENDPOINT", "R2_ACCOUNT_ID")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")
_KEY_PART = re.compile(r"^[A-Za-z0-9._=-]{1,200}$")


class R2Error(RuntimeError):
    """Public reason code only; never a credential, URL query or response body."""


def credentials_present(environ=None):
    env = os.environ if environ is None else environ
    return all(env.get(name) for name in CREDENTIAL_ENV) and any(
        env.get(name) for name in ENDPOINT_ENV
    )


def endpoint(environ=None):
    env = os.environ if environ is None else environ
    explicit = env.get("R2_ENDPOINT", "")
    if explicit:
        if not re.fullmatch(
            r"https://[a-z0-9]{32}\.r2\.cloudflarestorage\.com", explicit
        ):
            raise R2Error("R2_ENDPOINT_NOT_ACCOUNT_ENDPOINT")
        return explicit
    account = env.get("R2_ACCOUNT_ID", "")
    if not re.fullmatch(r"[a-z0-9]{32}", account):
        raise R2Error("R2_ACCOUNT_ID_INVALID")
    return f"https://{account}.r2.cloudflarestorage.com"


def validate_location(bucket, *parts):
    if type(bucket) is not str or not _BUCKET.fullmatch(bucket):
        raise R2Error("R2_BUCKET_INVALID")
    for part in parts:
        for segment in str(part).split("/"):
            if not _KEY_PART.fullmatch(segment) or segment in {".", ".."}:
                raise R2Error("R2_KEY_SEGMENT_INVALID")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_directory(source, destination_dir):
    """Deterministic-order tar of one checkpoint directory; no symlinks followed."""

    source = Path(source)
    if not source.is_dir():
        raise R2Error("CHECKPOINT_DIR_MISSING")
    fd, name = tempfile.mkstemp(suffix=".tar", dir=destination_dir)
    os.close(fd)
    try:
        with tarfile.open(name, "w") as archive:
            for path in sorted(source.rglob("*")):
                if path.is_symlink():
                    raise R2Error("CHECKPOINT_CONTAINS_SYMLINK")
                archive.add(
                    path, arcname=str(path.relative_to(source)), recursive=False
                )
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


class R2CheckpointStore:
    """Real uploader. Constructed only when a pool is executing, never in plan mode."""

    def __init__(self, bucket, prefix, environ=None):
        validate_location(bucket, prefix)
        env = os.environ if environ is None else environ
        if not credentials_present(env):
            raise R2Error("R2_CREDENTIALS_MISSING")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._endpoint = endpoint(env)
        self._env = env
        self._client = None

    def _s3(self):
        if self._client is None:
            try:
                import boto3  # noqa: PLC0415 - optional dependency on GPU nodes only
            except ImportError as exc:
                raise R2Error("R2_CLIENT_UNAVAILABLE") from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint,
                aws_access_key_id=self._env["R2_ACCESS_KEY_ID"],
                aws_secret_access_key=self._env["R2_SECRET_ACCESS_KEY"],
                region_name="auto",
            )
        return self._client

    def upload_checkpoint(self, run_id, label, checkpoint_dir, work_dir):
        validate_location(self.bucket, self.prefix, run_id, label)
        archive = archive_directory(checkpoint_dir, work_dir)
        try:
            sha = sha256_file(archive)
            size = archive.stat().st_size
            key = f"{self.prefix}/{run_id}/{label}.tar"
            try:
                client = self._s3()
                client.upload_file(
                    str(archive),
                    self.bucket,
                    key,
                    ExtraArgs={"Metadata": {"sha256": sha}},
                )
                head = client.head_object(Bucket=self.bucket, Key=key)
            except R2Error:
                raise
            except Exception as exc:  # noqa: BLE001 - never echo provider text
                raise R2Error(f"R2_UPLOAD_FAILED:{type(exc).__name__}") from None
            if int(head.get("ContentLength", -1)) != size:
                raise R2Error("R2_READBACK_SIZE_MISMATCH")
            if head.get("Metadata", {}).get("sha256") != sha:
                raise R2Error("R2_READBACK_SHA_MISMATCH")
            return {"uri": f"r2://{self.bucket}/{key}", "sha256": sha, "bytes": size}
        finally:
            archive.unlink(missing_ok=True)
