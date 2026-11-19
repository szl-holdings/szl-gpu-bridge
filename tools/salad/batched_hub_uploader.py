"""Batched, rate-limit-aware Hub uploads for Salad lanes (receipts, logs, heartbeats).

Why this exists: the Hub caps commit creation per repository (320 per hour on
``SZLHOLDINGS/frontier-54-runs``). Every ``HfApi.upload_file`` call is one commit,
so a fleet that uploads one file per receipt or heartbeat exhausts the cap and
every writer then sees HTTP 429. This helper turns "one commit per file" into
"one commit per batch":

* ``add()`` writes the bytes to a local spool first (recovery evidence) and only
  then queues them; a batch goes out as ONE ``HfApi.create_commit`` once it holds
  ``max_files`` files or is ``max_interval_s`` old.
* A non-final flush makes a single attempt. On HTTP 429 (or any transient
  failure) the batch stays spooled, the uploader cools down for ``Retry-After``
  (or exponential backoff), and the caller keeps running. Throttling never raises.
* ``flush(final=True)`` / ``close()`` is the only call that can fail. It retries
  transient failures with exponential backoff, honors ``Retry-After``, and raises
  ``UploadFlushError`` only when the retry budget is spent (or on a permanent 4xx).
* An atomic status file (``uploader-status.json``) tells a watcher whether the
  process is alive and only throttled, so it does not re-launch a healthy run.

Honesty scope: this is transport plumbing. A flushed batch is not proof of model
quality; it does not replace per-receipt hashing, and it must not be used for
evidence that has to be byte-verified before the run may claim success unless the
caller reads the returned commit back itself. Exception bodies and headers are
never stored or printed; only the HTTP status label is.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

STATUS_FILE = "uploader-status.json"
STATUS_SCHEMA = "szl.batched-hub-uploader-status/v1"
TRANSIENT_STATUSES = frozenset((408, 429, 500, 502, 503, 504))


class UploadFlushError(RuntimeError):
    """The final flush could not publish the pending batch; spool files remain."""


def transport_label(error: BaseException) -> str:
    """Status code or type name only: SDK strings can carry URLs and headers."""
    status = _status(error)
    return f"HTTP {status}" if status is not None else type(error).__name__


def _status(error: BaseException):
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status if type(status) is int else None


def retry_after_seconds(error: BaseException, cap: float):
    """Honor a numeric Retry-After header; ignore anything else."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    try:
        raw = headers.get("Retry-After") if headers is not None else None
        value = float(raw)
    except (AttributeError, TypeError, ValueError):
        return None
    if value != value or value < 0:  # NaN or negative
        return None
    return min(value, cap)


def _default_operation(path_in_repo: str, data: bytes):
    from huggingface_hub import CommitOperationAdd

    return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=data)


class BatchedHubUploader:
    def __init__(
        self,
        api,
        repo_id: str,
        spool_dir,
        *,
        repo_type: str = "dataset",
        max_files: int = 50,
        max_interval_s: float = 300.0,
        commit_message: str = "Batch upload {count} files",
        max_attempts: int = 8,
        base_delay_s: float = 20.0,
        max_delay_s: float = 600.0,
        max_retry_after_s: float = 3600.0,
        operation_factory=_default_operation,
        clock=time.monotonic,
        sleep=time.sleep,
        wall_clock=time.time,
    ):
        if max_files < 1 or max_attempts < 1:
            raise ValueError("max_files and max_attempts must be >= 1")
        self.api, self.repo_id, self.repo_type = api, repo_id, repo_type
        self.spool = Path(spool_dir)
        self.spool.mkdir(parents=True, exist_ok=True)
        self.max_files, self.max_interval_s = max_files, max_interval_s
        self.commit_message = commit_message
        self.max_attempts, self.base_delay_s = max_attempts, base_delay_s
        self.max_delay_s, self.max_retry_after_s = max_delay_s, max_retry_after_s
        self._op, self._clock, self._sleep = operation_factory, clock, sleep
        self._wall = wall_clock
        self._lock = threading.RLock()
        self._pending: dict[str, Path] = {}  # path_in_repo -> spool file (latest wins)
        self._batch_started = None
        self._cooldown_until = 0.0
        self.commits = 0
        self.files_committed = 0
        self.throttled_count = 0
        self.last_commit = None
        self.last_error = None
        self._write_status("IDLE")

    # ---- queueing -------------------------------------------------------------
    def add(self, path_in_repo: str, data: bytes) -> bool:
        """Spool one file; flush opportunistically. Never raises for upload failures."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if (
            not path_in_repo
            or path_in_repo.startswith("/")
            or ".." in Path(path_in_repo).parts
        ):
            raise ValueError("path_in_repo must be a relative path without '..'")
        name = hashlib.sha256(path_in_repo.encode()).hexdigest() + ".spool"
        target = self.spool / name
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        with self._lock:
            if not self._pending:
                self._batch_started = self._clock()
            self._pending[path_in_repo] = target
        return self.maybe_flush()

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def maybe_flush(self) -> bool:
        """Flush if the batch is full or old and the cool-down has passed."""
        with self._lock:
            if not self._pending:
                return False
            now = self._clock()
            full = len(self._pending) >= self.max_files
            old = (
                self._batch_started is not None
                and now - self._batch_started >= self.max_interval_s
            )
            if not (full or old) or now < self._cooldown_until:
                return False
            return self._attempt_once(now)

    # ---- flushing -------------------------------------------------------------
    def flush(self, final: bool = False) -> bool:
        """Publish everything pending. Only ``final=True`` can raise."""
        with self._lock:
            if not self._pending:
                self._write_status("IDLE" if not final else "FLUSHED")
                return True
            if not final:
                now = self._clock()
                return now >= self._cooldown_until and self._attempt_once(now)
            return self._final_flush()

    def close(self) -> None:
        """Final flush; raises UploadFlushError if the retry budget is spent."""
        self.flush(final=True)

    def _commit_pending(self):
        paths = sorted(self._pending)
        ops = [self._op(p, self._pending[p].read_bytes()) for p in paths]
        message = self.commit_message.format(count=len(ops))
        info = self.api.create_commit(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            operations=ops,
            commit_message=message,
        )
        return paths, getattr(info, "oid", None)

    def _on_success(self, paths, oid):
        for p in paths:
            spool_file = self._pending.pop(p, None)
            if spool_file is not None:
                try:
                    spool_file.unlink()
                except OSError:
                    pass
        self.commits += 1
        self.files_committed += len(paths)
        self.last_commit = oid if isinstance(oid, str) else None
        self.last_error = None
        self._batch_started = self._clock() if self._pending else None
        self._cooldown_until = 0.0
        self._write_status("OK")

    def _delay(self, error, attempt: int) -> float:
        hinted = retry_after_seconds(error, self.max_retry_after_s)
        if hinted is not None:
            return hinted
        return min(self.base_delay_s * (2**attempt), self.max_delay_s)

    def _attempt_once(self, now: float) -> bool:
        try:
            paths, oid = self._commit_pending()
        except Exception as error:  # noqa: BLE001 - transport errors are classified
            status = _status(error)
            self.last_error = transport_label(error)
            if status == 429:
                self.throttled_count += 1
            self._cooldown_until = now + self._delay(error, 0)
            state = "THROTTLED" if status == 429 else "DEGRADED"
            self._write_status(state)
            return False
        self._on_success(paths, oid)
        return True

    def _final_flush(self) -> bool:
        last = None
        for attempt in range(self.max_attempts):
            try:
                paths, oid = self._commit_pending()
            except Exception as error:  # noqa: BLE001
                status = _status(error)
                last = transport_label(error)
                self.last_error = last
                if status == 429:
                    self.throttled_count += 1
                permanent = (
                    status is not None
                    and 400 <= status < 500
                    and status not in TRANSIENT_STATUSES
                )
                if permanent or attempt == self.max_attempts - 1:
                    self._write_status("FAILED")
                    raise UploadFlushError(
                        f"final flush failed ({last}); {len(self._pending)} "
                        f"file(s) remain in the spool"
                    ) from None
                delay = self._delay(error, attempt)
                self._write_status("THROTTLED" if status == 429 else "DEGRADED")
                self._sleep(delay)
                continue
            self._on_success(paths, oid)
            self._write_status("FLUSHED")
            return True
        raise UploadFlushError("final flush failed")  # pragma: no cover

    # ---- status for watchers --------------------------------------------------
    def status(self) -> dict:
        retry_in = max(0.0, self._cooldown_until - self._clock())
        return {
            "schema": STATUS_SCHEMA,
            "updated_unix": int(self._wall()),
            "pending_files": len(self._pending),
            "commits": self.commits,
            "files_committed": self.files_committed,
            "throttled_count": self.throttled_count,
            "last_commit": self.last_commit,
            "last_error": self.last_error,
            "retry_in_s": round(retry_in, 1),
            "scope": "UPLOAD_TRANSPORT_ONLY_NOT_RUN_FAILURE_NOT_MODEL_QUALITY",
        }

    def _write_status(self, state: str) -> None:
        value = {"state": state, **self.status()}
        target = self.spool / STATUS_FILE
        tmp = target.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
            os.replace(tmp, target)
        except OSError:
            pass  # status is advisory; never fail a run for it
