"""Read-only loopback view of the existing qualified training observer.

GETs read one bounded local observation and emit an allowlisted projection.
No provider clients, credentials, signatures, training or promotion actions.
"""

import argparse
import datetime as dt
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import stat
from urllib.parse import urlsplit

from tools.salad.artifact_qualification import RELEASE_BLOCKERS, strict_object
from tools.salad.training_snapshot import EvidenceHold

INPUT_SCHEMA = "szl.training-observation/v1"
SCHEMA = "szl.training-dashboard/v1"
MAX_BYTES = 256 * 1024
ASSETS = Path(__file__).resolve().parent
OBSERVATION_STATES = {
    "UNAVAILABLE",
    "NO_CHECKPOINT_OBSERVED",
    "TARGET_CONTRACT_MISMATCH",
    "TERMINAL_STEP_OVERSHOOT",
    "CHECKPOINT_PROGRESS_OBSERVED",
    "TERMINAL_METADATA_PRESENT",
    "TARGET_CHECKPOINT_WITHOUT_FINAL_BUNDLE",
}
PROVIDER_STATES = {
    "allocating",
    "allocated",
    "creating",
    "running",
    "stopping",
    "stopped",
    "failed",
    "error",
    "downloading",
    "starting",
    "pending",
    "queued",
    "provisioning",
    "succeeded",
    "unknown",
}
FLAGS = (
    "training_complete_verified",
    "promotion_authorized",
    "trainer_process_liveness_verified",
)


class InvalidObservation(ValueError):
    pass


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|\+00:00)",
        value,
    ):
        raise InvalidObservation("TIMESTAMP_INVALID")
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidObservation("TIMESTAMP_INVALID") from exc


def integer(value, *, minimum=0, maximum=1_000_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise InvalidObservation("INTEGER_INVALID")
    return value


def optional_integer(value):
    return None if value is None else integer(value)


def false_flags(value, names=FLAGS):
    if any(value.get(name) is not False for name in names):
        raise InvalidObservation("OBSERVATION_AUTHORITY_MISMATCH")


def result_base(now):
    return {
        "schema": SCHEMA,
        "status": "UNAVAILABLE",
        "generated_at_utc": now.isoformat(),
        "source": None,
        "scope": "SELECTED_CANDIDATES",
        "candidates": [],
        "training_complete_verified": False,
        "promotion_authorized": False,
        "trainer_process_liveness_verified": False,
        "reason": None,
    }


def provider_projection(value, freshness):
    if not isinstance(value, dict):
        raise InvalidObservation("PROVIDER_INVALID")
    false_flags(value, ("trainer_process_liveness_verified",))
    state = value.get("state")
    if state not in {"OBSERVED", "UNAVAILABLE"}:
        raise InvalidObservation("PROVIDER_INVALID")
    output = {
        "state": state if freshness == "CURRENT" else "STALE",
        "observed_state": state,
        "instance_count": None,
        "instance_states": [],
        "ready_count": None,
        "trainer_process_liveness_verified": False,
    }
    if state == "OBSERVED":
        count = integer(value.get("instance_count"), maximum=1000)
        ready = integer(value.get("ready_count"), maximum=count)
        states = value.get("instance_states")
        if not isinstance(states, list) or len(states) != count:
            raise InvalidObservation("PROVIDER_COUNTS_INVALID")
        if any(not isinstance(s, str) for s in states):
            raise InvalidObservation("PROVIDER_INVALID")
        output.update(
            instance_count=count,
            ready_count=ready,
            instance_states=[s if s in PROVIDER_STATES else "unknown" for s in states],
        )
    return output


def candidate_projection(value, now, report_time, started, ttl):
    if not isinstance(value, dict):
        raise InvalidObservation("CANDIDATE_INVALID")
    false_flags(value)
    repo = value.get("repo")
    if not isinstance(repo, str) or not re.fullmatch(
        r"SZLHOLDINGS/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", repo
    ):
        raise InvalidObservation("CANDIDATE_ID_INVALID")
    target = integer(value.get("target_steps"), minimum=1)
    state = value.get("state")
    if state not in OBSERVATION_STATES:
        raise InvalidObservation("CANDIDATE_STATE_INVALID")
    observed = timestamp(value.get("observed_at_utc"))
    if not started <= observed <= report_time:
        raise InvalidObservation("OBSERVATION_TIME_ORDER_INVALID")
    age = (now - observed).total_seconds()
    freshness = "CURRENT" if age <= ttl else "STALE"
    snapshot = value.get("snapshot", {})
    trainer = value.get("trainer_state", {})
    if not isinstance(snapshot, dict) or not isinstance(trainer, dict):
        raise InvalidObservation("CANDIDATE_EVIDENCE_INVALID")
    if "snapshot" in value and (
        snapshot.get("id") != repo or snapshot.get("status") != "OBSERVED"
    ):
        raise InvalidObservation("SNAPSHOT_IDENTITY_INVALID")
    revision = snapshot.get("revision")
    if revision is not None and (
        not isinstance(revision, str)
        or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision)
    ):
        raise InvalidObservation("REVISION_INVALID")
    checkpoint = optional_integer(snapshot.get("latest_checkpoint"))
    if state == "NO_CHECKPOINT_OBSERVED" and checkpoint is not None:
        raise InvalidObservation("CHECKPOINT_STATE_CONTRADICTION")
    if state == "TERMINAL_METADATA_PRESENT" and any(
        snapshot.get(name) is not True
        for name in ("final_adapter", "final_config", "final_tokenizer")
    ):
        raise InvalidObservation("TERMINAL_METADATA_CONTRADICTION")
    step = optional_integer(trainer.get("global_step"))
    maximum = optional_integer(trainer.get("max_steps"))
    integrity = trainer.get("integrity")
    trainer_digest = trainer.get("trainer_state_sha256")
    if integrity is not None and (
        not isinstance(trainer_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", trainer_digest)
    ):
        raise InvalidObservation("TRAINER_STATE_DIGEST_INVALID")
    if integrity not in {None, "PINNED_OBJECT_BYTES_VERIFIED"}:
        raise InvalidObservation("TRAINER_STATE_INTEGRITY_INVALID")
    if step is not None and (
        step != checkpoint or revision is None or integrity is None
    ):
        raise InvalidObservation("TRAINER_STATE_BINDING_INVALID")
    if state in {
        "CHECKPOINT_PROGRESS_OBSERVED",
        "TERMINAL_METADATA_PRESENT",
        "TARGET_CHECKPOINT_WITHOUT_FINAL_BUNDLE",
        "TERMINAL_STEP_OVERSHOOT",
        "TARGET_CONTRACT_MISMATCH",
    } and (step is None or maximum is None):
        raise InvalidObservation("TRAINER_STATE_MISSING")
    if state == "CHECKPOINT_PROGRESS_OBSERVED" and not (step < target == maximum):
        raise InvalidObservation("TRAINER_STATE_BINDING_INVALID")
    if state in {"TERMINAL_METADATA_PRESENT", "TARGET_CHECKPOINT_WITHOUT_FINAL_BUNDLE"}:
        if not step == target == maximum:
            raise InvalidObservation("TRAINER_STATE_BINDING_INVALID")
    if state == "TERMINAL_STEP_OVERSHOOT" and not (step > target == maximum):
        raise InvalidObservation("TRAINER_STATE_BINDING_INVALID")
    if state == "TARGET_CONTRACT_MISMATCH" and maximum == target:
        raise InvalidObservation("TRAINER_STATE_BINDING_INVALID")
    return {
        "repo": repo,
        "state": state if freshness == "CURRENT" else "STALE",
        "observed_state": state,
        "freshness": freshness,
        "observed_at_utc": observed.isoformat(),
        "age_seconds": round(age, 3),
        "revision": revision,
        "latest_checkpoint": checkpoint,
        "target_steps": target,
        "global_step": step,
        "trainer_max_steps": maximum,
        "trainer_state_integrity": integrity,
        "trainer_state_sha256": trainer_digest if integrity is not None else None,
        "provider": provider_projection(value.get("provider"), freshness),
        "training_complete_verified": False,
        "promotion_authorized": False,
        "trainer_process_liveness_verified": False,
    }


def project_observation(raw, *, now=None, max_age_seconds=900):
    """Project source-reported evidence, not independently qualify an artifact."""
    now = utc_now() if now is None else now
    output = result_base(now)
    try:
        integer(max_age_seconds, minimum=60, maximum=3600)
        if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
            raise InvalidObservation("SOURCE_BYTE_BUDGET")
        source = strict_object(raw)
        if source.get("schema") != INPUT_SCHEMA:
            raise InvalidObservation("SOURCE_SCHEMA_INVALID")
        false_flags(source, ("training_complete_verified", "promotion_authorized"))
        observed = timestamp(source.get("observed_at_utc"))
        started = timestamp(source.get("started_at_utc"))
        if observed > now or started > observed:
            raise InvalidObservation("OBSERVATION_TIME_ORDER_INVALID")
        declared_ttl = integer(
            source.get("stale_after_seconds"), minimum=1, maximum=7260
        )
        ttl = min(declared_ttl, max_age_seconds)
        candidates = source.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 6:
            raise InvalidObservation("CANDIDATE_SCOPE_INVALID")
        rows = [
            candidate_projection(candidate, now, observed, started, ttl)
            for candidate in candidates
        ]
        if len({row["repo"] for row in rows}) != len(rows):
            raise InvalidObservation("DUPLICATE_CANDIDATE")
        age = (now - observed).total_seconds()
        current = age <= ttl and all(row["freshness"] == "CURRENT" for row in rows)
        output.update(
            status="CURRENT" if current else "STALE",
            source={
                "schema": INPUT_SCHEMA,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "observed_at_utc": observed.isoformat(),
                "age_seconds": round(age, 3),
                "stale_after_seconds": ttl,
                "declared_stale_after_seconds": declared_ttl,
            },
            candidates=rows,
            reason=None if current else "OBSERVATION_EXPIRED",
        )
    except (
        InvalidObservation,
        EvidenceHold,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ):
        # Never echo arbitrary source text, exceptions, credentials or local paths.
        output.update(status="INVALID", reason="OBSERVATION_CONTRACT_INVALID")
    return output


def qualification_summary(
    state="NOT_CONFIGURED", reason="QUALIFICATION_NOT_CONFIGURED"
):
    """Display a recorded check without inheriting quality or execution authority."""
    return {
        "state": state,
        "revision": None,
        "observed_at_utc": None,
        "qualification_sha256": None,
        "artifact_sha256": None,
        "target_steps": None,
        "base_repo": None,
        "hold_reasons": [reason] if reason else [],
        "release_blockers": list(RELEASE_BLOCKERS),
        "artifact_observation_bound": False,
        "authenticity_verified": False,
        "quality_qualification": "NOT_ESTABLISHED",
        "compute_authorized": False,
        "promotion_authorized": False,
    }


def attach_qualification(projection, directory, expected_sha256, now):
    rows = projection["candidates"]
    for row in rows:
        row["qualification"] = qualification_summary()
    if not rows or directory is None and expected_sha256 is None:
        return projection
    from tools.salad.evaluation_binding import (
        BindingError,
        load_qualification_bundle,
        validate_candidate,
    )

    try:
        if directory is None or expected_sha256 is None:
            raise ValueError("partial_configuration")
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError("bundle_unavailable")
        if directory.is_symlink():
            raise ValueError("bundle_not_regular")
        for name in (
            "qualification.json",
            "manifest.json",
            "artifact_qualification.py",
            "training_snapshot.py",
        ):
            if not (directory / name).is_file():
                raise FileNotFoundError("bundle_member_unavailable")
        report, _ = load_qualification_bundle(directory, expected_sha256)
        if timestamp(report.get("observed_at_utc")) > now:
            raise ValueError("future_qualification")
    except OSError:
        for row in rows:
            row["qualification"] = qualification_summary(
                "UNAVAILABLE", "QUALIFICATION_UNAVAILABLE"
            )
        return projection
    except BindingError as error:
        unavailable = error.reason_code == "QUALIFICATION_UNAVAILABLE"
        for row in rows:
            row["qualification"] = qualification_summary(
                "UNAVAILABLE" if unavailable else "INVALID",
                "QUALIFICATION_UNAVAILABLE" if unavailable else "QUALIFICATION_INVALID",
            )
        return projection
    except (ValueError, TypeError):
        for row in rows:
            row["qualification"] = qualification_summary(
                "INVALID", "QUALIFICATION_INVALID"
            )
        return projection
    selected = {candidate["repo"]: candidate for candidate in report["candidates"]}
    for row in rows:
        candidate = selected.get(row["repo"])
        if candidate is None:
            row["qualification"] = qualification_summary(
                "NOT_ASSESSED", "QUALIFICATION_CANDIDATE_NOT_ASSESSED"
            )
            row["qualification"]["qualification_sha256"] = expected_sha256
            continue
        if candidate.get("status") == "HOLD":
            row["qualification"] = qualification_summary(
                "ARTIFACT_HOLD", "QUALIFICATION_ARTIFACT_HELD"
            )
            row["qualification"]["qualification_sha256"] = expected_sha256
            continue
        try:
            checked = validate_candidate(candidate)
            if timestamp(checked["observed_at_utc"]) > min(
                now, timestamp(report["observed_at_utc"])
            ):
                raise ValueError("inconsistent_qualification_time")
        except (BindingError, ValueError, TypeError):
            row["qualification"] = qualification_summary(
                "INVALID", "QUALIFICATION_INVALID"
            )
            continue
        qualified = qualification_summary("ARTIFACT_OBSERVATION_MATCHED", None)
        qualified.update(
            revision=checked["revision"],
            observed_at_utc=checked["observed_at_utc"],
            qualification_sha256=expected_sha256,
            artifact_sha256=checked["artifact_sha256"],
            target_steps=checked["target_steps"],
            base_repo=checked["base_repo"],
            release_blockers=checked["release_blockers"],
        )
        if checked["revision"] != row["revision"]:
            qualified.update(
                state="REVISION_MISMATCH",
                hold_reasons=["QUALIFICATION_REVISION_MISMATCH"],
            )
        elif checked["target_steps"] != row["target_steps"]:
            qualified.update(
                state="INVALID",
                hold_reasons=["QUALIFICATION_TARGET_MISMATCH"],
            )
        else:
            qualified["artifact_observation_bound"] = True
        row["qualification"] = qualified
    return projection


def read_projection(
    path,
    *,
    now=None,
    max_age_seconds=900,
    qualification_bundle=None,
    qualification_sha256=None,
):
    now = utc_now() if now is None else now
    output = result_base(now)
    try:
        path = Path(path)
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            output.update(status="INVALID", reason="SOURCE_NOT_REGULAR_FILE")
            return output
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
    except FileNotFoundError:
        output["reason"] = "SOURCE_NOT_FOUND"
        return output
    except OSError:
        output["reason"] = "SOURCE_UNREADABLE"
        return output
    projection = project_observation(raw, now=now, max_age_seconds=max_age_seconds)
    return attach_qualification(
        projection, qualification_bundle, qualification_sha256, now
    )


def make_server(
    status_path,
    *,
    port=8767,
    max_age_seconds=900,
    qualification_bundle=None,
    qualification_sha256=None,
):
    """Bind only loopback. Paths and provider credentials are never request inputs."""
    integer(port, minimum=0, maximum=65535)
    integer(max_age_seconds, minimum=60, maximum=3600)
    status_path = Path(status_path)

    class Handler(BaseHTTPRequestHandler):
        server_version = "SZLTrainingView/1"
        sys_version = ""

        def log_message(self, *_):
            # No request paths, headers or private candidate IDs in access logs.
            pass

        def send_payload(self, status_code, body, content_type):
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; "
                "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
                "form-action 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status_code, value):
            self.send_payload(
                status_code,
                json.dumps(value, allow_nan=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def local_request(self):
            port = self.server.server_port
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origins = {f"http://{host}" for host in hosts}
            return (
                self.headers.get("Host") in hosts
                and self.headers.get("Origin") in {None, *origins}
                and len(self.headers.get_all("Host", [])) == 1
            )

        def do_GET(self):
            if not self.local_request():
                self.send_json(403, {"error": "LOCAL_ORIGIN_REQUIRED"})
                return
            try:
                url = urlsplit(self.path)
            except ValueError:
                self.send_json(400, {"error": "INVALID_REQUEST"})
                return
            if url.scheme or url.netloc or url.query or url.fragment:
                self.send_json(404, {"error": "NOT_FOUND"})
                return
            if url.path == "/api/training/status":
                self.send_json(
                    200,
                    read_projection(
                        status_path,
                        max_age_seconds=max_age_seconds,
                        qualification_bundle=qualification_bundle,
                        qualification_sha256=qualification_sha256,
                    ),
                )
                return
            if url.path == "/healthz":
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "component": "training-observation-view",
                        "training_complete_verified": False,
                        "promotion_authorized": False,
                    },
                )
                return
            assets = {
                "/": ("training_command.html", "text/html; charset=utf-8"),
                "/training": ("training_command.html", "text/html; charset=utf-8"),
                "/training_command.js": (
                    "training_command.js",
                    "text/javascript; charset=utf-8",
                ),
            }
            asset = assets.get(url.path)
            if asset is None:
                self.send_json(404, {"error": "NOT_FOUND"})
                return
            try:
                body = (ASSETS / asset[0]).read_bytes()
            except OSError:
                self.send_json(503, {"error": "ASSET_UNAVAILABLE"})
                return
            self.send_payload(200, body, asset[1])

        def deny_write(self):
            self.send_json(405, {"error": "READ_ONLY"})

        do_POST = deny_write
        do_PUT = deny_write
        do_PATCH = deny_write
        do_DELETE = deny_write
        do_OPTIONS = deny_write

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", required=True, help="Existing local observer JSON")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--max-age-seconds", type=int, default=900)
    parser.add_argument("--qualification-bundle", type=Path)
    parser.add_argument("--qualification-sha256")

    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 60 <= args.max_age_seconds <= 3600:
        parser.error("port must be 1024-65535 and max age 60-3600 seconds")
    if (args.qualification_bundle is None) != (args.qualification_sha256 is None):
        parser.error(
            "qualification bundle and reviewed report SHA-256 are required together"
        )
    if (
        args.qualification_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", args.qualification_sha256) is None
    ):
        parser.error(
            "qualification SHA-256 must be 64 lowercase hexadecimal characters"
        )
    with make_server(
        args.status,
        port=args.port,
        max_age_seconds=args.max_age_seconds,
        qualification_bundle=args.qualification_bundle,
        qualification_sha256=args.qualification_sha256,
    ) as server:
        print(
            f"Training observations: http://127.0.0.1:{args.port}/training", flush=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
