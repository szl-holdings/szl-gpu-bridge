"""Bounded read-only training telemetry. Provider-running is not trainer liveness.

No start/stop, signing, upload, completion flags, or promotion effects. One
cycle by default; daemon mode is bounded and intended for an existing supervisor.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import tempfile
import time
import urllib.error
import urllib.request

from tools.salad.artifact_qualification import strict_object
from tools.salad.training_snapshot import (
    EvidenceHold,
    model_snapshot,
    verified_trainer_state,
)

SALAD_BASE = "https://api.salad.com/api/public/organizations/alloyscape/projects/default/containers/"
ACCESS_STOP_CODES = frozenset((401, 403, 429))


def _http_status(exc):
    """Typed status only; never log provider exception text or signed URLs."""
    code = getattr(exc, "code", None)
    if type(code) is int:
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if type(code) is int else None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def provider_open(request, timeout):
    # Never forward a provider credential through an HTTP redirect.
    return urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout)


def provider_observation(group, token, *, opener=provider_open):
    unavailable = {"state": "UNAVAILABLE", "trainer_process_liveness_verified": False}
    if (
        not isinstance(group, str)
        or re.fullmatch(r"szl-[a-z0-9-]{1,58}", group) is None
        or not token
    ):
        return {**unavailable, "reason": "PROVIDER_OBSERVATION_NOT_CONFIGURED"}
    request = urllib.request.Request(
        SALAD_BASE + group + "/instances",
        headers={
            "Salad-Api-Key": token,
            "User-Agent": "szl-training-observer/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with opener(request, timeout=12) as response:
            if response.status != 200:
                status = response.status
                if type(status) is int and status in ACCESS_STOP_CODES:
                    return {
                        **unavailable,
                        "reason": "PROVIDER_HTTP_" + str(status),
                        "http_status": status,
                        "access_stop_required": True,
                    }
                return {**unavailable, "reason": "PROVIDER_HTTP_ERROR"}
            raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                return {**unavailable, "reason": "PROVIDER_RESPONSE_BYTE_BUDGET"}
            data = strict_object(raw)
        items = data.get("instances")
        if not isinstance(items, list) or any(
            not isinstance(i, dict) or not isinstance(i.get("state"), str)
            for i in items
        ):
            return {**unavailable, "reason": "INVALID_PROVIDER_RESPONSE"}
        return {
            "state": "OBSERVED",
            "instance_count": len(items),
            "instance_states": [i["state"] for i in items],
            "ready_count": sum(i.get("ready") is True for i in items),
            "trainer_process_liveness_verified": False,
        }
    except urllib.error.HTTPError as exc:
        code = _http_status(exc)
        return {
            **unavailable,
            "reason": "PROVIDER_HTTP_" + str(code),
            "http_status": code,
            "access_stop_required": code in ACCESS_STOP_CODES,
        }
    except Exception as exc:
        return {**unavailable, "reason": "PROVIDER_UNAVAILABLE:" + type(exc).__name__}


def observe_training(api, fetch, candidate):
    repo, target = candidate.get("repo"), candidate.get("target_steps")
    result = {
        "repo": repo,
        "target_steps": target,
        "state": "UNAVAILABLE",
        "training_complete_verified": False,
        "promotion_authorized": False,
        "trainer_process_liveness_verified": False,
    }
    try:
        if (
            not isinstance(repo, str)
            or re.fullmatch(r"SZLHOLDINGS/[A-Za-z0-9_.-]+", repo) is None
            or type(target) is not int
            or target < 1
        ):
            raise EvidenceHold("INVALID_OBSERVATION_CONTRACT")
        _, snapshot, entries = model_snapshot(api, repo)
        result["snapshot"] = snapshot
        step = snapshot["latest_checkpoint"]
        if step is None:
            result["state"] = "NO_CHECKPOINT_OBSERVED"
            return result
        name = f"checkpoint-{step}/trainer_state.json"
        if name not in entries:
            raise EvidenceHold("TRAINER_STATE_MISSING")
        if (
            type(entries[name].size) is not int
            or not 0 < entries[name].size <= 2_000_000
        ):
            raise EvidenceHold("TRAINER_JSON_BYTE_BUDGET")
        with tempfile.TemporaryDirectory(prefix="szl-observe-") as temp:
            path = Path(
                fetch(
                    repo_id=repo,
                    filename=name,
                    revision=snapshot["revision"],
                    local_dir=temp,
                    etag_timeout=12,
                )
            )
            if path.is_symlink() or Path(temp).resolve() not in path.resolve().parents:
                raise EvidenceHold("DOWNLOAD_PATH_ESCAPE")
            with path.open("rb") as stream:
                state = verified_trainer_state(
                    stream.read(2_000_001), entries[name], step
                )
        result["trainer_state"] = state
        if state["max_steps"] != target:
            result["state"] = "TARGET_CONTRACT_MISMATCH"
        elif step > target:
            result["state"] = "TERMINAL_STEP_OVERSHOOT"
        elif step < target:
            result["state"] = "CHECKPOINT_PROGRESS_OBSERVED"
        elif all(
            name in entries
            for name in (
                "final/adapter_model.safetensors",
                "final/adapter_config.json",
                "final/tokenizer.json",
                "final/tokenizer_config.json",
            )
        ):
            result["state"] = "TERMINAL_METADATA_PRESENT"
        else:
            result["state"] = "TARGET_CHECKPOINT_WITHOUT_FINAL_BUNDLE"
    except EvidenceHold as exc:
        result["reason"] = str(exc)
    except Exception as exc:
        code = _http_status(exc)
        if code in ACCESS_STOP_CODES:
            result["reason"] = "HUB_HTTP_" + str(code)
            result["http_status"] = code
            result["access_stop_required"] = True
        else:
            result["reason"] = "OBSERVATION_UNAVAILABLE:" + type(exc).__name__
    return result


def atomic_status(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("status path must not be a symlink")
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 300 or not 60 <= args.interval <= 3600:
        parser.error("cycles must be 1-300 and interval 60-3600 seconds")
    candidates = strict_object(Path(args.spec).read_bytes()).get("candidates")
    if (
        not isinstance(candidates, list)
        or not 1 <= len(candidates) <= 6
        or any(not isinstance(c, dict) for c in candidates)
    ):
        parser.error("1-6 candidate objects required")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    # These stop flags persist across bounded cycles. No alternate credential
    # path or repeated request is permitted after a provider's 401/403/429.
    hub_access_stopped = False
    salad_access_stopped = False
    for cycle in range(1, args.cycles + 1):
        report = {
            "schema": "szl.training-observation/v1",
            "cycle": cycle,
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "candidates": [],
            "promotion_authorized": False,
            "training_complete_verified": False,
        }
        for candidate in candidates:
            if hub_access_stopped:
                entry = {
                    "repo": candidate.get("repo"),
                    "target_steps": candidate.get("target_steps"),
                    "state": "UNAVAILABLE",
                    "reason": "HUB_ACCESS_STOPPED_AFTER_DENIAL",
                    "training_complete_verified": False,
                    "promotion_authorized": False,
                    "trainer_process_liveness_verified": False,
                }
            else:
                entry = observe_training(api, hf_hub_download, candidate)
                if entry.get("access_stop_required") is True:
                    hub_access_stopped = True

            if salad_access_stopped:
                entry["provider"] = {
                    "state": "UNAVAILABLE",
                    "reason": "PROVIDER_ACCESS_STOPPED_AFTER_DENIAL",
                    "trainer_process_liveness_verified": False,
                }
            else:
                entry["provider"] = provider_observation(
                    candidate.get("salad_group"), os.environ.get("SALAD_API_KEY")
                )
                if entry["provider"].get("access_stop_required") is True:
                    salad_access_stopped = True
            entry["observed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            report["candidates"].append(entry)
        report["provider_lane_stops"] = {
            "huggingface": hub_access_stopped,
            "salad": salad_access_stopped,
        }
        report["observed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        report["stale_after_seconds"] = 2 * args.interval + 60
        atomic_status(args.status, report)
        with Path(args.history).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(
            json.dumps(
                {
                    "cycle": cycle,
                    "states": {r["repo"]: r["state"] for r in report["candidates"]},
                }
            ),
            flush=True,
        )
        if cycle < args.cycles:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
