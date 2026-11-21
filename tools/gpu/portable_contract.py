"""Review an offline placement sidecar. No result authorizes an effect."""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re

from jsonschema import Draft202012Validator

from tools.gpu.checkpoint_contract import (
    CheckpointContract,
    Hold,
    digest,
    document,
    require,
)
from tools.massed.admission import (
    INTENT_FIELDS,
    INTENT_SCHEMA,
    MAX_DEADLINE_S,
    MIN_DEADLINE_S,
)


@lru_cache(maxsize=3)
def _validator(name):
    # Private allowlist: never resolve a schema/ref supplied by a caller.
    require(
        name in ("portable-workload.v1", "jobspec.v1", "jobspec.v2"), "INVALID_KIND"
    )
    path = Path(__file__).resolve().parents[2] / "schema" / (name + ".json")
    return Draft202012Validator(document(path.read_bytes(), allow_float=True))


def _check_workload(spec, job, now):
    w, p, b = spec["workload"], spec["placement"], spec["budget"]
    if w["kind"] == "massed-launch.v1":
        require(
            set(job) == set(INTENT_FIELDS)
            and job["schema"] == INTENT_SCHEMA
            and job["op"] == "launch"
            and job["intent_id"] == w["runId"]
            and p["sites"] == ["massed"]
            and p["allocation"] == "new-paid"
            and job["lane_id"] == p["resourceId"]
            and job["product_name"] == p["sku"]
            and job["region_name"] == p["region"]
            and type(job["hard_deadline_s"]) is int
            and job["hard_deadline_s"] == b["wallSeconds"],
            "LAUNCH_INTENT_BINDING",
        )
        require(
            MIN_DEADLINE_S <= b["wallSeconds"] <= MAX_DEADLINE_S
            and type(job["image_id"]) is int
            and job["image_id"] > 0
            and all(type(job[k]) is str and job[k] for k in ("group", "actor"))
            and (
                job["prev_intent_sha256"] is None
                or (
                    type(job["prev_intent_sha256"]) is str
                    and re.fullmatch("[0-9a-f]{64}", job["prev_intent_sha256"])
                )
            ),
            "LAUNCH_INTENT_BINDING",
        )
        return
    require(
        _validator(w["kind"]).is_valid(job) and job["jobId"] == w["runId"],
        "JOBSPEC_BINDING",
    )
    try:
        start, end = (
            datetime.fromisoformat(job[k].replace("Z", "+00:00"))
            for k in ("createdAt", "expiresAt")
        )
        require(
            start.tzinfo is not None
            and end.tzinfo is not None
            and start.timestamp()
            <= now
            < spec["controls"]["stopBy"]
            <= end.timestamp(),
            "JOB_TIME_WINDOW",
        )
    except Hold:
        raise
    except (ValueError, OverflowError, OSError) as exc:
        raise Hold("JOB_TIME_WINDOW") from exc
    asset = spec["asset"]
    for repo_type, key in (("model", "base"), ("dataset", "dataset")):
        ref = job[key]
        require(re.fullmatch("[0-9a-f]{40}", ref["revision"]), "UNPINNED_JOB_ASSET")
        if asset["repoType"] == repo_type:
            require(
                (asset["repoId"], asset["revision"])
                == (ref["repoId"], ref["revision"]),
                "JOB_ASSET_MISMATCH",
            )
    gates = job["gates"]
    require(
        0 < gates["minFreeVramGb"] * 1024 <= p["minVramMiB"]
        and gates["maxWallclockMinutes"] * 60 == b["wallSeconds"],
        "JOB_GATE_MISMATCH",
    )


def _check_evidence(spec, obs, now):
    p, w, controls = spec["placement"], spec["workload"], spec["controls"]
    require(
        set(obs)
        == {
            "schema",
            "workload",
            "source",
            "asset",
            "checkoutState",
            "assetAccess",
            "costCeilings",
            "site",
            "allocation",
            "resourceId",
            "sku",
            "region",
            "devices",
            "storage",
            "cancelHookId",
            "validFrom",
            "validThrough",
        }
        and obs["schema"] == "szl.portable-placement-evidence/v1",
        "INVALID_PLACEMENT_EVIDENCE",
    )
    require(
        obs["checkoutState"] == "clean-at-pin"
        and obs["assetAccess"] == "readable-at-pin",
        "SOURCE_OR_ASSET_UNVERIFIED",
    )
    require(
        type(now) is int
        and now >= 0
        and type(obs["validFrom"]) is int
        and type(obs["validThrough"]) is int
        and obs["validFrom"] <= now < controls["stopBy"] <= obs["validThrough"],
        "STALE_PLACEMENT_EVIDENCE",
    )
    require(
        obs["workload"] == w
        and type(obs["workload"].get("attempt")) is int
        and obs["source"] == spec["source"]
        and obs["asset"] == spec["asset"]
        and obs["site"] == p["sites"][0]
        and all(obs[k] == p[k] for k in ("allocation", "resourceId", "sku", "region"))
        and obs["cancelHookId"] == controls["cancelHookId"],
        "PLACEMENT_EVIDENCE_MISMATCH",
    )
    devices = obs["devices"]
    require(type(devices) is list and 0 < len(devices) <= 256, "DEVICE_EVIDENCE")
    by_id = {}
    for d in devices:
        require(
            type(d) is dict
            and set(d) == {"deviceId", "usableVramMiB", "toolchainSha256"}
            and type(d["deviceId"]) is str
            and d["deviceId"] not in by_id
            and type(d["usableVramMiB"]) is int
            and d["usableVramMiB"] >= 0
            and type(d["toolchainSha256"]) is str
            and re.fullmatch("[0-9a-f]{64}", d["toolchainSha256"]),
            "DEVICE_EVIDENCE",
        )
        by_id[d["deviceId"]] = d
    d = by_id.get(p["deviceId"])
    require(
        d is not None and d["usableVramMiB"] >= p["minVramMiB"], "SINGLE_DEVICE_VRAM"
    )
    require(d["toolchainSha256"] == p["toolchainSha256"], "TOOLCHAIN_MISMATCH")
    store = obs["storage"]
    require(
        type(store) is dict
        and set(store)
        == {"persistence", "storeId", "versionId", "manifestSha256", "expectedPaths"}
        and store["persistence"] == "durable-versioned"
        and all(
            store[k] == spec["checkpoint"][k]
            for k in ("storeId", "versionId", "manifestSha256")
        ),
        "STORAGE_UNVERIFIED",
    )
    costs, budget = obs["costCeilings"], spec["budget"]
    require(
        type(costs) is dict
        and set(costs) == {"wallSeconds", "compute", "storage", "egress"}
        and type(costs["wallSeconds"]) is int
        and costs["wallSeconds"] == budget["wallSeconds"]
        and all(
            type(costs[k]) is int and 0 <= costs[k] <= budget[k]
            for k in ("compute", "storage", "egress")
        )
        and budget["validThrough"] <= obs["validThrough"],
        "COST_EVIDENCE_MISMATCH",
    )
    return store["expectedPaths"]


def review(
    *,
    sidecar_bytes,
    workload_bytes,
    evidence_bytes,
    manifest_bytes,
    readbacks,
    owner_contract_sha256,
    trusted_evidence_sha256,
    now,
):
    """Pins come from separate trusted records, NOT from the untrusted input bundle."""
    new_paid = False
    proposal = None
    try:
        spec = document(sidecar_bytes)
        p = spec.get("placement")
        new_paid = type(p) is dict and p.get("allocation") == "new-paid"
        job = document(workload_bytes, allow_float=True)
        if job.get("schema") == INTENT_SCHEMA and job.get("op") == "launch":
            new_paid = True  # An account/lane is not an approved existing instance.
        require(type(p) is dict, "INVALID_PLACEMENT")
        sites = p.get("sites")
        require(
            not (type(sites) is list and len(sites) > 1)
            and p.get("collective", "none") == "none"
            and p.get("poolVram", False) is False,
            "INVALID_FABRIC_POOLING",
        )
        require(
            _validator("portable-workload.v1").is_valid(spec),
            "INVALID_PORTABLE_CONTRACT",
        )
        require(
            digest(workload_bytes) == spec["workload"]["sha256"],
            "WORKLOAD_DIGEST_MISMATCH",
        )
        require(type(now) is int and now >= 0, "INVALID_CLOCK")
        _check_workload(spec, job, now)
        obs = document(evidence_bytes)
        require(
            digest(evidence_bytes)
            == trusted_evidence_sha256
            == spec["placementReceiptSha256"],
            "UNTRUSTED_PLACEMENT_EVIDENCE",
        )
        new_paid |= obs.get("allocation") == "new-paid"
        expected = _check_evidence(spec, obs, now)
        b, c, w = spec["budget"], spec["controls"], spec["workload"]
        require(
            sum(b[k] for k in ("compute", "storage", "egress", "contingency"))
            <= b["maxMicroUsd"]
            and (not new_paid or b["compute"] > 0)
            and c["stopBy"] <= min(now + b["wallSeconds"], b["validThrough"]),
            "BUDGET_NOT_CAPPED",
        )
        contract_pin = digest(sidecar_bytes)
        CheckpointContract.verify(
            manifest_bytes,
            spec["checkpoint"]["manifestSha256"],
            w["runId"],
            w["attempt"],
            p["sites"][0],
            expected,
            readbacks,
        )
        if new_paid:
            proposal = {
                "contractSha256": contract_pin,
                "site": p["sites"][0],
                "sku": p["sku"],
                "region": p["region"],
                "quantity": 1,
                "budget": b,
                "controls": c,
            }
        require(owner_contract_sha256 == contract_pin, "OWNER_CAP_DIGEST_REQUIRED")
        status, reason = (
            ("PROPOSAL_REQUIRED_CAPPED", "NEW_PAID_NEVER_AUTHORIZED")
            if new_paid
            else ("EXISTING_CONTRACT_VERIFIED", "OFFLINE_EVIDENCE_ONLY")
        )
    except Hold as exc:
        status, reason = ("PROPOSAL_REQUIRED_CAPPED" if new_paid else "HOLD"), str(exc)
    return {
        "status": status,
        "reason": reason,
        "proposal": proposal,
        "mayDispatch": False,
        "maySpend": False,
    }
