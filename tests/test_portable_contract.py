"""Offline evidence mutations: re-pin altered bundles so inner gates really run."""

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unittest

from tools.gpu.portable_contract import review


ROOT = Path(__file__).resolve().parents[1]
NOW = 1_785_000_000  # historical replay clock, not a current placement


def raw(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Bundle:
    def __init__(self, site="massed", mode="existing-approved"):
        signed = json.loads(
            (ROOT / "queue/pending/job-0001-szl-quant-1p5b.json").read_bytes()
        )
        self.job = base64.b64decode(signed["payload"], validate=True)
        old = json.loads(self.job)
        self.chunks = [("model/part-1", [b"abc", b"d"]), ("config.json", [b"{}"])]
        rows = [
            {
                "path": path,
                "bytes": len(b"".join(chunks)),
                "sha256": sha(b"".join(chunks)),
            }
            for path, chunks in self.chunks
        ]
        self.manifest = {
            "schema": "szl.checkpoint-shards/v1",
            "runId": old["jobId"],
            "site": site,
            "attempt": 1,
            "step": 7,
            "totalBytes": 6,
            "shards": rows,
        }
        self.spec = {
            "schema": "szl.portable-workload/v1",
            "workload": {
                "kind": "jobspec.v1",
                "sha256": sha(self.job),
                "runId": old["jobId"],
                "attempt": 1,
            },
            "source": {"githubRepo": "example/trainer", "commit": "a" * 40},
            "asset": {
                "repoType": "model",
                "repoId": old["base"]["repoId"],
                "revision": old["base"]["revision"],
            },
            "placement": {
                "allocation": mode,
                "sites": [site],
                "resourceId": "resource-1",
                "sku": "gpu_1x_a6000_high_ram",
                "region": "us-central-1",
                "deviceId": "d1",
                "minVramMiB": 8000,
                "toolchainSha256": "d" * 64,
                "collective": "none",
                "poolVram": False,
            },
            "checkpoint": {
                "storeId": "archive",
                "versionId": "version-7",
                "manifestSha256": sha(raw(self.manifest)),
                "recoveryPolicy": "verify-before-resume",
            },
            "budget": {
                "currency": "USD",
                "maxMicroUsd": 150,
                "compute": 100,
                "storage": 20,
                "egress": 8,
                "contingency": 22,
                "wallSeconds": 14400,
                "validThrough": NOW + 14400,
            },
            "controls": {
                "idempotencyKey": "logical-effect-1",
                "cancelHookId": "stop-lease-1",
                "cancelOwner": "owner",
                "stopBy": NOW + 14400,
            },
        }
        p = self.spec["placement"]
        self.evidence = {
            "schema": "szl.portable-placement-evidence/v1",
            "checkoutState": "clean-at-pin",
            "assetAccess": "readable-at-pin",
            "costCeilings": {
                "wallSeconds": 14400,
                "compute": 100,
                "storage": 20,
                "egress": 8,
            },
            **{k: deepcopy(self.spec[k]) for k in ("workload", "source", "asset")},
            **{k: p[k] for k in ("resourceId", "sku", "region", "allocation")},
            "site": site,
            "validFrom": NOW - 30,
            "validThrough": NOW + 14400,
            "cancelHookId": "stop-lease-1",
            "devices": [
                {"deviceId": d, "usableVramMiB": 10000, "toolchainSha256": "d" * 64}
                for d in ("d1", "d2")
            ],
            "storage": {
                "persistence": "durable-versioned",
                **{
                    k: self.spec["checkpoint"][k]
                    for k in ("storeId", "versionId", "manifestSha256")
                },
                "expectedPaths": [path for path, _ in self.chunks],
            },
        }

    def replace_job(self, job, kind):
        self.job = raw(job)
        self.spec["workload"].update(kind=kind, sha256=sha(self.job))
        self.evidence["workload"] = deepcopy(self.spec["workload"])

    def args(self):
        m = raw(self.manifest)
        self.spec["checkpoint"]["manifestSha256"] = sha(m)
        self.evidence["storage"]["manifestSha256"] = sha(m)
        e = raw(self.evidence)
        self.spec["placementReceiptSha256"] = sha(e)
        s = raw(self.spec)
        return {
            "sidecar_bytes": s,
            "workload_bytes": self.job,
            "evidence_bytes": e,
            "manifest_bytes": m,
            "readbacks": self.chunks,
            "owner_contract_sha256": sha(s),
            "trusted_evidence_sha256": sha(e),
            "now": NOW,
        }


class PortableContractTests(unittest.TestCase):
    def check(self, args, status, reason):
        result = review(**args)
        self.assertEqual((result["status"], result["reason"]), (status, reason))
        self.assertIs(result["maySpend"], False)
        self.assertIs(result["mayDispatch"], False)
        return result

    def test_existing_single_sites_and_exact_legacy_bytes(self):
        for site in ("hq", "mac", "massed", "salad", "hpc-slurm"):
            with self.subTest(site=site):
                b = Bundle(site)
                self.check(
                    b.args(), "EXISTING_CONTRACT_VERIFIED", "OFFLINE_EVIDENCE_ONLY"
                )
                self.check(
                    b.args() | {"workload_bytes": b.job + b"\n"},
                    "HOLD",
                    "WORKLOAD_DIGEST_MISMATCH",
                )

    def test_typed_assets(self):
        for kind in ("dataset", "space"):
            b = Bundle()
            data = json.loads(b.job)["dataset"]
            b.spec["asset"] = {
                "repoType": kind,
                "repoId": data["repoId"] if kind == "dataset" else "example/space",
                "revision": data["revision"] if kind == "dataset" else "c" * 40,
            }
            b.evidence["asset"] = deepcopy(b.spec["asset"])
            self.check(b.args(), "EXISTING_CONTRACT_VERIFIED", "OFFLINE_EVIDENCE_ONLY")
        b.spec["asset"]["repoType"] = "model"
        self.check(b.args(), "HOLD", "JOB_ASSET_MISMATCH")

    def test_jobspec_v2_and_unchanged_signed_schema_boundary(self):
        b = Bundle()
        j = json.loads(b.job)
        j["kind"] = "unsloth-frontier-sft-v2"
        j["jobId"] = "job-2026-test"
        del j["base"]["licenseNote"]
        j["base"]["licenseId"] = "synthetic"
        j["dataset"].update(format="messages-jsonl", licenseId="synthetic")
        j["recipe"].update(
            packingStrategy="ffd",
            assistantOnlyLoss=False,
            useRsLoRA=True,
            warmupRatio=0.01,
            weightDecay=0,
            lrSchedulerType="cosine",
            expectedChatTemplateSha256="e" * 64,
        )
        j["gates"]["maxDatasetRows"] = 10
        j["outputs"]["exports"] = {
            "adapter": True,
            "merged16bit": False,
            "ggufQuantizations": [],
            "requireReloadSmoke": True,
        }
        j["eval"].update(
            suite="frontier-heldout-v2",
            maxGenerations=1,
            maxNewTokens=1,
            requiredJsonKeys=["test"],
            convictionCeiling=1,
            maxDegenerateRate=0,
            minJsonValidRate=1,
            minRequiredKeysRate=1,
            minCeilingRespectRate=1,
        )
        b.spec["workload"]["runId"] = b.manifest["runId"] = j["jobId"]
        b.replace_job(j, "jobspec.v2")
        self.check(b.args(), "EXISTING_CONTRACT_VERIFIED", "OFFLINE_EVIDENCE_ONLY")

        # The unselected dataset is valid for legacy v2 (40..64 hex) but
        # not for the portable v1 SHA-1 profile. Keep the selected model pinned.
        wider = deepcopy(j)
        wider["dataset"]["revision"] = "e" * 64
        b.replace_job(wider, "jobspec.v2")
        self.check(b.args(), "HOLD", "UNPINNED_JOB_ASSET")

        j["placement"] = b.spec["placement"]
        b.replace_job(j, "jobspec.v2")
        self.check(b.args(), "HOLD", "JOBSPEC_BINDING")

    def test_complete_and_incomplete_new_paid_proposals(self):
        b = Bundle(mode="new-paid")
        args = b.args()
        for owner, reason in [
            (None, "OWNER_CAP_DIGEST_REQUIRED"),
            ("0" * 64, "OWNER_CAP_DIGEST_REQUIRED"),
            (args["owner_contract_sha256"], "NEW_PAID_NEVER_AUTHORIZED"),
        ]:
            with self.subTest(owner=owner):
                r = self.check(
                    args | {"owner_contract_sha256": owner},
                    "PROPOSAL_REQUIRED_CAPPED",
                    reason,
                )
                self.assertEqual(
                    r["proposal"]["contractSha256"], sha(args["sidecar_bytes"])
                )
                self.assertEqual(r["proposal"]["budget"]["maxMicroUsd"], 150)
                self.assertEqual(r["proposal"]["quantity"], 1)
        del b.spec["budget"]
        self.assertIsNone(
            self.check(
                b.args(), "PROPOSAL_REQUIRED_CAPPED", "INVALID_PORTABLE_CONTRACT"
            )["proposal"]
        )

    def test_massed_launch_cannot_borrow_existing_account_approval(self):
        b = Bundle(mode="new-paid")
        p = b.spec["placement"]
        intent = {
            "schema": "szl.massed-admission-intent/v1",
            "intent_id": b.spec["workload"]["runId"],
            "lane_id": p["resourceId"],
            "product_name": p["sku"],
            "region_name": p["region"],
            "op": "launch",
            "actor": "review",
            "group": "fixture",
            "image_id": 1,
            "hard_deadline_s": 14400,
            "prev_intent_sha256": None,
        }
        b.replace_job(intent, "massed-launch.v1")
        self.check(b.args(), "PROPOSAL_REQUIRED_CAPPED", "NEW_PAID_NEVER_AUTHORIZED")
        for field, value in [
            ("intent_id", "other-run"),
            ("lane_id", "other-lane"),
            ("prev_intent_sha256", "a" * 63),
            ("hard_deadline_s", 59),
            ("hard_deadline_s", 21_601),
        ]:
            with self.subTest(field=field, value=value):
                bad = deepcopy(b)
                if field == "hard_deadline_s":
                    # Every other deadline/quote agrees, including the short one.
                    bad.spec["budget"]["wallSeconds"] = value
                    bad.evidence["costCeilings"]["wallSeconds"] = value
                    bad.spec["controls"]["stopBy"] = NOW + value
                    bad.spec["budget"]["validThrough"] = NOW + value
                    bad.evidence["validThrough"] = NOW + value
                bad.replace_job(intent | {field: value}, "massed-launch.v1")
                result = self.check(
                    bad.args(), "PROPOSAL_REQUIRED_CAPPED", "LAUNCH_INTENT_BINDING"
                )
                self.assertIsNone(result["proposal"])
        p["allocation"] = b.evidence["allocation"] = "existing-approved"
        self.check(b.args(), "PROPOSAL_REQUIRED_CAPPED", "LAUNCH_INTENT_BINDING")

    def test_job_identity_time_window_and_requested_vram(self):
        def iso(timestamp):
            return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()

        for field, value, reason in [
            ("jobId", "job-0002-other", "JOBSPEC_BINDING"),
            ("createdAt", iso(NOW + 1), "JOB_TIME_WINDOW"),
            ("expiresAt", iso(NOW + 14_400 - 1), "JOB_TIME_WINDOW"),
            ("minFreeVramGb", 8, "JOB_GATE_MISMATCH"),  # 8192 MiB > 8000 requested
        ]:
            with self.subTest(field=field):
                b = Bundle()
                job = json.loads(b.job)
                target = job["gates"] if field == "minFreeVramGb" else job
                target[field] = value
                b.replace_job(job, "jobspec.v1")
                self.check(b.args(), "HOLD", reason)

    def test_independently_pinned_placement_evidence(self):
        for section, field, value, reason in [
            (None, "validFrom", NOW + 1, "STALE_PLACEMENT_EVIDENCE"),
            ("workload", "attempt", 2, "PLACEMENT_EVIDENCE_MISMATCH"),
            ("source", "commit", "b" * 40, "PLACEMENT_EVIDENCE_MISMATCH"),
            ("asset", "revision", "f" * 40, "PLACEMENT_EVIDENCE_MISMATCH"),
            ("costCeilings", "wallSeconds", 14_399, "COST_EVIDENCE_MISMATCH"),
        ]:
            with self.subTest(section=section, field=field):
                b = Bundle()
                record = b.evidence if section is None else b.evidence[section]
                record[field] = value
                self.check(b.args(), "HOLD", reason)
        b = Bundle()
        b.evidence["devices"].append(deepcopy(b.evidence["devices"][0]))
        self.check(b.args(), "HOLD", "DEVICE_EVIDENCE")

    def test_budget_lifetime_and_positive_new_paid_compute(self):
        b = Bundle()
        b.spec["budget"]["validThrough"] = b.evidence["validThrough"] + 1
        self.check(b.args(), "HOLD", "COST_EVIDENCE_MISMATCH")

        b = Bundle()
        late = NOW + b.spec["budget"]["wallSeconds"] + 1
        b.spec["budget"]["validThrough"] = b.evidence["validThrough"] = late
        b.spec["controls"]["stopBy"] = late
        self.check(b.args(), "HOLD", "BUDGET_NOT_CAPPED")

        b = Bundle(mode="new-paid")
        b.spec["budget"]["compute"] = b.evidence["costCeilings"]["compute"] = 0
        result = self.check(b.args(), "PROPOSAL_REQUIRED_CAPPED", "BUDGET_NOT_CAPPED")
        self.assertIsNone(result["proposal"])

    def test_checkpoint_types_paths_and_duplicate_readback(self):
        b = Bundle()
        b.manifest["attempt"] = True  # bool compares equal to reviewed attempt 1
        self.check(b.args(), "HOLD", "INVALID_CHECKPOINT")

        b = Bundle()
        path = "config.json."
        b.manifest["shards"][1]["path"] = b.evidence["storage"]["expectedPaths"][1] = (
            path
        )
        b.chunks[1] = (path, [b"{}"])
        self.check(b.args(), "HOLD", "INVALID_SHARD_PATH")

        b = Bundle()
        # One real byte: deleting the type check must not be masked by size/hash.
        b.manifest["shards"][1].update(bytes=True, sha256=sha(b"x"))
        b.manifest["totalBytes"] = 5
        b.chunks[1] = ("config.json", [b"x"])
        self.check(b.args(), "HOLD", "INVALID_SHARD_RECORD")

        b = Bundle()
        b.chunks.append(deepcopy(b.chunks[0]))  # complete inventory, plus one repeat
        self.check(b.args(), "HOLD", "READBACK_INVENTORY")

    def test_required_fields_and_pins(self):
        for section, key, value in [
            ("source", "commit", "main"),
            ("source", "githubRepo", "https://not-github.invalid/x"),
            ("asset", "revision", "latest"),
            ("asset", "repoType", "kernel"),
            ("asset", "repoId", "../file"),
            ("budget", "maxMicroUsd", None),
            ("budget", "egress", None),
            ("budget", "storage", -1),
            ("budget", "compute", True),
            ("controls", "idempotencyKey", None),
            ("controls", "cancelHookId", None),
            ("controls", "cancelOwner", None),
            ("controls", "stopBy", None),
            ("checkpoint", "recoveryPolicy", None),
            ("placement", "toolchainSha256", "latest"),
            ("placement", "minVramMiB", 0),
            ("placement", "sites", ["unknown"]),
            ("placement", "environment", {"MASTER_ADDR": "other-site"}),
        ]:
            with self.subTest(section=section, key=key):
                b = Bundle()
                if value is None:
                    del b.spec[section][key]
                else:
                    b.spec[section][key] = value
                self.check(b.args(), "HOLD", "INVALID_PORTABLE_CONTRACT")

    def test_no_pooling_or_adding_device_memory(self):
        for flags in [
            {"sites": ["hq", "mac"]},
            {"sites": ["massed", "salad"], "collective": "nccl"},
            {"sites": ["hq", "hpc-slurm"], "poolVram": True},
            {"poolVram": True},
            {"collective": "nccl"},
        ]:
            with self.subTest(flags=flags):
                b = Bundle()
                b.spec["placement"].update(flags)
                self.check(b.args(), "HOLD", "INVALID_FABRIC_POOLING")
        b = Bundle()
        b.spec["placement"]["minVramMiB"] = 15000  # 2x 10000 is not 1x 20000
        self.check(b.args(), "HOLD", "SINGLE_DEVICE_VRAM")

    def test_coherently_repinned_partial_and_duplicate_manifests(self):
        b = Bundle()
        b.manifest["shards"].pop()
        b.manifest["totalBytes"] = 4
        b.chunks.pop()
        self.check(b.args(), "HOLD", "PARTIAL_MANIFEST")
        for path, reason in [
            ("model/part-1", "DUPLICATE_SHARD_PATH"),
            ("MODEL/part-1", "DUPLICATE_SHARD_PATH"),
            ("../escape", "INVALID_SHARD_PATH"),
            ("model", "INVALID_SHARD_PATH"),
            ("config.json.bak", "PARTIAL_MANIFEST"),
            ("aux.bin", "INVALID_SHARD_PATH"),
        ]:
            with self.subTest(path=path):
                b = Bundle()
                b.manifest["shards"][1]["path"] = path
                self.check(b.args(), "HOLD", reason)

    def test_readback_bytes_and_exact_inventory(self):
        for data, reason in [
            ([("model/part-1", [b"ab"])], "SHARD_SIZE_MISMATCH"),
            ([("model/part-1", [b"abcd!"])], "SHARD_SIZE_MISMATCH"),
            ([("model/part-1", [b"ABCD"])], "SHARD_HASH_MISMATCH"),
            ([("model/part-1", [b"abcd"])], "READBACK_INVENTORY"),
            ([("extra", [b"abcd"])], "READBACK_INVENTORY"),
            ([("config.json", [b"{}"])] * 2, "READBACK_INVENTORY"),
            ([("config.json", [b""])], "INVALID_READBACK_CHUNK"),
            (None, "READBACK_UNAVAILABLE"),
        ]:
            with self.subTest(reason=reason, data=data):
                self.check(Bundle().args() | {"readbacks": data}, "HOLD", reason)
        b = Bundle()
        del b.manifest["shards"][0]["sha256"]
        self.check(b.args(), "HOLD", "INVALID_SHARD_RECORD")
        b = Bundle()
        b.manifest["totalBytes"] += 1
        self.check(b.args(), "HOLD", "CHECKPOINT_TOTAL_MISMATCH")

    def test_observation_and_budget_rebinding(self):
        for part, name, value, reason in [
            ("storage", "persistence", "UNKNOWN", "STORAGE_UNVERIFIED"),
            ("storage", "versionId", "replacement", "STORAGE_UNVERIFIED"),
            ("evidence", "site", "mac", "PLACEMENT_EVIDENCE_MISMATCH"),
            ("evidence", "checkoutState", "dirty", "SOURCE_OR_ASSET_UNVERIFIED"),
            ("evidence", "assetAccess", "held", "SOURCE_OR_ASSET_UNVERIFIED"),
            (
                "evidence",
                "costCeilings",
                {"wallSeconds": 14400, "compute": 101, "storage": 20, "egress": 8},
                "COST_EVIDENCE_MISMATCH",
            ),
            ("evidence", "cancelHookId", "other", "PLACEMENT_EVIDENCE_MISMATCH"),
            ("evidence", "validThrough", NOW - 1, "STALE_PLACEMENT_EVIDENCE"),
            ("device", "toolchainSha256", "a" * 64, "TOOLCHAIN_MISMATCH"),
            ("budget", "maxMicroUsd", 149, "BUDGET_NOT_CAPPED"),
            ("budget", "validThrough", NOW, "BUDGET_NOT_CAPPED"),
            ("budget", "wallSeconds", 1, "JOB_GATE_MISMATCH"),
            ("manifest", "attempt", 2, "INVALID_CHECKPOINT"),
            ("manifest", "site", "salad", "INVALID_CHECKPOINT"),
            ("manifest", "runId", "another-run", "INVALID_CHECKPOINT"),
        ]:
            with self.subTest(part=part, field=name):
                b = Bundle()
                target = {
                    "storage": b.evidence["storage"],
                    "evidence": b.evidence,
                    "device": b.evidence["devices"][0],
                    "budget": b.spec["budget"],
                    "manifest": b.manifest,
                }[part]
                target[name] = value
                self.check(b.args(), "HOLD", reason)
        b = Bundle()
        b.evidence["allocation"] = "new-paid"
        self.check(b.args(), "PROPOSAL_REQUIRED_CAPPED", "PLACEMENT_EVIDENCE_MISMATCH")

    def test_pins_are_external_and_duplicate_keys_are_not_last_write_wins(self):
        b = Bundle()
        pins = b.args()
        b.spec["controls"]["idempotencyKey"] = "another-effect"
        self.check(
            b.args() | {"owner_contract_sha256": pins["owner_contract_sha256"]},
            "HOLD",
            "OWNER_CAP_DIGEST_REQUIRED",
        )
        b.evidence["devices"][0]["usableVramMiB"] += 1
        self.check(
            b.args() | {"trusted_evidence_sha256": pins["trusted_evidence_sha256"]},
            "HOLD",
            "UNTRUSTED_PLACEMENT_EVIDENCE",
        )
        self.check(
            pins | {"manifest_bytes": pins["manifest_bytes"] + b"\n"},
            "HOLD",
            "MANIFEST_DIGEST_MISMATCH",
        )
        self.check(
            pins | {"sidecar_bytes": b'{"placement":null,' + pins["sidecar_bytes"][1:]},
            "HOLD",
            "DUPLICATE_JSON_KEY",
        )
        self.check(pins | {"now": True}, "HOLD", "INVALID_CLOCK")
