#!/usr/bin/env python3
"""Verify-first contract for a single governed SZL-Nemo v3 GPU attempt.

This module has no network or ML-framework imports.  The dispatcher and runner
load it only after the DSSE envelope has verified against the pinned engine key.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from datetime import datetime
from typing import Any

from frontier_contract import ContractError

NEMO_V3_PAYLOAD_TYPE = "application/vnd.szl.gpu-bridge.nemo-v3.jobspec.v1+json"
NEMO_V3_KIND = "szl-nemo-governed-v3"
_SHA = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_JOB = re.compile(r"^job-[0-9]{4}-nemo-v3-[a-z0-9][a-z0-9-]{2,64}$")
_ATTEMPT_JOB = re.compile(
    r"^job-[0-9]{4}-nemo-v3-governed-attempt-(?P<generation>[1-9][0-9]*)$"
)
_ENGINE_KEY_ID = re.compile(r"^[0-9a-f]{16}$")
_SAFE_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]*$")
_HOLDOUT_NAMES = ("original-v2", "shadow-v2", "challenge-v3")
LEGACY_ENGINE_KEY_ID = "5c6cf59741ade920"
ATTEMPT_1_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-1"
SUCCESSOR_2_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-successor-2"
PROVISIONAL_ENGINE_KEY_ID = "815714c8d4ae3e4d"
COORDINATED_ENGINE_KEY_ID = "b8041281c81c4caa"
COORDINATED_ENGINE_SPKI_SHA256 = (
    "b8041281c81c4caaea18112df5e8c99ea8472f0711fc796fc3072c27398af2cf"
)
SETTLED_A11OY_SOURCE_REVISION = "5f98d90a42e021cf29948457a2404a159f236487"
SETTLED_OWNER_WORKFLOW_BLOB = "7e08ffc8aa87b78d0fa1618d7d3c3e68cb81ca33"
SETTLED_A11OY_RELOCK_RUN_URL = (
    "https://github.com/szl-holdings/a11oy/actions/runs/30561614589"
)
CORRECTED_BRIDGE_REVISION = "2237bb3f36663343ace29d98cda6c32e165450a0"
NEXT_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-4"
FINAL_A11OY_SOURCE_REVISION = "e3d4a46724b222c8a5b2b6f04877bc115a6c82cb"
FINAL_OWNER_WORKFLOW_BLOB = "2522d3b54eeb7adc37ffc47e7c685a5ce7edf68f"
FINAL_A11OY_RELOCK_RUN_URL = (
    "https://github.com/szl-holdings/a11oy/actions/runs/30588489971"
)
FINAL_CORRECTED_BRIDGE_REVISION = "a2015accc0be8060c4084455e829a9373e5c99e2"
ATTEMPT_5_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-5"
EXECUTION_A11OY_SOURCE_REVISION = "78b35d244b89c7663063372ff459894bab2977b6"
EXECUTION_OWNER_WORKFLOW_BLOB = "d29d937b2d398e9c207777a9a819aadd050ac231"
EXECUTION_A11OY_RELOCK_RUN_URL = (
    "https://github.com/szl-holdings/a11oy/actions/runs/30592401025"
)
ATTEMPT_6_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-6"
SUCCESSOR_A11OY_SOURCE_REVISION = "2b190b3806a5d2b3faa58f34c2db41c5dc4668fa"
SUCCESSOR_OWNER_WORKFLOW_BLOB = "d29d937b2d398e9c207777a9a819aadd050ac231"
SUCCESSOR_A11OY_RELOCK_RUN_URL = (
    "https://github.com/szl-holdings/a11oy/actions/runs/30601635066"
)
SUCCESSOR_CORRECTED_BRIDGE_REVISION = "2f33607d8fcbec76fe98290258ec3dfa728fb509"
FUTURE_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-7"
NEXT_RUNTIME_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-8"
NEXT_RUNTIME_CORRECTED_BRIDGE_REVISION = "dc36af2b264bbdb4cc101593c54c5b2c24c1d9cf"
RECOVERY_A11OY_SOURCE_REVISION = "c6aa4f08f752a22bbae35cf5a618a81811494a43"
RECOVERY_OWNER_WORKFLOW_BLOB = "f0ab364e1db9c48a0d8f49c7f0c17b5e44cad99d"
RECOVERY_A11OY_RELOCK_RUN_URL = (
    "https://github.com/szl-holdings/a11oy/actions/runs/30607399378"
)
ATTEMPT_9_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-9"
ATTEMPT_9_CORRECTED_BRIDGE_REVISION = "eeabd1b52380d2b24439e53d5e4ad38f8114556c"
ATTEMPT_10_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-10"
ATTEMPT_10_CORRECTED_BRIDGE_REVISION = "37479c23af3228a57ad6018b3f9134186e6d7fa7"
EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION = "434d653eaf100b9b3e5484687db1e6e6ca7116c9"
EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB = "7cf0c877399471a084d3e70638ef50ec28d7f646"
EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL = (
    "https://github.com/szl-holdings/a11oy/actions/runs/30613619902"
)
ATTEMPT_17_A11OY_SOURCE_REVISION = "cad529a2cef4cb43024bf4974ae155d89f33fa5b"
ATTEMPT_17_OWNER_WORKFLOW_BLOB = "7cf0c877399471a084d3e70638ef50ec28d7f646"
ATTEMPT_17_A11OY_RELOCK_RUN_URL = (
    "https://github.com/szl-holdings/a11oy/actions/runs/30706177629"
)
ATTEMPT_11_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-11"
ATTEMPT_11_CORRECTED_BRIDGE_REVISION = "f07263bc37ef6e90b313ba5576ef425d845cf287"
ATTEMPT_12_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-12"
ATTEMPT_12_CORRECTED_BRIDGE_REVISION = "d110abb8ea48c9382a70c3eead22dddf555f292b"
ATTEMPT_13_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-13"
ATTEMPT_14_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-14"
ATTEMPT_15_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-15"
ATTEMPT_16_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-16"
ATTEMPT_17_REVIEWED_JOB_ID = "job-2026-nemo-v3-governed-attempt-17"
ATTEMPT_17_CORRECTED_BRIDGE_REVISION = "120a49206354ad98779ac46a65ca1fae45131e1c"
_PREDECESSOR_SUCCESSOR_REPLACEMENT = {
    "reviewedJobId": SUCCESSOR_2_REVIEWED_JOB_ID,
    "reviewedSpecPath": "jobspecs/nemo-v3-20260729-successor-2-reviewed.json",
    "reviewedSpecSha256": (
        "9d58f752c26ac37ae7fa4999e33a6f136d060e97704124df26f0ee7948a11746"
    ),
    "automaticRetry": False,
}
_PREDECESSOR_EXECUTION_EVIDENCE = {
    "predecessorJobId": ATTEMPT_1_REVIEWED_JOB_ID,
    "predecessorClaimSha256": (
        "77fd63583bf11f1d7416cea7e6e0c02b230973d4773f9c409ce18aa83140f10b"
    ),
    "predecessorEnvelopeSha256": (
        "09187c0a724c8caf8a11dcd492d3f284af8a18791adac7e1a98b9a21bf81591b"
    ),
    "predecessorBridgeRevision": "114c3030763291009d665ae88cb3d6537fccacef",
    "predecessorImageId": (
        "sha256:537e4a25a503d202ec75dbb9035bd9688ba2ae8d8a7555466840e581d5109f28"
    ),
    "predecessorClaimedAt": "2026-07-29T16:41:34.8842570+00:00",
    "incidentUrl": (
        "https://github.com/szl-holdings/szl-gpu-bridge/issues/4"
        "#issuecomment-5120817312"
    ),
    "failurePhase": "PRE_TRAINING_RUNTIME_SOURCE_PARSE",
    "successorGeneration": 2,
    "automaticRetry": False,
    "trainingStarted": False,
    "modelRepositoryCodeImported": False,
    "holdoutsAccessed": False,
    "candidateProduced": False,
    "receiptIntentProduced": False,
    "terminalLedgerWritten": False,
    "scienceInputsReused": True,
}
_ATTEMPT_4_REPLACEMENT = {
    "sourceRevision": SETTLED_A11OY_SOURCE_REVISION,
    "workflowBlob": SETTLED_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": NEXT_REVIEWED_JOB_ID,
}
_ATTEMPT_5_REPLACEMENT = {
    "sourceRevision": FINAL_A11OY_SOURCE_REVISION,
    "workflowBlob": FINAL_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": ATTEMPT_5_REVIEWED_JOB_ID,
}
_ATTEMPT_6_REPLACEMENT = {
    "sourceRevision": EXECUTION_A11OY_SOURCE_REVISION,
    "workflowBlob": EXECUTION_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": ATTEMPT_6_REVIEWED_JOB_ID,
}
_ATTEMPT_7_REPLACEMENT = {
    "sourceRevision": SUCCESSOR_A11OY_SOURCE_REVISION,
    "workflowBlob": SUCCESSOR_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": FUTURE_REVIEWED_JOB_ID,
}
_ATTEMPT_8_REPLACEMENT = {
    "sourceRevision": SUCCESSOR_A11OY_SOURCE_REVISION,
    "workflowBlob": SUCCESSOR_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": NEXT_RUNTIME_REVIEWED_JOB_ID,
}
_ATTEMPT_9_REPLACEMENT = {
    "sourceRevision": RECOVERY_A11OY_SOURCE_REVISION,
    "workflowBlob": RECOVERY_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": ATTEMPT_9_REVIEWED_JOB_ID,
}
_ATTEMPT_10_REPLACEMENT = {
    "sourceRevision": RECOVERY_A11OY_SOURCE_REVISION,
    "workflowBlob": RECOVERY_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": ATTEMPT_10_REVIEWED_JOB_ID,
}
_ATTEMPT_11_REPLACEMENT = {
    "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    "workflowBlob": EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": ATTEMPT_11_REVIEWED_JOB_ID,
}
_ATTEMPT_12_REPLACEMENT = {
    "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    "workflowBlob": EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": ATTEMPT_12_REVIEWED_JOB_ID,
}
_ATTEMPT_13_REPLACEMENT = {
    "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    "workflowBlob": EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": ATTEMPT_13_REVIEWED_JOB_ID,
}
_ATTEMPT_14_REPLACEMENT = {
    "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    "workflowBlob": EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": ATTEMPT_14_REVIEWED_JOB_ID,
}
_ATTEMPT_15_REPLACEMENT = {
    "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    "workflowBlob": EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": ATTEMPT_15_REVIEWED_JOB_ID,
}
_ATTEMPT_16_REPLACEMENT = {
    "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    "workflowBlob": EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    "workflowVersion": "nemo-v3-owner-dispatch.v4",
    "settledA11oyRelockRunUrl": EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": ATTEMPT_16_REVIEWED_JOB_ID,
    "successorGeneration": 16,
}
_ATTEMPT_17_REPLACEMENT = {
    "sourceRevision": ATTEMPT_17_A11OY_SOURCE_REVISION,
    "workflowBlob": ATTEMPT_17_OWNER_WORKFLOW_BLOB,
    "workflowVersion": "nemo-v3-owner-dispatch.v4",
    "settledA11oyRelockRunUrl": ATTEMPT_17_A11OY_RELOCK_RUN_URL,
    "engineKeyId": COORDINATED_ENGINE_KEY_ID,
    "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
    "reviewedJobId": ATTEMPT_17_REVIEWED_JOB_ID,
    "successorGeneration": 17,
}
QUARANTINE_POLICIES: dict[str, dict[str, Any]] = {
    ATTEMPT_1_REVIEWED_JOB_ID: {
        "statuses": (
            "PRE_TRAINING_RUNTIME_SOURCE_PARSE",
            "POST_CLAIM",
            "NEVER_DISPATCH",
            "NEVER_RESEND",
            "NEVER_RESIGN",
        ),
        "queue_file_sha256": (
            "0686889c3abcf54e3f6b2151bc60155176e1eccb25af7b01d9f1fbf05080d80d"
        ),
        "payload_sha256": (
            "8a5c2e3f99711be84e45371824ca737d480e587ff61c55cc3d30ad96d2c62055"
        ),
        "engine_key_id": LEGACY_ENGINE_KEY_ID,
        "source_revision": "a5351c8e37a7cfe54e0c3cf53c8bbd460a16c11c",
        "replacement": _PREDECESSOR_SUCCESSOR_REPLACEMENT,
        "execution_evidence_path": (
            "queue/evidence/job-2026-nemo-v3-governed-attempt-1.json"
        ),
        "execution_evidence_sha256": (
            "d3f28fd63ee4c84ecf7aa72300a7fe55a29033953906356a83fdf089f47aaed6"
        ),
        "execution_evidence": _PREDECESSOR_EXECUTION_EVIDENCE,
    },
    "job-2026-nemo-v3-governed-attempt-2": {
        "statuses": ("STALE_SOURCE", "RETIRED_KEY", "NEVER_DISPATCH"),
        "queue_file_sha256": (
            "e74ecaea040c2abb52a5613c32e0648994f96ff39910c70e1fcc3e23fc053724"
        ),
        "payload_sha256": (
            "84a808615ba1693935eee8cc9fa1a4c5a83d119b79ad7e9437380ec73756b90d"
        ),
        "engine_key_id": LEGACY_ENGINE_KEY_ID,
        "source_revision": "b21b8fb65400e7eb39595365c5f54c80ed78aa67",
        "replacement": _ATTEMPT_4_REPLACEMENT,
    },
    "job-2026-nemo-v3-governed-successor-3": {
        "statuses": (
            "UNAUTHORIZED_PROVISIONAL_KEY",
            "STALE_SOURCE",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "bb624d301f23552617566f57167a12360bbba27afebee086a8262b1be7ee6eaa"
        ),
        "payload_sha256": (
            "f20bf865dca5413262e5fd3733df112486aec72bb9b47932083ffecb2470a415"
        ),
        "engine_key_id": PROVISIONAL_ENGINE_KEY_ID,
        "source_revision": "a5351c8e37a7cfe54e0c3cf53c8bbd460a16c11c",
        "replacement": _ATTEMPT_4_REPLACEMENT,
    },
    "job-2026-nemo-v3-governed-attempt-4": {
        "statuses": (
            "STALE_SOURCE",
            "TRANSPORT_UNREPRESENTABLE",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "e240a176849b1f6c0d453ac55277cd7732b3a302ea9679db78d3c612501f27f2"
        ),
        "payload_sha256": (
            "14441cf982b177c1b613e56e63eae8be3e589ae35444826b40731c32312268e5"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": SETTLED_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_5_REPLACEMENT,
    },
    ATTEMPT_5_REVIEWED_JOB_ID: {
        "statuses": (
            "STALE_SOURCE",
            "HOST_EXECUTION_POLICY_BLOCKED",
            "PRE_ADMISSION",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "30549fc522238193b4985dbf96a690518bad2ae8c399dc3ee78fb9dd7f551009"
        ),
        "payload_sha256": (
            "374901dec6923e0c28688407e581d374827d76f7567970d8ec481b6bf140c67b"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": FINAL_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_6_REPLACEMENT,
    },
    ATTEMPT_6_REVIEWED_JOB_ID: {
        "statuses": (
            "STALE_SOURCE",
            "PRE_DISPATCH_VALIDATOR_REJECTED",
            "PRE_EVENT",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "c68e1ecf380d7023c27439e9988ca182ebd9b2446dc769269d4de1c48d507d70"
        ),
        "payload_sha256": (
            "d0fa9bd15f8e576411b643858d650470b6f1d5ddd56003cd53eda28d83dd914d"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": EXECUTION_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_7_REPLACEMENT,
    },
    FUTURE_REVIEWED_JOB_ID: {
        "statuses": (
            "RUNTIME_CONTRACT_BINDING_REJECTED",
            "PRE_CLAIM",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "8c1e333f797a8de634217b19cd140994a1d4f3920afebdf6f658dcc984188a96"
        ),
        "payload_sha256": (
            "0fa239d3e14f0644d26b76c0e605ea8068b305cd4d96ea41385cad38fbdfbde7"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": SUCCESSOR_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_8_REPLACEMENT,
    },
    NEXT_RUNTIME_REVIEWED_JOB_ID: {
        "statuses": (
            "TRUSTED_PREFETCH_DIRTIED_EXECUTION_CHECKOUT",
            "PRE_CLAIM",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "b2db463661ab9e16bf24267c82ee104cf25344e7b4addbd2e9867e7e33be3719"
        ),
        "payload_sha256": (
            "3372fff9c21a73ee140598c152b728b4d7694fb0a066c80e8b55e09832a0769d"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": SUCCESSOR_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_9_REPLACEMENT,
    },
    ATTEMPT_9_REVIEWED_JOB_ID: {
        "statuses": (
            "ISOLATED_HF_CACHE_ROOT_PERMISSION_BLOCKED",
            "TRUSTED_FINALIZER_RUNTIME_BINDING_REJECTED",
            "POST_CLAIM",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "a7b67f1245137b3422d6e2ce5cf379aa9adb193e1f1d9db0dec8abf92bf5fa49"
        ),
        "payload_sha256": (
            "f8ec93b0a2967e548ba2222cbf8a69abbe89987c98e695688c39c0e0d3827c5b"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": RECOVERY_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_10_REPLACEMENT,
    },
    ATTEMPT_10_REVIEWED_JOB_ID: {
        "statuses": (
            "IMMUTABLE_RUNTIME_JOB_BINDING_REJECTED",
            "PRE_CLAIM",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "b354d34dcc6487e311b2d40413de4920ef8646d3f40e9d7442d366152aac901b"
        ),
        "payload_sha256": (
            "2287b1be69239ec0f577ee6e712e0093345e46640485dc6fefa88e8104d727c9"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": RECOVERY_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_11_REPLACEMENT,
    },
    ATTEMPT_11_REVIEWED_JOB_ID: {
        "statuses": (
            "TOKENIZER_LOAD_BLOCKED",
            "POST_CLAIM",
            "SIGNED_BLOCKED_RECEIPT",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "7b9af824b529fa80ec51e060cd0fa14f1af8acc8ded5fff5b10f159acb861918"
        ),
        "payload_sha256": (
            "85f08bc171370b25606915008d1b96ff50f670d09e20eb631b4c1ebeb108d994"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_12_REPLACEMENT,
        "execution_evidence_path": (
            "queue/evidence/job-2026-nemo-v3-governed-attempt-11.json"
        ),
        "execution_evidence_sha256": (
            "ab8876488cb198718b576c53db427242b85f5152628bae2c0d040ce8f82a4908"
        ),
        "execution_evidence": {
            "workflowRunId": "30620232291",
            "workflowRunAttempt": 1,
            "runtimeClaimSha256": (
                "f73c18a970d5b99ea8f567ff682eb9c8b7e1ba9f1e769b8c3f6ce4ad93765cc2"
            ),
            "attemptClaimSha256": (
                "3b0caf335622a1034d5e5ce31dd81d4b66819f520805c3cfe1f10c634a7d1f80"
            ),
            "receiptRevision": "1a74ad3f5fc2682e6bbdd034a68399dee7e79525",
            "receiptFileSha256": (
                "f6f1c5af7c8a47c4c4a4ce35ccb9d2859cf3177c06c439bd529c901308aeb9e3"
            ),
            "receiptKeyId": "167c14fbddbe97cc",
            "receiptVerdict": "BLOCKED",
            "receiptReason": (
                "RuntimeError: Unsloth: The tokenizer is weirdly not loaded? "
                "Please check if there is one."
            ),
            "trainingStarted": False,
            "candidateUploaded": False,
            "adapterUploaded": False,
            "modelCardUploaded": False,
            "datasetUploaded": False,
            "deployed": False,
            "promoted": False,
        },
    },
    ATTEMPT_12_REVIEWED_JOB_ID: {
        "statuses": (
            "RUNTIME_JOB_BINDING_REJECTED",
            "PRE_CLAIM",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "a1c9f3d909b120d3675efe2cee0ba06b1c92c950f3a9ed4cc4e5b242971ed70f"
        ),
        "payload_sha256": (
            "a5e04951412bb0c4d085e567e4e869d52bdf6987546b16ffcd6d2bcb72768ce8"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_13_REPLACEMENT,
        "execution_evidence_path": (
            "queue/evidence/job-2026-nemo-v3-governed-attempt-12.json"
        ),
        "execution_evidence_sha256": (
            "0d5caab31736bf00fd8e6457aa437edc8d2a86466c5f6c8bb31fe67d63274215"
        ),
        "execution_evidence": {
            "workflowRunId": "30626533443",
            "workflowRunAttempt": 1,
            "workflowJobId": "91142994672",
            "failurePhase": "PRE_CLAIM_AUTHENTICATED_PREFETCH_RUNTIME_BINDING",
            "errorType": "frontier_contract.ContractError",
            "error": (
                "coordinated authorization requires an exact reviewed job binding"
            ),
            "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
            "envelopeRevision": ("6b21684b64bf01971f3c3aac71493bba8078e532"),
            "executionBridgeRevision": ATTEMPT_12_CORRECTED_BRIDGE_REVISION,
            "claimCreated": False,
            "jobDirectoryCreated": False,
            "prefetchReceiptCreated": False,
            "trainingStarted": False,
            "receiptIntentProduced": False,
            "receiptUploaded": False,
            "candidateUploaded": False,
            "adapterUploaded": False,
            "modelCardUploaded": False,
            "datasetUploaded": False,
            "deployed": False,
            "promoted": False,
        },
    },
    ATTEMPT_13_REVIEWED_JOB_ID: {
        "statuses": (
            "SFTCONFIG_STRATEGY_KEY_BLOCKED",
            "POST_CLAIM",
            "PRE_TRAINING",
            "SIGNED_BLOCKED_RECEIPT",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "de31cbb574cdeeaaf611a25fe1e40616b7fe8d4f6e2e138b66697474f5d800b0"
        ),
        "payload_sha256": (
            "82f619eb1fff6a7617b5761358d2f5c1d8ca62a306eb7cb1bf2570e096b2b9fc"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_14_REPLACEMENT,
        "execution_evidence_path": (
            "queue/evidence/job-2026-nemo-v3-governed-attempt-13.json"
        ),
        "execution_evidence_sha256": (
            "8d9c7e4b37138a2b61de9e15f3c622dc1291f2c38f909a7a9f48115385831c4a"
        ),
        "execution_evidence": {
            "workflowRunId": "30629929196",
            "workflowRunAttempt": 1,
            "workflowJobId": "91153664576",
            "failurePhase": "POST_CLAIM_SFTCONFIG_STRATEGY_COMPATIBILITY",
            "errorType": "TypeError",
            "error": (
                "SFTConfig.__init__() got an unexpected keyword argument "
                "'evaluation_strategy'"
            ),
            "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
            "envelopeRevision": "b929bae4230ffe39ee63b34b8e9f9974cffc66ca",
            "executionBridgeRevision": "2783b3518abcec9f38d3f6504c06e305a4723801",
            "attemptClaimSha256": (
                "bb1fd12fb73289864503d5f8d65aacb4b34d0db0d0ba2fcce73a975c71364293"
            ),
            "prefetchReceiptSha256": (
                "b290716a5bc9427a20bf954893770a5401d9b70c0530c51cf0e958aadc9e3e64"
            ),
            "claimCreated": True,
            "jobDirectoryCreated": True,
            "prefetchReceiptCreated": True,
            "trainingStarted": False,
            "receiptIntentProduced": True,
            "receiptUploaded": True,
            "receiptRevision": "ac219fe87da9acf57141ff24ffbd330216584f7c",
            "receiptFileSha256": (
                "384e64b0ebd43fcfd2f52a3b1139cf1bca04f23c43ccfd9738af3a1fdfe46d02"
            ),
            "receiptBodySha256": (
                "ec5f8b173f3e8f13c252bf9c7eb52625210b3bf936c7dec88fc640e032275876"
            ),
            "receiptKeyId": "167c14fbddbe97cc",
            "receiptVerdict": "BLOCKED",
            "receiptReason": (
                "TypeError: SFTConfig.__init__() got an unexpected keyword argument "
                "'evaluation_strategy'"
            ),
            "candidateUploaded": False,
            "adapterUploaded": False,
            "modelCardUploaded": False,
            "datasetUploaded": False,
            "deployed": False,
            "promoted": False,
        },
    },
    ATTEMPT_14_REVIEWED_JOB_ID: {
        "statuses": (
            "META_TENSOR_MATERIALIZATION_BLOCKED",
            "POST_CLAIM",
            "PRE_TRAINING",
            "SIGNED_BLOCKED_RECEIPT",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "207f0c58525f042d31a748404d0acb678f5fd83722d2a3eacf8399e4e34c9f82"
        ),
        "payload_sha256": (
            "162354602784e8a1cbcecbbfc8a5d7cc9af6be2dd58c66fae442d4f5a292f1da"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_15_REPLACEMENT,
        "execution_evidence_path": (
            "queue/evidence/job-2026-nemo-v3-governed-attempt-14.json"
        ),
        "execution_evidence_sha256": (
            "430aa2494b6b1bbcae45f99409075cfbe525ab628582806e3be1c8ae18204bc4"
        ),
        "execution_evidence": {
            "workflowRunId": "30634484969",
            "workflowRunAttempt": 1,
            "workflowJobId": "91168515330",
            "failurePhase": "POST_CLAIM_TRAINER_META_TENSOR",
            "errorType": "NotImplementedError",
            "error": "Cannot copy out of meta tensor; no data!",
            "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
            "envelopeRevision": "fd97065eb2aa9fc3299706c531597538a65eb735",
            "executionBridgeRevision": "e150711a6ba6a0c29109a00da7fc82af2967f588",
            "attemptClaimSha256": (
                "fc93d880beb0ff183e7da4f7a9a42f0fd075addfc07056e9d260539d9f1dfd92"
            ),
            "runtimeClaimSha256": (
                "b0f7c68e357692a68f8d436c417e7c76a852ab0635c3e4a49c3713a53dc16243"
            ),
            "prefetchReceiptSha256": (
                "bd315a4a97356451781ceee3e390b847c559381c69eaf50d5efed8c191d2e28c"
            ),
            "claimCreated": True,
            "jobDirectoryCreated": True,
            "prefetchReceiptCreated": True,
            "modelRepositoryCodeImported": True,
            "holdoutsAccessed": True,
            "trainingStarted": False,
            "receiptIntentProduced": True,
            "receiptUploaded": True,
            "receiptRevision": "8c504d466d6b1b3fb0a755768341a34e58b82c11",
            "receiptFileSha256": (
                "f45c7b319f5f762d03b100149732a4287dfda0d7c91046f21d580fc6f7684ecd"
            ),
            "receiptBodySha256": (
                "cb4dc5cce83797f5d39f86f1c7078230344dc176c854dd3f07988177cafd2500"
            ),
            "receiptKeyId": "167c14fbddbe97cc",
            "receiptVerdict": "BLOCKED",
            "receiptReason": (
                "NotImplementedError: Cannot copy out of meta tensor; no data!"
            ),
            "candidateUploaded": False,
            "adapterUploaded": False,
            "modelCardUploaded": False,
            "datasetUploaded": False,
            "deployed": False,
            "promoted": False,
        },
    },
    ATTEMPT_15_REVIEWED_JOB_ID: {
        "statuses": (
            "RUNTIME_JOB_BINDING_REJECTED",
            "PRE_CLAIM",
            "NEVER_DISPATCH",
        ),
        "queue_file_sha256": (
            "93d5effe94740af9135c3ffa379c85df1aa88e6ad5717bc6421266d21bb9dbe7"
        ),
        "payload_sha256": (
            "9c55b95627b93e522eaebec5cb9e837b46d8e368065470aa45f55f488aeff873"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_16_REPLACEMENT,
        "execution_evidence_path": (
            "queue/evidence/job-2026-nemo-v3-governed-attempt-15.json"
        ),
        "execution_evidence_sha256": (
            "a5af132a89fdf26f2857c06891711e56843c6708d5db14d1f6bf20fc3cf81779"
        ),
        "execution_evidence": {
            "workflowRunId": "30641766033",
            "workflowRunAttempt": 1,
            "workflowJobId": "91193214499",
            "failurePhase": "PRE_CLAIM_AUTHENTICATED_PREFETCH_RUNTIME_BINDING",
            "errorType": "frontier_contract.ContractError",
            "error": (
                "coordinated authorization requires an exact reviewed job binding"
            ),
            "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
            "envelopeRevision": "7f42bad2cb7c762f8eb771922a0ba6e94c96e908",
            "executionBridgeRevision": ("60b9894efe9e0e782999aaa4ee5b0d668e7a9b63"),
            "claimCreated": False,
            "jobDirectoryCreated": False,
            "prefetchReceiptCreated": False,
            "trainingStarted": False,
            "receiptIntentProduced": False,
            "receiptUploaded": False,
            "candidateUploaded": False,
            "adapterUploaded": False,
            "modelCardUploaded": False,
            "datasetUploaded": False,
            "deployed": False,
            "promoted": False,
        },
    },
    ATTEMPT_16_REVIEWED_JOB_ID: {
        "statuses": (
            "STALE_SOURCE",
            "PRE_DISPATCH_VALIDATOR_REJECTED",
            "PRE_EVENT",
            "NEVER_DISPATCH",
            "NEVER_RESEND",
            "NEVER_RESIGN",
        ),
        "queue_file_sha256": (
            "5f657aebb650c6a9c19b4b52e710236220fe7ab89e6a50488ee270017a78f756"
        ),
        "payload_sha256": (
            "0b80bc0e42edd75de9e63f9f74f53df1d10c328d89b84c8481834a27fa4111f8"
        ),
        "engine_key_id": COORDINATED_ENGINE_KEY_ID,
        "source_revision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        "replacement": _ATTEMPT_17_REPLACEMENT,
        "pre_event_evidence_path": (
            "queue/evidence/job-2026-nemo-v3-governed-attempt-16.json"
        ),
        "pre_event_evidence_sha256": (
            "efff8d60590b317a873772e72e401165300331daf5431136ec18a1ddcab85389"
        ),
        "pre_event_evidence": {
            "failurePhase": "PRE_DISPATCH_VALIDATOR_REJECTED",
            "evidenceUrl": "https://github.com/szl-holdings/a11oy/pull/1217",
            "errorType": "DispatchValidationError",
            "error": (
                "predecessor quarantine replacement contains unsupported fields: "
                "['settledA11oyRelockRunUrl', 'successorGeneration', "
                "'workflowVersion']"
            ),
            "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
            "supersedingSourceRevision": ("a7e70c2b3dd198b9368d31382b25fddbd8caad89"),
            "envelopeRevision": "0939008a73fa8b1912c842a304c5d0204a5b9d57",
            "executionBridgeRevision": ("b99f37260bcabf7f5c98cddbc5988a3ba87b766e"),
            "eventCreated": False,
            "workflowRunCreated": False,
            "claimCreated": False,
            "jobDirectoryCreated": False,
            "prefetchReceiptCreated": False,
            "trainingStarted": False,
            "modelRepositoryCodeImported": False,
            "holdoutsAccessed": False,
            "receiptIntentProduced": False,
            "receiptUploaded": False,
            "candidateUploaded": False,
            "adapterUploaded": False,
            "modelCardUploaded": False,
            "datasetUploaded": False,
            "deployed": False,
            "promoted": False,
        },
    },
}
QUARANTINED_NEMO_JOB_IDS = frozenset(QUARANTINE_POLICIES)
_OWNER_WORKFLOW_IDENTITY = (
    "szl-holdings/a11oy/"
    ".github/workflows/nemo-v3-isolated-owner-dispatch.yml"
    "@refs/heads/main"
)
_OWNER_WORKFLOW_VERSION = "nemo-v3-owner-dispatch.v2"
_FINAL_OWNER_WORKFLOW_VERSION = "nemo-v3-owner-dispatch.v4"
_OWNER_TRAINING_IMAGE = (
    "unsloth/unsloth@"
    "sha256:9cc97606fc386b4b13455285eb7bd2668f51530988a9c2578707fe6cdfc46123"
)
_OWNER_RECEIPTS_REPO = "SZLHOLDINGS/szl-training-receipts"
_COORDINATED_JOB_BINDINGS = {
    NEXT_REVIEWED_JOB_ID: {
        "sourceRevision": SETTLED_A11OY_SOURCE_REVISION,
        "workflowBlob": SETTLED_OWNER_WORKFLOW_BLOB,
        "workflowVersion": _OWNER_WORKFLOW_VERSION,
        "relockRunUrl": SETTLED_A11OY_RELOCK_RUN_URL,
        "correctedBridgeRevision": CORRECTED_BRIDGE_REVISION,
        "successorGeneration": 4,
    },
    ATTEMPT_5_REVIEWED_JOB_ID: {
        "sourceRevision": FINAL_A11OY_SOURCE_REVISION,
        "workflowBlob": FINAL_OWNER_WORKFLOW_BLOB,
        "workflowVersion": _FINAL_OWNER_WORKFLOW_VERSION,
        "relockRunUrl": FINAL_A11OY_RELOCK_RUN_URL,
        "correctedBridgeRevision": FINAL_CORRECTED_BRIDGE_REVISION,
        "successorGeneration": 5,
    },
    ATTEMPT_6_REVIEWED_JOB_ID: {
        "sourceRevision": EXECUTION_A11OY_SOURCE_REVISION,
        "workflowBlob": EXECUTION_OWNER_WORKFLOW_BLOB,
        "workflowVersion": _FINAL_OWNER_WORKFLOW_VERSION,
        "relockRunUrl": EXECUTION_A11OY_RELOCK_RUN_URL,
        "correctedBridgeRevision": "69a097d2eb0619506d673464353f1aea7174cf05",
        "successorGeneration": 6,
    },
    FUTURE_REVIEWED_JOB_ID: {
        "sourceRevision": SUCCESSOR_A11OY_SOURCE_REVISION,
        "workflowBlob": SUCCESSOR_OWNER_WORKFLOW_BLOB,
        "workflowVersion": _FINAL_OWNER_WORKFLOW_VERSION,
        "relockRunUrl": SUCCESSOR_A11OY_RELOCK_RUN_URL,
        "correctedBridgeRevision": SUCCESSOR_CORRECTED_BRIDGE_REVISION,
        "successorGeneration": 7,
    },
    NEXT_RUNTIME_REVIEWED_JOB_ID: {
        "sourceRevision": SUCCESSOR_A11OY_SOURCE_REVISION,
        "workflowBlob": SUCCESSOR_OWNER_WORKFLOW_BLOB,
        "workflowVersion": _FINAL_OWNER_WORKFLOW_VERSION,
        "relockRunUrl": SUCCESSOR_A11OY_RELOCK_RUN_URL,
        "correctedBridgeRevision": NEXT_RUNTIME_CORRECTED_BRIDGE_REVISION,
        "successorGeneration": 8,
    },
    ATTEMPT_9_REVIEWED_JOB_ID: {
        "sourceRevision": RECOVERY_A11OY_SOURCE_REVISION,
        "workflowBlob": RECOVERY_OWNER_WORKFLOW_BLOB,
        "workflowVersion": _FINAL_OWNER_WORKFLOW_VERSION,
        "relockRunUrl": RECOVERY_A11OY_RELOCK_RUN_URL,
        "correctedBridgeRevision": ATTEMPT_9_CORRECTED_BRIDGE_REVISION,
        "successorGeneration": 9,
    },
    ATTEMPT_10_REVIEWED_JOB_ID: {
        "sourceRevision": RECOVERY_A11OY_SOURCE_REVISION,
        "workflowBlob": RECOVERY_OWNER_WORKFLOW_BLOB,
        "workflowVersion": _FINAL_OWNER_WORKFLOW_VERSION,
        "relockRunUrl": RECOVERY_A11OY_RELOCK_RUN_URL,
        "correctedBridgeRevision": ATTEMPT_10_CORRECTED_BRIDGE_REVISION,
        "successorGeneration": 10,
    },
    ATTEMPT_11_REVIEWED_JOB_ID: {
        "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        "workflowBlob": EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
        "workflowVersion": _FINAL_OWNER_WORKFLOW_VERSION,
        "relockRunUrl": EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
        "correctedBridgeRevision": ATTEMPT_11_CORRECTED_BRIDGE_REVISION,
        "successorGeneration": 11,
    },
    ATTEMPT_12_REVIEWED_JOB_ID: {
        "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        "workflowBlob": EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
        "workflowVersion": _FINAL_OWNER_WORKFLOW_VERSION,
        "relockRunUrl": EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
        "correctedBridgeRevision": ATTEMPT_12_CORRECTED_BRIDGE_REVISION,
        "successorGeneration": 12,
    },
}
_ADMITTED_A11OY_CONTEXTS = {
    (
        binding["sourceRevision"],
        binding["workflowBlob"],
        binding["workflowVersion"],
        binding["relockRunUrl"],
    )
    for binding in _COORDINATED_JOB_BINDINGS.values()
}
_ADMITTED_A11OY_CONTEXTS.add(
    (
        ATTEMPT_17_A11OY_SOURCE_REVISION,
        ATTEMPT_17_OWNER_WORKFLOW_BLOB,
        _FINAL_OWNER_WORKFLOW_VERSION,
        ATTEMPT_17_A11OY_RELOCK_RUN_URL,
    )
)
_RUNTIME_BOUND_EXECUTION_REVISIONS = {
    ATTEMPT_17_REVIEWED_JOB_ID: ATTEMPT_17_CORRECTED_BRIDGE_REVISION,
}
_REPLACEMENT_FIELDS = frozenset(
    {
        "sourceRevision",
        "workflowBlob",
        "engineKeyId",
        "enginePublicKeySpkiSha256",
        "reviewedJobId",
    }
)
_RUNTIME_REPLACEMENT_FIELDS = _REPLACEMENT_FIELDS | {
    "workflowVersion",
    "settledA11oyRelockRunUrl",
    "successorGeneration",
}
_ALLOWED_TOP = {
    "jobId",
    "kind",
    "createdAt",
    "expiresAt",
    "source",
    "base",
    "dataset",
    "recipe",
    "gates",
    "outputs",
    "evaluation",
    "notes",
    "lineage",
    "ownerDispatch",
    "authorization",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def record_ids_sha256(ids: list[str]) -> str:
    return sha256_bytes(("\n".join(ids) + "\n").encode("utf-8"))


def _object(
    parent: dict[str, Any], key: str, required: set[str], allowed: set[str]
) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ContractError(f"{key} must be an object")
    missing = required - value.keys()
    extra = value.keys() - allowed
    if missing:
        raise ContractError(f"{key} missing required fields: {sorted(missing)}")
    if extra:
        raise ContractError(f"{key} contains unsupported fields: {sorted(extra)}")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ContractError(f"{field} must be ISO-8601")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{field} is not valid ISO-8601") from exc


def _repo(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _REPO.fullmatch(value):
        raise ContractError(f"{field} must be owner/name")
    return value


def _revision(value: Any, field: str, *, exact40: bool = False) -> str:
    if (
        not isinstance(value, str)
        or not _SHA.fullmatch(value)
        or (exact40 and len(value) != 40)
    ):
        suffix = "exactly 40" if exact40 else "40-64"
        raise ContractError(f"{field} must be {suffix} lowercase hex")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ContractError(f"{field} must be lowercase sha256 hex")
    return value


def _path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_PATH.fullmatch(value):
        raise ContractError(f"{field} must be a safe relative path")
    candidate = pathlib.PurePosixPath(value)
    if value.startswith("/") or ".." in candidate.parts:
        raise ContractError(f"{field} must not be absolute or traverse parents")
    return value


def _pinned_file(
    value: Any, field: str, *, require_records: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{field} must be an object")
    required = {"path", "sha256", "bytes"}
    if require_records:
        required |= {"name", "recordIds", "recordIdsSha256"}
    missing = required - value.keys()
    if missing:
        raise ContractError(f"{field} missing fields: {sorted(missing)}")
    _path(value.get("path"), f"{field}.path")
    _sha256(value.get("sha256"), f"{field}.sha256")
    if not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
        raise ContractError(f"{field}.bytes must be a positive integer")
    if require_records:
        ids = value.get("recordIds")
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise ContractError(f"{field}.recordIds must be a non-empty unique array")
        if not all(isinstance(item, str) and item for item in ids):
            raise ContractError(f"{field}.recordIds contains an invalid identifier")
        _sha256(value.get("recordIdsSha256"), f"{field}.recordIdsSha256")
        if record_ids_sha256(ids) != value["recordIdsSha256"]:
            raise ContractError(f"{field}.recordIdsSha256 does not bind recordIds")
    return value


def expected_engine_key_id(spec: dict[str, Any]) -> str:
    authorization = spec.get("authorization")
    if authorization is None:
        return LEGACY_ENGINE_KEY_ID
    return str(authorization["engineKeyId"])


def quarantine_policy(spec_or_job_id: dict[str, Any] | str) -> dict[str, Any] | None:
    """Return the exact immutable quarantine policy for a reviewed job."""

    job_id = (
        spec_or_job_id.get("jobId")
        if isinstance(spec_or_job_id, dict)
        else spec_or_job_id
    )
    return QUARANTINE_POLICIES.get(str(job_id))


def _attempt_generation(job_id: Any, field: str) -> int:
    if not isinstance(job_id, str):
        raise ContractError(f"{field} is not an exact governed attempt ID")
    match = _ATTEMPT_JOB.fullmatch(job_id)
    if match is None:
        raise ContractError(f"{field} is not an exact governed attempt ID")
    return int(match.group("generation"))


def _derived_runtime_binding(spec: dict[str, Any]) -> dict[str, Any]:
    """Derive one runtime successor from its protected predecessor quarantine."""

    job_id = spec.get("jobId")
    generation = _attempt_generation(job_id, "jobId")
    lineage = spec.get("lineage")
    if not isinstance(lineage, dict):
        raise ContractError(
            "runtime-bound coordinated authorization requires exact lineage"
        )
    predecessor_id = lineage.get("predecessorJobId")
    predecessor_generation = _attempt_generation(
        predecessor_id, "lineage.predecessorJobId"
    )
    if predecessor_generation + 1 != generation:
        raise ContractError(
            "runtime-bound lineage cannot skip a generation; exact reviewed job "
            "binding refused"
        )
    if lineage.get("successorGeneration") != generation:
        raise ContractError(
            "runtime-bound coordinated authorization has mismatched generation"
        )

    predecessor_policy = QUARANTINE_POLICIES.get(predecessor_id)
    if not isinstance(predecessor_policy, dict):
        raise ContractError(
            "runtime-bound coordinated authorization requires a protected "
            "predecessor quarantine"
        )
    replacement = predecessor_policy.get("replacement")
    replacement_fields = (
        frozenset(replacement) if isinstance(replacement, dict) else frozenset()
    )
    if not isinstance(replacement, dict) or replacement_fields not in {
        _REPLACEMENT_FIELDS,
        _RUNTIME_REPLACEMENT_FIELDS,
    }:
        raise ContractError(
            "protected predecessor replacement has an invalid exact shape"
        )
    statuses = predecessor_policy.get("statuses")
    if not isinstance(statuses, (tuple, list)) or "NEVER_DISPATCH" not in statuses:
        raise ContractError(
            "runtime-bound predecessor is not protected as never dispatch"
        )
    if replacement.get("reviewedJobId") != job_id:
        raise ContractError(
            "protected predecessor replacement does not authorize this reviewed job"
        )
    if (
        predecessor_policy.get("engine_key_id") != COORDINATED_ENGINE_KEY_ID
        or replacement.get("engineKeyId") != COORDINATED_ENGINE_KEY_ID
        or replacement.get("enginePublicKeySpkiSha256")
        != COORDINATED_ENGINE_SPKI_SHA256
    ):
        raise ContractError(
            "protected predecessor replacement does not bind the final trust root"
        )

    source_revision = replacement.get("sourceRevision")
    workflow_blob = replacement.get("workflowBlob")
    _revision(source_revision, "predecessor replacement sourceRevision", exact40=True)
    _revision(workflow_blob, "predecessor replacement workflowBlob", exact40=True)
    matching_contexts = {
        (workflow_version, relock_run_url)
        for (
            context_source,
            context_workflow,
            workflow_version,
            relock_run_url,
        ) in _ADMITTED_A11OY_CONTEXTS
        if context_source == source_revision and context_workflow == workflow_blob
    }
    if len(matching_contexts) != 1:
        raise ContractError(
            "protected predecessor replacement does not bind one admitted A11oy context"
        )
    workflow_version, relock_run_url = matching_contexts.pop()
    if replacement_fields == _RUNTIME_REPLACEMENT_FIELDS and (
        replacement["workflowVersion"] != workflow_version
        or replacement["settledA11oyRelockRunUrl"] != relock_run_url
        or replacement["successorGeneration"] != generation
    ):
        raise ContractError(
            "protected predecessor replacement has mismatched runtime context"
        )
    if generation >= 16 and replacement_fields != _RUNTIME_REPLACEMENT_FIELDS:
        raise ContractError(
            "new runtime successor replacement lacks exact context and generation"
        )
    corrected_bridge_revision = _RUNTIME_BOUND_EXECUTION_REVISIONS.get(job_id)
    if generation >= 17 and corrected_bridge_revision is None:
        raise ContractError(
            "runtime-bound successor lacks an exact protected execution revision"
        )
    return {
        "sourceRevision": source_revision,
        "workflowBlob": workflow_blob,
        "workflowVersion": workflow_version,
        "relockRunUrl": relock_run_url,
        "correctedBridgeRevision": corrected_bridge_revision,
        "runtimeBound": True,
        "successorGeneration": generation,
        "predecessorJobId": predecessor_id,
        "predecessorPolicy": predecessor_policy,
    }


def _coordinated_job_binding(spec: dict[str, Any]) -> dict[str, Any]:
    static_binding = _COORDINATED_JOB_BINDINGS.get(spec.get("jobId"))
    if static_binding is not None:
        return static_binding
    return _derived_runtime_binding(spec)


def _runtime_predecessor_evidence(
    binding: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    policy = binding.get("predecessorPolicy")
    if not isinstance(policy, dict):
        raise ContractError("runtime-bound successor lacks a predecessor policy")
    evidence = policy.get("execution_evidence")
    if isinstance(evidence, dict):
        workflow_run_id = evidence.get("workflowRunId")
        if not isinstance(workflow_run_id, str) or not workflow_run_id.isdigit():
            raise ContractError(
                "protected predecessor execution evidence has an invalid workflow run"
            )
        normalized = {
            **evidence,
            "eventCreated": True,
            "workflowRunCreated": True,
        }
        return (
            normalized,
            "https://github.com/szl-holdings/a11oy/actions/runs/" + workflow_run_id,
        )

    evidence = policy.get("pre_event_evidence")
    if not isinstance(evidence, dict):
        raise ContractError(
            "protected predecessor quarantine lacks exact execution or pre-event evidence"
        )
    evidence_url = evidence.get("evidenceUrl")
    zero_effect_fields = (
        "eventCreated",
        "workflowRunCreated",
        "claimCreated",
        "jobDirectoryCreated",
        "prefetchReceiptCreated",
        "trainingStarted",
        "modelRepositoryCodeImported",
        "holdoutsAccessed",
        "receiptIntentProduced",
        "receiptUploaded",
        "candidateUploaded",
        "adapterUploaded",
        "modelCardUploaded",
        "datasetUploaded",
        "deployed",
        "promoted",
    )
    if (
        not isinstance(evidence_url, str)
        or not evidence_url.startswith("https://github.com/szl-holdings/")
        or any(evidence.get(field) is not False for field in zero_effect_fields)
    ):
        raise ContractError(
            "protected predecessor pre-event evidence is not an exact zero-event boundary"
        )
    return evidence, evidence_url


def _require_exact_runtime_predecessor_lineage(
    lineage: dict[str, Any], binding: dict[str, Any]
) -> None:
    if not binding.get("runtimeBound"):
        return
    policy = binding.get("predecessorPolicy")
    if not isinstance(policy, dict):
        return
    evidence, transport_evidence_url = _runtime_predecessor_evidence(binding)
    predecessor_source_revision = policy.get("source_revision")
    _revision(
        predecessor_source_revision,
        "protected predecessor source revision",
        exact40=True,
    )
    if evidence.get("sourceRevision") != predecessor_source_revision:
        raise ContractError(
            "protected predecessor execution evidence has mismatched source"
        )
    expected = {
        "predecessorJobId": binding["predecessorJobId"],
        "predecessorEnvelopeSha256": policy.get("queue_file_sha256"),
        "predecessorPayloadSha256": policy.get("payload_sha256"),
        "predecessorEnvelopeRevision": evidence.get("envelopeRevision"),
        "predecessorExecutionBridgeRevision": evidence.get("executionBridgeRevision"),
        "transportEvidenceUrl": transport_evidence_url,
        "failurePhase": evidence.get("failurePhase"),
        "successorGeneration": binding["successorGeneration"],
        "automaticRetry": False,
        "eventCreated": evidence.get("eventCreated"),
        "workflowRunCreated": evidence.get("workflowRunCreated"),
        "candidateProduced": False,
        "scienceInputsReused": True,
    }
    evidence_boundaries = {
        "claimCreated": "claimCreated",
        "trainingStarted": "trainingStarted",
        "modelRepositoryCodeImported": "modelRepositoryCodeImported",
        "holdoutsAccessed": "holdoutsAccessed",
        "receiptIntentProduced": "receiptIntentProduced",
        "terminalLedgerWritten": "receiptUploaded",
    }
    for lineage_field, evidence_field in evidence_boundaries.items():
        if evidence_field in evidence:
            expected[lineage_field] = evidence[evidence_field]
    for field, value in expected.items():
        if lineage.get(field) != value:
            raise ContractError(
                "runtime-bound successor lineage does not match protected "
                "predecessor evidence"
            )


def require_nemo_v3_dispatchable(
    spec: dict[str, Any],
    *,
    expected_execution_bridge_revision: str | None = None,
) -> None:
    """Reject immutable historical envelopes before any execution side effect."""

    policy = quarantine_policy(spec)
    if policy is not None:
        raise ContractError(
            "Nemo v3 job is quarantined: " + " + ".join(policy["statuses"])
        )
    authorization = spec.get("authorization")
    if (
        isinstance(authorization, dict)
        and authorization.get("rotationMode")
        == "COORDINATED_FINAL_TRUST_ROOT_NEW_GENERATION"
        and _coordinated_job_binding(spec).get("runtimeBound")
    ):
        expected = _revision(
            expected_execution_bridge_revision,
            "expected execution Bridge revision",
            exact40=True,
        )
        if authorization.get("correctedBridgeRevision") != expected:
            raise ContractError(
                "runtime-bound successor does not match the exact execution "
                "Bridge revision"
            )


def validate_nemo_v3_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ContractError("Nemo v3 spec must be an object")
    required = _ALLOWED_TOP - {
        "notes",
        "lineage",
        "ownerDispatch",
        "authorization",
    }
    missing = required - spec.keys()
    extra = spec.keys() - _ALLOWED_TOP
    if missing:
        raise ContractError(f"spec missing required fields: {sorted(missing)}")
    if extra:
        raise ContractError(f"spec contains unsupported fields: {sorted(extra)}")
    if spec.get("kind") != NEMO_V3_KIND:
        raise ContractError(f"kind must be {NEMO_V3_KIND!r}")
    if not isinstance(spec.get("jobId"), str) or not _JOB.fullmatch(spec["jobId"]):
        raise ContractError("jobId does not match the Nemo v3 idempotency pattern")
    created = _timestamp(spec.get("createdAt"), "createdAt")
    expires = _timestamp(spec.get("expiresAt"), "expiresAt")
    if expires <= created:
        raise ContractError("expiresAt must be later than createdAt")

    coordinated_authorization = False
    coordinated_binding: dict[str, Any] | None = None
    if "authorization" in spec:
        legacy_authorization_fields = {
            "engineKeyId",
            "previousEngineKeyId",
            "recoveryIssueUrl",
            "rotationMode",
            "oldKeyStatus",
            "decisionAt",
        }
        coordinated_authorization_fields = legacy_authorization_fields | {
            "enginePublicKeySpkiSha256",
            "provisionalEngineKeyId",
            "provisionalKeyStatus",
            "coordinationMode",
            "settledA11oyRelockRunUrl",
            "cryptographicContinuityClaimed",
            "correctedBridgeRevision",
        }
        raw_authorization = spec.get("authorization")
        if not isinstance(raw_authorization, dict):
            raise ContractError("authorization must be an object")
        coordinated_authorization = (
            raw_authorization.get("rotationMode")
            == "COORDINATED_FINAL_TRUST_ROOT_NEW_GENERATION"
        )
        if coordinated_authorization:
            coordinated_binding = _coordinated_job_binding(spec)
        authorization_fields = (
            coordinated_authorization_fields
            if coordinated_authorization
            else legacy_authorization_fields
        )
        authorization = _object(
            spec,
            "authorization",
            authorization_fields,
            authorization_fields,
        )
        for field in ("engineKeyId", "previousEngineKeyId"):
            if not isinstance(
                authorization[field], str
            ) or not _ENGINE_KEY_ID.fullmatch(authorization[field]):
                raise ContractError(f"authorization.{field} must be lowercase 16-hex")
        if authorization["engineKeyId"] == authorization["previousEngineKeyId"]:
            raise ContractError("authorization must enroll a distinct engine key")
        if authorization["previousEngineKeyId"] != LEGACY_ENGINE_KEY_ID:
            raise ContractError(
                "authorization does not preserve the historical trust root"
            )
        if authorization["rotationMode"] not in {
            "LOST_PRIVATE_KEY_NEW_GENERATION",
            "COORDINATED_FINAL_TRUST_ROOT_NEW_GENERATION",
        }:
            raise ContractError("authorization.rotationMode is not admitted")
        if authorization["oldKeyStatus"] != "VERIFY_ONLY":
            raise ContractError("authorization.oldKeyStatus must be VERIFY_ONLY")
        if authorization["recoveryIssueUrl"] != (
            "https://github.com/szl-holdings/szl-gpu-bridge/issues/25"
        ):
            raise ContractError(
                "authorization.recoveryIssueUrl is not the recorded incident"
            )
        _timestamp(authorization["decisionAt"], "authorization.decisionAt")
        if coordinated_authorization:
            if authorization["engineKeyId"] != COORDINATED_ENGINE_KEY_ID:
                raise ContractError(
                    "coordinated authorization does not use the final engine key"
                )
            if (
                authorization["enginePublicKeySpkiSha256"]
                != COORDINATED_ENGINE_SPKI_SHA256
            ):
                raise ContractError(
                    "coordinated authorization does not bind the final SPKI hash"
                )
            if authorization["provisionalEngineKeyId"] != PROVISIONAL_ENGINE_KEY_ID:
                raise ContractError(
                    "coordinated authorization does not identify the provisional key"
                )
            if authorization["provisionalKeyStatus"] != "VERIFY_ONLY":
                raise ContractError(
                    "coordinated authorization must retire the provisional key"
                )
            if authorization["coordinationMode"] != "FINAL_ACTIVE_TRUST_ROOT":
                raise ContractError(
                    "coordinated authorization mode is not the final trust root"
                )
            if (
                authorization["settledA11oyRelockRunUrl"]
                != coordinated_binding["relockRunUrl"]
            ):
                raise ContractError(
                    "coordinated authorization does not bind the terminal A11oy relock"
                )
            if authorization["cryptographicContinuityClaimed"] is not False:
                raise ContractError(
                    "administrative recovery must not claim cryptographic continuity"
                )
            _revision(
                authorization["correctedBridgeRevision"],
                "authorization.correctedBridgeRevision",
                exact40=True,
            )
            expected_bridge_revision = coordinated_binding.get(
                "correctedBridgeRevision"
            )
            if (
                expected_bridge_revision is not None
                and authorization["correctedBridgeRevision"] != expected_bridge_revision
            ):
                raise ContractError(
                    "coordinated authorization does not bind corrected bridge main"
                )

    if "ownerDispatch" in spec:
        owner_fields = {
            "workflowIdentity",
            "workflowBlob",
            "workflowVersion",
            "trainingImage",
            "candidateUpload",
            "modelCardUpload",
            "datasetUpload",
            "receiptsRepoId",
        }
        owner_dispatch = _object(
            spec,
            "ownerDispatch",
            owner_fields,
            owner_fields,
        )
        expected_workflow_version = (
            coordinated_binding["workflowVersion"]
            if coordinated_binding is not None
            else _OWNER_WORKFLOW_VERSION
        )
        if (
            owner_dispatch["workflowIdentity"] != _OWNER_WORKFLOW_IDENTITY
            or owner_dispatch["workflowVersion"] != expected_workflow_version
        ):
            raise ContractError("ownerDispatch workflow identity is not admitted")
        _revision(
            owner_dispatch["workflowBlob"],
            "ownerDispatch.workflowBlob",
            exact40=True,
        )
        if owner_dispatch["trainingImage"] != _OWNER_TRAINING_IMAGE:
            raise ContractError("ownerDispatch training image is not admitted")
        for field in ("candidateUpload", "modelCardUpload", "datasetUpload"):
            if owner_dispatch[field] is not False:
                raise ContractError(f"ownerDispatch.{field} must remain false")
        if owner_dispatch["receiptsRepoId"] != _OWNER_RECEIPTS_REPO:
            raise ContractError("ownerDispatch receipts repository is not admitted")

    if "lineage" in spec:
        legacy_lineage_fields = {
            "predecessorJobId",
            "predecessorClaimSha256",
            "predecessorEnvelopeSha256",
            "predecessorBridgeRevision",
            "predecessorImageId",
            "predecessorClaimedAt",
            "incidentUrl",
            "failurePhase",
            "successorGeneration",
            "automaticRetry",
            "trainingStarted",
            "modelRepositoryCodeImported",
            "holdoutsAccessed",
            "candidateProduced",
            "receiptIntentProduced",
            "terminalLedgerWritten",
            "scienceInputsReused",
        }
        transport_lineage_fields = {
            "predecessorJobId",
            "predecessorEnvelopeSha256",
            "predecessorPayloadSha256",
            "predecessorEnvelopeRevision",
            "predecessorExecutionBridgeRevision",
            "transportEvidenceUrl",
            "failurePhase",
            "successorGeneration",
            "automaticRetry",
            "eventCreated",
            "workflowRunCreated",
            "claimCreated",
            "trainingStarted",
            "modelRepositoryCodeImported",
            "holdoutsAccessed",
            "candidateProduced",
            "receiptIntentProduced",
            "terminalLedgerWritten",
            "scienceInputsReused",
        }
        raw_lineage = spec.get("lineage")
        if not isinstance(raw_lineage, dict):
            raise ContractError("lineage must be an object")
        lineage_keys = set(raw_lineage)
        transport_lineage = lineage_keys == transport_lineage_fields
        if lineage_keys == legacy_lineage_fields:
            lineage = raw_lineage
        elif transport_lineage:
            lineage = raw_lineage
        else:
            raise ContractError("lineage fields must match an admitted exact shape")
        predecessor = lineage["predecessorJobId"]
        if not isinstance(predecessor, str) or not _JOB.fullmatch(predecessor):
            raise ContractError("lineage.predecessorJobId is invalid")
        if predecessor == spec["jobId"]:
            raise ContractError("successor jobId must differ from its predecessor")
        if coordinated_binding is not None:
            _require_exact_runtime_predecessor_lineage(lineage, coordinated_binding)
        _sha256(
            lineage["predecessorEnvelopeSha256"],
            "lineage.predecessorEnvelopeSha256",
        )
        expected_claim_created = False
        expected_holdouts_accessed = False
        expected_receipt_intent_produced = False
        expected_model_repository_code_imported = False
        expected_terminal_ledger_written = False
        if transport_lineage:
            _sha256(
                lineage["predecessorPayloadSha256"],
                "lineage.predecessorPayloadSha256",
            )
            for field in (
                "predecessorEnvelopeRevision",
                "predecessorExecutionBridgeRevision",
            ):
                _revision(lineage[field], f"lineage.{field}", exact40=True)
            if predecessor == NEXT_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/szl-gpu-bridge/issues/32"
                )
                expected_failure_phase = "PRE_EVENT_TRANSPORT_VALIDATION"
                expected_event_created = False
            elif predecessor == ATTEMPT_5_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30591897165"
                )
                expected_failure_phase = "PRE_ADMISSION_HOST_EXECUTION_POLICY"
                expected_event_created = True
            elif predecessor == ATTEMPT_6_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/szl-gpu-bridge/issues/41"
                )
                expected_failure_phase = "PRE_DISPATCH_VALIDATOR_REJECTION"
                expected_event_created = False
            elif predecessor == FUTURE_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30605081533"
                )
                expected_failure_phase = "PRE_CLAIM_RUNTIME_CONTRACT_VALIDATION"
                expected_event_created = True
            elif predecessor == NEXT_RUNTIME_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30606664591"
                )
                expected_failure_phase = "PRE_CLAIM_DIRTY_EXECUTION_CHECKOUT"
                expected_event_created = True
            elif predecessor == ATTEMPT_9_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30609977388"
                )
                expected_failure_phase = (
                    "POST_CLAIM_CACHE_LICENSE_AND_FINALIZER_BINDING"
                )
                expected_event_created = True
                expected_claim_created = True
                expected_holdouts_accessed = True
                expected_receipt_intent_produced = True
            elif predecessor == ATTEMPT_10_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30612658302"
                )
                expected_failure_phase = (
                    "PRE_CLAIM_IMMUTABLE_RUNTIME_JOB_BINDING_VALIDATION"
                )
                expected_event_created = True
            elif predecessor == ATTEMPT_11_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30620232291"
                )
                expected_failure_phase = "POST_CLAIM_TOKENIZER_LOAD"
                expected_event_created = True
                expected_claim_created = True
                expected_holdouts_accessed = True
                expected_receipt_intent_produced = True
                expected_model_repository_code_imported = True
                expected_terminal_ledger_written = True
            elif predecessor == ATTEMPT_12_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30626533443"
                )
                expected_failure_phase = (
                    "PRE_CLAIM_AUTHENTICATED_PREFETCH_RUNTIME_BINDING"
                )
                expected_event_created = True
            elif predecessor == ATTEMPT_13_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30629929196"
                )
                expected_failure_phase = "POST_CLAIM_SFTCONFIG_STRATEGY_COMPATIBILITY"
                expected_event_created = True
                expected_claim_created = True
                expected_holdouts_accessed = True
                expected_receipt_intent_produced = True
                expected_model_repository_code_imported = True
                expected_terminal_ledger_written = True
            elif predecessor == ATTEMPT_14_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30634484969"
                )
                expected_failure_phase = "POST_CLAIM_TRAINER_META_TENSOR"
                expected_event_created = True
                expected_claim_created = True
                expected_holdouts_accessed = True
                expected_receipt_intent_produced = True
                expected_model_repository_code_imported = True
                expected_terminal_ledger_written = True
            elif predecessor == ATTEMPT_15_REVIEWED_JOB_ID:
                expected_transport_evidence = (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30641766033"
                )
                expected_failure_phase = (
                    "PRE_CLAIM_AUTHENTICATED_PREFETCH_RUNTIME_BINDING"
                )
                expected_event_created = True
            elif coordinated_binding is not None and coordinated_binding.get(
                "runtimeBound"
            ):
                evidence, expected_transport_evidence = _runtime_predecessor_evidence(
                    coordinated_binding
                )
                expected_failure_phase = evidence["failurePhase"]
                expected_event_created = evidence["eventCreated"]
                expected_claim_created = evidence.get("claimCreated", False)
                expected_holdouts_accessed = evidence.get("holdoutsAccessed", False)
                expected_receipt_intent_produced = evidence.get(
                    "receiptIntentProduced", False
                )
                expected_model_repository_code_imported = evidence.get(
                    "modelRepositoryCodeImported", False
                )
                expected_terminal_ledger_written = evidence.get(
                    "receiptUploaded", False
                )
            else:
                raise ContractError(
                    "lineage predecessor is not an admitted transport recovery"
                )
            if lineage["transportEvidenceUrl"] != expected_transport_evidence:
                raise ContractError(
                    "lineage.transportEvidenceUrl is not the recorded transport evidence"
                )
            if lineage["failurePhase"] != expected_failure_phase:
                raise ContractError(
                    "lineage.failurePhase is not an admitted transport recovery phase"
                )
        else:
            _sha256(
                lineage["predecessorClaimSha256"],
                "lineage.predecessorClaimSha256",
            )
            _revision(
                lineage["predecessorBridgeRevision"],
                "lineage.predecessorBridgeRevision",
                exact40=True,
            )
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", lineage["predecessorImageId"]):
                raise ContractError(
                    "lineage.predecessorImageId must be an exact image ID"
                )
            _timestamp(lineage["predecessorClaimedAt"], "lineage.predecessorClaimedAt")
            if not re.fullmatch(
                r"https://github\.com/szl-holdings/szl-gpu-bridge/issues/"
                r"[0-9]+#issuecomment-[0-9]+",
                lineage["incidentUrl"],
            ):
                raise ContractError(
                    "lineage.incidentUrl must identify the recorded incident"
                )
            if lineage["failurePhase"] != "PRE_TRAINING_RUNTIME_SOURCE_PARSE":
                raise ContractError(
                    "lineage.failurePhase is not an admitted recovery phase"
                )
        if (
            not isinstance(lineage["successorGeneration"], int)
            or lineage["successorGeneration"] < 2
        ):
            raise ContractError("lineage.successorGeneration must be at least 2")
        expected_boundaries = {
            "automaticRetry": False,
            "trainingStarted": False,
            "modelRepositoryCodeImported": (expected_model_repository_code_imported),
            "holdoutsAccessed": expected_holdouts_accessed,
            "candidateProduced": False,
            "receiptIntentProduced": expected_receipt_intent_produced,
            "terminalLedgerWritten": expected_terminal_ledger_written,
            "scienceInputsReused": True,
        }
        if transport_lineage:
            expected_boundaries |= {
                "eventCreated": expected_event_created,
                "workflowRunCreated": expected_event_created,
                "claimCreated": expected_claim_created,
            }
        for field, expected in expected_boundaries.items():
            if lineage[field] is not expected:
                raise ContractError(
                    f"lineage.{field} must remain {str(expected).lower()}"
                )

    source = _object(
        spec,
        "source",
        {"repoId", "revision", "licenseId"},
        {"repoId", "revision", "licenseId"},
    )
    if (
        source["repoId"] != "szl-holdings/a11oy"
        or source["licenseId"].lower() != "apache-2.0"
    ):
        raise ContractError("source must be Apache-2.0 szl-holdings/a11oy")
    _revision(source["revision"], "source.revision", exact40=True)
    if coordinated_authorization:
        owner_dispatch = spec.get("ownerDispatch")
        lineage = spec.get("lineage")
        if source["revision"] != coordinated_binding["sourceRevision"]:
            raise ContractError(
                "coordinated recovery must bind the settled A11oy source"
            )
        if (
            not isinstance(owner_dispatch, dict)
            or owner_dispatch.get("workflowBlob") != coordinated_binding["workflowBlob"]
        ):
            raise ContractError(
                "coordinated recovery must bind the settled owner workflow"
            )
        if (
            not isinstance(lineage, dict)
            or lineage.get("successorGeneration")
            != coordinated_binding["successorGeneration"]
        ):
            raise ContractError(
                "coordinated recovery requires its exact successor generation"
            )
        if spec["jobId"] == ATTEMPT_5_REVIEWED_JOB_ID:
            exact_transport_lineage = {
                "predecessorJobId": NEXT_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "e240a176849b1f6c0d453ac55277cd7732b3a302ea9679db78d3c612501f27f2"
                ),
                "predecessorPayloadSha256": (
                    "14441cf982b177c1b613e56e63eae8be3e589ae35444826b40731c32312268e5"
                ),
                "predecessorEnvelopeRevision": (
                    "7045fe223703ba8fb2d710a59989f971080e7702"
                ),
                "predecessorExecutionBridgeRevision": CORRECTED_BRIDGE_REVISION,
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/szl-gpu-bridge/issues/32"
                ),
                "failurePhase": "PRE_EVENT_TRANSPORT_VALIDATION",
                "successorGeneration": 5,
                "automaticRetry": False,
                "eventCreated": False,
                "workflowRunCreated": False,
                "claimCreated": False,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": False,
                "candidateProduced": False,
                "receiptIntentProduced": False,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            }
            if lineage != exact_transport_lineage:
                raise ContractError("attempt-5 transport recovery lineage is not exact")
        if spec["jobId"] == ATTEMPT_6_REVIEWED_JOB_ID:
            exact_host_policy_lineage = {
                "predecessorJobId": ATTEMPT_5_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "30549fc522238193b4985dbf96a690518bad2ae8c399dc3ee78fb9dd7f551009"
                ),
                "predecessorPayloadSha256": (
                    "374901dec6923e0c28688407e581d374827d76f7567970d8ec481b6bf140c67b"
                ),
                "predecessorEnvelopeRevision": (
                    "d127d7bcd734235fba83e786de923787ab90c51b"
                ),
                "predecessorExecutionBridgeRevision": FINAL_CORRECTED_BRIDGE_REVISION,
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30591897165"
                ),
                "failurePhase": "PRE_ADMISSION_HOST_EXECUTION_POLICY",
                "successorGeneration": 6,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": False,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": False,
                "candidateProduced": False,
                "receiptIntentProduced": False,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            }
            if lineage != exact_host_policy_lineage:
                raise ContractError(
                    "attempt-6 host-policy recovery lineage is not exact"
                )
        if spec["jobId"] == FUTURE_REVIEWED_JOB_ID:
            exact_validator_rejection_lineage = {
                "predecessorJobId": ATTEMPT_6_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "c68e1ecf380d7023c27439e9988ca182ebd9b2446dc769269d4de1c48d507d70"
                ),
                "predecessorPayloadSha256": (
                    "d0fa9bd15f8e576411b643858d650470b6f1d5ddd56003cd53eda28d83dd914d"
                ),
                "predecessorEnvelopeRevision": (
                    "72f9bf650b081fec0a016825f2cb7f962c52242d"
                ),
                "predecessorExecutionBridgeRevision": (
                    "69a097d2eb0619506d673464353f1aea7174cf05"
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/szl-gpu-bridge/issues/41"
                ),
                "failurePhase": "PRE_DISPATCH_VALIDATOR_REJECTION",
                "successorGeneration": 7,
                "automaticRetry": False,
                "eventCreated": False,
                "workflowRunCreated": False,
                "claimCreated": False,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": False,
                "candidateProduced": False,
                "receiptIntentProduced": False,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            }
            if lineage != exact_validator_rejection_lineage:
                raise ContractError(
                    "attempt-7 validator-rejection recovery lineage is not exact"
                )
        if spec["jobId"] == NEXT_RUNTIME_REVIEWED_JOB_ID:
            exact_runtime_binding_lineage = {
                "predecessorJobId": FUTURE_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "8c1e333f797a8de634217b19cd140994a1d4f3920afebdf6f658dcc984188a96"
                ),
                "predecessorPayloadSha256": (
                    "0fa239d3e14f0644d26b76c0e605ea8068b305cd4d96ea41385cad38fbdfbde7"
                ),
                "predecessorEnvelopeRevision": (
                    "21553a898db76dddba3227e91518835185b55a6f"
                ),
                "predecessorExecutionBridgeRevision": (
                    "2f33607d8fcbec76fe98290258ec3dfa728fb509"
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30605081533"
                ),
                "failurePhase": "PRE_CLAIM_RUNTIME_CONTRACT_VALIDATION",
                "successorGeneration": 8,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": False,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": False,
                "candidateProduced": False,
                "receiptIntentProduced": False,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            }
            if lineage != exact_runtime_binding_lineage:
                raise ContractError(
                    "attempt-8 runtime-binding recovery lineage is not exact"
                )
        if spec["jobId"] == ATTEMPT_9_REVIEWED_JOB_ID:
            exact_prefetch_recovery_lineage = {
                "predecessorJobId": NEXT_RUNTIME_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "b2db463661ab9e16bf24267c82ee104cf25344e7b4addbd2e9867e7e33be3719"
                ),
                "predecessorPayloadSha256": (
                    "3372fff9c21a73ee140598c152b728b4d7694fb0a066c80e8b55e09832a0769d"
                ),
                "predecessorEnvelopeRevision": (
                    "08b1bd8bc0659b939d3d6d08c2ee7c670f82cd09"
                ),
                "predecessorExecutionBridgeRevision": (
                    NEXT_RUNTIME_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30606664591"
                ),
                "failurePhase": "PRE_CLAIM_DIRTY_EXECUTION_CHECKOUT",
                "successorGeneration": 9,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": False,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": False,
                "candidateProduced": False,
                "receiptIntentProduced": False,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            }
            if lineage != exact_prefetch_recovery_lineage:
                raise ContractError(
                    "attempt-9 prefetch-checkout recovery lineage is not exact"
                )
        if spec["jobId"] == ATTEMPT_10_REVIEWED_JOB_ID:
            exact_cache_license_finalizer_recovery_lineage = {
                "predecessorJobId": ATTEMPT_9_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "a7b67f1245137b3422d6e2ce5cf379aa9adb193e1f1d9db0dec8abf92bf5fa49"
                ),
                "predecessorPayloadSha256": (
                    "f8ec93b0a2967e548ba2222cbf8a69abbe89987c98e695688c39c0e0d3827c5b"
                ),
                "predecessorEnvelopeRevision": (
                    "4fa21a298e9b8f8dd6827f6dd0406ba6de02421e"
                ),
                "predecessorExecutionBridgeRevision": (
                    ATTEMPT_9_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30609977388"
                ),
                "failurePhase": "POST_CLAIM_CACHE_LICENSE_AND_FINALIZER_BINDING",
                "successorGeneration": 10,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": True,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": True,
                "candidateProduced": False,
                "receiptIntentProduced": True,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            }
            if lineage != exact_cache_license_finalizer_recovery_lineage:
                raise ContractError(
                    "attempt-10 cache/license/finalizer recovery lineage is not exact"
                )
        if spec["jobId"] == ATTEMPT_11_REVIEWED_JOB_ID:
            exact_runtime_admission_recovery_lineage = {
                "predecessorJobId": ATTEMPT_10_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "b354d34dcc6487e311b2d40413de4920ef8646d3f40e9d7442d366152aac901b"
                ),
                "predecessorPayloadSha256": (
                    "2287b1be69239ec0f577ee6e712e0093345e46640485dc6fefa88e8104d727c9"
                ),
                "predecessorEnvelopeRevision": (
                    "5c0aa8e9949b1cf2593acc269eb3fefffeaa36e1"
                ),
                "predecessorExecutionBridgeRevision": (
                    ATTEMPT_10_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30612658302"
                ),
                "failurePhase": ("PRE_CLAIM_IMMUTABLE_RUNTIME_JOB_BINDING_VALIDATION"),
                "successorGeneration": 11,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": False,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": False,
                "candidateProduced": False,
                "receiptIntentProduced": False,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            }
            if lineage != exact_runtime_admission_recovery_lineage:
                raise ContractError(
                    "attempt-11 runtime-admission recovery lineage is not exact"
                )
        if spec["jobId"] == ATTEMPT_12_REVIEWED_JOB_ID:
            exact_tokenizer_recovery_lineage = {
                "predecessorJobId": ATTEMPT_11_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "7b9af824b529fa80ec51e060cd0fa14f1af8acc8ded5fff5b10f159acb861918"
                ),
                "predecessorPayloadSha256": (
                    "85f08bc171370b25606915008d1b96ff50f670d09e20eb631b4c1ebeb108d994"
                ),
                "predecessorEnvelopeRevision": (
                    "61bb29bdad1e6b76bf3d818428c1d81149a6e72f"
                ),
                "predecessorExecutionBridgeRevision": (
                    ATTEMPT_11_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30620232291"
                ),
                "failurePhase": "POST_CLAIM_TOKENIZER_LOAD",
                "successorGeneration": 12,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": True,
                "trainingStarted": False,
                "modelRepositoryCodeImported": True,
                "holdoutsAccessed": True,
                "candidateProduced": False,
                "receiptIntentProduced": True,
                "terminalLedgerWritten": True,
                "scienceInputsReused": True,
            }
            if lineage != exact_tokenizer_recovery_lineage:
                raise ContractError(
                    "attempt-12 tokenizer recovery lineage is not exact"
                )
        if spec["jobId"] == ATTEMPT_13_REVIEWED_JOB_ID:
            exact_runtime_binding_recovery_lineage = {
                "predecessorJobId": ATTEMPT_12_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "a1c9f3d909b120d3675efe2cee0ba06b1c92c950f3a9ed4cc4e5b242971ed70f"
                ),
                "predecessorPayloadSha256": (
                    "a5e04951412bb0c4d085e567e4e869d52bdf6987546b16ffcd6d2bcb72768ce8"
                ),
                "predecessorEnvelopeRevision": (
                    "6b21684b64bf01971f3c3aac71493bba8078e532"
                ),
                "predecessorExecutionBridgeRevision": (
                    ATTEMPT_12_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30626533443"
                ),
                "failurePhase": ("PRE_CLAIM_AUTHENTICATED_PREFETCH_RUNTIME_BINDING"),
                "successorGeneration": 13,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": False,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": False,
                "candidateProduced": False,
                "receiptIntentProduced": False,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            }
            if lineage != exact_runtime_binding_recovery_lineage:
                raise ContractError(
                    "attempt-13 runtime-binding recovery lineage is not exact"
                )
        if spec["jobId"] == ATTEMPT_14_REVIEWED_JOB_ID:
            exact_sftconfig_recovery_lineage = {
                "predecessorJobId": ATTEMPT_13_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "de31cbb574cdeeaaf611a25fe1e40616b7fe8d4f6e2e138b66697474f5d800b0"
                ),
                "predecessorPayloadSha256": (
                    "82f619eb1fff6a7617b5761358d2f5c1d8ca62a306eb7cb1bf2570e096b2b9fc"
                ),
                "predecessorEnvelopeRevision": (
                    "b929bae4230ffe39ee63b34b8e9f9974cffc66ca"
                ),
                "predecessorExecutionBridgeRevision": (
                    "2783b3518abcec9f38d3f6504c06e305a4723801"
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30629929196"
                ),
                "failurePhase": "POST_CLAIM_SFTCONFIG_STRATEGY_COMPATIBILITY",
                "successorGeneration": 14,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": True,
                "trainingStarted": False,
                "modelRepositoryCodeImported": True,
                "holdoutsAccessed": True,
                "candidateProduced": False,
                "receiptIntentProduced": True,
                "terminalLedgerWritten": True,
                "scienceInputsReused": True,
            }
            if lineage != exact_sftconfig_recovery_lineage:
                raise ContractError(
                    "attempt-14 SFTConfig recovery lineage is not exact"
                )
        if spec["jobId"] == ATTEMPT_15_REVIEWED_JOB_ID:
            exact_meta_tensor_recovery_lineage = {
                "predecessorJobId": ATTEMPT_14_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "207f0c58525f042d31a748404d0acb678f5fd83722d2a3eacf8399e4e34c9f82"
                ),
                "predecessorPayloadSha256": (
                    "162354602784e8a1cbcecbbfc8a5d7cc9af6be2dd58c66fae442d4f5a292f1da"
                ),
                "predecessorEnvelopeRevision": (
                    "fd97065eb2aa9fc3299706c531597538a65eb735"
                ),
                "predecessorExecutionBridgeRevision": (
                    "e150711a6ba6a0c29109a00da7fc82af2967f588"
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30634484969"
                ),
                "failurePhase": "POST_CLAIM_TRAINER_META_TENSOR",
                "successorGeneration": 15,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": True,
                "trainingStarted": False,
                "modelRepositoryCodeImported": True,
                "holdoutsAccessed": True,
                "candidateProduced": False,
                "receiptIntentProduced": True,
                "terminalLedgerWritten": True,
                "scienceInputsReused": True,
            }
            if lineage != exact_meta_tensor_recovery_lineage:
                raise ContractError(
                    "attempt-15 meta-tensor recovery lineage is not exact"
                )
        if spec["jobId"] == ATTEMPT_16_REVIEWED_JOB_ID:
            exact_runtime_binding_recovery_lineage = {
                "predecessorJobId": ATTEMPT_15_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "93d5effe94740af9135c3ffa379c85df1aa88e6ad5717bc6421266d21bb9dbe7"
                ),
                "predecessorPayloadSha256": (
                    "9c55b95627b93e522eaebec5cb9e837b46d8e368065470aa45f55f488aeff873"
                ),
                "predecessorEnvelopeRevision": (
                    "7f42bad2cb7c762f8eb771922a0ba6e94c96e908"
                ),
                "predecessorExecutionBridgeRevision": (
                    "60b9894efe9e0e782999aaa4ee5b0d668e7a9b63"
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30641766033"
                ),
                "failurePhase": ("PRE_CLAIM_AUTHENTICATED_PREFETCH_RUNTIME_BINDING"),
                "successorGeneration": 16,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": False,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": False,
                "candidateProduced": False,
                "receiptIntentProduced": False,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            }
            if lineage != exact_runtime_binding_recovery_lineage:
                raise ContractError(
                    "attempt-16 runtime-binding recovery lineage is not exact"
                )

    base = _object(
        spec,
        "base",
        {
            "repoId",
            "revision",
            "licenseId",
            "licenseAcknowledgement",
            "trustRemoteCode",
        },
        {
            "repoId",
            "revision",
            "licenseId",
            "licenseAcknowledgement",
            "trustRemoteCode",
        },
    )
    _repo(base["repoId"], "base.repoId")
    _revision(base["revision"], "base.revision")
    if not isinstance(base["licenseId"], str) or not base["licenseId"].strip():
        raise ContractError("base.licenseId is required")
    if (
        coordinated_binding is not None
        and coordinated_binding["successorGeneration"] >= 10
        and base["licenseId"] != "nvidia-nemotron-open-model-license"
    ):
        raise ContractError(
            "runtime recovery must bind the exact immutable custom license ID"
        )
    if (
        not isinstance(base["licenseAcknowledgement"], str)
        or len(base["licenseAcknowledgement"].strip()) < 20
    ):
        raise ContractError("base.licenseAcknowledgement must be explicit")
    if not isinstance(base["trustRemoteCode"], bool):
        raise ContractError("base.trustRemoteCode must be boolean")

    dataset = _object(
        spec,
        "dataset",
        {"provenance", "rightsBasis", "train", "holdouts", "preregistration"},
        {"provenance", "rightsBasis", "train", "holdouts", "preregistration"},
    )
    if dataset["rightsBasis"] != "PROJECT_AUTHORED_SCENARIOS":
        raise ContractError("dataset rights basis is not admitted")
    if (
        not isinstance(dataset["provenance"], str)
        or len(dataset["provenance"].strip()) < 40
    ):
        raise ContractError("dataset provenance must be auditable")
    _pinned_file(dataset["train"], "dataset.train")
    _pinned_file(dataset["preregistration"], "dataset.preregistration")
    holdouts = dataset["holdouts"]
    if not isinstance(holdouts, list) or len(holdouts) != 3:
        raise ContractError("dataset.holdouts must contain exactly three frozen suites")
    names: list[str] = []
    ids: list[str] = []
    for index, value in enumerate(holdouts):
        item = _pinned_file(value, f"dataset.holdouts[{index}]", require_records=True)
        if item.get("name") not in _HOLDOUT_NAMES:
            raise ContractError("holdout suite name is not allowed")
        names.append(item["name"])
        ids.extend(item["recordIds"])
    if tuple(names) != _HOLDOUT_NAMES:
        raise ContractError(
            "holdout suite order must be original-v2, shadow-v2, challenge-v3"
        )
    if len(ids) != len(set(ids)):
        raise ContractError("record identifiers overlap across holdout suites")

    recipe_required = {
        "maxSeqLength",
        "loraR",
        "loraAlpha",
        "loraDropout",
        "targetModules",
        "batchSize",
        "gradAccum",
        "epochs",
        "learningRate",
        "optimizer",
        "gradientCheckpointing",
        "seed",
        "warmupRatio",
        "weightDecay",
        "lrSchedulerType",
    }
    recipe = _object(spec, "recipe", recipe_required, recipe_required)
    integer_ranges = {
        "maxSeqLength": (256, 4096),
        "loraR": (1, 64),
        "loraAlpha": (1, 256),
        "batchSize": (1, 1),
        "gradAccum": (1, 64),
    }
    for field, (minimum, maximum) in integer_ranges.items():
        value = recipe[field]
        if not isinstance(value, int) or not minimum <= value <= maximum:
            raise ContractError(f"recipe.{field} is outside the fixed range")
    if (
        not isinstance(recipe["loraDropout"], (int, float))
        or not 0 <= recipe["loraDropout"] <= 0.2
    ):
        raise ContractError("recipe.loraDropout is outside the fixed range")
    if (
        not isinstance(recipe["targetModules"], list)
        or not recipe["targetModules"]
        or len(recipe["targetModules"]) != len(set(recipe["targetModules"]))
    ):
        raise ContractError("recipe.targetModules must be a non-empty unique array")
    if not all(isinstance(item, str) and item for item in recipe["targetModules"]):
        raise ContractError("recipe.targetModules contains an invalid item")
    if not isinstance(recipe["epochs"], (int, float)) or not 0 < recipe["epochs"] <= 8:
        raise ContractError("recipe.epochs is outside the fixed range")
    if (
        not isinstance(recipe["learningRate"], (int, float))
        or not 0 < recipe["learningRate"] <= 0.001
    ):
        raise ContractError("recipe.learningRate is outside the fixed range")
    if recipe["optimizer"] not in {"adamw_8bit", "paged_adamw_8bit"}:
        raise ContractError("recipe.optimizer is unsupported")
    if recipe["gradientCheckpointing"] not in {"unsloth", "true"}:
        raise ContractError("recipe.gradientCheckpointing is unsupported")
    if not isinstance(recipe["seed"], int):
        raise ContractError("recipe.seed must be an integer")
    for field, maximum in (("warmupRatio", 0.2), ("weightDecay", 0.2)):
        value = recipe[field]
        if not isinstance(value, (int, float)) or not 0 <= value <= maximum:
            raise ContractError(f"recipe.{field} is outside the fixed range")
    if recipe["lrSchedulerType"] not in {"linear", "cosine"}:
        raise ContractError("recipe.lrSchedulerType is unsupported")

    gates_required = {
        "minFreeVramGb",
        "minFreeDiskGb",
        "maxWallclockMinutes",
        "maxDatasetRows",
        "maxTemperatureC",
        "maxUtilizationPct",
    }
    gates = _object(spec, "gates", gates_required, gates_required)
    if (
        not isinstance(gates["minFreeVramGb"], (int, float))
        or gates["minFreeVramGb"] < 5
    ):
        raise ContractError("gates.minFreeVramGb cannot be weakened below 5 GB")
    if (
        not isinstance(gates["minFreeDiskGb"], (int, float))
        or gates["minFreeDiskGb"] < 20
    ):
        raise ContractError("gates.minFreeDiskGb cannot be weakened below 20 GB")
    if (
        not isinstance(gates["maxWallclockMinutes"], int)
        or not 10 <= gates["maxWallclockMinutes"] <= 360
    ):
        raise ContractError("gates.maxWallclockMinutes is outside the fixed range")
    if (
        not isinstance(gates["maxDatasetRows"], int)
        or not 24 <= gates["maxDatasetRows"] <= 5000
    ):
        raise ContractError("gates.maxDatasetRows is outside the fixed range")
    if (
        not isinstance(gates["maxTemperatureC"], int)
        or not 40 <= gates["maxTemperatureC"] <= 80
    ):
        raise ContractError("gates.maxTemperatureC is outside the fixed range")
    if (
        not isinstance(gates["maxUtilizationPct"], int)
        or not 0 <= gates["maxUtilizationPct"] <= 30
    ):
        raise ContractError("gates.maxUtilizationPct is outside the fixed range")

    outputs = _object(
        spec,
        "outputs",
        {"candidateId", "receiptsRepoId", "private", "publishCandidate"},
        {"candidateId", "receiptsRepoId", "private", "publishCandidate"},
    )
    if not isinstance(outputs["candidateId"], str) or not outputs[
        "candidateId"
    ].startswith("SZL-Nemo-v3-"):
        raise ContractError("outputs.candidateId must identify SZL-Nemo-v3")
    _repo(outputs["receiptsRepoId"], "outputs.receiptsRepoId")
    if outputs["private"] is not True or outputs["publishCandidate"] is not False:
        raise ContractError("Nemo v3 candidate publication must remain disabled")

    evaluation = _object(
        spec,
        "evaluation",
        {
            "requiredPassRate",
            "maxDegenerateRate",
            "maxNewTokens",
            "requireExactRecordOrder",
        },
        {
            "requiredPassRate",
            "maxDegenerateRate",
            "maxNewTokens",
            "requireExactRecordOrder",
        },
    )
    if evaluation["requiredPassRate"] != 1.0 or evaluation["maxDegenerateRate"] != 0.0:
        raise ContractError(
            "Nemo v3 requires all holdouts to pass with no degeneration"
        )
    if (
        not isinstance(evaluation["maxNewTokens"], int)
        or not 32 <= evaluation["maxNewTokens"] <= 512
    ):
        raise ContractError("evaluation.maxNewTokens is outside the fixed range")
    if evaluation["requireExactRecordOrder"] is not True:
        raise ContractError("exact holdout record order is mandatory")
    return spec
