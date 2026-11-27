"""Fail-closed admission for a NeMo Framework trainer backend.

The existing trainer remains Hugging Face / PEFT in
``tools/salad/salad_worker.py``. This module records an additional backend.
It does not import NeMo, does not train, and does not launch.

``laptop/nemo_v3_contract.py`` owns the recipe ranges and the
``licenseId`` gate. This module reads neither file at runtime and does not
widen those bounds. Tests pin the bounds to that source.
"""

from __future__ import annotations

from typing import Any

SCHEMA = "szl.nemo-framework-admission/v0"
HF_PEFT_BACKEND = "hf-peft"
NEMO_FRAMEWORK_BACKEND = "nemo-framework"
HF_PEFT_MODULE = "tools/salad/salad_worker.py"
PINNED_TRAIN_IMAGE = (
    "ghcr.io/szl-holdings/szl-train-pinned"
    "@sha256:8eb919654e0da406d41a2e307e7705cce8f6b79ad0be9556d74d4c4640e63235"
)
SUCCESSOR_DIGEST = "UNKNOWN"
LAUNCHER = "saywa"
PROVIDERS = ("massed", "salad")
V3_LICENSE_ID = "nvidia-nemotron-open-model-license"
NAMED_OPEN_LICENSE_NAMES = frozenset(
    {
        "nvidia-nemotron-open-model-license",
        "nvidia-open-model-license",
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
    }
)
# Mirrors laptop/nemo_v3_contract.py. A test fails if either side moves.
RECIPE_INTEGER_RANGES = {
    "maxSeqLength": (256, 4096),
    "loraR": (1, 64),
    "loraAlpha": (1, 256),
    "batchSize": (1, 1),
    "gradAccum": (1, 64),
}
RECIPE_UNIT_INTERVAL_MAX = {
    "loraDropout": 0.2,
    "warmupRatio": 0.2,
    "weightDecay": 0.2,
}
EPOCHS_MAX = 8
LEARNING_RATE_MAX = 0.001
OPTIMIZERS = frozenset({"adamw_8bit", "paged_adamw_8bit"})
GRADIENT_CHECKPOINTING = frozenset({"unsloth", "true"})
SCHEDULERS = frozenset({"linear", "cosine"})
USD_TO_UUSD = 1_000_000
CAPS = {
    "perRunUsd": 100,
    "perDayUsd": 500,
    "perMonthUsd": 5000,
    "perRunUusd": 100 * USD_TO_UUSD,
    "perDayUusd": 500 * USD_TO_UUSD,
    "perMonthUusd": 5000 * USD_TO_UUSD,
}
FORBIDDEN_IMAGE_MARKERS = (
    "nvcr.io",
    "ngc.nvidia.com",
    "nvidia-ai-enterprise",
    "nvidia_ai_enterprise",
)
LICENSE_OBSERVED_ON = "2026-11-27"

FAMILIES: dict[str, dict[str, Any]] = {
    "nano-4b": {
        "internalName": "Nemotron Nano-4B",
        "hfRepoId": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "hfRevision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        "hfApiLicense": "other",
        "hfApiLicenseName": "nvidia-nemotron-open-model-license",
        "hfApiLicenseLink": (
            "https://www.nvidia.com/en-us/agreements/enterprise-software/"
            "nvidia-nemotron-open-model-license/"
        ),
        "licenseObservedOn": LICENSE_OBSERVED_ON,
        "amaruLicenseCheck": "PENDING",
        "matchesV3LicenseGate": True,
    },
    "nano-9b-v2": {
        "internalName": "Nemotron Nano-9B-v2",
        "hfRepoId": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "hfRevision": "6533e8de2c68e4536bf7c411d7a3ce5734111476",
        "hfApiLicense": "other",
        "hfApiLicenseName": "nvidia-open-model-license",
        "hfApiLicenseLink": (
            "https://www.nvidia.com/en-us/agreements/enterprise-software/"
            "nvidia-open-model-license/"
        ),
        "licenseObservedOn": LICENSE_OBSERVED_ON,
        "amaruLicenseCheck": "PENDING",
        "matchesV3LicenseGate": False,
    },
    "illapa": {
        "internalName": "Illapa",
        "role": "Nemotron flagship, internal name only",
        "hfRepoId": "UNKNOWN",
        "hfRevision": "UNKNOWN",
        "hfApiLicense": "UNKNOWN",
        "hfApiLicenseName": "UNKNOWN",
        "hfApiLicenseLink": "UNKNOWN",
        "licenseObservedOn": LICENSE_OBSERVED_ON,
        "amaruLicenseCheck": "PENDING",
        "matchesV3LicenseGate": False,
        "unboundCandidatesObserved": [
            {
                "hfRepoId": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
                "hfRevision": "77df655d5e9f8362164ed14dd8b48f8bce657498",
                "hfApiLicense": "other",
                "hfApiLicenseName": "openmdw-1.1",
                "adopted": False,
            },
            {
                "hfRepoId": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
                "hfRevision": "2dc98e2afe4face0e4ce40972a915c45368bd34a",
                "hfApiLicense": "other",
                "hfApiLicenseName": "nvidia-nemotron-open-model-license",
                "adopted": False,
            },
        ],
    },
}


class AdmissionError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _object(value: Any, code: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise AdmissionError(code)
    return value


def _exact(value: Any, expected: Any, code: str) -> None:
    if value != expected:
        raise AdmissionError(code)


def _reject_enterprise_image(image: str) -> None:
    if type(image) is not str or not image:
        raise AdmissionError("IMAGE_INVALID")
    lowered = image.lower()
    for marker in FORBIDDEN_IMAGE_MARKERS:
        if marker in lowered:
            raise AdmissionError("ENTERPRISE_IMAGE_FORBIDDEN")


def _number(value: Any) -> bool:
    return type(value) in (int, float) and type(value) is not bool


def recipe_within_contract(recipe: Any) -> None:
    """Reject a recipe outside the fixed nemo v3 bounds. Does not widen them."""

    body = _object(recipe, "RECIPE_SHAPE")
    required = (
        set(RECIPE_INTEGER_RANGES)
        | set(RECIPE_UNIT_INTERVAL_MAX)
        | {
            "targetModules",
            "epochs",
            "learningRate",
            "optimizer",
            "gradientCheckpointing",
            "seed",
            "lrSchedulerType",
        }
    )
    if set(body) != required:
        raise AdmissionError("RECIPE_SHAPE")
    for field, (low, high) in RECIPE_INTEGER_RANGES.items():
        value = body[field]
        if type(value) is not int or not low <= value <= high:
            raise AdmissionError("RECIPE_RANGE")
    for field, high in RECIPE_UNIT_INTERVAL_MAX.items():
        value = body[field]
        if not _number(value) or not 0 <= value <= high:
            raise AdmissionError("RECIPE_RANGE")
    modules = body["targetModules"]
    if (
        type(modules) is not list
        or not modules
        or len(modules) != len(set(modules))
        or not all(type(item) is str and item for item in modules)
    ):
        raise AdmissionError("RECIPE_RANGE")
    epochs = body["epochs"]
    if not _number(epochs) or not 0 < epochs <= EPOCHS_MAX:
        raise AdmissionError("RECIPE_RANGE")
    learning_rate = body["learningRate"]
    if not _number(learning_rate) or not 0 < learning_rate <= LEARNING_RATE_MAX:
        raise AdmissionError("RECIPE_RANGE")
    if type(body["seed"]) is not int:
        raise AdmissionError("RECIPE_RANGE")
    if body["optimizer"] not in OPTIMIZERS:
        raise AdmissionError("RECIPE_RANGE")
    if body["gradientCheckpointing"] not in GRADIENT_CHECKPOINTING:
        raise AdmissionError("RECIPE_RANGE")
    if body["lrSchedulerType"] not in SCHEDULERS:
        raise AdmissionError("RECIPE_RANGE")


def _family_block(family_id: str) -> dict[str, Any]:
    family = FAMILIES[family_id]
    reasons = ["NEMO_LAYER_DIGEST_UNKNOWN", "AMARU_LICENSE_CHECK_PENDING"]
    if family_id == "nano-9b-v2":
        reasons.insert(0, "LICENSE_NAME_DIFFERS_FROM_V3_GATE")
    if family_id == "illapa":
        reasons.insert(0, "HF_REPO_UNKNOWN")
    license_name = family["hfApiLicenseName"]
    named_open = license_name in NAMED_OPEN_LICENSE_NAMES
    return {
        "family": family_id,
        "hfRepoId": family["hfRepoId"],
        "hfApiLicense": family["hfApiLicense"],
        "hfApiLicenseName": license_name,
        "amaruLicenseCheck": "PENDING",
        "namedOpenLicense": named_open,
        "admittedForTraining": False,
        "reasons": reasons,
    }


def framework_draft() -> dict[str, Any]:
    """Non-dispatchable draft. A successor image digest is not assigned."""

    return {
        "schema": "szl.nemo-framework-trainer-draft/v0",
        "backend": NEMO_FRAMEWORK_BACKEND,
        "existingBackend": HF_PEFT_BACKEND,
        "existingBackendModule": HF_PEFT_MODULE,
        "baseImage": PINNED_TRAIN_IMAGE,
        "baseImageRegistryVerified": "UNKNOWN",
        "successorImageDigest": SUCCESSOR_DIGEST,
        "successorImageStatus": "NOT_BUILT",
        "nemoFramework": {
            "source": "https://github.com/NVIDIA-NeMo/NeMo",
            "githubApiSpdx": "Apache-2.0",
            "revision": "UNKNOWN",
            "ngcContainer": False,
            "compatibilityWithPinnedTorch": "UNKNOWN",
        },
        "launcher": LAUNCHER,
        "launcherImplementation": "UNKNOWN",
        "providers": list(PROVIDERS),
        "caps": dict(CAPS),
        "launch": False,
        "families": [family_id for family_id in FAMILIES],
        "licenseRegister": FAMILIES,
        "recipe": {
            "maxSeqLength": 768,
            "loraR": 16,
            "loraAlpha": 32,
            "loraDropout": 0.05,
            "targetModules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "batchSize": 1,
            "gradAccum": 8,
            "epochs": 2,
            "learningRate": 0.0001,
            "optimizer": "paged_adamw_8bit",
            "gradientCheckpointing": "unsloth",
            "seed": 3407,
            "warmupRatio": 0.05,
            "weightDecay": 0.01,
            "lrSchedulerType": "linear",
        },
    }


def admit(request: Any) -> dict[str, Any]:
    """Return a receipt that never authorizes a launch."""

    body = _object(request, "REQUEST_NOT_OBJECT")
    if body.get("launch") is not False:
        raise AdmissionError("LAUNCH_REFUSED")
    _exact(body.get("launcher"), LAUNCHER, "LAUNCHER_NOT_SAYWA")
    _exact(list(body.get("providers") or ()), list(PROVIDERS), "PROVIDER_NOT_ALLOWED")
    _exact(body.get("caps"), CAPS, "CAP_MISMATCH")
    backend = body.get("backend")
    if backend not in (HF_PEFT_BACKEND, NEMO_FRAMEWORK_BACKEND):
        raise AdmissionError("BACKEND_UNKNOWN")
    recipe = body.get("recipe")
    if recipe is not None:
        recipe_within_contract(recipe)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "backend": backend,
        "existingBackendRetained": HF_PEFT_BACKEND,
        "existingBackendModule": HF_PEFT_MODULE,
        "launchAuthorized": False,
        "trainingStarted": False,
        "imagePushed": False,
        "launcher": LAUNCHER,
        "launcherImplementation": "UNKNOWN",
        "caps": dict(CAPS),
        "providers": list(PROVIDERS),
    }
    if backend == HF_PEFT_BACKEND:
        receipt["reasons"] = ["HF_PEFT_PATH_UNCHANGED", "THIS_MODULE_DOES_NOT_LAUNCH"]
        return receipt
    image = body.get("baseImage")
    _reject_enterprise_image(image)
    _exact(image, PINNED_TRAIN_IMAGE, "BASE_IMAGE_NOT_PINNED")
    successor = body.get("successorImageDigest")
    _reject_enterprise_image(str(successor) if successor is not None else "")
    _exact(successor, SUCCESSOR_DIGEST, "SUCCESSOR_DIGEST_NOT_UNKNOWN")
    families = body.get("families")
    if type(families) is not list or not families:
        raise AdmissionError("FAMILY_REQUIRED")
    if any(type(item) is not str or item not in FAMILIES for item in families):
        raise AdmissionError("FAMILY_UNKNOWN")
    receipt["baseImage"] = PINNED_TRAIN_IMAGE
    receipt["successorImageDigest"] = SUCCESSOR_DIGEST
    receipt["successorImageStatus"] = "NOT_BUILT"
    receipt["families"] = [_family_block(item) for item in families]
    receipt["reasons"] = ["NEMO_LAYER_DIGEST_UNKNOWN"]
    return receipt
