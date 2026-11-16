"""Read-only preflight for an admitted, prebuilt training environment.

Never install packages, hydrate a model, start training, or publish artifacts.
A matching manifest proves neither source admission nor model qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable

SCHEMA = "szl.training-runtime/v1"
REQUIRED = frozenset(
    {
        "torch",
        "transformers",
        "trl",
        "datasets",
        "peft",
        "accelerate",
        "huggingface-hub",
        "safetensors",
        "nvidia-ml-py",
    }
)
OPTIONAL = frozenset({"torchvision", "torchaudio"})
CONFIG_FIELDS = frozenset(
    {
        "max_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_steps",
        "logging_steps",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "bf16",
        "optim",
        "num_train_epochs",
        "report_to",
        "packing",
        "seed",
        "output_dir",
    }
)


class RuntimeHold(ValueError):
    """A fixed, secret-free failure code."""


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise RuntimeHold("DUPLICATE_CONTRACT_KEY")
        result[key] = value
    return result


def _constant(value):
    raise RuntimeHold("NONFINITE_CONTRACT")


def parse_contract(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 32768:
        raise RuntimeHold("CONTRACT_BYTE_BUDGET")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeHold("INVALID_CONTRACT_JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeHold("CONTRACT_NOT_OBJECT")
    return value


def validate_contract(value: dict, recipe: bytes, python_version: str) -> dict:
    keys = {
        "schema",
        "repository",
        "source_revision",
        "recipe_sha256",
        "python",
        "distributions",
    }
    if set(value) != keys or value.get("schema") != SCHEMA:
        raise RuntimeHold("CONTRACT_SCHEMA_MISMATCH")
    if value["repository"] != "szl-holdings/szl-gpu-bridge":
        raise RuntimeHold("SOURCE_REPOSITORY_MISMATCH")
    if not re.fullmatch(r"[0-9a-f]{40}", str(value["source_revision"])):
        raise RuntimeHold("SOURCE_REVISION_NOT_PINNED")
    if value["recipe_sha256"] != hashlib.sha256(recipe).hexdigest():
        raise RuntimeHold("RECIPE_BYTES_CHANGED")
    if not isinstance(value["python"], str) or value["python"] != python_version:
        raise RuntimeHold("PYTHON_VERSION_MISMATCH")
    pins = value["distributions"]
    if not isinstance(pins, dict) or not REQUIRED <= set(pins) <= REQUIRED | OPTIONAL:
        raise RuntimeHold("DISTRIBUTION_COVERAGE_MISMATCH")
    if any(
        not isinstance(v, str)
        or len(v) > 128
        or not re.fullmatch(r"[0-9][0-9A-Za-z.+!_-]*", v)
        for v in pins.values()
    ):
        raise RuntimeHold("DISTRIBUTIONS_REQUIRE_EXACT_VERSIONS")
    return pins


def verify_versions(pins: dict, getter: Callable = importlib.metadata.version) -> dict:
    observed = {}
    for name in sorted(REQUIRED | OPTIONAL):
        try:
            version = getter(name)
        except importlib.metadata.PackageNotFoundError:
            if name in pins:
                raise RuntimeHold("REQUIRED_DISTRIBUTION_MISSING") from None
            continue
        if name not in pins:
            raise RuntimeHold("UNDECLARED_OPTIONAL_DISTRIBUTION")
        if version != pins[name]:
            raise RuntimeHold("DISTRIBUTION_VERSION_MISMATCH")
        observed[name] = version
    return observed


SMOKE = r"""
import dataclasses, importlib.util, json
import torch
if importlib.util.find_spec("torchvision") is not None:
    import torchvision
if importlib.util.find_spec("torchaudio") is not None:
    import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import pynvml
assert callable(pynvml.nvmlDeviceGetPowerUsage)
print(json.dumps({"sft_config_fields": sorted(f.name for f in dataclasses.fields(SFTConfig))}))
"""


def run_checks(runner: Callable = subprocess.run) -> dict:
    env = dict(os.environ)
    for name in ("HF_TOKEN", "HF_TOKEN_INJECT", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(name, None)
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    try:
        check = runner(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if check.returncode:
            raise RuntimeHold("PACKAGE_DEPENDENCY_CONFLICT")
        smoke = runner(
            [sys.executable, "-I", "-c", SMOKE],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        if smoke.returncode:
            raise RuntimeHold("TRAINING_IMPORT_SMOKE_FAILED")
        if len(smoke.stdout) > 32768:
            raise RuntimeHold("SMOKE_OUTPUT_BYTE_BUDGET")
        output = json.loads(smoke.stdout)
        fields = output.get("sft_config_fields") if isinstance(output, dict) else None
        if (
            not isinstance(fields, list)
            or not all(isinstance(v, str) for v in fields)
            or not CONFIG_FIELDS <= set(fields)
        ):
            raise RuntimeHold("TRAINER_CONFIG_API_MISMATCH")
    except subprocess.TimeoutExpired as exc:
        raise RuntimeHold("RUNTIME_CHECK_TIMEOUT") from exc
    except (ValueError, AttributeError, TypeError) as exc:
        if isinstance(exc, RuntimeHold):
            raise
        raise RuntimeHold("INVALID_SMOKE_RESPONSE") from exc
    return {
        "pip_check": "PASS",
        "training_import_smoke": "PASS",
        "recipe_config_api": "PASS",
        "model_hydrated": False,
        "training_started": False,
        "packages_installed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    args = parser.parse_args()
    try:
        if (
            args.contract.is_symlink()
            or args.recipe.is_symlink()
            or args.contract.stat().st_size > 32768
            or args.recipe.stat().st_size > 200000
        ):
            raise RuntimeHold("INPUT_PATH_OR_SIZE_REJECTED")
        raw, recipe = args.contract.read_bytes(), args.recipe.read_bytes()
        contract = parse_contract(raw)
        pins = validate_contract(
            contract, recipe, f"{sys.version_info.major}.{sys.version_info.minor}"
        )
        versions = verify_versions(pins)
        report = run_checks()
        report.update(
            schema=SCHEMA,
            status="RUNTIME_PREFLIGHT_PASSED",
            source_revision=contract["source_revision"],
            source_admission_verified=False,
            runtime_contract_sha256=hashlib.sha256(raw).hexdigest(),
            recipe_sha256=hashlib.sha256(recipe).hexdigest(),
            distributions=versions,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    except (RuntimeHold, OSError) as exc:
        code = str(exc) if isinstance(exc, RuntimeHold) else "INPUT_UNAVAILABLE"
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "HOLD",
                    "reason": code,
                    "training_started": False,
                    "packages_installed": False,
                }
            )
        )
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
