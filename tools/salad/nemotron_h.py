#!/usr/bin/env python3
"""Nemotron-H (hybrid Mamba-2 + attention) LoRA/QLoRA SFT helpers.

Target modules come from the model config. An unknown block type or pattern
character refuses the run. The license id and acknowledgement length are the
same values laptop/nemo_v3_contract.py enforces; this module does not widen them.
Nothing here downloads weights, starts a GPU job, or reads a credential.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

LICENSE_ID = "nvidia-nemotron-open-model-license"
ACK_MIN_CHARS = 20
THINK_SYSTEM_PROMPT = "/think"
NO_THINK_SYSTEM_PROMPT = "/no_think"
ALLOWED_PROVIDERS = ("salad", "massed")
TRAIN_IMAGE = (
    "ghcr.io/szl-holdings/szl-train-pinned"
    "@sha256:8eb919654e0da406d41a2e307e7705cce8f6b79ad0be9556d74d4c4640e63235"
)
METRICS_ENV = "SZL_POOL_METRICS_FILE"
CHECKPOINT_REQUEST_ENV = "SZL_POOL_CHECKPOINT_REQUEST"

# Image pins that training-image/requirements.lock is expected to match.
# mamba-ssm and causal-conv1d are the wheels the N-14 overlay records for this
# torch build. They are not in the train lock; requirement_pin_mismatches
# reports that absence instead of inventing a lock line.
IMAGE_LOCK_PINS = {
    "torch": "2.14.1+cu126",
    "transformers": "5.22.2",
    "peft": "0.21.2",
    "trl": "1.17.1",
}
IMAGE_ABSENT_PINS = {
    "mamba-ssm": (
        "2.3.2.post1+cu.12.6.torch.2.14",
        "329557d964d58ebfcaf44cb17dc043356a24dd5c8ab9b68d96dc0fbf89911614",
    ),
    "causal-conv1d": (
        "1.7.0+cu.12.6.torch.2.14",
        "75f42d6ab5c87490d04a9a58f9e3c191c9cd88b31e8b626fc057099e01832d60",
    ),
}
ROOT_PACKAGE_PINS = {
    "transformers": "5.17.0",
    "peft": "0.21.0",
    "trl": "1.14.0",
}

_PATTERN = {"M": "mamba", "*": "attention", "-": "mlp"}
_BLOCK = {
    "mamba": "mamba",
    "linear_attention": "mamba",
    "attention": "attention",
    "full_attention": "attention",
    "mlp": "mlp",
}
_FAMILY_MODULES = {
    "mamba": ("in_proj", "out_proj"),
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("up_proj", "down_proj"),
}


class NemotronHError(RuntimeError):
    """Fail closed. The message is a reason, not a retry hint."""


def _get(config, key):
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _require_nemotron_h(config) -> None:
    model_type = _get(config, "model_type")
    architectures = list(_get(config, "architectures") or [])
    if model_type == "nemotron_h":
        return
    if "NemotronHForCausalLM" in architectures:
        return
    raise NemotronHError(
        "unknown architecture "
        f"model_type={model_type!r} architectures={architectures!r}"
    )


def block_families(config) -> list[str]:
    """Map a Nemotron-H config onto mamba / attention / mlp, or refuse."""

    _require_nemotron_h(config)
    pattern = _get(config, "hybrid_override_pattern")
    layers = _get(config, "layers_block_type")
    if layers is None:
        layers = _get(config, "layer_types")
    from_pattern = None
    if isinstance(pattern, str) and pattern:
        mapped = []
        for char in pattern:
            if char not in _PATTERN:
                raise NemotronHError(f"unknown hybrid pattern character {char!r}")
            mapped.append(_PATTERN[char])
        from_pattern = mapped
    from_layers = None
    if layers is not None:
        if (
            isinstance(layers, str)
            or not isinstance(layers, (list, tuple))
            or not layers
        ):
            raise NemotronHError("layers_block_type is not a non-empty list")
        mapped = []
        for item in layers:
            if item not in _BLOCK:
                raise NemotronHError(f"unknown block type {item!r}")
            mapped.append(_BLOCK[item])
        from_layers = mapped
    if from_pattern is None and from_layers is None:
        raise NemotronHError("nemotron_h config has no hybrid pattern")
    if (
        from_pattern is not None
        and from_layers is not None
        and from_pattern != from_layers
    ):
        raise NemotronHError("hybrid pattern disagrees with layers_block_type")
    families = from_layers if from_layers is not None else from_pattern
    if not families:
        raise NemotronHError("nemotron_h config has no blocks")
    return list(families)


def lora_target_modules(config) -> list[str]:
    """Mamba in/out, attention q/k/v/o, and MLP up/down, for families present."""

    present = set(block_families(config))
    names: list[str] = []
    for family in ("mamba", "attention", "mlp"):
        if family in present:
            names.extend(_FAMILY_MODULES[family])
    if not names:
        raise NemotronHError("no LoRA projections for this Nemotron-H config")
    return names


def base_requires_nemotron_h_preflight(base: str) -> bool:
    """True only for Nemotron-H hub ids, not every name that contains nemotron.

    Llama-Nemotron and Nemotron-4 stay on the worker's ordinary LoRA targets.
    """

    name = base.lower()
    markers = (
        "nemotron-h",
        "nemotron_h",
        "nemotronh",
        "nvidia-nemotron-3-nano",
        "nvidia-nemotron-nano-9b",
    )
    return any(marker in name for marker in markers)


def use_eager_out_proj(model=None) -> None:
    """Make Mamba out_proj LoRA execute when mamba-ssm is installed.

    The mixer calls ``mamba2_split_conv1d_scan_combined`` and, only when that
    returns a tensor, uses it as the layer output. ``None`` is the documented
    signal that the fused kernel is unavailable, after which the same forward
    calls ``self.out_proj``. Replacing the module global forces that branch
    even if the import-time wrapper closed over mamba-ssm. A mixer whose
    instance forward is no longer the class method is pointed back at the
    class method, which reads the global. ``conv1d`` stays excluded.
    """

    import transformers.models.nemotron_h.modeling_nemotron_h as modeling

    def _pytorch_fallback(*args, **kwargs):
        return None

    modeling.mamba2_split_conv1d_scan_combined = _pytorch_fallback
    if model is None:
        return
    mixer_type = modeling.NemotronHMamba2Mixer
    class_forward = mixer_type.forward

    def _class_forward(module):
        def forward(*args, **kwargs):
            return class_forward(module, *args, **kwargs)

        forward._szl_eager_out_proj = True
        return forward

    for module in model.modules():
        if not isinstance(module, mixer_type):
            continue
        current = module.forward
        if getattr(current, "_szl_eager_out_proj", False):
            continue
        if getattr(current, "__func__", None) is not class_forward:
            module.forward = _class_forward(module)


def attach_lora(model, *, r, lora_alpha, lora_dropout, target_modules):
    """Attach LoRA, including Mamba out_proj, and keep model_type nemotron_h."""

    from peft import LoraConfig, get_peft_model

    use_eager_out_proj(model)
    config = model.config
    original = getattr(config, "model_type", None)
    renamed = original == "nemotron_h" and "out_proj" in list(target_modules)
    if renamed:
        config.model_type = "nemotron_h_lora_out_proj"
    try:
        adapted = get_peft_model(
            model,
            LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=list(target_modules),
                task_type="CAUSAL_LM",
            ),
        )
    finally:
        if renamed:
            config.model_type = original
    adapted_config = getattr(adapted, "config", None)
    if getattr(adapted_config, "model_type", None) == "nemotron_h_lora_out_proj":
        adapted_config.model_type = original
    return adapted


def assert_lora_modules_present(model, names: list[str]) -> list[str]:
    import torch.nn as nn

    found = {name: [] for name in names}
    for module_name, module in model.named_modules():
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf in found and isinstance(module, nn.Linear):
            found[leaf].append(module_name)
    missing = [name for name, hits in found.items() if not hits]
    if missing:
        raise NemotronHError(f"detected LoRA modules are absent: {missing}")
    return list(names)


def lora_targets_for_model(model, requested: list[str]) -> list[str]:
    """Use the recipe list for other architectures. Nemotron-H must match detection."""

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    architectures = list(getattr(config, "architectures", None) or [])
    if model_type != "nemotron_h" and "NemotronHForCausalLM" not in architectures:
        return list(requested)
    detected = lora_target_modules(config)
    assert_lora_modules_present(model, detected)
    if set(requested) != set(detected):
        raise NemotronHError(
            f"recipe.targetModules {list(requested)} != detected {detected}"
        )
    return detected


def messages_for_thinking(messages, enable_thinking: bool) -> list[dict]:
    """Apply Nemotron's /think or /no_think system prompt from the recipe field."""

    if not isinstance(enable_thinking, bool):
        raise NemotronHError("enableThinking must be boolean")
    if not isinstance(messages, list) or not messages:
        raise NemotronHError("messages must be a non-empty list")
    prompt = THINK_SYSTEM_PROMPT if enable_thinking else NO_THINK_SYSTEM_PROMPT
    copied = [dict(item) for item in messages]
    for item in copied:
        if not isinstance(item, dict) or "role" not in item:
            raise NemotronHError("each message needs a role")
    if copied[0].get("role") != "system":
        copied.insert(0, {"role": "system", "content": prompt})
        return copied
    content = copied[0].get("content")
    if not isinstance(content, str):
        raise NemotronHError("system message content must be a string")
    stripped = content.strip()
    for token in (THINK_SYSTEM_PROMPT, NO_THINK_SYSTEM_PROMPT):
        if stripped == token:
            stripped = ""
        elif stripped.endswith("\n" + token) or stripped.endswith(" " + token):
            stripped = stripped[: -(len(token) + 1)].rstrip()
    if THINK_SYSTEM_PROMPT in stripped or NO_THINK_SYSTEM_PROMPT in stripped:
        raise NemotronHError("system message contains a conflicting thinking token")
    copied[0]["content"] = f"{stripped}\n{prompt}".strip() if stripped else prompt
    return copied


def preflight_license(base: dict, notice_path) -> Path:
    """Acknowledgement, NOTICE file, and the exact Nemotron Open Model license id."""

    if not isinstance(base, dict):
        raise NemotronHError("base must be an object")
    if base.get("licenseId") != LICENSE_ID:
        raise NemotronHError(
            "runtime recovery must bind the exact immutable custom license ID"
        )
    acknowledgement = base.get("licenseAcknowledgement")
    if (
        not isinstance(acknowledgement, str)
        or len(acknowledgement.strip()) < ACK_MIN_CHARS
    ):
        raise NemotronHError("base.licenseAcknowledgement must be explicit")
    path = Path(notice_path)
    if path.is_symlink() or not path.is_file():
        raise NemotronHError("NVIDIA NOTICE file is absent")
    if not path.read_bytes().strip():
        raise NemotronHError("NVIDIA NOTICE file is empty")
    return path


def admit_provider(spec: dict, provider: str) -> str:
    providers = spec.get("providers")
    if providers != list(ALLOWED_PROVIDERS):
        raise NemotronHError("providers must be salad and massed only")
    if provider not in ALLOWED_PROVIDERS:
        raise NemotronHError("provider is not salad or massed")
    return provider


def _pool_runner():
    path = Path(__file__).resolve().parents[1] / "gpu" / "pool_runner.py"
    spec = importlib.util.spec_from_file_location("szl_pool_runner", path)
    if spec is None or spec.loader is None:
        raise NemotronHError("swap-pool runner is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def resolved_caps(spec: dict, repo_root, extra_sources=()):
    """Read caps from the named config. Extra sources apply; the stricter value wins."""

    caps = spec.get("caps")
    if not isinstance(caps, dict) or caps.get("stricterWins") is not True:
        raise NemotronHError(
            "caps must be read from config with the stricter value winning"
        )
    if any(key in caps for key in ("per_run_usd", "per_day_usd", "per_month_usd")):
        raise NemotronHError("cap amounts must not be hardcoded on the jobspec")
    relative = caps.get("config")
    if not isinstance(relative, str) or not relative or relative.startswith("/"):
        raise NemotronHError("caps.config must be a repo-relative file")
    path = Path(repo_root) / relative
    pool = _pool_runner()
    dispatch = pool.load_dispatch_caps(path)
    sources = [(f"dispatch:{relative}", dispatch), *list(extra_sources)]
    return pool.effective_caps(sources)


def _fixed_recipe_ranges(recipe: dict) -> None:
    path = Path(__file__).resolve().parents[2] / "laptop" / "nemo_v3_contract.py"
    if not path.is_file():
        raise NemotronHError("fixed recipe ranges are unavailable")
    spec = importlib.util.spec_from_file_location("nemo_v3_contract_ranges", path)
    if spec is None or spec.loader is None:
        raise NemotronHError("fixed recipe ranges are unavailable")
    laptop = str(path.parent)
    inserted = laptop not in sys.path
    if inserted:
        sys.path.insert(0, laptop)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted and sys.path and sys.path[0] == laptop:
            sys.path.pop(0)
    try:
        module.validate_fixed_recipe_ranges(recipe)
    except Exception as exc:
        raise NemotronHError(str(exc)) from exc


def review_launch_spec(spec: dict, repo_root, *, notice_path, provider: str):
    if spec.get("trainingImage") != TRAIN_IMAGE:
        raise NemotronHError("training image is not the pinned digest")
    admit_provider(spec, provider)
    base = spec.get("base")
    preflight_license(base, notice_path)
    recipe = spec.get("recipe")
    if not isinstance(recipe, dict) or not isinstance(
        recipe.get("enableThinking"), bool
    ):
        raise NemotronHError("recipe.enableThinking must be boolean")
    _fixed_recipe_ranges(recipe)
    detected = lora_target_modules(
        {
            "model_type": "nemotron_h",
            "hybrid_override_pattern": spec.get("expectedHybridPattern"),
            "layers_block_type": spec.get("expectedLayersBlockType"),
        }
    )
    if list(recipe.get("targetModules") or []) != detected:
        raise NemotronHError(
            f"recipe.targetModules {recipe.get('targetModules')!r} != detected {detected}"
        )
    return resolved_caps(spec, repo_root)


def append_eval_record(step, eval_loss, *, environ=None) -> Path | None:
    """Append one pool metrics line. No env var means the pool is not attached."""

    environ = os.environ if environ is None else environ
    raw = environ.get(METRICS_ENV)
    if not raw:
        return None
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise NemotronHError("eval step is invalid")
    try:
        loss = float(eval_loss)
    except (TypeError, ValueError) as exc:
        raise NemotronHError("eval_loss is not a float") from exc
    if not math.isfinite(loss):
        raise NemotronHError("eval_loss is not finite")
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"eval_loss": loss, "step": step}, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
    return path


def honor_checkpoint_request(save, *, environ=None):
    """Save into the pool's requested directory and write COMPLETE."""

    environ = os.environ if environ is None else environ
    raw = environ.get(CHECKPOINT_REQUEST_ENV)
    if not raw:
        return None
    request_path = Path(raw)
    if not request_path.is_file():
        return None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NemotronHError("checkpoint request is unreadable") from exc
    if not isinstance(request, dict):
        raise NemotronHError("checkpoint request is not an object")
    for key in ("seq", "label", "dir"):
        if key not in request:
            raise NemotronHError(f"checkpoint request missing {key}")
    if isinstance(request["seq"], bool) or not isinstance(request["seq"], int):
        raise NemotronHError("checkpoint request seq is invalid")
    if not isinstance(request["label"], str) or not request["label"]:
        raise NemotronHError("checkpoint request label is invalid")
    if not isinstance(request["dir"], str) or not request["dir"]:
        raise NemotronHError("checkpoint request dir is invalid")
    target = Path(request["dir"])
    if not target.is_absolute():
        raise NemotronHError("checkpoint dir must be absolute")
    if target.is_symlink():
        raise NemotronHError("checkpoint dir is a symlink")
    if (target / "COMPLETE").is_file():
        return target
    target.mkdir(parents=True, exist_ok=True)
    save(target)
    payload = json.dumps(
        {"seq": request["seq"], "label": request["label"]}, sort_keys=True
    )
    temporary = target / "COMPLETE.tmp"
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, target / "COMPLETE")
    return target


def pool_callback_class():
    from transformers import TrainerCallback

    class PoolCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if isinstance(metrics, dict) and "eval_loss" in metrics:
                append_eval_record(state.global_step, metrics["eval_loss"])
            return control

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if model is None:
                return control

            def save(directory):
                model.save_pretrained(str(directory))

            honor_checkpoint_request(save)
            return control

    return PoolCallback


def requirement_pin_mismatches(
    lock_text: str, root_text: str, project_text: str
) -> list[str]:
    """Exact mismatches between the named train image and this repo's pins."""

    mismatches = []
    for name, version in IMAGE_LOCK_PINS.items():
        needle = f"{name}=={version}"
        if needle not in lock_text:
            mismatches.append(f"{name}: training-image lock does not contain {needle}")
    for name, (version, digest) in IMAGE_ABSENT_PINS.items():
        if f"{name}==" not in lock_text:
            mismatches.append(
                f"{name}: absent from training-image/requirements.lock. "
                f"The N-14 overlay records {version} wheel sha256 {digest} "
                "for this torch build. Digest 8eb91965 was not readable "
                "anonymously, so this is the overlay record, not a manifest read-back."
            )
        else:
            mismatches.append(
                f"{name}: training-image lock contains a pin; the expected absence is gone"
            )
    for name, version in ROOT_PACKAGE_PINS.items():
        image = IMAGE_LOCK_PINS[name]
        in_root = f"{name}=={version}" in root_text
        in_project = f"{name}=={version}" in project_text
        if in_root and in_project and version != image:
            mismatches.append(
                f"{name}: requirements.txt and pyproject.toml pin {version}; "
                f"image pins {image}"
            )
        elif in_root and version != image:
            mismatches.append(
                f"{name}: requirements.txt pins {version}; image pins {image}"
            )
        elif in_project and version != image:
            mismatches.append(
                f"{name}: pyproject.toml pins {version}; image pins {image}"
            )
    if "torch>=" in root_text or "torch>=" in project_text:
        mismatches.append(
            "torch: requirements.txt and pyproject.toml do not pin 2.14.1+cu126 "
            "(image pin). The CPU dry-run installs torch==2.14.1+cpu."
        )
    return mismatches


def cpu_dry_run(work_dir) -> dict:
    """Random tiny NemotronH, LoRA on the detected modules, two CPU steps, save."""

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    previous_metrics = os.environ.get(METRICS_ENV)
    previous_request = os.environ.get(CHECKPOINT_REQUEST_ENV)
    os.environ[METRICS_ENV] = str(work / "metrics.jsonl")
    os.environ.pop(CHECKPOINT_REQUEST_ENV, None)
    try:
        return _cpu_dry_run(work)
    finally:
        if previous_metrics is None:
            os.environ.pop(METRICS_ENV, None)
        else:
            os.environ[METRICS_ENV] = previous_metrics
        if previous_request is None:
            os.environ.pop(CHECKPOINT_REQUEST_ENV, None)
        else:
            os.environ[CHECKPOINT_REQUEST_ENV] = previous_request


def _cpu_dry_run(work: Path) -> dict:
    import torch
    from transformers import (
        NemotronHConfig,
        NemotronHForCausalLM,
        PreTrainedTokenizerFast,
    )
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    notice = work / "NOTICE"
    notice.write_text(
        "NVIDIA Nemotron Open Model License notice. Retain this file.\n",
        encoding="utf-8",
    )
    base = {
        "licenseId": LICENSE_ID,
        "licenseAcknowledgement": (
            "I accept the pinned NVIDIA Nemotron Open Model License and "
            "preserve the NOTICE file."
        ),
    }
    preflight_license(base, notice)
    config = NemotronHConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=2,
        ssm_state_size=8,
        conv_kernel=4,
        chunk_size=8,
        num_nextn_predict_layers=0,
        hybrid_override_pattern="M*-",
        mlp_hidden_act="silu",
        use_cache=False,
    )
    detected = lora_target_modules(config)
    torch.manual_seed(0)
    model = NemotronHForCausalLM(config)
    model.config.use_cache = False
    assert_lora_modules_present(model, detected)
    adapted = attach_lora(
        model, r=4, lora_alpha=8, lora_dropout=0.0, target_modules=detected
    )
    adapted.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in adapted.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    losses = []
    for step in (1, 2):
        if step == 2:
            request_dir = work / "checkpoints" / "0002-dry-run"
            request = {
                "seq": 2,
                "label": "0002-dry-run",
                "dir": str(request_dir.resolve()),
            }
            request_path = work / "checkpoint.request"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            os.environ[CHECKPOINT_REQUEST_ENV] = str(request_path)
        optimizer.zero_grad(set_to_none=True)
        output = adapted(input_ids=input_ids, labels=input_ids.clone())
        loss = output.loss
        if loss is None or not math.isfinite(float(loss.detach())):
            raise NemotronHError("dry-run loss is not finite")
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        append_eval_record(step, value)
        honor_checkpoint_request(lambda directory: adapted.save_pretrained(directory))
    lora_updated = False
    out_proj_updated = False
    for name, parameter in adapted.named_parameters():
        if "lora_B" not in name or parameter.detach().abs().sum().item() == 0:
            continue
        lora_updated = True
        if ".out_proj." in f".{name}.":
            out_proj_updated = True
    if not lora_updated or not out_proj_updated:
        raise NemotronHError("LoRA weights did not move during the dry-run")
    adapter_dir = work / "adapter"
    adapted.save_pretrained(adapter_dir)
    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise NemotronHError("adapter_model.safetensors is absent")
    vocab = {"<pad>": 0, "<eos>": 1}
    backend = Tokenizer(WordLevel(vocab, unk_token="<pad>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<pad>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<pad>",
    )
    tokenizer.chat_template = (
        "{% if enable_thinking is not defined %}"
        "{% set enable_thinking = true %}{% endif %}"
        "{{ messages[0]['content'] }}|thinking={{ enable_thinking }}"
    )
    rendered = {}
    for flag in (True, False):
        rendered[str(flag).lower()] = tokenizer.apply_chat_template(
            messages_for_thinking([{"role": "user", "content": "hello"}], flag),
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=flag,
        )
    from trl import SFTConfig

    bound = SFTConfig(
        output_dir=str(work / "sft-bind"),
        max_length=8,
        max_steps=2,
        eval_strategy="steps",
        eval_steps=1,
        report_to=[],
        bf16=False,
        fp16=False,
    )
    if bound.max_length != 8 or bound.eval_strategy != "steps":
        raise NemotronHError("TRL did not bind max_length and eval_strategy")
    metrics = (work / "metrics.jsonl").read_text(encoding="utf-8")
    return {
        "detected_modules": detected,
        "losses": losses,
        "global_steps": 2,
        "adapter_config": str(adapter_dir / "adapter_config.json"),
        "adapter_target_modules": list(adapter_config.get("target_modules") or []),
        "adapter_safetensors": True,
        "metrics": metrics,
        "checkpoint_complete": (
            work / "checkpoints" / "0002-dry-run" / "COMPLETE"
        ).is_file(),
        "thinking_on": rendered["true"],
        "thinking_off": rendered["false"],
        "lora_updated": True,
        "weights_downloaded": False,
        "gpu_job_launched": False,
        "quantization": "cpu-dry-run-is-fp32-lora; QLoRA remains --load-in-4bit on the worker",
    }
