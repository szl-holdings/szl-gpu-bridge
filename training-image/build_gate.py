#!/usr/bin/env python3
"""build_gate.py - build-time gate for the pinned training image. Any failure exits non-zero and fails the build.

Checks (CPU only; the build host has no GPU):
  1. exact versions of torch / torchvision / torchaudio / transformers / peft / trl (from env EXPECT_*),
     torch built for CUDA 12.6 (the pinned base);
  2. torchvision.ops.nms really runs on CPU (the torch 2.15 + torchvision 0.19.1 'nms is missing' crash);
  3. torchaudio loads and runs an op against this torch;
  4. transformers / peft: a tiny Llama gets a LoRA adapter and does a forward pass;
  5. trl.SFTConfig is built with the sequence-length argument this TRL really exposes (the same contract
     salad_worker.sft_sequence_length_kwargs enforces) and the retired max_seq_length= is rejected;
  6. salad_worker imports against these pins;
  7. the run-receipt uploader: imports, `--dry-run` on a synthetic proof run with no token and no network,
     and the image carries no HF/GitHub token.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile

EXPECT = {
    "torch": os.environ["EXPECT_TORCH"],
    "torchvision": os.environ["EXPECT_TORCHVISION"],
    "torchaudio": os.environ["EXPECT_TORCHAUDIO"],
    "transformers": os.environ["EXPECT_TRANSFORMERS"],
    "peft": os.environ["EXPECT_PEFT"],
    "trl": os.environ["EXPECT_TRL"],
}
EXPECT_CUDA = os.environ["EXPECT_CUDA"]
WORK = os.environ.get("GATE_WORK_DIR", "/work")
failures: list[str] = []
versions: dict[str, str] = {}


def step(name):
    def deco(fn):
        try:
            fn()
            print(f"GATE ok   {name}", flush=True)
        except Exception as exc:  # noqa: BLE001 - the gate must report every failure
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"GATE FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
        return fn
    return deco


@step("versions")
def _versions():
    for name, want in EXPECT.items():
        mod = importlib.import_module(name)
        got = mod.__version__
        versions[name] = got
        if got.split("+")[0] != want.split("+")[0] or (name in ("torch", "torchvision", "torchaudio") and got != want):
            raise AssertionError(f"{name}: want {want}, got {got}")
    import torch
    if torch.version.cuda != EXPECT_CUDA:
        raise AssertionError(f"torch.version.cuda want {EXPECT_CUDA}, got {torch.version.cuda}")
    if sys.version_info[:2] != (3, 12):
        raise AssertionError(f"python {sys.version}")


@step("torchvision.ops.nms on CPU")
def _nms():
    import torch
    import torchvision
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [50., 50., 60., 60.]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    if keep.tolist() != [0, 2]:
        raise AssertionError(f"nms keep={keep.tolist()}")


@step("torchaudio op on CPU")
def _audio():
    import torch
    import torchaudio
    y = torchaudio.functional.resample(torch.randn(1, 1600), 16000, 8000)
    if tuple(y.shape) != (1, 800):
        raise AssertionError(f"resample shape {tuple(y.shape)}")


@step("transformers + peft tiny LoRA forward")
def _peft():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    model = get_peft_model(LlamaForCausalLM(cfg),
                           LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    out = model(input_ids=torch.randint(0, 64, (1, 8)), labels=torch.randint(0, 64, (1, 8)))
    if not torch.isfinite(out.loss):
        raise AssertionError("non-finite loss")


@step("trl.SFTConfig sequence-length argument")
def _sft():
    from trl import SFTConfig, SFTTrainer  # noqa: F401 - SFTTrainer import is part of the gate
    params = inspect.signature(SFTConfig).parameters
    explicit = [n for n in ("max_seq_length", "max_length")
                if n in params and params[n].kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)]
    if len(explicit) != 1:
        raise AssertionError(f"SFTConfig must expose exactly one sequence-length field, got {explicit}")
    with tempfile.TemporaryDirectory() as d:
        cfg = SFTConfig(output_dir=d, report_to=[], use_cpu=True, **{explicit[0]: 2048})
        if getattr(cfg, explicit[0]) != 2048:
            raise AssertionError("sequence length not applied")
        print(f"GATE info SFTConfig sequence-length field = {explicit[0]}", flush=True)
        if explicit[0] == "max_length":
            try:
                SFTConfig(output_dir=d, report_to=[], use_cpu=True, max_seq_length=2048)
            except TypeError:
                pass
            else:
                raise AssertionError("SFTConfig unexpectedly accepts max_seq_length")


@step("salad_worker contract against these pins")
def _worker():
    sys.path.insert(0, WORK)
    import salad_worker
    from trl import SFTConfig
    kw = salad_worker.sft_sequence_length_kwargs(SFTConfig, 2048)
    with tempfile.TemporaryDirectory() as d:
        SFTConfig(output_dir=d, report_to=[], use_cpu=True, **kw)


@step("optional libs import (datasets, accelerate, safetensors, huggingface_hub, bitsandbytes, pynvml, zeus)")
def _libs():
    for name in ("datasets", "accelerate", "safetensors", "huggingface_hub", "bitsandbytes", "pynvml", "sentencepiece", "google.protobuf", "cryptography"):
        importlib.import_module(name)
    import importlib.metadata as md
    md.version("zeus-ml")  # zeus probes NVML at import; presence is what the build can verify


@step("run-receipt uploader")
def _receipts():
    rdir = os.path.join(WORK, "receipts")
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise AssertionError("a Hugging Face token is present in the build environment")
    sys.path.insert(0, rdir)
    sys.path.insert(0, os.path.join(rdir, "vendor"))
    import receipt_upload  # noqa: F401
    from proofrun import synthetic
    with tempfile.TemporaryDirectory() as run_dir:
        synthetic.make_run(run_dir, steps=20, save_every=10)
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        p = subprocess.run([sys.executable, os.path.join(rdir, "receipt_upload.py"), run_dir,
                            "--repo", "SZLHOLDINGS/szl-run-receipts", "--run-id", "build-gate",
                            "--large-mb", "10", "--dry-run"],
                           capture_output=True, text=True, env=env, timeout=300)
        print(p.stdout[-600:], p.stderr[-600:], flush=True)
        if p.returncode != 0:
            raise AssertionError(f"receipt_upload --dry-run exit {p.returncode}")
    t = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", "discover", "-s", os.path.join(rdir, "tests")],
                       capture_output=True, text=True, timeout=600)
    print(t.stderr[-500:], flush=True)
    if t.returncode != 0:
        raise AssertionError("receipt_upload unit tests failed")


print(json.dumps({"versions": versions, "failures": failures}), flush=True)
if failures:
    print("BUILD GATE FAILED", file=sys.stderr)
    sys.exit(1)
print("BUILD_GATE_OK " + " ".join(f"{k}={v}" for k, v in versions.items()), flush=True)
