#!/usr/bin/env python
"""runjob.py — verify a DSSE job spec, gate, train (Unsloth QLoRA), eval,
sign receipts, upload. Fail closed at every step; failures become honest
signed BLOCKED receipts wherever a signature is already established.

Exit codes: 0 = receipts uploaded (success OR honest BLOCKED); nonzero = local
infrastructure failure before receipts were possible (daemon retries).
"""
import base64
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(r"C:\szl-bridge")
PAYLOAD_TYPE = "application/vnd.szl.gpu-bridge.jobspec.v1+json"
ENGINE_PIN = json.loads((ROOT / "keys" / "engine_pubkey.json").read_text())


def canonicalize(v):
    """Sorted-keys, no-whitespace JSON — mirrors the cloud signer exactly."""
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pae(payload_type: str, payload: bytes) -> bytes:
    pt = payload_type.encode()
    return b"DSSEv1 %d %s %d %s" % (len(pt), pt, len(payload), payload)


def verify_envelope(env: dict) -> dict:
    """Verify BEFORE trusting any field. Returns the spec or raises."""
    from nacl.signing import VerifyKey

    if env.get("payloadType") != PAYLOAD_TYPE:
        raise ValueError(f"wrong payloadType {env.get('payloadType')!r}")
    spki = base64.b64decode(env["publicKeySpkiBase64"])
    key_id = hashlib.sha256(spki).hexdigest()[:16]
    if key_id != ENGINE_PIN["keyId"]:
        raise ValueError(f"envelope key {key_id} ≠ pinned engine {ENGINE_PIN['keyId']}")
    if env["publicKeySpkiBase64"] != ENGINE_PIN["publicKeySpkiBase64"]:
        raise ValueError("SPKI bytes differ from pin (keyId collision attempt?)")
    payload = base64.b64decode(env["payload"])
    sig = base64.b64decode(env["signatures"][0]["sig"])
    VerifyKey(spki[-32:]).verify(pae(PAYLOAD_TYPE, payload), sig)  # raises on bad sig
    return json.loads(payload)


def laptop_keys():
    from nacl.signing import SigningKey

    pem = (ROOT / "keys" / "laptop_key.pem").read_text()
    seed = base64.b64decode("".join(l for l in pem.splitlines() if "-----" not in l))
    sk = SigningKey(seed)
    pub = json.loads((ROOT / "keys" / "laptop_pubkey.json").read_text())
    return sk, pub


def sign_receipt(receipt: dict) -> dict:
    sk, pub = laptop_keys()
    body = canonicalize(receipt).encode()
    sig = sk.sign(body).signature
    return {
        "receipt": receipt,
        "signatureBase64": base64.b64encode(sig).decode(),
        "publicKeySpkiBase64": pub["publicKeySpkiBase64"],
        "keyId": pub["keyId"],
        "scheme": "ed25519-over-canonical-json",
    }


def upload_receipt(signed: dict, name: str, spec: dict):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(spec["outputs"]["receiptsRepoId"], repo_type="dataset", exist_ok=True, private=False)
    p = ROOT / "jobs" / name
    p.write_text(json.dumps(signed, indent=2))
    api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{spec['jobId']}/{name}", repo_id=spec["outputs"]["receiptsRepoId"], repo_type="dataset")
    print(f"receipt uploaded: {spec['jobId']}/{name}")


def blocked(spec: dict, stage: str, reason: str, extra=None):
    """Honest BLOCKED verdict — signed and uploaded, then exit 0 (job consumed)."""
    r = {
        "kind": "szl-bridge-blocked",
        "jobId": spec["jobId"],
        "verdict": "BLOCKED",
        "stage": stage,
        "reason": reason,
        "at": now_iso(),
        "extra": extra or {},
        "doctrine": {"failClosed": True, "note": "a refused job is a result, not an error to hide"},
    }
    upload_receipt(sign_receipt(r), "blocked_receipt.signed.json", spec)
    sys.exit(0)


def probe_vram_gb():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    )
    return int(out.strip().splitlines()[0]) / 1024.0


def main(spec_path: str):
    env = json.loads(pathlib.Path(spec_path).read_text())

    # 1 — verify signature before reading ANY field
    try:
        spec = verify_envelope(env)
    except Exception as e:
        # cannot sign a per-job BLOCKED without trusting jobId from an unverified
        # spec; log-only refusal, exit 0 so the daemon ledgers it as consumed.
        print(f"REFUSED (unverified spec): {e}")
        return 0

    # 2 — replay/expiry + gates (each failure = signed BLOCKED receipt)
    if datetime.fromisoformat(spec["expiresAt"].replace("Z", "+00:00")) < datetime.now(timezone.utc):
        blocked(spec, "expiry", f"spec expired at {spec['expiresAt']}")
    free_disk_gb = shutil.disk_usage(str(ROOT)).free / 1e9
    if free_disk_gb < spec["gates"]["minFreeDiskGb"]:
        blocked(spec, "gate:disk", f"free disk {free_disk_gb:.1f} GB < required {spec['gates']['minFreeDiskGb']} GB")
    try:
        free_vram = probe_vram_gb()
    except Exception as e:
        blocked(spec, "gate:vram-probe", f"nvidia-smi probe failed: {e}")
    if free_vram < spec["gates"]["minFreeVramGb"]:
        blocked(spec, "gate:vram", f"free VRAM {free_vram:.1f} GB < required {spec['gates']['minFreeVramGb']} GB")

    # 3 — dataset download + hash pin (REPORTED input, pinned)
    from huggingface_hub import hf_hub_download

    ds_path = hf_hub_download(
        repo_id=spec["dataset"]["repoId"], filename=spec["dataset"]["file"],
        repo_type="dataset", revision=spec["dataset"]["revision"],
    )
    got = hashlib.sha256(pathlib.Path(ds_path).read_bytes()).hexdigest()
    if got != spec["dataset"]["sha256"]:
        blocked(spec, "gate:dataset-hash", f"dataset sha {got[:12]}… ≠ pinned {spec['dataset']['sha256'][:12]}…")

    # 4 — train (Unsloth QLoRA; NaN loss aborts to BLOCKED)
    t0 = time.time()
    r = spec["recipe"]
    from unsloth import FastLanguageModel  # noqa — imported late so gates run without GPU deps
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec["base"]["repoId"], revision=spec["base"]["revision"],
        max_seq_length=r["maxSeqLength"], load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=r["loraR"], lora_alpha=r["loraAlpha"], lora_dropout=r["loraDropout"],
        target_modules=r["targetModules"], use_gradient_checkpointing=r["gradientCheckpointing"],
        random_state=r["seed"],
    )
    full = load_dataset("json", data_files=ds_path, split="train")
    split = full.train_test_split(test_size=spec["eval"]["heldOutFraction"], seed=spec["eval"].get("seed", 7))
    train_ds, eval_ds = split["train"], split["test"]

    def to_text(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

    train_ds = train_ds.map(to_text, remove_columns=[c for c in train_ds.column_names if c != "text"])
    eval_txt = eval_ds.map(to_text, remove_columns=[c for c in eval_ds.column_names if c != "text"])

    class WallclockNan:
        def __init__(self, max_min): self.deadline = t0 + max_min * 60
        def __call__(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs and (logs["loss"] != logs["loss"]):
                blocked(spec, "train:nan-loss", f"NaN loss at step {state.global_step}")
            if time.time() > self.deadline:
                blocked(spec, "train:wallclock", f"exceeded {spec['gates']['maxWallclockMinutes']} min")
            return control

    from transformers import TrainerCallback

    class GateCallback(TrainerCallback):
        def __init__(self): self.g = WallclockNan(spec["gates"]["maxWallclockMinutes"])
        def on_log(self, args, state, control, logs=None, **kw): return self.g(args, state, control, logs, **kw)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=eval_txt,
        callbacks=[GateCallback()],
        args=SFTConfig(
            per_device_train_batch_size=r["batchSize"], gradient_accumulation_steps=r["gradAccum"],
            num_train_epochs=r["epochs"], learning_rate=r["learningRate"], optim=r["optimizer"],
            seed=r["seed"], output_dir=str(ROOT / "jobs" / spec["jobId"] / "out"),
            logging_steps=10, save_strategy="epoch", report_to=[], bf16=torch.cuda.is_bf16_supported(),
        ),
    )
    train_out = trainer.train()
    train_minutes = (time.time() - t0) / 60
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    # 5 — eval (MEASURED on held-out split; doctrine-form checks on generations)
    eval_metrics = trainer.evaluate()
    FastLanguageModel.for_inference(model)
    import re
    n_gen, json_ok, ceiling_ok, abstain_seen = 0, 0, 0, 0
    for ex in eval_ds.select(range(min(50, len(eval_ds)))):
        prompt = tokenizer.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=400, do_sample=False)
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        n_gen += 1
        try:
            obj = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            json_ok += 1
            conv = obj.get("conviction")
            if conv is None or (isinstance(conv, (int, float)) and conv <= 0.97):
                ceiling_ok += 1
            if obj.get("action") == "ABSTAIN":
                abstain_seen += 1
        except Exception:
            pass

    # 6 — signed receipts (forge trio convention, laptop key), then upload
    adapter_dir = ROOT / "jobs" / spec["jobId"] / "adapter"
    model.save_pretrained(str(adapter_dir)); tokenizer.save_pretrained(str(adapter_dir))
    adapter_sha = hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()

    training_receipt = {
        "kind": "szl-bridge-training-receipt", "v": 1, "jobId": spec["jobId"],
        "specPayloadSha256": hashlib.sha256(base64.b64decode(env["payload"])).hexdigest(),
        "base": spec["base"], "dataset": spec["dataset"], "recipe": r,
        "measured": {
            "label": "MEASURED",
            "finalTrainLoss": float(train_out.training_loss),
            "trainMinutes": round(train_minutes, 2),
            "peakVramGb": round(peak_vram_gb, 2),
            "steps": int(train_out.global_step),
        },
        "adapterSha256": adapter_sha, "at": now_iso(),
        "stack": {"note": "pinned via bootstrap.ps1; unsloth as dependency (Apache core, LGPL zoo unvendored)"},
    }
    signed_training = sign_receipt(training_receipt)
    tr_sha = hashlib.sha256(canonicalize(signed_training).encode()).hexdigest()

    eval_receipt = {
        "kind": "szl-bridge-eval-receipt", "v": 1, "jobId": spec["jobId"],
        "trainingReceiptSha256": tr_sha,  # eval→training chain (forge convention)
        "suite": spec["eval"]["suite"],
        "measured": {
            "label": "MEASURED",
            "heldOutLoss": float(eval_metrics.get("eval_loss", float("nan"))),
            "heldOutRows": len(eval_ds),
            "generationsChecked": n_gen,
            "jsonValidRate": (json_ok / n_gen) if n_gen else None,
            "ceilingRespectRate": (ceiling_ok / n_gen) if n_gen else None,
            "abstainGenerations": abstain_seen,
        },
        "limits": "single held-out split; n is small; form-checks measure doctrine compliance, not market skill",
        "at": now_iso(),
    }
    signed_eval = sign_receipt(eval_receipt)

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(spec["outputs"]["modelRepoId"], exist_ok=True, private=spec["outputs"].get("private", True))
    api.upload_folder(folder_path=str(adapter_dir), repo_id=spec["outputs"]["modelRepoId"])
    for name, obj in [("training_receipt.signed.json", signed_training), ("eval_receipt.signed.json", signed_eval)]:
        p = ROOT / "jobs" / name
        p.write_text(json.dumps(obj, indent=2))
        api.upload_file(path_or_fileobj=str(p), path_in_repo=name, repo_id=spec["outputs"]["modelRepoId"])
        upload_receipt(obj, name, spec)
    (ROOT / "keys" / "laptop_pubkey.json").exists() and api.upload_file(
        path_or_fileobj=str(ROOT / "keys" / "laptop_pubkey.json"), path_in_repo="owner_pubkey.json",
        repo_id=spec["outputs"]["modelRepoId"])
    print(f"job {spec['jobId']} COMPLETE — adapter + receipt trio uploaded")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
