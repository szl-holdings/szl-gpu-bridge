#!/usr/bin/env python3
"""train_khipu_abstain.py — honest retrain of the SZL-Khipu-1.5B-abstain curriculum.

PROPOSAL-GRADE CODE. Runs only on owner metal under a signed szl-gpu-bridge
jobspec. This script:
  - FAILS CLOSED if any dataset file's sha256 does not match the jobspec;
  - never touches the held-out adversarial refusal set during training;
  - emits an UNSIGNED-honest training receipt (owner signs it out-of-band);
  - reports energy as MEASURED only if pynvml works, else UNAVAILABLE (null).

Doctrine v11. Λ = Conjecture 1. Never overwrite the abstain lineage repo.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def energy_joules_start():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None

def read_energy_joules(nvml, handle):
    if nvml is None or handle is None:
        return None
    try:
        return nvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0
    except Exception:
        return None

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobspec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    spec = json.loads(args.jobspec.read_text())
    if spec.get("signature_state") != "SIGNED":
        print("FAIL-CLOSED: jobspec is not owner-signed (signature_state != SIGNED)", file=sys.stderr)
        return 2

    # Preflight: every dataset byte-verified before any token is trained on.
    datasets = {}
    for entry in spec["datasets"]:
        p = Path(entry["path"])
        actual = sha256_file(p)
        if actual != entry["sha256"]:
            print(f"FAIL-CLOSED: {p} sha256 mismatch: {actual} != {entry['sha256']}", file=sys.stderr)
            return 3
        datasets[entry["role"]] = p
    if "heldout_refusals" in datasets:
        heldout_sha = sha256_file(datasets["heldout_refusals"])
    else:
        print("FAIL-CLOSED: no held-out refusal set declared", file=sys.stderr)
        return 4

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base = spec["base_model"]
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")

    train = load_dataset("json", data_files=str(datasets["train"]), split="train")

    def to_response_only(ex):
        # Loss on the assistant span only; prompt tokens masked to -100.
        prompt = tok.apply_chat_template([ex["messages"][0]], tokenize=True, add_generation_prompt=True)
        full = tok.apply_chat_template(ex["messages"], tokenize=True)
        labels = [-100] * len(prompt) + full[len(prompt):]
        return {"input_ids": full, "labels": labels}

    train = train.map(to_response_only, remove_columns=train.column_names)

    nvml = energy_joules_start()
    handle = nvml.nvmlDeviceGetHandleByIndex(0) if nvml else None
    j0, t0 = read_energy_joules(nvml, handle), time.time()

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(args.out), per_device_train_batch_size=spec.get("batch_size", 4),
            num_train_epochs=spec.get("epochs", 2), learning_rate=spec.get("lr", 2e-4),
            logging_steps=10, save_strategy="no", report_to=[],
        ),
        train_dataset=train,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear"),
    )
    result = trainer.train()
    trainer.save_model(args.out)

    j1, t1 = read_energy_joules(nvml, handle), time.time()
    energy = None if (j0 is None or j1 is None) else round(j1 - j0, 3)

    receipt = {
        "kind": "szl.training-receipt/v1",
        "signature_state": "UNSIGNED",
        "signature": None,
        "base_model": base,
        "datasets": {e["role"]: {"path": e["path"], "sha256": e["sha256"]} for e in spec["datasets"]},
        "heldout_refusals_sha256": heldout_sha,
        "final_train_loss": result.training_loss,
        "energy_joules": energy,
        "energy_state": "MEASURED" if energy is not None else "UNAVAILABLE",
        "wall_seconds": round(t1 - t0, 3),
        "notes": "Owner signs this canonical JSON out-of-band; eval receipt chains to sha256 of the signed training receipt.",
    }
    (args.out / "training_receipt.unsigned.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({"ok": True, "loss": result.training_loss, "energy_state": receipt["energy_state"]}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
