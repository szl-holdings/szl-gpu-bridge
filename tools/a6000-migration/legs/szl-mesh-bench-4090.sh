
set -e
export HF_HOME=/workspace/hf PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q transformers datasets peft trl accelerate huggingface_hub safetensors pynvml
python3 - <<'BENCH'
import os, time, json, hashlib, threading, subprocess, torch
import pynvml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from huggingface_hub import HfApi

TOK = os.environ["HF_TOKEN"]; api = HfApi(token=TOK)
SEED = 20261109; MAX_STEPS = 2000
torch.manual_seed(SEED)
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# ---- NVML energy sampler: integrate power.draw -> joules ----
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
_joules = [0.0]; _watts = []; _stop = [False]
def sampler():
    last = time.time()
    while not _stop[0]:
        try:
            w = pynvml.nvmlDeviceGetPower(handle) / 1000.0  # W
        except Exception:
            w = 0.0
        now = time.time()
        _joules[0] += w * (now - last); last = now
        _watts.append(w)
        time.sleep(1.0)
th = threading.Thread(target=sampler, daemon=True); th.start()

t0 = time.time()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16, device_map="auto")
ds = load_dataset("SZLHOLDINGS/abstention-doctrine-sft-v1", split="train").shuffle(seed=SEED).select(range(2000))
def fmt(ex):
    if "text" in ex: return {"text": ex["text"]}
    if "messages" in ex: return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
    if "prompt" in ex and "completion" in ex: return {"text": ex["prompt"] + ex["completion"]}
    return {"text": json.dumps(ex)}
ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c != "text"])

peft = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type="CAUSAL_LM")
cfg = SFTConfig(output_dir="/workspace/ckpt", max_seq_length=2048,
    per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    logging_steps=50, save_strategy="no", bf16=True, optim="adamw_torch",
    max_steps=MAX_STEPS, report_to=[], packing=True, seed=SEED)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft)

t_train = time.time()
trainer.train()
train_s = time.time() - t_train
_stop[0] = True; time.sleep(1.2)
joules = _joules[0]
avg_w = sum(_watts)/len(_watts) if _watts else 0

# tokens processed estimate: packed 2048-token seqs, batch 4 x accum 8 = 32 seqs/step
tokens_est = 32 * 2048 * trainer.state.global_step
logs = trainer.state.log_history
loss_curve = [{"step": l.get("step"), "loss": l.get("loss")} for l in logs if "loss" in l]

# honest carbon join: measured joules x published US grid average (EPA eGRID ~0.385 kgCO2/kWh)
KWH = joules / 3.6e6
CARBON_KG_PER_KWH = 0.385  # EPA eGRID US average; labeled average, NOT node-specific
receipt = {
  "schema": "szl.gpu-mesh-benchmark/v1",
  "gpu": gpu, "vram_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 0,
  "workload": {"model": "Qwen/Qwen3-1.7B", "method": "LoRA r32 a64 bf16 packing",
               "max_steps": MAX_STEPS, "steps_done": trainer.state.global_step,
               "corpus": "SZLHOLDINGS/abstention-doctrine-sft-v1 (2000 examples, seed 20261109)"},
  "throughput": {"steps_per_sec": round(trainer.state.global_step/train_s,4),
                 "sec_per_step": round(train_s/trainer.state.global_step,4),
                 "wall_seconds": round(train_s,1)},
  "energy": {
    "joules_measured": round(joules,1),
    "method": "NVML power.draw integrated at 1Hz (pynvml)",
    "avg_watts": round(avg_w,1),
    "kwh": round(KWH,6),
    "joules_per_token": round(joules/max(tokens_est,1),4),
    "tokens_estimated": tokens_est,
    "carbon_note": "MEASURED joules x PUBLISHED US grid average (EPA eGRID 0.385 kgCO2/kWh). Node-region carbon NOT claimed: SaladCloud node region undisclosed.",
    "co2_kg_estimated_us_average": round(KWH * CARBON_KG_PER_KWH, 6),
  },
  "loss": {"first": loss_curve[0]["loss"] if loss_curve else None,
           "last": loss_curve[-1]["loss"] if loss_curve else None},
  "provider": "SaladCloud", "torch": torch.__version__,
  "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
}
blob = json.dumps(receipt, indent=2)
receipt["receipt_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
api.upload_file(path_or_fileobj=json.dumps(receipt, indent=2).encode(),
                path_in_repo="benchmark/4090-leg.json",
                repo_id="SZLHOLDINGS/frontier-54-runs", repo_type="dataset")
print("BENCH_4090_DONE", json.dumps(receipt["throughput"]), json.dumps(receipt["energy"]))
BENCH
