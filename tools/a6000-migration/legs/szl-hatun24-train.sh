
set -e
export HF_HOME=${HF_HOME:-$HOME/szl-legs/hf-cache} PIP_BREAK_SYSTEM_PACKAGES=1
mkdir -p $HOME/szl-workspace/out
pip install -q --upgrade pip
pip install -q transformers datasets peft trl accelerate bitsandbytes huggingface_hub safetensors
python3 - <<'TRAIN'
import os, time, json, torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
BASE = "Qwen/Qwen3-1.7B"
CANDIDATE = "SZLHOLDINGS/hatun-24-doctrine-math-r1-candidate"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
ds_math = load_dataset("SZLHOLDINGS/ancient-math-reasoning-sft-v1", split="train")
ds_doc  = load_dataset("SZLHOLDINGS/abstention-doctrine-sft-v1", split="train")
try:
    ds_gov = load_dataset("SZLHOLDINGS/governed-distill-4b-sft-v3", split="train")
    ds = concatenate_datasets([ds_math, ds_doc, ds_gov])
except Exception as e:
    print("gov dataset unavailable, using 2:", str(e)[:120])
    ds = concatenate_datasets([ds_math, ds_doc])
print("corpus:", len(ds), "examples")
def fmt(ex):
    if "text" in ex: return {"text": ex["text"]}
    if "messages" in ex: return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
    if "prompt" in ex and "completion" in ex: return {"text": ex["prompt"] + ex["completion"]}
    return {"text": json.dumps(ex)}
ds = ds.map(fmt, remove_columns=ds.column_names)
peft_cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type="CAUSAL_LM")
cfg = SFTConfig(
    output_dir=os.path.expanduser("~/szl-workspace/out/hatun24"),
    max_length=2048,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_steps=500,
    save_total_limit=6,
    bf16=True,
    optim="adamw_torch",
    num_train_epochs=8,
    report_to=[],
    packing=True,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg)
trainer.train()
print("TRAIN_DONE hours=", round((time.time()-t0)/3600, 2))
from huggingface_hub import HfApi, create_repo
api = HfApi(token=os.environ["HF_TOKEN"])
create_repo(CANDIDATE, private=False, exist_ok=True, token=os.environ["HF_TOKEN"])
trainer.save_model(os.path.expanduser("~/szl-workspace/out/hatun24/final"))
api.upload_folder(folder_path=os.path.expanduser("~/szl-workspace/out/hatun24/final"), repo_id=CANDIDATE, repo_type="model")
report = {"run":"hatun-24","base":BASE,"corpus_examples":len(ds),
         "datasets":["ancient-math-reasoning-sft-v1","abstention-doctrine-sft-v1","governed-distill-4b-sft-v3"],
         "wall_hours": round((time.time()-t0)/3600,2),
         "gpu":"RTX4090-24GB","method":"LoRA r32 a64 bf16","started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))}
open(os.path.expanduser("~/szl-workspace/out/hatun24/training_report.json"),"w").write(json.dumps(report, indent=2))
api.upload_file(path_or_fileobj=os.path.expanduser("~/szl-workspace/out/hatun24/training_report.json"), path_in_repo="training_report.json", repo_id=CANDIDATE, repo_type="model")
print("PUBLISHED", CANDIDATE)
TRAIN
echo HATUN24_COMPLETE
