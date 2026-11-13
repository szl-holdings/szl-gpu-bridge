# Offline: stub torch/transformers/trl/peft/datasets so the module imports without GPU deps.
import sys, types, os, fnmatch, pathlib, importlib.util
for name in ["torch", "transformers", "trl", "peft", "datasets", "huggingface_hub"]:
    m = types.ModuleType(name); sys.modules[name] = m
sys.modules["transformers"].AutoModelForCausalLM = object; sys.modules["transformers"].AutoTokenizer = object
sys.modules["trl"].SFTConfig = object; sys.modules["trl"].SFTTrainer = object
sys.modules["peft"].LoraConfig = object; sys.modules["datasets"].load_dataset = None
sys.modules["huggingface_hub"].HfApi = object
spec = importlib.util.spec_from_file_location("sw", pathlib.Path(__file__).resolve().parent.parent / "salad_worker.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)

CHECKPOINT_FILES = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "rng_state.pth",
                    "scheduler.pt", "training_args.bin", "tokenizer.json", "trainer_state.json"]

def test_pickle_patterns_exclude_trainer_state_but_keep_adapter():
    excluded = {f for f in CHECKPOINT_FILES if any(fnmatch.fnmatch(f, p) for p in sw.PICKLE_PATTERNS)}
    assert excluded == {"optimizer.pt", "rng_state.pth", "scheduler.pt", "training_args.bin"}

class FakeApi:
    def __init__(self, files): self.files = files; self.calls = []
    def list_repo_files(self, repo_id, repo_type): return self.files
    def snapshot_download(self, **kw):
        self.calls.append(kw); os.makedirs(os.path.join(kw["local_dir"], kw["allow_patterns"][0].split("/")[0]), exist_ok=True)

def test_resume_downloads_newest_checkpoint_to_local_dir(tmp_path):
    api = FakeApi(["checkpoint-50/a", "checkpoint-150/a", "checkpoint-100/a", "final/x"])
    got = sw.fetch_resume_checkpoint(api, "r", str(tmp_path))
    assert got == str(tmp_path / "checkpoint-150") and api.calls[0]["allow_patterns"] == ["checkpoint-150/*"]

def test_no_checkpoint_means_genesis(tmp_path):
    assert sw.fetch_resume_checkpoint(FakeApi(["final/x"]), "r", str(tmp_path)) is None

def test_working_repo_created_private():
    src = (pathlib.Path(__file__).resolve().parent.parent / "salad_worker.py").read_text()
    assert "create_repo(a.adapter_out, exist_ok=True, private=True" in src
