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
    assert "ensure_private_repo(api, a.adapter_out, a.hf_token)" in src


# --- fail-closed visibility verification (create_repo(exist_ok=True) never flips an existing public repo) ---
import pytest

class _Info:
    def __init__(self, private): self.private = private

class VisApi:
    """Fake Hub: `state` is the repo's visibility as the Hub would report it."""
    def __init__(self, state=None, exists=True, info_exc=None, flip_to=None):
        self.state, self.exists, self.info_exc, self.flip_to = state, exists, info_exc, flip_to
        self.created = []
    def create_repo(self, repo_id, exist_ok=False, private=None, token=None, repo_type=None):
        self.created.append((repo_id, private))
        if not self.exists:           # a new repo honours private=True
            self.exists, self.state = True, private
        # an existing repo keeps its visibility, exactly like huggingface_hub
        if self.flip_to is not None:  # race: visibility flipped between create and verify
            self.state = self.flip_to
    def repo_info(self, repo_id, repo_type="model"):
        if self.info_exc: raise self.info_exc
        return _Info(self.state)

def test_new_repo_is_created_private_and_verified():
    api = VisApi(exists=False)
    sw.ensure_private_repo(api, "o/r", "t")
    assert api.created == [("o/r", True)]

def test_existing_private_repo_passes():
    sw.ensure_private_repo(VisApi(state=True), "o/r", "t")

def test_existing_public_repo_is_refused():
    with pytest.raises(sw.VisibilityError, match="not private"):
        sw.ensure_private_repo(VisApi(state=False), "o/r", "t")

def test_missing_private_field_is_refused():
    with pytest.raises(sw.VisibilityError):
        sw.ensure_private_repo(VisApi(state=None), "o/r", "t")

def test_inaccessible_repo_is_refused():
    with pytest.raises(sw.VisibilityError, match="cannot verify"):
        sw.ensure_private_repo(VisApi(state=True, info_exc=PermissionError("403")), "o/r", "t")

def test_network_error_is_refused():
    with pytest.raises(sw.VisibilityError, match="cannot verify"):
        sw.ensure_private_repo(VisApi(state=True, info_exc=ConnectionError("reset")), "o/r", "t")

def test_race_flip_to_public_after_create_is_refused():
    with pytest.raises(sw.VisibilityError):
        sw.ensure_private_repo(VisApi(exists=False, flip_to=False), "o/r", "t")

def test_every_upload_is_preceded_by_a_visibility_check():
    src = (pathlib.Path(__file__).resolve().parent.parent / "salad_worker.py").read_text()
    on_save = src.split("def on_save", 1)[1].split("trainer.add_callback", 1)[0]
    assert on_save.index("verify_private(api, a.adapter_out)") < on_save.index("api.upload_folder(")
    tail = src.split("trainer.save_model(final)", 1)[1]
    assert tail.index("verify_private(api, a.adapter_out)") < tail.index("api.upload_folder(")
    assert tail.index("ensure_private_repo(api, a.publish_final_to") < tail.rindex("api.upload_folder(")
