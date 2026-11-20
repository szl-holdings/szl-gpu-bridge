"""Exercise real worker terminal branches with synthetic, network-denied adapters."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

WORKER = Path(__file__).resolve().parents[1] / "tools" / "salad" / "salad_worker.py"


class Dataset:
    column_names = ["text"]

    def __len__(self):
        return 8

    def select(self, rows):
        return self

    def map(self, function, remove_columns, **kwargs):
        return self


class TerminalContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {"HF_TOKEN": ""}, clear=True))
        self.stack.enter_context(
            patch.object(
                socket.socket,
                "connect",
                side_effect=AssertionError("network forbidden"),
            )
        )
        self.stack.enter_context(patch.object(signal, "signal"))
        self.events = []
        self.uploads = []
        self.uploaded_file_snapshots = []
        self.saved = []
        self.trained = []
        self.api_calls = []
        self.loaded = []
        self.receipt_uploads = []
        self.receipt_downloads = []
        self.receipt_bytes = {}
        self.receipt_failures = []
        self.fail_receipt_event = None
        self.readback_error = None
        self.corrupt_readback = False
        self.receipt_revision = "b" * 40
        self.flip_receipt_visibility = False
        self.invoke_heartbeat = False
        self.invoke_checkpoint = False
        self.trainer_configs = []
        self.trainer_inputs = []
        self.callbacks = []
        self.actual = 200
        self.interrupted = False
        self.final_present = True
        self.flip_visibility_on_list = False
        self.upload_error = None
        self.fail_upload_repo = None
        self.private = True
        self.lose_visibility_on_train = False
        fixture = self

        class API:
            def __init__(self, **kwargs):
                fixture.api_calls.append("api")

            def create_repo(self, *args, **kwargs):
                fixture.api_calls.append("create")

            def repo_info(self, *args, **kwargs):
                fixture.api_calls.append("visibility")
                return types.SimpleNamespace(private=fixture.private)

            def list_repo_files(self, *args, **kwargs):
                if fixture.flip_visibility_on_list:
                    fixture.private = False
                return (
                    ["final/adapter_model.safetensors"] if fixture.final_present else []
                )

            def upload_folder(self, **kwargs):
                directory = Path(kwargs.get("folder_path", ""))
                fixture.uploaded_file_snapshots.append(
                    {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
                    if directory.is_dir()
                    else None
                )
                fixture.uploads.append(kwargs)
                if fixture.upload_error and (
                    fixture.fail_upload_repo is None
                    or kwargs["repo_id"] == fixture.fail_upload_repo
                ):
                    raise fixture.upload_error
                return types.SimpleNamespace(oid="a" * 40)

            def upload_file(self, **kwargs):
                fixture.receipt_uploads.append(kwargs)
                value = json.loads(kwargs["path_or_fileobj"])
                if value["event"] == fixture.fail_receipt_event:
                    error = PermissionError(
                        "synthetic-sensitive-url-must-not-be-logged"
                    )
                    error.response = types.SimpleNamespace(status_code=403)
                    raise error
                if fixture.receipt_failures:
                    error = fixture.receipt_failures.pop(0)
                    if error:
                        raise error
                fixture.receipt_bytes[kwargs["path_in_repo"]] = kwargs[
                    "path_or_fileobj"
                ]
                if fixture.flip_receipt_visibility:
                    fixture.private = False
                return types.SimpleNamespace(oid=fixture.receipt_revision)

        def fetch(url, token, max_bytes):
            # v3.11 seam: the worker readback goes through its module-level
            # _readback_fetch(url, token, max_bytes) network boundary; this
            # fixture substitutes it (no socket may be opened in tests).
            fixture.receipt_downloads.append(
                {"url": url, "token": token, "max_bytes": max_bytes}
            )
            if fixture.readback_error:
                raise fixture.readback_error
            path_in_repo = url[url.index("mesh/") :]
            content = fixture.receipt_bytes[path_in_repo]
            if fixture.corrupt_readback:
                content += b"corrupt"
            return content[:max_bytes]

        class Trainer:
            def __init__(self, **kwargs):
                self.state = types.SimpleNamespace(global_step=fixture.actual)
                self.args = kwargs["args"]
                fixture.trainer_configs.append(kwargs["args"])
                fixture.trainer_inputs.append(kwargs)

            def add_callback(self, callback):
                self.callback = callback
                fixture.callbacks.append(callback)

            def train(self, **kwargs):
                fixture.trained.append(kwargs)
                if fixture.lose_visibility_on_train:
                    fixture.private = False
                if fixture.interrupted:
                    fixture.worker.STOP["flag"] = True
                fixture.control = types.SimpleNamespace(
                    should_training_stop=False, should_save=False
                )
                if fixture.invoke_heartbeat:
                    self.callback.last_heartbeat = -1000
                    self.callback.on_step_end(self.args, self.state, fixture.control)
                if fixture.invoke_checkpoint or fixture.control.should_save:
                    (Path(self.args.output_dir) / f"checkpoint-{fixture.actual}").mkdir(
                        parents=True, exist_ok=True
                    )
                    self.callback.on_save(self.args, self.state, fixture.control)
                return types.SimpleNamespace(global_step=fixture.actual)

            def save_model(self, path):
                fixture.saved.append(path)

        modules = {
            name: types.ModuleType(name)
            for name in (
                "torch",
                "transformers",
                "trl",
                "peft",
                "datasets",
                "huggingface_hub",
            )
        }
        modules["torch"].cuda = types.SimpleNamespace(
            is_available=lambda: True, get_device_name=lambda _: "SYNTHETIC"
        )
        modules["torch"].bfloat16 = "SYNTHETIC"
        modules["transformers"].AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: object()
        )
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(use_cache=True))
        modules["transformers"].AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: fixture.loaded.append(True) or self.model
        )
        modules["transformers"].TrainerCallback = object

        def config(*, max_length, **kwargs):
            # Emulate modern TRL with an explicit required sequence keyword.
            # A **kwargs-only double would mask unsupported worker arguments.
            fixture.config_kwargs = {"max_length": max_length, **kwargs}
            return types.SimpleNamespace(**fixture.config_kwargs)

        modules["trl"].SFTConfig = config
        modules["trl"].SFTTrainer = Trainer
        modules["peft"].LoraConfig = lambda **kwargs: kwargs
        modules["datasets"].load_dataset = lambda *a, **k: Dataset()
        modules["huggingface_hub"].HfApi = API

        self.stack.enter_context(patch.dict(sys.modules, modules))
        spec = importlib.util.spec_from_file_location(
            "salad_terminal_under_test", WORKER
        )
        self.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.worker)
        self.original_receipt = self.worker.build_receipt
        self.real_receipt = self.worker.receipt
        self.sleep = self.stack.enter_context(patch.object(self.worker.time, "sleep"))
        # v3.11: substitute the receipt-readback network boundary.
        self.stack.enter_context(patch.object(self.worker, "_readback_fetch", fetch))
        self.stack.enter_context(
            patch.object(
                self.worker,
                "receipt",
                side_effect=lambda kind, detail, extra=None: self.events.append(
                    {"event": kind, "detail": detail, **(extra or {})}
                ),
            )
        )
        self.stack.enter_context(
            patch.object(self.worker, "fetch_resume_checkpoint", return_value=None)
        )
        self.stack.enter_context(
            patch.object(self.worker, "load_train_split", return_value=Dataset())
        )
        self.stack.enter_context(patch.object(self.worker, "cuda_arch_precheck"))
        self.stack.enter_context(
            patch.object(
                self.worker,
                "sample_energy_j",
                return_value={"label": "UNAVAILABLE", "energy_j": None},
            )
        )
        self.stack.enter_context(
            patch.object(
                self.worker, "zeus_begin_window", return_value={"label": "UNAVAILABLE"}
            )
        )
        self.stack.enter_context(
            patch.object(
                self.worker, "zeus_end_window", return_value={"label": "UNAVAILABLE"}
            )
        )
        self.stack.enter_context(
            patch.object(
                self.worker.os.path,
                "expanduser",
                side_effect=lambda p: str(Path(self.temp.name) / p.removeprefix("~/")),
            )
        )

    def run_worker(self, *extra):
        argv = [
            "salad_worker.py",
            "--base",
            "fixture/base",
            "--dataset",
            "fixture/data",
            "--adapter-out",
            "fixture/private-adapter",
            "--hf-token",
            "unit-test-not-a-credential",
            *extra,
        ]
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch.object(sys, "argv", argv),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            try:
                return self.worker.main()
            except SystemExit as error:
                return error.code
            finally:
                self.stdout, self.stderr = output.getvalue(), errors.getvalue()

    def assert_incomplete(self, expected_event):
        self.assertEqual(self.run_worker(), 7)
        self.assertEqual(self.saved, [])
        self.assertEqual(self.uploads, [])
        self.assertFalse(any(e["event"] == "complete" for e in self.events))
        terminal = self.events[-1]
        self.assertEqual(terminal["event"], expected_event)
        self.assertEqual(terminal["target_steps"], 200)
        self.assertIs(terminal["completion_qualified"], False)

    def test_direct_script_resolves_resume_verifier(self):
        root = WORKER.parents[2]
        isolated = [path for path in sys.path if Path(path or ".").resolve() != root]
        with patch.object(sys, "path", isolated):
            self.assertNotIn(str(root), sys.path)
            spec = importlib.util.spec_from_file_location(
                "salad_direct_script_launch_under_test", WORKER
            )
            direct = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(direct)
            self.assertIn(str(root), sys.path)

            class Hub:
                def model_info(self, repo_id, *, files_metadata, timeout):
                    return types.SimpleNamespace(
                        sha="a" * 40, private=True, siblings=[]
                    )

            self.assertIsNone(direct.latest_checkpoint(Hub(), "fixture/private"))

    def test_modern_recipe_keeps_explicit_sequence_bound(self):
        self.assertEqual(self.run_worker(), 0)
        self.assertEqual(self.config_kwargs["max_length"], 2048)
        self.assertNotIn("max_seq_length", self.config_kwargs)

    def test_legacy_pinned_image_uses_legacy_explicit_field(self):
        def legacy(*, max_seq_length, **kwargs):
            fixture = {"max_seq_length": max_seq_length, **kwargs}
            self.config_kwargs = fixture
            return types.SimpleNamespace(**fixture)

        self.assertEqual(
            self.worker.sft_sequence_length_kwargs(legacy, 2048),
            {"max_seq_length": 2048},
        )
        with patch.object(self.worker, "SFTConfig", legacy):
            self.assertEqual(self.run_worker(), 0)
        self.assertEqual(self.config_kwargs["max_seq_length"], 2048)
        self.assertNotIn("max_length", self.config_kwargs)

    def test_unknown_trl_signature_stops_before_any_provider_effect(self):
        # A broken training runtime must not initialize even the Hub client or
        # private adapter repository. A published receipt is not fabricated.
        for constructor in (
            lambda **kwargs: None,
            lambda *, max_length=None, max_seq_length=None: None,
        ):
            with self.subTest(signature=str(constructor)):
                self.api_calls.clear()
                self.events.clear()
                with (
                    patch.object(self.worker, "SFTConfig", constructor),
                    patch.object(
                        self.worker,
                        "ReceiptChannel",
                        side_effect=AssertionError("unexpected channel"),
                    ),
                ):
                    self.assertEqual(self.run_worker(), 8)
                self.assertEqual(self.api_calls, [])
                self.assertEqual(self.events, [])
                self.assertEqual(self.uploads, [])
                self.assertEqual(self.receipt_uploads, [])
                self.assertEqual(self.loaded, [])
                self.assertEqual(self.trained, [])
                self.assertIn("SFTCONFIG_API_CONTRACT_UNSUPPORTED", self.stderr)
                self.assertNotIn("unit-test-not-a-credential", self.stderr)

    def test_ambiguous_or_kwargs_only_sft_api_fails_closed(self):
        for constructor in (
            lambda **kwargs: None,
            lambda *, max_length=None, max_seq_length=None: None,
        ):
            with self.subTest(signature=str(constructor)):
                with self.assertRaisesRegex(RuntimeError, "exactly one explicit"):
                    self.worker.sft_sequence_length_kwargs(constructor, 2048)

    def test_interrupted_run_cannot_publish_final_or_claim_target(self):
        self.actual, self.interrupted = 50, True
        self.assert_incomplete("interrupted")
        self.assertEqual(self.events[-1]["actual_steps"], 50)

    def test_early_stop_without_signal_is_incomplete(self):
        self.actual = 50
        self.assert_incomplete("incomplete")
        self.assertEqual(self.events[-1]["actual_steps"], 50)

    def test_overshoot_is_not_an_exact_target_completion(self):
        self.actual = 201
        self.assert_incomplete("incomplete")

    def test_missing_trainer_step_is_unknown_not_requested_target(self):
        self.actual = None
        self.assert_incomplete("incomplete")
        self.assertIsNone(self.events[-1]["actual_steps"])

    def test_boolean_trainer_step_is_not_a_measured_count(self):
        self.actual = True
        self.assert_incomplete("incomplete")
        self.assertIsNone(self.events[-1]["actual_steps"])

    def test_string_trainer_step_is_not_a_measured_count(self):
        self.actual = "200"
        self.assert_incomplete("incomplete")

    def test_float_trainer_step_is_not_a_measured_count(self):
        self.actual = 200.0
        self.assert_incomplete("incomplete")

    def test_negative_trainer_step_is_not_a_measured_count(self):
        self.actual = -1
        self.assert_incomplete("incomplete")

    def test_exact_target_publishes_and_records_actual_not_just_requested(self):
        self.assertEqual(self.run_worker(), 0)
        self.assertEqual(len(self.saved), 1)
        self.assertEqual(len(self.uploads), 1)
        terminal = self.events[-1]
        self.assertEqual(terminal["event"], "complete")
        self.assertEqual(terminal["steps"], 200)
        self.assertEqual(terminal["actual_steps"], 200)
        self.assertEqual(terminal["target_steps"], 200)
        self.assertIs(terminal["completion_qualified"], True)
        self.assertIn("*.pt", self.uploads[0]["ignore_patterns"])

    def test_signal_at_exact_target_preserves_observed_completion(self):
        self.interrupted = True
        self.assertEqual(self.run_worker(), 0)
        self.assertEqual(self.events[-1]["event"], "complete")

    def test_upload_failure_never_emits_successful_completion(self):
        self.upload_error = OSError("synthetic-sensitive-url-must-not-be-logged")
        self.assertEqual(self.run_worker(), 9)
        self.assertNotIn("synthetic-sensitive", self.stderr + self.stdout)
        self.assertFalse(any(e["event"] == "complete" for e in self.events))

    def test_optional_artifact_failure_is_sanitized_without_completion(self):
        self.fail_upload_repo = "fixture/second-private"
        self.upload_error = OSError("synthetic-sensitive-url-must-not-be-logged")
        self.assertEqual(
            self.run_worker("--publish-final-to", self.fail_upload_repo), 9
        )
        self.assertEqual(len(self.uploads), 2)
        self.assertNotIn("synthetic-sensitive", self.stderr + self.stdout)
        self.assertFalse(any(e["event"] == "complete" for e in self.events))

    def make_checkpoint(self, *names):
        # v3.10: the F10 restore branch reads terminal checkpoint bytes, so the
        # restore tests need a real directory instead of a bare string path.
        checkpoint = Path(self.temp.name) / "checkpoint-200"
        checkpoint.mkdir(parents=True, exist_ok=True)
        for name in names:
            (checkpoint / name).write_bytes(b"synthetic-" + name.encode())
        return checkpoint

    def test_resume_without_final_and_missing_checkpoint_directory_fails_closed(self):
        self.final_present = False
        with (
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value="checkpoint-200"
            ),
            patch.object(self.worker, "resume_global_step", return_value=200),
        ):
            self.assertEqual(self.run_worker(), 7)
        self.assertEqual(self.trained, [])
        self.assertEqual(self.saved, [])
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.events[-1]["event"], "fail-closed")
        self.assertIs(self.events[-1]["final_adapter_present"], False)
        self.assertFalse(any(e["event"] == "complete" for e in self.events))

    def test_resume_without_final_and_without_adapter_weights_fails_closed(self):
        self.final_present = False
        checkpoint = self.make_checkpoint("adapter_config.json", "optimizer.pt")
        with (
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value=str(checkpoint)
            ),
            patch.object(self.worker, "resume_global_step", return_value=200),
        ):
            self.assertEqual(self.run_worker(), 7)
        self.assertEqual(self.trained, [])
        self.assertEqual(self.saved, [])
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.events[-1]["event"], "fail-closed")
        self.assertIs(self.events[-1]["final_adapter_present"], False)
        self.assertFalse(any(e["event"] == "complete" for e in self.events))

    def test_resume_without_final_restores_terminal_checkpoint_bytes_as_final(self):
        self.final_present = False
        checkpoint = self.make_checkpoint(
            "adapter_model.safetensors",
            "adapter_config.json",
            "training_format.json",
            "optimizer.pt",
            "scheduler.pt",
        )
        with (
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value=str(checkpoint)
            ),
            patch.object(self.worker, "resume_global_step", return_value=200),
        ):
            self.assertEqual(self.run_worker(), 0)
        self.assertEqual(self.trained, [])
        self.assertEqual(self.saved, [])
        terminal = self.events[-1]
        self.assertEqual(terminal["event"], "already-complete")
        self.assertIs(terminal["final_adapter_present"], True)
        self.assertIs(terminal["new_steps_this_boot"], 0)
        restored = terminal["final_restored"]
        self.assertEqual(restored["restored_from"], "checkpoint-200")
        self.assertEqual(
            sorted(restored["files"]),
            [
                "adapter_config.json",
                "adapter_model.safetensors",
                "training_format.json",
            ],
        )
        self.assertEqual(
            restored["scope"],
            "TERMINAL_CHECKPOINT_BYTES_REPUBLISHED_NOT_NEW_TRAINING",
        )
        self.assertEqual(restored["final_revision"], "a" * 40)
        self.assertEqual(len(self.uploads), 1)
        upload = self.uploads[0]
        self.assertEqual(upload["path_in_repo"], "final")
        self.assertEqual(upload["repo_id"], "fixture/private-adapter")
        self.assertIn("*.pt", upload["ignore_patterns"])
        staging = Path(upload["folder_path"])
        self.assertFalse(staging.exists(), "exclusive staging must be cleaned")
        self.assertEqual(
            sorted(self.uploaded_file_snapshots[0]),
            [
                "adapter_config.json",
                "adapter_model.safetensors",
                "training_format.json",
            ],
        )
        self.assertEqual(
            self.uploaded_file_snapshots[0]["adapter_model.safetensors"],
            b"synthetic-adapter_model.safetensors",
        )
        self.assertEqual(restored["staging"], "EXCLUSIVE_CLEAN_CHECKPOINT_ONLY")
        self.assertEqual(
            restored["artifact_readback"],
            "HUB_COMMIT_ACKNOWLEDGED_NOT_FULL_BYTE_READBACK",
        )
        self.assertFalse(any(e["event"] == "complete" for e in self.events))

    def test_stale_local_final_is_not_uploaded_or_deleted(self):
        self.final_present = False
        checkpoint = self.make_checkpoint(
            "adapter_model.safetensors", "adapter_config.json"
        )
        old_final = Path(self.temp.name) / "ckpt" / "final"
        old_final.mkdir(parents=True, exist_ok=True)
        stale = old_final / "stale-other-run.json"
        stale.write_bytes(b"DO_NOT_UPLOAD_OR_DELETE")
        with (
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value=str(checkpoint)
            ),
            patch.object(self.worker, "resume_global_step", return_value=200),
        ):
            self.assertEqual(self.run_worker(), 0)
        self.assertEqual(len(self.uploads), 1)
        self.assertNotIn("stale-other-run.json", self.uploaded_file_snapshots[0])
        self.assertEqual(stale.read_bytes(), b"DO_NOT_UPLOAD_OR_DELETE")
        self.assertFalse(Path(self.uploads[0]["folder_path"]).exists())
        self.assertEqual(self.trained, [])

    def test_terminal_checkpoint_symlink_is_refused_before_upload(self):
        self.final_present = False
        checkpoint = self.make_checkpoint("adapter_model.safetensors")
        original = self.worker.os.path.islink

        def fake_symlink(path):
            return str(path).endswith("adapter_model.safetensors") or original(path)

        with (
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value=str(checkpoint)
            ),
            patch.object(self.worker, "resume_global_step", return_value=200),
            patch.object(self.worker.os.path, "islink", side_effect=fake_symlink),
        ):
            self.assertEqual(self.run_worker(), 7)
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.trained, [])
        self.assertEqual(self.events[-1]["recovery_refusal"], "ResumeError")
        self.assertFalse(any(x["event"] == "complete" for x in self.events))

    def test_repo_visibility_flip_refuses_terminal_upload(self):
        self.final_present = False
        self.flip_visibility_on_list = True
        checkpoint = self.make_checkpoint("adapter_model.safetensors")
        with (
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value=str(checkpoint)
            ),
            patch.object(self.worker, "resume_global_step", return_value=200),
        ):
            self.assertEqual(self.run_worker(), 3)
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.trained, [])
        self.assertEqual(self.events[-1]["recovery_refusal"], "VisibilityError")
        self.assertFalse(any(x["event"] == "complete" for x in self.events))

    def test_final_restore_upload_failure_is_sanitized_artifact_failure(self):
        self.final_present = False
        self.upload_error = OSError("synthetic-sensitive-url-must-not-be-logged")
        checkpoint = self.make_checkpoint("adapter_model.safetensors")
        with (
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value=str(checkpoint)
            ),
            patch.object(self.worker, "resume_global_step", return_value=200),
        ):
            self.assertEqual(self.run_worker(), 9)
        self.assertEqual(self.trained, [])
        self.assertEqual(self.saved, [])
        self.assertNotIn("synthetic-sensitive", self.stderr + self.stdout)
        self.assertFalse(any(e["event"] == "complete" for e in self.events))

    def test_resume_at_target_with_final_does_not_retrain(self):
        with (
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value="checkpoint-200"
            ),
            patch.object(self.worker, "resume_global_step", return_value=200),
            patch.object(
                self.worker,
                "cuda_arch_precheck",
                side_effect=AssertionError("terminal resume must not probe CUDA"),
            ) as precheck,
        ):
            self.assertEqual(self.run_worker(), 0)
        precheck.assert_not_called()
        self.assertEqual(self.trained, [])
        self.assertEqual(self.events[-1]["event"], "already-complete")

    def test_cuda_precheck_failure_prevents_model_load_and_training(self):
        with (
            patch.object(self.worker, "cuda_arch_precheck", side_effect=SystemExit(4)),
            patch.object(
                self.worker.AutoModelForCausalLM, "from_pretrained"
            ) as load_model,
        ):
            self.assertEqual(self.run_worker(), 4)
        load_model.assert_not_called()
        self.assertEqual(self.trained, [])
        self.assertEqual(self.uploads, [])

    def test_ambiguous_prompt_join_fails_before_model_load_or_training(self):
        class AmbiguousDataset(Dataset):
            column_names = ["prompt", "completion"]

            def map(self, function, remove_columns, **kwargs):
                self.assert_cache_disabled = kwargs["load_from_cache_file"] is False
                function({"prompt": "Complete the proof := by", "completion": "by rfl"})
                return self

        dataset = AmbiguousDataset()
        with (
            patch.object(self.worker, "load_train_split", return_value=dataset),
            patch.object(self.worker.AutoModelForCausalLM, "from_pretrained") as load,
        ):
            self.assertEqual(self.run_worker(), 9)
        load.assert_not_called()
        self.assertEqual(self.trained, [])
        self.assertEqual(self.uploads, [])
        self.assertTrue(dataset.assert_cache_disabled)
        self.assertFalse(self.events[-1]["training_started"])
        self.assertNotIn("Complete the proof", json.dumps(self.events))

    def test_chat_conversion_records_and_saves_checkpoint_format(self):
        rendered = []

        class PromptDataset(Dataset):
            column_names = ["prompt", "completion"]

            def map(self, function, remove_columns, **kwargs):
                rendered.append(
                    function(
                        {"prompt": "Complete the proof := by", "completion": "by rfl"}
                    )
                )
                return self

        tokenizer = types.SimpleNamespace(
            get_chat_template=lambda: "synthetic template",
            apply_chat_template=lambda *a, **kw: {
                "input_ids": [100, 200, 300],
                "attention_mask": [1, 1, 1],
            },
        )
        with (
            patch.object(self.worker, "load_train_split", return_value=PromptDataset()),
            patch.object(
                self.worker.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
        ):
            self.assertEqual(self.run_worker("--prompt-completion-format", "chat"), 0)
        self.assertEqual(
            rendered, [{"input_ids": [100, 200, 300], "attention_mask": [1, 1, 1]}]
        )
        dataset_event = next(e for e in self.events if e["event"] == "dataset")
        contract = dataset_event["training_format"]
        self.assertEqual(contract["mode"], "prompt_completion_chat")
        self.assertEqual(self.events[-1]["training_format"], contract)
        final = Path(self.saved[0]) / "training_format.json"
        self.assertEqual(json.loads(final.read_text()), contract)

        output = Path(self.trainer_configs[0].output_dir)
        checkpoint = output / "checkpoint-50"
        checkpoint.mkdir(parents=True)
        self.callbacks[0].on_save(
            types.SimpleNamespace(output_dir=str(output)),
            types.SimpleNamespace(global_step=50),
            types.SimpleNamespace(should_training_stop=False),
        )
        self.assertEqual(
            json.loads((checkpoint / "training_format.json").read_text()), contract
        )
        self.assertEqual(self.uploads[-1]["folder_path"], str(checkpoint))

    def test_chat_conversion_cannot_resume_an_unmarked_legacy_checkpoint(self):
        class PromptDataset(Dataset):
            column_names = ["prompt", "completion"]

        with (
            patch.object(self.worker, "load_train_split", return_value=PromptDataset()),
            patch.object(
                self.worker.AutoTokenizer,
                "from_pretrained",
                return_value=types.SimpleNamespace(
                    get_chat_template=lambda: "template"
                ),
            ),
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value=self.temp.name
            ),
            patch.object(self.worker, "resume_global_step", return_value=50),
            patch.object(self.worker.AutoModelForCausalLM, "from_pretrained") as load,
        ):
            self.assertEqual(self.run_worker("--prompt-completion-format", "chat"), 9)
        load.assert_not_called()
        self.assertEqual(self.trained, [])
        self.assertEqual(self.uploads, [])

    def test_formatting_exception_cannot_echo_private_row_content(self):
        class FailingTemplateDataset(Dataset):
            def map(self, function, remove_columns, **kwargs):
                raise RuntimeError("SYNTHETIC_PRIVATE_ROW_CONTENT")

        with (
            patch.object(
                self.worker, "load_train_split", return_value=FailingTemplateDataset()
            ),
            patch.object(self.worker.AutoModelForCausalLM, "from_pretrained") as load,
        ):
            self.assertEqual(self.run_worker(), 9)
        load.assert_not_called()
        self.assertNotIn("SYNTHETIC_PRIVATE_ROW_CONTENT", json.dumps(self.events))
        self.assertEqual(self.trained, [])

    def test_nonpositive_limits_fail_before_api_effects(self):
        for flag in (
            "--max-steps",
            "--save-steps",
            "--max-examples",
            "--min-examples",
            "--heartbeat-seconds",
        ):
            with self.subTest(flag=flag):
                self.api_calls.clear()
                self.assertEqual(self.run_worker(flag, "0"), 2)
                self.assertEqual(self.api_calls, [])

    def test_invalid_shared_gpu_limits_fail_before_api_effects(self):
        for flag in ("--gpu-memory-budget-mib", "--gpu-reserve-mib"):
            with self.subTest(flag=flag):
                self.assertEqual(self.run_worker(flag, "-1"), 2)
                self.assertEqual(self.api_calls, [])

    def test_mutable_explicit_revision_is_refused_before_api_effects(self):
        for flag in ("--base-revision", "--dataset-revision"):
            with self.subTest(flag=flag):
                self.assertEqual(self.run_worker(flag, "main"), 2)
                self.assertEqual(self.api_calls, [])

    def test_base_revision_pins_tokenizer_and_weights(self):
        revision = "a" * 40
        with (
            patch.object(self.worker.AutoTokenizer, "from_pretrained") as tokenizer,
            patch.object(
                self.worker.AutoModelForCausalLM,
                "from_pretrained",
                return_value=self.model,
            ) as model,
        ):
            self.assertEqual(self.run_worker("--base-revision", revision), 0)
        self.assertEqual(tokenizer.call_args.kwargs["revision"], revision)
        self.assertEqual(model.call_args.kwargs["revision"], revision)
        self.assertIs(
            self.trainer_inputs[0]["processing_class"], tokenizer.return_value
        )
        self.assertIn("torch_dtype", model.call_args.kwargs)
        boot = next(event for event in self.events if event["event"] == "boot")
        self.assertEqual(boot["base_revision"], revision)

    def test_shared_gpu_refusal_prevents_model_load_and_training(self):
        with (
            patch.object(
                self.worker,
                "configure_shared_gpu_budget",
                side_effect=RuntimeError("no headroom"),
            ),
            patch.object(self.worker.AutoModelForCausalLM, "from_pretrained") as load,
        ):
            self.assertEqual(self.run_worker("--gpu-memory-budget-mib", "24576"), 8)
        load.assert_not_called()
        self.assertEqual(self.trained, [])
        self.assertEqual(self.uploads, [])

    def test_lane_output_directory_applies_to_resume_training_and_final(self):
        with patch.object(
            self.worker, "fetch_resume_checkpoint", return_value=None
        ) as resume:
            self.assertEqual(self.run_worker("--output-dir", "~/lean-v2/ckpt"), 0)
        expected = str(Path(self.temp.name) / "lean-v2" / "ckpt")
        self.assertEqual(resume.call_args.args[2], expected)
        self.assertEqual(self.trainer_configs[0].output_dir, expected)
        self.assertEqual(self.saved, [str(Path(expected) / "final")])

    def test_checkpointing_opt_in_disables_incompatible_model_cache(self):
        self.assertEqual(self.run_worker("--gradient-checkpointing"), 0)
        cfg = self.trainer_configs[0]
        self.assertTrue(cfg.gradient_checkpointing)
        self.assertEqual(cfg.gradient_checkpointing_kwargs, {"use_reentrant": False})
        self.assertFalse(self.model.config.use_cache)

    def test_stop_signal_requests_save_and_stop_at_next_optimizer_step(self):
        self.assertEqual(self.run_worker(), 0)
        callback = self.callbacks[0]
        control = types.SimpleNamespace(should_save=False, should_training_stop=False)
        callback.on_step_end(None, None, control)
        self.assertFalse(control.should_save)
        self.assertFalse(control.should_training_stop)
        self.worker.STOP["flag"] = True
        self.assertIs(callback.on_step_end(None, None, control), control)
        self.assertTrue(control.should_save)
        self.assertTrue(control.should_training_stop)

    def test_signal_handler_never_waits_for_telemetry_or_network(self):
        with patch.object(
            self.worker,
            "receipt",
            side_effect=AssertionError("signal telemetry blocked"),
        ) as telemetry:
            self.worker.on_term(signal.SIGTERM, None)
            self.worker.on_term(signal.SIGINT, None)
        telemetry.assert_not_called()
        self.assertTrue(self.worker.STOP["flag"])
        self.assertEqual(self.worker.STOP["signal"], signal.SIGTERM)

    def test_prebuilt_runtime_does_not_install_missing_energy_libraries(self):
        with (
            patch.dict(os.environ, {"SZL_DISABLE_RUNTIME_INSTALL": "1"}),
            patch.dict(
                sys.modules, {"pynvml": None, "zeus": None, "zeus.monitor": None}
            ),
            patch(
                "subprocess.run",
                side_effect=AssertionError("runtime installation forbidden"),
            ),
        ):
            self.assertIsNone(self.worker._ensure_pynvml())
            self.assertIsNone(self.worker._ensure_zeus())

    def test_max_examples_below_minimum_cannot_silently_train_stub(self):
        self.assertEqual(
            self.run_worker("--max-examples", "1", "--min-examples", "8"), 2
        )
        self.assertEqual(self.api_calls, [])

    def test_receipt_keeps_schema_and_machine_readable_event(self):
        with redirect_stdout(io.StringIO()):
            value = self.original_receipt(
                "interrupted", "synthetic event", {"actual_steps": 50}
            )
        self.assertEqual(value["kind"], "szl.salad-worker/v1")
        self.assertEqual(value["event"], "interrupted")
        digest = value.pop("receipt_sha256")
        self.assertEqual(
            digest,
            hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(),
        )

    def test_extra_fields_cannot_override_receipt_identity(self):
        with redirect_stdout(io.StringIO()):
            value = self.original_receipt(
                "incomplete",
                "truth",
                {
                    "kind": "forged",
                    "event": "complete",
                    "detail": "false",
                    "ts": "yesterday",
                    "receipt_sha256": "fabricated",
                },
            )
        self.assertEqual(value["kind"], "szl.salad-worker/v1")
        self.assertEqual(value["event"], "incomplete")
        self.assertEqual(value["detail"], "truth")
        self.assertNotEqual(value["ts"], "yesterday")
        digest = value.pop("receipt_sha256")
        self.assertEqual(
            digest,
            hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(),
        )

    def test_visibility_loss_blocks_final_and_success_receipt(self):
        self.lose_visibility_on_train = True
        self.assertEqual(self.run_worker(), 3)
        self.assertEqual(self.uploads, [])
        self.assertFalse(any(e["event"] == "complete" for e in self.events))

    def test_private_check_still_precedes_training(self):
        self.private = False
        self.assertEqual(self.run_worker(), 3)
        self.assertEqual(self.trained, [])
        self.assertEqual(self.uploads, [])

    def test_interruption_closes_energy_window_without_final(self):
        self.actual, self.interrupted = 50, True
        with patch.object(
            self.worker, "zeus_end_window", return_value={"label": "UNAVAILABLE"}
        ) as close:
            self.assertEqual(self.run_worker(), 7)
        close.assert_called_once()
        self.assertEqual(self.uploads, [])

    def test_incomplete_run_cannot_publish_to_optional_target(self):
        self.actual = 50
        self.assertEqual(
            self.run_worker("--publish-final-to", "fixture/second-private"), 7
        )
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.saved, [])

    def test_missing_final_closes_energy_window_without_retraining(self):
        self.final_present = False
        checkpoint = self.make_checkpoint("adapter_config.json")
        with (
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value=str(checkpoint)
            ),
            patch.object(self.worker, "resume_global_step", return_value=200),
            patch.object(
                self.worker, "zeus_end_window", return_value={"label": "UNAVAILABLE"}
            ) as close,
        ):
            self.assertEqual(self.run_worker(), 7)
        close.assert_called_once()
        self.assertEqual(self.trained, [])

    def channel(self, repo_type="dataset"):
        channel = self.worker.ReceiptChannel(
            self.worker.HfApi(),
            "unit-test-not-a-credential",
            "fixture/receipts",
            repo_type,
            self.temp.name,
            "fixture/private-adapter",
        )
        self.worker.RECEIPTS = channel
        return channel

    def real_run(self, *extra):
        with patch.object(self.worker, "receipt", self.real_receipt):
            return self.run_worker(*extra)

    def published_events(self):
        return [
            value
            for line in self.stdout.splitlines()
            if line.startswith("{")
            for value in [json.loads(line)]
            if value.get("kind") == "szl.salad-worker/v1"
        ]

    def test_unconfigured_or_missing_token_cannot_emit_local_success(self):
        with redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(self.worker.ReceiptError):
                self.real_receipt("complete", "unpublished")
        self.assertEqual(output.getvalue(), "")
        with self.assertRaises(self.worker.ReceiptError):
            self.worker.ReceiptChannel(None, "", "fixture/r", "model", ".", "fixture/a")
        self.assertEqual(self.receipt_uploads, [])

    def test_receipt_outbox_cannot_be_placed_in_final_artifact_directory(self):
        self.assertEqual(self.real_run("--receipt-outbox", "~/ckpt/final"), 8)
        self.assertEqual(self.receipt_uploads, [])
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.loaded, [])

    def test_custom_output_directory_cannot_contain_receipt_outbox(self):
        self.assertEqual(
            self.run_worker(
                "--output-dir",
                "~/lane/ckpt",
                "--receipt-outbox",
                "~/lane/ckpt/final/receipts",
            ),
            8,
        )
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.receipt_uploads, [])
        self.assertIn("outside the artifact directory", self.stderr)

    def test_default_outbox_is_outside_all_artifact_uploads(self):
        self.assertEqual(self.real_run(), 0)
        for upload in self.uploads:
            self.assertFalse(
                self.worker.RECEIPTS.outbox.resolve().is_relative_to(
                    Path(upload["folder_path"]).resolve()
                )
            )

    def test_outbox_symlink_into_artifacts_is_refused(self):
        target = Path(self.temp.name) / "ckpt" / "final"
        target.mkdir(parents=True)
        link = Path(self.temp.name) / "outside-link"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest("this environment does not allow symlink creation")
        self.assertEqual(self.real_run("--receipt-outbox", str(link)), 8)
        self.assertEqual(self.receipt_uploads, [])
        self.assertEqual(self.uploads, [])

    def test_cli_token_is_used_for_receipts_without_environment_token(self):
        self.assertEqual(self.real_run(), 0)
        self.assertEqual(
            [e["event"] for e in self.published_events()],
            ["boot", "dataset", "complete"],
        )
        self.assertTrue(self.receipt_downloads)
        self.assertTrue(
            all(
                d["token"] == "unit-test-not-a-credential"
                for d in self.receipt_downloads
            )
        )

    def test_boot_permission_failure_stops_before_model_loading(self):
        self.fail_receipt_event = "boot"
        self.assertEqual(self.real_run(), 8)
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.trained, [])
        self.assertEqual(self.uploads, [])
        self.assertEqual(len(self.receipt_uploads), 1)
        self.assertEqual(self.published_events(), [])
        self.assertNotIn("synthetic-sensitive", self.stderr + self.stdout)
        self.sleep.assert_not_called()

    def test_failed_final_receipt_keeps_adapter_but_cannot_claim_success(self):
        self.fail_receipt_event = "complete"
        self.assertEqual(self.real_run(), 8)
        self.assertEqual([u["path_in_repo"] for u in self.uploads], ["final"])
        self.assertFalse(any(e["event"] == "complete" for e in self.published_events()))
        pending = list(self.worker.RECEIPTS.outbox.glob("*.pending.json"))
        self.assertEqual(len(pending), 3)
        self.assertEqual(
            len(list(self.worker.RECEIPTS.outbox.glob("*.publication.json"))), 2
        )
        self.assertNotIn("synthetic-sensitive", self.stderr + self.stdout)

    def test_missing_commit_revision_is_not_accepted_as_publication(self):
        self.receipt_revision = None
        self.assertEqual(self.real_run(), 8)
        self.assertEqual(self.receipt_downloads, [])
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.published_events(), [])

    def test_wrong_bytes_fail_without_success_or_another_upload(self):
        self.corrupt_readback = True
        self.assertEqual(self.real_run(), 8)
        self.assertEqual(len(self.receipt_uploads), 1)
        self.assertEqual(len(self.receipt_downloads), 1)
        self.assertEqual(self.published_events(), [])
        self.sleep.assert_not_called()

    def test_readback_outage_is_bounded_and_does_not_repeat_upload(self):
        self.readback_error = OSError("synthetic-sensitive-url-must-not-be-logged")
        self.assertEqual(self.real_run(), 8)
        self.assertEqual(len(self.receipt_uploads), 1)
        # v3.9: the transient-retry budget is 6 attempts (5 exponential backoffs
        # capped at 240s) so a multi-minute Hub 429 storm cannot fail-close a
        # whole run; the outage stays bounded and never repeats the upload.
        self.assertEqual(len(self.receipt_downloads), 6)
        self.assertEqual(self.sleep.call_count, 5)
        self.assertEqual(self.published_events(), [])
        self.assertNotIn("synthetic-sensitive", self.stderr + self.stdout)

    def test_transient_upload_retries_identical_bytes_and_reads_exact_commit(self):
        self.channel()
        error = OSError("synthetic rate limit")
        error.response = types.SimpleNamespace(status_code=429)
        self.receipt_failures = [error, error]
        with redirect_stdout(io.StringIO()):
            value = self.real_receipt(
                "heartbeat", "step observed", {"actual_steps": 50}
            )
        self.assertEqual(len(self.receipt_uploads), 3)
        self.assertEqual(len({u["path_in_repo"] for u in self.receipt_uploads}), 1)
        self.assertEqual(len({u["path_or_fileobj"] for u in self.receipt_uploads}), 1)
        self.assertIn("b" * 40, self.receipt_downloads[0]["url"])
        self.assertIn("/resolve/", self.receipt_downloads[0]["url"])
        self.assertIn(value["receipt_sha256"], self.receipt_downloads[0]["url"])

    def test_explicit_private_model_channel_uses_only_named_destination(self):
        self.assertEqual(
            self.real_run(
                "--receipts-repo",
                "fixture/private-adapter",
                "--receipts-repo-type",
                "model",
            ),
            0,
        )
        self.assertTrue(
            all(
                u["repo_id"] == "fixture/private-adapter" and u["repo_type"] == "model"
                for u in self.receipt_uploads
            )
        )

    def test_model_channel_refuses_public_repo_without_upload_or_fallback(self):
        self.channel("model")
        self.private = False
        with self.assertRaises(self.worker.ReceiptError):
            self.real_receipt("boot", "synthetic")
        self.assertEqual(self.receipt_uploads, [])

    def test_model_visibility_loss_after_upload_withholds_acknowledgement(self):
        self.flip_receipt_visibility = True
        self.assertEqual(
            self.real_run(
                "--receipts-repo",
                "fixture/private-adapter",
                "--receipts-repo-type",
                "model",
            ),
            8,
        )
        self.assertEqual(len(self.receipt_uploads), 1)
        self.assertEqual(len(self.receipt_downloads), 1)
        self.assertEqual(self.published_events(), [])
        self.assertEqual(self.loaded, [])

    def test_heartbeat_failure_requests_checkpoint_and_prevents_final_even_at_target(
        self,
    ):
        self.invoke_heartbeat = True
        self.fail_receipt_event = "heartbeat"
        self.assertEqual(self.real_run(), 8)
        self.assertIs(self.control.should_training_stop, True)
        self.assertIs(self.control.should_save, True)
        self.assertEqual([u["path_in_repo"] for u in self.uploads], ["checkpoint-200"])
        self.assertEqual(self.saved, [])
        self.assertFalse(any(e["event"] == "complete" for e in self.published_events()))
        self.assertIs(self.published_events()[-1]["receipt_channel_failed"], True)

    def test_checkpoint_receipt_failure_requests_stop_without_final(self):
        self.invoke_checkpoint = True
        self.fail_receipt_event = "checkpoint"
        self.assertEqual(self.real_run(), 8)
        self.assertIs(self.control.should_training_stop, True)
        self.assertEqual([u["path_in_repo"] for u in self.uploads], ["checkpoint-200"])
        self.assertFalse(any(e["event"] == "complete" for e in self.published_events()))

    def test_checkpoint_upload_failure_stops_and_redacts_provider_exception(self):
        self.invoke_checkpoint = True
        self.upload_error = OSError("synthetic-sensitive-url-must-not-be-logged")
        self.assertEqual(self.real_run(), 7)
        self.assertIs(self.control.should_training_stop, True)
        self.assertEqual(self.saved, [])
        self.assertNotIn("synthetic-sensitive", self.stderr + self.stdout)
        self.assertIs(
            self.published_events()[-1]["checkpoint_publication_failed"], True
        )

    def test_verified_terminal_receipt_binds_artifact_and_publication_commits(self):
        self.assertEqual(self.real_run(), 0)
        complete = self.published_events()[-1]
        self.assertEqual(complete["final_artifact"]["revision"], "a" * 40)
        self.assertEqual(complete["final_artifact"]["directory"], "final")
        ack_path = (
            self.worker.RECEIPTS.outbox
            / f"{complete['receipt_sha256']}.publication.json"
        )
        ack = json.loads(ack_path.read_text())
        self.assertEqual(ack["revision"], "b" * 40)
        raw = self.receipt_bytes[ack["path"]]
        self.assertEqual(ack["file_sha256"], hashlib.sha256(raw).hexdigest())
        digest = complete.pop("receipt_sha256")
        self.assertEqual(
            digest,
            hashlib.sha256(json.dumps(complete, sort_keys=True).encode()).hexdigest(),
        )

    def test_heartbeat_is_durable_liveness_without_checkpoint_claim(self):
        self.invoke_heartbeat = True
        self.assertEqual(self.real_run(), 0)
        heartbeat = next(
            e for e in self.published_events() if e["event"] == "heartbeat"
        )
        self.assertEqual(heartbeat["actual_steps"], 200)
        self.assertEqual(
            heartbeat["scope"], "TRAINER_LIVENESS_NOT_CHECKPOINT_DURABILITY"
        )
        self.assertIs(self.control.should_training_stop, False)

    def test_signal_handler_never_performs_blocking_receipt_io(self):
        with redirect_stdout(io.StringIO()):
            self.worker.on_term(signal.SIGTERM, None)
        self.assertIs(self.worker.STOP["flag"], True)
        self.assertEqual(self.events, [])
        self.assertEqual(self.receipt_uploads, [])

    def test_already_complete_still_requires_durable_terminal_receipt(self):
        self.fail_receipt_event = "already-complete"
        with (
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value="checkpoint-200"
            ),
            patch.object(self.worker, "resume_global_step", return_value=200),
        ):
            self.assertEqual(self.real_run(), 8)
        self.assertEqual(self.trained, [])
        self.assertEqual(self.uploads, [])
        self.assertFalse(
            any(e["event"] == "already-complete" for e in self.published_events())
        )


if __name__ == "__main__":
    unittest.main()
