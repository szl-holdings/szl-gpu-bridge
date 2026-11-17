"""Exercise real worker terminal branches with synthetic, network-denied adapters."""

from contextlib import ExitStack, redirect_stdout
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
        self.saved = []
        self.trained = []
        self.api_calls = []
        self.trainer_configs = []
        self.trainer_inputs = []
        self.callbacks = []
        self.actual = 200
        self.interrupted = False
        self.final_present = True
        self.upload_error = None
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
                return (
                    ["final/adapter_model.safetensors"] if fixture.final_present else []
                )

            def upload_folder(self, **kwargs):
                fixture.uploads.append(kwargs)
                if fixture.upload_error:
                    raise fixture.upload_error

        class Trainer:
            def __init__(self, **kwargs):
                self.state = types.SimpleNamespace(global_step=fixture.actual)
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
            from_pretrained=lambda *a, **k: self.model
        )
        modules["transformers"].TrainerCallback = object
        modules["trl"].SFTConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
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
        self.original_receipt = self.worker.receipt
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
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            try:
                return self.worker.main()
            except SystemExit as error:
                return error.code

    def assert_incomplete(self, expected_event):
        self.assertEqual(self.run_worker(), 7)
        self.assertEqual(self.saved, [])
        self.assertEqual(self.uploads, [])
        self.assertFalse(any(e["event"] == "complete" for e in self.events))
        terminal = self.events[-1]
        self.assertEqual(terminal["event"], expected_event)
        self.assertEqual(terminal["target_steps"], 200)
        self.assertIs(terminal["completion_qualified"], False)

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
        self.upload_error = OSError("synthetic upload failure")
        with self.assertRaises(OSError):
            self.run_worker()
        self.assertFalse(any(e["event"] == "complete" for e in self.events))

    def test_resume_at_target_without_final_is_failure(self):
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
        self.assertIs(self.events[-1]["final_adapter_present"], False)
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
        for flag in ("--max-steps", "--save-steps", "--max-examples", "--min-examples"):
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
        with (
            patch.object(
                self.worker, "fetch_resume_checkpoint", return_value="checkpoint-200"
            ),
            patch.object(self.worker, "resume_global_step", return_value=200),
            patch.object(
                self.worker, "zeus_end_window", return_value={"label": "UNAVAILABLE"}
            ) as close,
        ):
            self.assertEqual(self.run_worker(), 7)
        close.assert_called_once()
        self.assertEqual(self.trained, [])


if __name__ == "__main__":
    unittest.main()
