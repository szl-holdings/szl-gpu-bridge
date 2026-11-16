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

    def map(self, function, remove_columns):
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

            def add_callback(self, callback):
                self.callback = callback

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
        modules["transformers"].AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: object()
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
        ):
            self.assertEqual(self.run_worker(), 0)
        self.assertEqual(self.trained, [])
        self.assertEqual(self.events[-1]["event"], "already-complete")

    def test_nonpositive_limits_fail_before_api_effects(self):
        for flag in ("--max-steps", "--save-steps", "--max-examples", "--min-examples"):
            with self.subTest(flag=flag):
                self.api_calls.clear()
                self.assertEqual(self.run_worker(flag, "0"), 2)
                self.assertEqual(self.api_calls, [])

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
