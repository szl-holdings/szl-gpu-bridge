"""CUDA admission contract tests; synthetic evidence is never a GPU qualification."""

import ast
from contextlib import nullcontext
import math
from pathlib import Path
import re
import types
import unittest
from unittest.mock import Mock


WORKER = Path(__file__).resolve().parents[1] / "tools" / "salad" / "salad_worker.py"
FUNCTIONS = {"_compatible_cuda_cubin", "_cuda_bf16_smoke", "cuda_arch_precheck"}


def load_functions(torch):
    """Load real admission code without importing training or network dependencies."""
    source = ast.parse(WORKER.read_text(encoding="utf-8"))
    module = ast.Module(
        body=[
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "re": re, "receipt": Mock()}
    exec(compile(module, str(WORKER), "exec"), namespace)
    return namespace


class CubinCompatibility(unittest.TestCase):
    def test_documented_binary_compatibility_boundaries(self):
        compatible = load_functions(None)["_compatible_cuda_cubin"]
        cases = [
            ((8, 9), ["sm_86", "sm_90"], "sm_86"),
            ((8, 9), ["sm_80", "sm_86", "sm_89"], "sm_89"),
            ((8, 6), ["sm_80", "sm_86"], "sm_86"),
            ((8, 0), ["sm_86"], None),
            ((9, 0), ["sm_86"], None),
            ((12, 0), ["sm_100", "sm_120"], "sm_120"),
            ((12, 0), ["sm_90", "sm_100"], None),
            ((10, 0), ["sm_90"], None),
            ((9, 0), ["sm_90a"], None),
            ((10, 3), ["sm_100f"], None),
            ((12, 0), ["compute_90"], None),
            ((8, 9), ["sm_89junk", "sm_８９"], None),
            ((8, 9), [], None),
        ]
        for cap, architectures, expected in cases:
            with self.subTest(cap=cap, architectures=architectures):
                self.assertEqual(compatible(cap, architectures), expected)


class AdmissionContract(unittest.TestCase):
    def setUp(self):
        self.caps = [(8, 9)]
        self.cuda = types.SimpleNamespace(
            is_available=Mock(return_value=True),
            get_arch_list=Mock(return_value=["sm_70", "sm_80", "sm_86", "sm_90"]),
            device_count=Mock(side_effect=lambda: len(self.caps)),
            get_device_capability=Mock(side_effect=lambda i: self.caps[i]),
            get_device_name=Mock(side_effect=lambda i: f"SYNTHETIC_GPU_{i}"),
        )
        torch = types.SimpleNamespace(__version__="SYNTHETIC", cuda=self.cuda)
        self.ns = load_functions(torch)
        self.smoke = Mock(return_value={"forward_backward": "SYNTHETIC_PASS"})
        self.ns["_cuda_bf16_smoke"] = self.smoke

    def assert_refused(self):
        with self.assertRaises(SystemExit) as error:
            self.ns["cuda_arch_precheck"]()
        self.assertEqual(error.exception.code, 4)
        self.assertEqual(self.ns["receipt"].call_count, 1)
        event, detail, evidence = self.ns["receipt"].call_args.args
        self.assertEqual(event, "fail-closed")
        self.assertNotIn("complete", event)
        return detail, evidence

    def test_ada_with_ampere_cubin_runs_smoke_and_records_exact_target(self):
        self.ns["cuda_arch_precheck"]()
        self.smoke.assert_called_once_with(0)
        event, detail, evidence = self.ns["receipt"].call_args.args
        self.assertEqual(event, "cuda-precheck")
        self.assertIn("whole-model qualification still required", detail)
        self.assertEqual(evidence["devices"][0]["sm"], "sm_89")
        self.assertEqual(evidence["devices"][0]["compatible_cubin"], "sm_86")

    def test_exact_architecture_still_requires_successful_runtime_probe(self):
        self.caps = [(8, 6)]
        self.smoke.side_effect = RuntimeError("no kernel image is available")
        detail, evidence = self.assert_refused()
        self.assertIn("no kernel image", detail)
        self.assertNotIn("bf16_smoke", evidence["devices"][0])

    def test_async_kernel_error_fails_closed(self):
        self.smoke.side_effect = RuntimeError("asynchronous launch failure")
        self.assertIn("asynchronous launch failure", self.assert_refused()[0])

    def test_bf16_failure_cannot_fall_back_silently_to_another_dtype(self):
        self.smoke.side_effect = RuntimeError("BF16 operation unsupported")
        self.assertIn("BF16", self.assert_refused()[0])
        self.assertEqual(self.smoke.call_count, 1)

    def test_every_device_available_to_auto_placement_is_probed(self):
        self.caps = [(8, 9), (8, 6)]
        self.ns["cuda_arch_precheck"]()
        self.assertEqual(
            [call.args for call in self.smoke.call_args_list], [(0,), (1,)]
        )
        self.assertEqual(len(self.ns["receipt"].call_args.args[2]["devices"]), 2)

    def test_bad_second_device_refuses_entire_training_admission(self):
        self.caps = [(8, 9), (12, 0)]
        detail, evidence = self.assert_refused()
        self.smoke.assert_called_once_with(0)
        self.assertIn("GPU 1", detail)
        self.assertIsNone(evidence["devices"][1]["compatible_cubin"])

    def test_blackwell_with_sm120_is_probed(self):
        self.caps = [(12, 0)]
        self.cuda.get_arch_list.return_value = ["sm_86", "sm_90", "sm_100", "sm_120"]
        self.ns["cuda_arch_precheck"]()
        self.smoke.assert_called_once_with(0)

    def test_ptx_only_is_explicitly_unqualified_without_claiming_impossibility(self):
        self.cuda.get_arch_list.return_value = ["compute_80"]
        detail, _ = self.assert_refused()
        self.assertIn("separate qualification", detail)
        self.assertNotIn("lacks", detail)
        self.smoke.assert_not_called()

    def test_no_cuda_refuses_without_allocating(self):
        self.cuda.is_available.return_value = False
        self.assert_refused()
        self.cuda.get_arch_list.assert_not_called()
        self.smoke.assert_not_called()

    def test_no_visible_devices_refuses_without_claiming_pass(self):
        self.caps = []
        self.assert_refused()
        self.smoke.assert_not_called()

    def test_device_query_failure_is_receipted_not_ignored(self):
        self.cuda.get_device_capability.side_effect = RuntimeError("driver unavailable")
        self.assertIn("driver unavailable", self.assert_refused()[0])


class Scalar:
    """Synthetic scalar for validation/error injection, not a CUDA math emulator."""

    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value

    def __eq__(self, expected):
        return Scalar(self.value == expected)


class SmokeValidation(unittest.TestCase):
    def fixture(self, loss=1.0, a_grad=1 / 64, b_grad=1 / 128, sync_error=None):
        result = Mock()
        result.float.return_value.square.return_value.mean.return_value = result
        result.item.return_value = loss
        a, b = Mock(), Mock()
        a.grad = None if a_grad is None else Scalar(a_grad)
        b.grad = None if b_grad is None else Scalar(b_grad)
        # The fixture injects externally chosen results, including corruption.
        a.__matmul__ = Mock(return_value=result)
        result.__eq__ = Mock(side_effect=lambda expected: Scalar(loss == expected))
        result.value = loss
        cuda = types.SimpleNamespace(
            device=Mock(return_value=nullcontext()),
            synchronize=Mock(side_effect=sync_error),
        )
        torch = types.SimpleNamespace(
            cuda=cuda,
            bfloat16="SYNTHETIC_BF16",
            enable_grad=nullcontext,
            full=Mock(side_effect=[a, b]),
            isfinite=lambda value: Scalar(math.isfinite(value.value)),
        )
        return load_functions(torch)["_cuda_bf16_smoke"], torch, result

    def test_correct_known_values_require_backward_and_synchronization(self):
        smoke, torch, loss = self.fixture()
        self.assertEqual(smoke(1)["forward_backward"], "PASS")
        loss.backward.assert_called_once()
        torch.cuda.synchronize.assert_called_once_with(1)
        for call in torch.full.call_args_list:
            self.assertEqual(call.kwargs["device"], "cuda:1")
            self.assertTrue(call.kwargs["requires_grad"])

    def test_nonfinite_missing_and_wrong_finite_results_are_rejected(self):
        for injected in (
            {"loss": float("nan")},
            {"loss": 2.0},
            {"a_grad": float("inf")},
            {"a_grad": None},
            {"b_grad": 0.0},
        ):
            with self.subTest(injected=injected):
                smoke, _, _ = self.fixture(**injected)
                with self.assertRaises(RuntimeError):
                    smoke(0)

    def test_synchronize_failure_cannot_return_pass(self):
        smoke, _, loss = self.fixture(sync_error=RuntimeError("kernel launch failed"))
        with self.assertRaisesRegex(RuntimeError, "kernel launch failed"):
            smoke(0)
        loss.backward.assert_called_once()


if __name__ == "__main__":
    unittest.main()
